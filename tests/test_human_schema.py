"""Canonical human-dataset schema hardening tests (no browser).

Covers the frozen action/observation identity written to every manifest and
game, the derived-field and reward-spec markers, legacy JSONL rejection, the
strengthened validator, backward-compatible manifest upgrade, and a
model-agnostic test showing that one raw transition is enough to reconstruct a
CNN encoding, a GNN view and a quantum-style feature encoding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from minesweeper_rl.agents.policy import board_to_tensor
from minesweeper_rl.data import (
    ACTION_SPACE,
    ACTION_TYPES_MAPPING,
    DERIVED_TRANSITION_FIELDS,
    HUMAN_DATASET_FORMAT,
    HUMAN_DATASET_VERSION,
    HumanDatasetError,
    HumanDatasetStore,
    HumanDemonstrationDataset,
    PersistentHumanRecorder,
    REWARD_SPEC,
    schema,
    validate_dataset,
)
from minesweeper_rl.data.human_dataset import MANIFEST_FILENAME
from minesweeper_rl.dom import FLAGGED, HIDDEN, GameStatus

WIDTH, HEIGHT = 4, 2
CELLS = WIDTH * HEIGHT


def empty_board(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    return np.full((height, width), HIDDEN, dtype=np.int8)


def make_store(tmp_path: Path) -> HumanDatasetStore:
    return HumanDatasetStore(
        tmp_path / "human", width=WIDTH, height=HEIGHT, mines=3, difficulty="expert"
    )


def record_open_game(store: HumanDatasetStore) -> str:
    recorder = PersistentHumanRecorder(
        store, width=WIDTH, height=HEIGHT, mines=3, difficulty="expert"
    )
    recorder.start_game()
    before = empty_board()
    after = before.copy()
    after[0, 0] = 1
    recorder.record_step(
        board_before=before, action=(0, 0), board_after=after,
        game_status=GameStatus.RUNNING, terminated=False,
    )
    return recorder.end_game("abandoned").game_id


# ---------------------------------------------------------------------------
# Identity fields
# ---------------------------------------------------------------------------


def test_manifest_has_machine_readable_identity(tmp_path) -> None:
    store = make_store(tmp_path)
    manifest = store.manifest
    assert manifest["format"] == HUMAN_DATASET_FORMAT
    assert manifest["format_version"] == HUMAN_DATASET_VERSION
    assert manifest["action_space"] == ACTION_SPACE
    assert manifest["action_types"] == ACTION_TYPES_MAPPING
    assert manifest["action_types"] == {"0": "open", "1": "flag", "2": "chord"}
    assert manifest["cells"] == CELLS
    assert manifest["num_actions"] == 3 * CELLS
    assert manifest["action_encoding"] == "action_type*cells+cell"
    assert manifest["observation_representation"] == "flag_aware_v1"
    assert manifest["observation_version"] == 1
    assert manifest["reward_spec"] == REWARD_SPEC
    assert manifest["derived_fields"] == list(DERIVED_TRANSITION_FIELDS)
    assert manifest["environment"] == "minesweeper.online"
    on_disk = json.loads((tmp_path / "human" / MANIFEST_FILENAME).read_text("utf-8"))
    assert on_disk["action_space"] == ACTION_SPACE


def test_default_expert_identity_is_1440(tmp_path) -> None:
    store = HumanDatasetStore(
        tmp_path / "expert", width=30, height=16, mines=99, difficulty="expert"
    )
    assert store.manifest["cells"] == 480
    assert store.manifest["num_actions"] == 1440
    assert store.manifest["action_types"]["2"] == "chord"


def test_game_is_self_describing(tmp_path) -> None:
    store = make_store(tmp_path)
    game_id = record_open_game(store)
    game = store.load_game(game_id)
    for key in (
        "format", "format_version", "observation_representation",
        "observation_version", "action_space", "action_encoding",
        "action_types", "num_actions", "cells", "reward_spec", "derived_fields",
        "environment", "environment_version", "recorder_version",
    ):
        assert key in game, key
    assert game["action_space"] == ACTION_SPACE
    assert game["action_types"] == ACTION_TYPES_MAPPING
    assert game["num_actions"] == 3 * CELLS
    assert game["cells"] == CELLS
    # A future loader can interpret the game without importing the package.
    assert game["action_encoding"] == "action_type*cells+cell"


def test_reward_and_coverage_are_marked_derived(tmp_path) -> None:
    store = make_store(tmp_path)
    game_id = record_open_game(store)
    game = store.load_game(game_id)
    assert set(DERIVED_TRANSITION_FIELDS).issubset(set(game["derived_fields"]))
    assert game["reward_spec"] == REWARD_SPEC
    step = game["transitions"][0]
    # Present for convenience but not part of the raw, model-independent facts.
    assert "reward" in step and "newly_revealed" in step and "opened_cells" in step


def test_transition_keeps_redundant_action_information(tmp_path) -> None:
    store = make_store(tmp_path)
    game_id = record_open_game(store)
    action = store.load_game(game_id)["transitions"][0]["action"]
    assert set(action) == {"type", "action_type", "x", "y", "cell_index", "action_id"}
    assert action["cell_index"] == action["y"] * WIDTH + action["x"]
    assert action["action_id"] == action["action_type"] * CELLS + action["cell_index"]


# ---------------------------------------------------------------------------
# Legacy protection
# ---------------------------------------------------------------------------


def write_legacy_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "games.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "transitions.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "meta.json").write_text(
        json.dumps({"name": "human_expert", "schema_version": 1}), encoding="utf-8"
    )


def test_store_rejects_legacy_dataset(tmp_path) -> None:
    legacy = tmp_path / "human_expert"
    write_legacy_dataset(legacy)
    with pytest.raises(HumanDatasetError, match="not a canonical human dataset"):
        HumanDatasetStore(legacy, width=WIDTH, height=HEIGHT)


def test_loader_rejects_legacy_dataset(tmp_path) -> None:
    legacy = tmp_path / "human_expert"
    write_legacy_dataset(legacy)
    with pytest.raises(schema.DataError, match="not a canonical human dataset"):
        HumanDemonstrationDataset(legacy)


def test_loader_rejects_non_flags_action_space(tmp_path) -> None:
    store = make_store(tmp_path)
    game_id = record_open_game(store)
    path = store.games_dir / f"{game_id}.json"
    data = json.loads(path.read_text("utf-8"))
    data["action_space"] = "open_only"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(schema.DataError, match="legacy chord-as-open"):
        HumanDemonstrationDataset(tmp_path / "human")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_accepts_identity_and_derived_fields(tmp_path) -> None:
    store = make_store(tmp_path)
    record_open_game(store)
    report = validate_dataset(tmp_path / "human")
    assert report.ok, report.errors


def mutate_game(store: HumanDatasetStore, game_id: str, **changes) -> None:
    path = store.games_dir / f"{game_id}.json"
    data = json.loads(path.read_text("utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"action_space": "open_only"}, "action_space"),
        ({"action_types": {"0": "reveal", "1": "flag"}}, "action_types"),
        ({"num_actions": 480}, "num_actions"),
        ({"cells": 999}, "cells"),
        ({"observation_version": 0}, "observation_version"),
        ({"format_version": 2, "action_space": None, "num_actions": None}, "missing"),
    ],
)
def test_validation_rejects_identity_mismatches(tmp_path, changes, expected) -> None:
    store = make_store(tmp_path)
    game_id = record_open_game(store)
    mutate_game(store, game_id, **changes)
    report = validate_dataset(tmp_path / "human")
    assert not report.ok
    assert any(expected in error for error in report.errors), report.errors


def test_validation_rejects_hidden_mine_layout(tmp_path) -> None:
    store = make_store(tmp_path)
    game_id = record_open_game(store)
    path = store.games_dir / f"{game_id}.json"
    data = json.loads(path.read_text("utf-8"))
    data["transitions"][0]["mine_positions"] = [[0, 1]]
    path.write_text(json.dumps(data), encoding="utf-8")
    report = validate_dataset(tmp_path / "human")
    assert not report.ok
    assert any("mine_positions" in error for error in report.errors)


def test_validation_rejects_legacy_root(tmp_path) -> None:
    legacy = tmp_path / "human_expert"
    write_legacy_dataset(legacy)
    report = validate_dataset(legacy)
    assert not report.ok
    assert any("legacy" in error for error in report.errors)


def test_manifest_v1_is_upgraded_in_place(tmp_path) -> None:
    store = make_store(tmp_path)
    path = store.manifest_path
    data = json.loads(path.read_text("utf-8"))
    for key in ("action_space", "action_types", "num_actions", "cells",
                "observation_version", "reward_spec"):
        data.pop(key, None)
    data["format_version"] = 1
    path.write_text(json.dumps(data), encoding="utf-8")

    reopened = make_store(tmp_path)
    assert reopened.manifest["format_version"] == HUMAN_DATASET_VERSION
    assert reopened.manifest["action_space"] == ACTION_SPACE
    assert reopened.manifest["num_actions"] == 3 * CELLS
    on_disk = json.loads(path.read_text("utf-8"))
    assert on_disk["action_space"] == ACTION_SPACE


# ---------------------------------------------------------------------------
# Model-agnostic consumption
# ---------------------------------------------------------------------------


def _canonical_board(transition: dict, width: int, height: int) -> np.ndarray:
    return schema.decode_board(
        transition["observation_before"], width=width, height=height
    )


def test_same_raw_transition_feeds_all_model_adapters(tmp_path) -> None:
    store = make_store(tmp_path)
    game_id = record_open_game(store)
    game = store.load_game(game_id)
    transition = game["transitions"][0]
    board = _canonical_board(transition, WIDTH, HEIGHT)
    action_id = transition["action"]["action_id"]

    # CNN adapter: canonical board -> 11 one-hot channels.
    channels = board_to_tensor(board, action_space="flags", width=WIDTH, height=HEIGHT)
    assert tuple(channels.shape) == (11, HEIGHT, WIDTH)
    assert int(channels.sum().item()) == CELLS  # exactly one class per cell

    # GNN adapter: canonical board -> per-cell node states + coordinates/edges.
    node_states = board.reshape(-1)
    assert node_states.shape == (CELLS,)
    coordinates = [(x, y) for y in range(HEIGHT) for x in range(WIDTH)]
    assert len(coordinates) == CELLS
    adjacency = np.zeros((CELLS, CELLS), dtype=bool)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if (dx or dy) and 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                        adjacency[y * WIDTH + x, ny * WIDTH + nx] = True
    assert adjacency[0, 1] and adjacency[0, WIDTH]
    assert adjacency.shape == (CELLS, CELLS)
    # Node class is the same canonical value that the CNN one-hot encodes.
    assert int(node_states[0]) == int(board[0, 0])

    # Quantum adapter: canonical board + action_id -> bounded feature encoding.
    value_index = {value: index for index, value in enumerate((FLAGGED, HIDDEN, *range(9)))}
    state_features = np.zeros(CELLS, dtype=np.float64)
    for cell, value in enumerate(node_states):
        state_features[cell] = value_index[int(value)]
    action_features = np.zeros(3 * CELLS, dtype=np.float64)
    action_features[action_id] = 1.0
    encoding = np.concatenate([state_features, action_features])
    assert encoding.shape == (4 * CELLS,)
    assert int(np.argmax(action_features)) == action_id

    # The raw record itself has no CNN-specific dependency.
    raw = json.dumps(game)
    for token in ("one_hot", "channels", "adjacency", "amplitude", "tensor", "embedding"):
        assert token not in raw
