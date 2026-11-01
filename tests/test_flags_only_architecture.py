"""FLAGS-only architecture contract tests (no browser, no live site).

These pin the architectural decision: the canonical action space is FLAGS with
1440 actions, the canonical observation is the 11-channel flag-aware encoding,
the main training/evaluation/reference entry points cannot be run OPEN-only, and
the canonical human dataset validator rejects anything else.

Some of these facts are also covered elsewhere (``test_policy_network``,
``test_human_schema``); this module states the end-to-end contract explicitly so
a regression in any one place is caught by name.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from minesweeper_rl.action_types import ACTION_CHORD, ACTION_FLAG, ACTION_OPEN
from minesweeper_rl.agents import (
    CELLS,
    FLAGS,
    NUM_ACTIONS_FLAGS,
    OBSERVATION_CHANNELS_FLAGS,
    PolicyValueNetwork,
    action_mask,
    encode_action,
    split_action,
)
from minesweeper_rl.agents.action_space import action_count
from minesweeper_rl.data import schema
from minesweeper_rl.data.human_dataset import (
    HumanDatasetStore,
    HUMAN_RECORDER_VERSION,
    PersistentHumanRecorder,
    validate_dataset,
)
from minesweeper_rl.dom import FLAGGED, HIDDEN

WIDTH, HEIGHT = 3, 2
CELLS_SMALL = WIDTH * HEIGHT


# ---------------------------------------------------------------------------
# Canonical action space: 1440 = 3 action types * 480 cells
# ---------------------------------------------------------------------------


def test_expert_action_count_is_1440() -> None:
    assert CELLS == 480
    assert NUM_ACTIONS_FLAGS == 1440
    assert action_count(FLAGS) == 1440


def test_open_flag_chord_blocks_are_contiguous() -> None:
    assert encode_action(0, action_type=ACTION_OPEN) == 0
    assert encode_action(479, action_type=ACTION_OPEN) == 479
    assert encode_action(0, action_type=ACTION_FLAG) == 480
    assert encode_action(479, action_type=ACTION_FLAG) == 959
    assert encode_action(0, action_type=ACTION_CHORD) == 960
    assert encode_action(479, action_type=ACTION_CHORD) == 1439


def test_every_action_round_trips() -> None:
    for action_type in (ACTION_OPEN, ACTION_FLAG, ACTION_CHORD):
        for cell in range(CELLS):
            action = encode_action(cell, action_type=action_type)
            assert 0 <= action < 1440
            assert split_action(action) == (action_type, cell)


# ---------------------------------------------------------------------------
# Canonical observation: 11-channel flag-aware encoding
# ---------------------------------------------------------------------------


def test_flags_observation_is_11_channels() -> None:
    assert OBSERVATION_CHANNELS_FLAGS == 11
    network = PolicyValueNetwork()
    assert network.observation_channels == 11
    assert network.encoder[0].in_channels == 11


def test_flags_one_hot_channel_order() -> None:
    board = np.array(
        [[FLAGGED, HIDDEN, 0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int8
    )
    tensor = PolicyValueNetwork(width=11, height=1).encode(board)
    assert tensor.shape == (11, 1, 11)
    for channel in range(11):
        assert tensor[channel, 0, channel].item() == 1.0
        assert tensor[:, 0, channel].sum().item() == 1.0


# ---------------------------------------------------------------------------
# Canonical policy: 1440 logits and a working forward/backward pass
# ---------------------------------------------------------------------------


def test_default_policy_outputs_1440_logits() -> None:
    network = PolicyValueNetwork()
    assert network.num_actions == 1440
    board = np.full((16, 30), HIDDEN, dtype=np.int8)
    board[8, 15] = 0
    logits, value = network(board)
    assert logits.shape == (1440,)
    assert value.shape == (1,)


def test_flags_policy_forward_backward_produces_gradients() -> None:
    network = PolicyValueNetwork()
    board = np.full((16, 30), FLAGGED, dtype=np.int8)
    board[0, 0] = HIDDEN
    board[1, 1] = 3
    mask = action_mask(board)
    assert mask.shape == (1440,)
    logits, value = network(board, action_mask=mask)
    (logits.sum() + value.sum()).backward()
    assert network.encoder[0].weight.grad is not None
    assert network.actor.weight.grad is not None
    assert network.critic[2].weight.grad is not None


# ---------------------------------------------------------------------------
# The main entry points are FLAGS-only (no OPEN-only switch)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path",
    [
        "minesweeper_rl.experiment.train",
        "minesweeper_rl.experiment.evaluate",
        "minesweeper_rl.agents.train_ppo",
        "minesweeper_rl.agents.experiment_ppo",
    ],
)
def test_entry_points_have_no_action_space_option(module_path: str) -> None:
    import importlib

    module = importlib.import_module(module_path)
    parser = module.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--action-space" not in option_strings


def test_legacy_human_recorder_cli_is_gone() -> None:
    import minesweeper_rl.data.record_human as record_human

    assert not hasattr(record_human, "build_parser")
    assert not hasattr(record_human, "main")


# ---------------------------------------------------------------------------
# Canonical human dataset identity + recorder semantics
# ---------------------------------------------------------------------------


def test_fresh_dataset_identity_is_flags_1440(tmp_path) -> None:
    store = HumanDatasetStore(tmp_path, width=30, height=16, mines=99)
    assert store.manifest["action_space"] == "flags"
    assert store.manifest["num_actions"] == 1440
    assert store.manifest["cells"] == 480
    assert store.manifest["recorder_version"] == HUMAN_RECORDER_VERSION == 3
    report = validate_dataset(tmp_path)
    assert report.ok
    assert report.games == 0


def _hidden() -> np.ndarray:
    return np.full((HEIGHT, WIDTH), HIDDEN, dtype=np.int8)


def test_recorder_classifies_open_flag_and_chord(tmp_path) -> None:
    store = HumanDatasetStore(tmp_path, width=WIDTH, height=HEIGHT, mines=1)
    recorder = PersistentHumanRecorder(store, width=WIDTH, height=HEIGHT, mines=1)

    b0 = _hidden()
    recorder.start_game(first_move=(0, 0), board=b0)

    b1 = b0.copy()
    b1[0, 0] = 1
    open_step = recorder.record_step(
        board_before=b0,
        action=(0, 0),
        board_after=b1,
        game_status="running",
        terminated=False,
        action_type=schema.ACTION_OPEN,
    )
    assert open_step["action"]["action_type"] == ACTION_OPEN
    assert open_step["action"]["action_id"] == 0

    b2 = b1.copy()
    b2[0, 1] = FLAGGED
    flag_step = recorder.record_step(
        board_before=b1,
        action=(1, 0),
        board_after=b2,
        game_status="running",
        terminated=False,
        action_type=schema.ACTION_FLAG,
    )
    assert flag_step["action"]["action_type"] == ACTION_FLAG
    assert flag_step["action"]["action_id"] == encode_action(
        1, action_type=ACTION_FLAG, width=WIDTH, height=HEIGHT
    )

    b3 = b2.copy()
    for x, y in ((2, 0), (0, 1), (1, 1), (2, 1)):
        b3[y, x] = 0
    chord_step = recorder.record_step(
        board_before=b2,
        action=(0, 0),
        board_after=b3,
        game_status="running",
        terminated=False,
        action_type=schema.ACTION_OPEN,  # a left-click; classified from the board
    )
    assert chord_step["action"]["action_type"] == ACTION_CHORD
    assert chord_step["action"]["action_id"] == encode_action(
        0, action_type=ACTION_CHORD, width=WIDTH, height=HEIGHT
    )
    assert chord_step["action"]["cell_index"] == 0


def test_loader_rejects_legacy_chord_as_open_dataset(tmp_path) -> None:
    from minesweeper_rl.data.human_demonstrations import HumanDemonstrationDataset

    (tmp_path / "games.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(schema.DataError):
        HumanDemonstrationDataset(tmp_path)


def test_validate_rejects_open_only_action_space(tmp_path) -> None:
    # A hand-written file that bypasses apply_canonical_identity must still be
    # rejected: the validator compares the declared identity, not just the
    # filename.
    import json

    HumanDatasetStore(tmp_path, width=WIDTH, height=HEIGHT, mines=1)
    game = {
        "format": "minesweeper_human_game",
        "format_version": 2,
        "game_id": "game_000001",
        "game_index": 1,
        "source": "human",
        "width": WIDTH,
        "height": HEIGHT,
        "mines": 1,
        "outcome": "incomplete",
        "complete": False,
        "result": "abandoned",
        "total_steps": 0,
        "started_at": 0.0,
        "ended_at": 1.0,
        "duration_s": 1.0,
        "action_space": "open_only",
        "action_types": {"0": "open", "1": "flag", "2": "chord"},
        "num_actions": WIDTH * HEIGHT,
        "cells": WIDTH * HEIGHT,
        "recorder_version": HUMAN_RECORDER_VERSION,
        "transitions": [],
    }
    (tmp_path / "games" / "game_000001.json").write_text(
        json.dumps(game), encoding="utf-8"
    )
    report = validate_dataset(tmp_path)
    assert not report.ok
    assert any("action_space" in error for error in report.errors)
