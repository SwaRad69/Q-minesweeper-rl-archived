"""Tests for human demonstration loading, splitting and imitation training.

No browser and no live data: a small synthetic human dataset is written with the
real persistent recorder, then loaded, split by game and used for a one-epoch
behavioral-cloning run in a temporary directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from minesweeper_rl.agents.policy import PolicyValueNetwork
from minesweeper_rl.data import (
    DatasetStatistics,
    HumanDatasetStore,
    HumanDemonstrationDataset,
    PersistentHumanRecorder,
    split_games,
)
from minesweeper_rl.dom import FLAGGED, HIDDEN, GameStatus
from minesweeper_rl.experiment import imitation_train as imitation
from minesweeper_rl.experiment.imitation_train import (
    BEST_FILENAME,
    HISTORY_FILE,
    IMITATION_CHECKPOINT_FORMAT,
    LATEST_FILENAME,
    RUN_CONFIG_FILE,
    ImitationError,
    build_imitation_checkpoint,
    load_imitation_checkpoint,
    read_imitation_checkpoint,
)

WIDTH, HEIGHT = 4, 2
CELLS = WIDTH * HEIGHT


def empty_board() -> np.ndarray:
    return np.full((HEIGHT, WIDTH), HIDDEN, dtype=np.int8)


def make_store(tmp_path: Path) -> HumanDatasetStore:
    return HumanDatasetStore(
        tmp_path / "human", width=WIDTH, height=HEIGHT, mines=3, difficulty="expert"
    )


def make_game(store: HumanDatasetStore, outcome: str) -> str:
    """Write one 3-step game (OPEN, FLAG, CHORD) with the given outcome."""

    recorder = PersistentHumanRecorder(
        store, width=WIDTH, height=HEIGHT, mines=3, difficulty="expert"
    )
    recorder.start_game()
    hidden = empty_board()
    opened = hidden.copy()
    opened[0, 0] = 1
    recorder.record_step(
        board_before=hidden, action=(0, 0), board_after=opened,
        game_status=GameStatus.RUNNING, terminated=False,
    )
    flagged = opened.copy()
    flagged[0, 1] = FLAGGED
    recorder.record_step(
        board_before=opened, action=(1, 0), board_after=flagged,
        game_status=GameStatus.RUNNING, terminated=False, action_type="flag",
    )
    final = flagged.copy()
    final[0, 1] = 0
    final[1, 1] = 0
    if outcome == "won":
        recorder.record_step(
            board_before=flagged, action=(0, 0), board_after=final,
            game_status=GameStatus.WON, terminated=True, action_type="open",
        )
        return recorder.end_game("won").game_id
    if outcome == "lost":
        recorder.record_step(
            board_before=flagged, action=(0, 0), board_after=final,
            game_status=GameStatus.LOST, terminated=True, action_type="open",
        )
        return recorder.end_game("lost").game_id
    recorder.record_step(
        board_before=flagged, action=(0, 0), board_after=final,
        game_status=GameStatus.RUNNING, terminated=False, action_type="open",
    )
    return recorder.end_game("abandoned").game_id


def make_dataset(tmp_path: Path, outcomes: list[str]) -> HumanDatasetStore:
    store = make_store(tmp_path)
    for outcome in outcomes:
        make_game(store, outcome)
    return store


def hash_tree(root: Path) -> dict[str, str]:
    digests = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digests[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return digests


# ---------------------------------------------------------------------------
# Loading / filtering
# ---------------------------------------------------------------------------


def test_loader_reads_all_games(tmp_path) -> None:
    make_dataset(tmp_path, ["won", "lost", "abandoned"])
    dataset = HumanDemonstrationDataset(tmp_path / "human")

    assert dataset.game_ids() == ["game_000001", "game_000002", "game_000003"]
    assert len(dataset.samples()) == 9
    assert dataset.width == WIDTH and dataset.height == HEIGHT
    assert dataset.num_actions == 3 * CELLS


def test_loader_supports_filters(tmp_path) -> None:
    make_dataset(tmp_path, ["won", "lost", "won", "abandoned"])

    winning = HumanDemonstrationDataset(tmp_path / "human", winning_only=True)
    assert len(winning.games) == 2

    complete = HumanDemonstrationDataset(tmp_path / "human", exclude_incomplete=True)
    assert len(complete.games) == 3

    selected = HumanDemonstrationDataset(
        tmp_path / "human", games=["game_000001", "game_000004"]
    )
    assert selected.game_ids() == ["game_000001", "game_000004"]


def test_loader_does_not_modify_raw_recordings(tmp_path) -> None:
    make_dataset(tmp_path, ["won", "lost"])
    before = hash_tree(tmp_path / "human")

    dataset = HumanDemonstrationDataset(tmp_path / "human")
    _ = dataset.statistics()
    _ = split_games(dataset.game_ids(), val_fraction=0.5, seed=1)

    assert hash_tree(tmp_path / "human") == before


def test_samples_have_observable_shape_and_valid_targets(tmp_path) -> None:
    make_dataset(tmp_path, ["won", "abandoned"])
    dataset = HumanDemonstrationDataset(tmp_path / "human")

    for sample in dataset.samples():
        assert sample.board.shape == (HEIGHT, WIDTH)
        assert 0 <= sample.action_id < 3 * CELLS
        assert sample.action_id == sample.action_type * CELLS + (
            sample.y * WIDTH + sample.x
        )
        assert sample.action_type in (0, 1, 2)
        allowed = np.isin(sample.board, np.arange(FLAGGED, 9)).all()
        assert bool(allowed)
        # Only the pre-action observation is used; no hidden mine field exists.
        assert not hasattr(sample, "mines")


def test_statistics_count_games_and_actions(tmp_path) -> None:
    make_dataset(tmp_path, ["won", "lost", "won", "abandoned"])
    stats = HumanDemonstrationDataset(tmp_path / "human").statistics()

    assert stats.games == 4
    assert stats.winning_games == 2
    assert stats.lost_games == 1
    assert stats.incomplete_games == 1
    assert stats.transitions == 12
    assert (stats.open_count, stats.flag_count, stats.chord_count) == (4, 4, 4)
    assert stats.average_transitions_per_game == pytest.approx(3.0)
    assert sum(stats.action_counts.values()) == 12
    assert "distribution" in stats.format()


def test_from_games_matches_dataset_statistics(tmp_path) -> None:
    make_dataset(tmp_path, ["won", "abandoned"])
    dataset = HumanDemonstrationDataset(tmp_path / "human")
    assert DatasetStatistics.from_games(dataset.games).transitions == 6


# ---------------------------------------------------------------------------
# Game-level split
# ---------------------------------------------------------------------------


def test_split_is_by_game_and_has_no_overlap(tmp_path) -> None:
    make_dataset(tmp_path, ["won", "lost", "won", "lost", "won"])
    dataset = HumanDemonstrationDataset(tmp_path / "human")
    ids = dataset.game_ids()

    train, val = split_games(ids, val_fraction=0.4, seed=7)

    assert not set(train) & set(val)
    assert set(train) | set(val) == set(ids)
    assert len(val) == 2  # round(5 * 0.4) == 2
    assert len(train) == 3


def test_split_is_deterministic_for_a_seed(tmp_path) -> None:
    ids = [f"game_{index:06d}" for index in range(1, 21)]
    first = split_games(ids, val_fraction=0.2, seed=123)
    second = split_games(ids, val_fraction=0.2, seed=123)
    assert first == second

    other = split_games(ids, val_fraction=0.2, seed=999)
    assert other != first


def test_split_handles_small_datasets() -> None:
    assert split_games([]) == ([], [])
    assert split_games(["a"], val_fraction=0.2) == (["a"], [])
    train, val = split_games(["a", "b"], val_fraction=0.2)
    assert len(train) == 1 and len(val) == 1
    # A zero validation fraction keeps everything in training.
    train, val = split_games(["a", "b", "c"], val_fraction=0.0)
    assert len(train) == 3 and val == []


# ---------------------------------------------------------------------------
# Model output
# ---------------------------------------------------------------------------


def test_default_policy_outputs_1440_logits() -> None:
    policy = PolicyValueNetwork()
    assert policy.num_actions == 1440
    assert policy.observation_channels == 11
    logits, value = policy(np.full((16, 30), HIDDEN, dtype=np.int8))
    assert logits.shape == (1440,)
    assert value.shape == (1,)


def test_small_policy_matches_dataset_action_count(tmp_path) -> None:
    make_dataset(tmp_path, ["won"])
    dataset = HumanDemonstrationDataset(tmp_path / "human")
    policy = PolicyValueNetwork(width=dataset.width, height=dataset.height)
    assert policy.num_actions == dataset.num_actions == 3 * CELLS


def test_encoding_matches_policy_observation_pipeline(tmp_path) -> None:
    make_dataset(tmp_path, ["won"])
    dataset = HumanDemonstrationDataset(tmp_path / "human")
    samples = dataset.samples()
    boards, actions = imitation.encode_samples(samples, width=WIDTH, height=HEIGHT)
    assert boards.shape == (len(samples), HEIGHT, WIDTH)
    assert boards.dtype == torch.int8
    assert actions.dtype == torch.int64
    assert int(actions.max()) < 3 * CELLS

    policy = PolicyValueNetwork(width=WIDTH, height=HEIGHT)
    logits, _ = policy(boards)
    assert logits.shape == (len(samples), 3 * CELLS)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


def test_checkpoint_save_and_load_round_trip(tmp_path) -> None:
    make_dataset(tmp_path, ["won", "lost"])
    dataset = HumanDemonstrationDataset(tmp_path / "human")
    policy = PolicyValueNetwork(width=WIDTH, height=HEIGHT)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    config = imitation.BCConfig(epochs=1)
    stats = dataset.statistics()

    payload = build_imitation_checkpoint(
        policy, optimizer, epoch=1, config=config, dataset_stats=stats,
        train_stats=stats, val_stats=stats, history=[],
    )
    path = tmp_path / "imitation.pt"
    imitation.atomic_save(payload, path)

    data = read_imitation_checkpoint(path)
    assert data["format"] == IMITATION_CHECKPOINT_FORMAT
    assert data["epoch"] == 1
    assert "optimizer" in data and "dataset" in data and "config" in data

    restored = PolicyValueNetwork(width=WIDTH, height=HEIGHT)
    load_imitation_checkpoint(path, restored)
    for a, b in zip(policy.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(a, b)


def test_checkpoint_rejects_incompatible_policy(tmp_path) -> None:
    make_dataset(tmp_path, ["won"])
    dataset = HumanDemonstrationDataset(tmp_path / "human")
    policy = PolicyValueNetwork(width=WIDTH, height=HEIGHT)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    stats = dataset.statistics()
    payload = build_imitation_checkpoint(
        policy, optimizer, epoch=1, config=imitation.BCConfig(),
        dataset_stats=stats, train_stats=stats, val_stats=stats, history=[],
    )
    path = tmp_path / "imitation.pt"
    imitation.atomic_save(payload, path)

    wrong = PolicyValueNetwork(width=WIDTH, height=HEIGHT, channels=32)
    with pytest.raises(ImitationError, match="channels"):
        load_imitation_checkpoint(path, wrong)


def test_imitation_checkpoint_is_not_a_ppo_checkpoint(tmp_path) -> None:
    from types import SimpleNamespace

    from minesweeper_rl.experiment.checkpoint import (
        CheckpointError,
        read_checkpoint,
        validate_checkpoint,
    )

    make_dataset(tmp_path, ["won"])
    dataset = HumanDemonstrationDataset(tmp_path / "human")
    policy = PolicyValueNetwork(width=WIDTH, height=HEIGHT)
    stats = dataset.statistics()
    payload = build_imitation_checkpoint(
        policy, torch.optim.Adam(policy.parameters()), epoch=1,
        config=imitation.BCConfig(), dataset_stats=stats,
        train_stats=stats, val_stats=stats, history=[],
    )
    path = tmp_path / "imitation.pt"
    imitation.atomic_save(payload, path)

    # The imitation format is distinct: the PPO validator rejects it clearly.
    data = read_checkpoint(path)
    assert data["format"] == IMITATION_CHECKPOINT_FORMAT
    with pytest.raises(CheckpointError, match="format"):
        validate_checkpoint(
            data,
            SimpleNamespace(policy=policy, num_actions=policy.num_actions),
        )



# ---------------------------------------------------------------------------
# End-to-end tiny training run
# ---------------------------------------------------------------------------


def test_imitation_train_end_to_end(tmp_path) -> None:
    make_dataset(tmp_path, ["won", "lost", "won", "lost", "won", "abandoned"])
    dataset_dir = tmp_path / "human"
    output = tmp_path / "out"

    code = imitation.main([
        "--dataset", str(dataset_dir),
        "--output", str(output),
        "--epochs", "2",
        "--batch-size", "4",
        "--val-split", "0.34",
        "--seed", "3",
        "--device", "cpu",
    ])

    assert code == 0
    assert (output / LATEST_FILENAME).exists()
    assert (output / BEST_FILENAME).exists()
    assert (output / RUN_CONFIG_FILE).exists()
    assert (output / HISTORY_FILE).exists()

    data = read_imitation_checkpoint(output / LATEST_FILENAME)
    assert data["format"] == IMITATION_CHECKPOINT_FORMAT
    assert data["epoch"] == 2
    assert data["model"]["num_actions"] == 3 * CELLS
    assert data["metrics"]["val_loss"] is not None
    assert data["metrics"]["val_accuracy"] is not None
    assert len(data["history"]) == 2

    run_config = json.loads((output / RUN_CONFIG_FILE).read_text(encoding="utf-8"))
    assert run_config["algorithm"] == "behavioral_cloning"
    assert run_config["dataset_statistics"]["games"] == 6

    history = json.loads((output / HISTORY_FILE).read_text(encoding="utf-8"))
    assert len(history["history"]) == 2
    for record in history["history"]:
        for split in ("train", "val"):
            assert record[split]["loss"] >= 0.0
            assert 0.0 <= record[split]["accuracy"] <= 1.0
            for name in ("open", "flag", "chord"):
                assert name in record[split]["per_type_accuracy"]


def test_imitation_train_filters_and_selected_games(tmp_path) -> None:
    make_dataset(tmp_path, ["won", "lost", "won", "abandoned"])
    output = tmp_path / "out"

    code = imitation.main([
        "--dataset", str(tmp_path / "human"),
        "--output", str(output),
        "--epochs", "1",
        "--batch-size", "2",
        "--val-split", "0.5",
        "--seed", "1",
        "--device", "cpu",
        "--winning-only",
        "--exclude-incomplete",
    ])
    assert code == 0
    run_config = json.loads((output / RUN_CONFIG_FILE).read_text(encoding="utf-8"))
    assert run_config["dataset_statistics"]["winning_games"] == 2
    assert run_config["dataset_statistics"]["incomplete_games"] == 0


def test_imitation_train_refuses_empty_dataset(tmp_path) -> None:
    HumanDatasetStore(
        tmp_path / "human", width=WIDTH, height=HEIGHT, mines=3, difficulty="expert"
    )
    code = imitation.main([
        "--dataset", str(tmp_path / "human"),
        "--output", str(tmp_path / "out"),
        "--epochs", "1",
        "--device", "cpu",
    ])
    assert code == 1


def test_imitation_resume_continues_history(tmp_path) -> None:
    make_dataset(tmp_path, ["won", "lost", "won", "lost"])
    dataset_dir = str(tmp_path / "human")
    output = str(tmp_path / "out")
    base = [
        "--dataset", dataset_dir, "--output", output,
        "--epochs", "1", "--batch-size", "4", "--val-split", "0.5",
        "--seed", "5", "--device", "cpu",
    ]

    assert imitation.main(base) == 0
    first = read_imitation_checkpoint(Path(output) / LATEST_FILENAME)
    assert first["epoch"] == 1 and len(first["history"]) == 1

    assert imitation.main(base + ["--resume"]) == 0
    resumed = read_imitation_checkpoint(Path(output) / LATEST_FILENAME)
    assert resumed["epoch"] == 2
    assert len(resumed["history"]) == 2
