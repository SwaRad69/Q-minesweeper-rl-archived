"""End-to-end flag-aware training integration test (no browser).

Drives the real ``MinesweeperPPOEnv`` / ``TrackedEnvironment`` / ``PPOTrainer`` /
``run_training`` / checkpoint stack with a deterministic fake that implements
``apply_action`` for OPEN, FLAG and CHORD.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from minesweeper_rl.agents import FLAGS, PPOConfig, PPOTrainer, PolicyValueNetwork
from minesweeper_rl.agents.minesweeper_env import MinesweeperPPOEnv
from minesweeper_rl.dom import FLAGGED, HIDDEN, GameStatus
from minesweeper_rl.experiment.checkpoint import (
    LATEST_FILENAME,
    CheckpointError,
    load_checkpoint,
    read_checkpoint,
)
from minesweeper_rl.experiment.metrics import MetricsLogger
from minesweeper_rl.experiment.tracking import TrackedEnvironment
from minesweeper_rl.experiment.train import run_training

WIDTH, HEIGHT = 3, 2
CELLS = WIDTH * HEIGHT


class FakeFlagBoardEnv:
    """Mine-free flag-aware board: OPEN reveals, FLAG toggles, CHORD floods."""

    def __init__(self, width: int = WIDTH, height: int = HEIGHT) -> None:
        self.config = SimpleNamespace(width=width, height=height, mines=0)
        self.width = width
        self.height = height
        self._board = np.full((height, width), HIDDEN, dtype=np.int8)

    def reset(self) -> np.ndarray:
        self._board = np.full((self.height, self.width), HIDDEN, dtype=np.int8)
        self._board[0, 0] = 0
        return self._board.copy()

    def apply_action(self, action_type: int, x: int, y: int):
        from minesweeper_rl.action_types import ACTION_CHORD, ACTION_FLAG, ACTION_OPEN

        self._board = self._board.copy()
        newly = 0
        if action_type == ACTION_OPEN and self._board[y, x] == HIDDEN:
            self._board[y, x] = 0
            newly = 1
        elif action_type == ACTION_FLAG and self._board[y, x] in (HIDDEN, FLAGGED):
            self._board[y, x] = (
                HIDDEN if self._board[y, x] == FLAGGED else FLAGGED
            )
        elif action_type == ACTION_CHORD:
            for ny in range(self.height):
                for nx in range(self.width):
                    if self._board[ny, nx] == HIDDEN:
                        self._board[ny, nx] = 0
                        newly += 1
        done = bool((self._board != HIDDEN).all())
        status = GameStatus.WON if done else GameStatus.RUNNING
        info = {
            "action": (x, y),
            "action_type": action_type,
            "operation": "chord" if action_type == ACTION_CHORD else (
                "flag" if action_type == ACTION_FLAG else "open"
            ),
            "game_status": status,
            "newly_revealed": newly,
            "hit_mine": False,
        }
        return self._board.copy(), 0.0 if action_type == ACTION_FLAG else float(newly), done, False, info


def make_trainer():
    env = MinesweeperPPOEnv(FakeFlagBoardEnv(), action_space=FLAGS)
    tracker = TrackedEnvironment(env, phase="train")
    policy = PolicyValueNetwork(
        width=WIDTH, height=HEIGHT, channels=8, action_space=FLAGS
    )
    config = PPOConfig(
        rollout_steps=1, num_epochs=1, minibatch_size=8, learning_rate=1e-2
    )
    trainer = PPOTrainer(policy, tracker, num_actions=env.num_actions, config=config)
    return trainer, tracker


def test_flag_aware_training_run_writes_compatible_checkpoint(tmp_path) -> None:
    torch.manual_seed(0)
    trainer, tracker = make_trainer()
    assert trainer.num_actions == 3 * CELLS

    report = run_training(
        trainer,
        tracker,
        MetricsLogger(tmp_path),
        games=3,
        checkpoint=tmp_path / LATEST_FILENAME,
        checkpoint_interval=1,
        emit=lambda _: None,
    )

    assert report.final_episode == 3
    assert not report.errors

    data = read_checkpoint(tmp_path / LATEST_FILENAME)
    assert data["action_space"] == FLAGS
    assert data["num_actions"] == 3 * CELLS
    assert data["model"]["observation_channels"] == 11

    restored, _ = make_trainer()
    load_checkpoint(restored, data)
    assert restored.num_actions == 3 * CELLS


def test_flag_aware_checkpoint_does_not_load_into_open_only(tmp_path) -> None:
    from minesweeper_rl.agents import OPEN_ONLY

    trainer, tracker = make_trainer()
    run_training(
        trainer,
        tracker,
        MetricsLogger(tmp_path),
        games=1,
        checkpoint=tmp_path / LATEST_FILENAME,
        emit=lambda _: None,
    )
    data = read_checkpoint(tmp_path / LATEST_FILENAME)

    open_policy = PolicyValueNetwork(
        width=WIDTH, height=HEIGHT, channels=8, action_space=OPEN_ONLY
    )
    open_trainer = PPOTrainer(
        open_policy,
        MinesweeperPPOEnv(FakeFlagBoardEnv(), action_space=OPEN_ONLY),
        num_actions=CELLS,
        config=PPOConfig(),
    )
    with pytest.raises(CheckpointError):
        load_checkpoint(open_trainer, data)


def test_tracking_counts_operations_from_info() -> None:
    tracker = TrackedEnvironment(
        MinesweeperPPOEnv(FakeFlagBoardEnv(), action_space=FLAGS)
    )
    tracker.reset()
    # OPEN(cell 1) then FLAG(cell 2); the episode finishes via CHORD.
    from minesweeper_rl.agents import ACTION_CHORD, ACTION_FLAG, ACTION_OPEN, encode_action

    tracker.step(encode_action(1, action_type=ACTION_OPEN, width=WIDTH, height=HEIGHT))
    tracker.step(encode_action(2, action_type=ACTION_FLAG, width=WIDTH, height=HEIGHT))
    # The flag is still on cell 2, so CHORD must not open it; remove then chord.
    tracker.step(encode_action(2, action_type=ACTION_FLAG, width=WIDTH, height=HEIGHT))
    tracker.step(encode_action(0, action_type=ACTION_CHORD, width=WIDTH, height=HEIGHT))

    episode = tracker.last_episode
    assert episode is not None
    assert episode.open_actions == 1
    assert episode.flag_actions == 2
    assert episode.chord_actions == 1
    assert episode.steps == 4
