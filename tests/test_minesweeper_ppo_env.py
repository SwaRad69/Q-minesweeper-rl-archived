"""Tests for the browser<->PPO adapter, using fakes (no browser)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from minesweeper_rl.agents import (
    ACTION_CHORD,
    ACTION_FLAG,
    ACTION_OPEN,
    FLAGS,
    OPEN_ONLY,
    MinesweeperPPOEnv,
    PPOConfig,
    PPOTrainer,
    PolicyValueNetwork,
    action_to_coordinate,
    coordinate_to_action,
    encode_action,
)
from minesweeper_rl.dom import FLAGGED, HIDDEN, GameStatus


class FakeMinesweeperEnvironment:
    """Minimal OPEN-only stand-in for MinesweeperEnvironment."""

    def __init__(self, width: int = 4, height: int = 2) -> None:
        self.config = SimpleNamespace(width=width, height=height)
        self.width = width
        self.height = height
        self.step_calls: list[tuple[int, int]] = []
        self._board = np.full((height, width), HIDDEN, dtype=np.int8)

    def reset(self) -> np.ndarray:
        self.step_calls.clear()
        self._board = np.full((self.height, self.width), HIDDEN, dtype=np.int8)
        return self._board.copy()

    def step(self, action: tuple[int, int]):
        x, y = action
        self.step_calls.append((x, y))
        if self._board[y, x] != HIDDEN:
            raise ValueError(f"cell ({x}, {y}) already revealed")
        self._board = self._board.copy()
        self._board[y, x] = np.int8(0)
        won = bool((self._board < 0).sum() == 0)
        reward = 100.0 if won else 1.0
        status = GameStatus.WON if won else GameStatus.RUNNING
        info = {
            "action": (x, y),
            "operation": "open",
            "game_status": status,
            "newly_revealed": 1,
            "hit_mine": False,
        }
        return self._board.copy(), reward, won, False, info


class FakeFlagEnvironment(FakeMinesweeperEnvironment):
    """Record the dispatch of OPEN, FLAG and CHORD and mutate the board."""

    def __init__(self, width: int = 3, height: int = 2) -> None:
        super().__init__(width=width, height=height)
        self.action_calls: list[tuple[int, int, int]] = []

    def reset(self) -> np.ndarray:
        self.action_calls.clear()
        self._board = np.full((self.height, self.width), HIDDEN, dtype=np.int8)
        return self._board.copy()

    def apply_action(self, action_type: int, x: int, y: int):
        self.action_calls.append((action_type, x, y))
        self._board = self._board.copy()
        if action_type == ACTION_OPEN:
            self._board[y, x] = 0
            operation = "open"
        elif action_type == ACTION_FLAG:
            self._board[y, x] = (
                HIDDEN if self._board[y, x] == FLAGGED else FLAGGED
            )
            operation = "flag"
        else:
            operation = "chord"
        info = {
            "action": (x, y),
            "action_type": action_type,
            "operation": operation,
            "game_status": GameStatus.RUNNING,
            "newly_revealed": 0,
            "hit_mine": False,
        }
        return self._board.copy(), 0.0, False, False, info


def test_adapter_exposes_flat_action_count() -> None:
    env = MinesweeperPPOEnv(
        FakeMinesweeperEnvironment(width=4, height=2), action_space=OPEN_ONLY
    )
    assert env.num_actions == 8  # one OPEN per cell
    assert env.reset().shape == (2, 4)


def test_adapter_dispatches_open() -> None:
    backend = FakeMinesweeperEnvironment(width=4, height=2)
    env = MinesweeperPPOEnv(backend, action_space=OPEN_ONLY)
    env.reset()

    env.step(0)      # OPEN(cell 0) -> (0, 0)
    env.step(7)      # OPEN(cell 7) -> (3, 1)

    assert backend.step_calls == [(0, 0), (3, 1)]


def test_adapter_cell_mapping_round_trip() -> None:
    for index in range(8):
        x, y = action_to_coordinate(index, width=4, height=2)
        assert coordinate_to_action(x, y, width=4, height=2) == index


def test_adapter_mask_hidden_open_flagged() -> None:
    env = MinesweeperPPOEnv(
        FakeMinesweeperEnvironment(width=3, height=2), action_space=OPEN_ONLY
    )
    board = np.array(
        [[1, FLAGGED, HIDDEN], [HIDDEN, HIDDEN, HIDDEN]], dtype=np.int8
    )
    cells = 6

    mask = env.action_mask(board)

    assert mask.shape == (cells,)
    assert not mask[0], "an opened cell cannot be opened again"
    assert not mask[1], "a flagged cell is not a legal OPEN target"
    assert mask[2], "hidden cells allow OPEN"


def test_adapter_records_last_info() -> None:
    env = MinesweeperPPOEnv(
        FakeMinesweeperEnvironment(width=4, height=2), action_space=OPEN_ONLY
    )
    env.reset()
    assert env.last_info == {}

    env.step(0)

    assert env.last_info["operation"] == "open"
    assert env.last_info["action"] == (0, 0)


# ---------------------------------------------------------------------------
# Flag-aware adapter
# ---------------------------------------------------------------------------


def test_flag_adapter_action_count_is_three_per_cell() -> None:
    env = MinesweeperPPOEnv(FakeFlagEnvironment(width=3, height=2))
    assert env.action_space == FLAGS
    assert env.num_actions == 18


def test_flag_adapter_dispatches_open_flag_chord() -> None:
    backend = FakeFlagEnvironment(width=3, height=2)
    env = MinesweeperPPOEnv(backend, action_space=FLAGS)
    env.reset()

    env.step(encode_action(0, action_type=ACTION_OPEN, width=3, height=2))
    env.step(encode_action(1, action_type=ACTION_FLAG, width=3, height=2))
    env.step(encode_action(0, action_type=ACTION_CHORD, width=3, height=2))

    assert backend.action_calls == [
        (ACTION_OPEN, 0, 0),
        (ACTION_FLAG, 1, 0),
        (ACTION_CHORD, 0, 0),
    ]
    assert env.last_info["operation"] == "chord"


def test_flag_adapter_mask_blocks_illegal_actions() -> None:
    env = MinesweeperPPOEnv(FakeFlagEnvironment(width=3, height=2))
    board = np.array(
        [[1, FLAGGED, HIDDEN], [HIDDEN, HIDDEN, HIDDEN]], dtype=np.int8
    )
    cells = 6
    cell = lambda index: cells + index  # noqa: E731 - compact readable helper

    mask = env.action_mask(board)
    assert mask.shape == (18,)

    # OPEN: only hidden cells 2..5.
    assert not mask[0] and not mask[1]
    assert mask[2] and mask[3] and mask[4] and mask[5]
    # FLAG: hidden or already flagged (cell 1).
    assert mask[cell(1)]
    assert mask[cell(2)]
    assert not mask[cell(0)], "an opened cell cannot be flagged"
    # CHORD: cell 0 is an opened 1 with exactly one flagged neighbour and a
    # hidden neighbour, so a chord is legal there and nowhere else.
    chord_base = 2 * cells
    assert mask[chord_base + 0]
    assert not mask[chord_base + 1]
    assert not mask[chord_base + 2]


def test_ppo_runs_through_adapter() -> None:
    backend = FakeMinesweeperEnvironment(width=4, height=2)
    env = MinesweeperPPOEnv(backend, action_space=OPEN_ONLY)
    policy = PolicyValueNetwork(
        width=4, height=2, channels=8, action_space=OPEN_ONLY
    )
    config = PPOConfig(rollout_steps=16, num_epochs=2, minibatch_size=8)
    trainer = PPOTrainer(policy, env, num_actions=env.num_actions, config=config)

    stats = trainer.collect_rollout()
    assert stats.steps == 16
    for action, mask in zip(trainer._buffer.actions, trainer._buffer.masks):
        assert mask[action]
        assert 0 <= action < env.num_actions

    metrics = trainer.update()
    assert np.isfinite(metrics["total_loss"])


def test_ppo_runs_through_flag_adapter() -> None:
    env = MinesweeperPPOEnv(FakeFlagEnvironment(width=3, height=2))
    policy = PolicyValueNetwork(width=3, height=2, channels=8, action_space=FLAGS)
    config = PPOConfig(rollout_steps=12, num_epochs=2, minibatch_size=6)
    trainer = PPOTrainer(policy, env, num_actions=env.num_actions, config=config)

    trainer.collect_rollout()
    assert trainer._buffer.actions
    for action, mask in zip(trainer._buffer.actions, trainer._buffer.masks):
        assert mask[action]

    metrics = trainer.update()
    assert np.isfinite(metrics["total_loss"])


class _SingleActionEnv:
    """Env whose mask allows only one flat action, to force a sampled id."""

    def __init__(self, flat_action: int, num_actions: int = 8) -> None:
        self.num_actions = num_actions
        self.flat_action = flat_action
        self._board = np.full((2, 4), HIDDEN, dtype=np.int8)

    def reset(self) -> np.ndarray:
        return self._board.copy()

    def action_mask(self, observation=None) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=bool)
        mask[self.flat_action] = True
        return mask

    def step(self, action: int):
        info = {"game_status": GameStatus.RUNNING, "hit_mine": False}
        return self._board.copy(), 0.0, True, False, info


def test_ppo_rollout_stores_open_action_id() -> None:
    env = _SingleActionEnv(flat_action=5)  # OPEN(cell 5)
    policy = PolicyValueNetwork(
        width=4, height=2, channels=8, action_space=OPEN_ONLY
    )
    trainer = PPOTrainer(
        policy, env, num_actions=env.num_actions, config=PPOConfig(rollout_steps=3)
    )

    trainer.collect_rollout()

    assert trainer._buffer.actions == [5, 5, 5]
    assert all(mask[action] for action, mask in zip(trainer._buffer.actions, trainer._buffer.masks))
