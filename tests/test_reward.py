"""Tests for the approved PRIMARY reward: revealed-cell progress + terminal.

The reward is defined once in :func:`minesweeper_rl.environment.transition_reward`
and shared by the autonomous OPEN-only environment.  It depends only on the
observable board (``board >= 0`` counts as revealed) and the terminal status;
the hidden mine layout is never consulted.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from minesweeper_rl.agents.experiment_ppo import _episode_outcome, classify_outcome
from minesweeper_rl.dom import HIDDEN, GameStatus
from minesweeper_rl.environment import (
    MINE_REWARD,
    WIN_REWARD,
    revealed_cells,
    transition_reward,
)


def board(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


HIDDEN_BOARD = board([[HIDDEN, HIDDEN], [HIDDEN, HIDDEN]])


# ---------------------------------------------------------------------------
# A/B - revealed-cell progress
# ---------------------------------------------------------------------------


def test_safe_single_open_is_plus_one() -> None:
    after = board([[0, HIDDEN], [HIDDEN, HIDDEN]])
    assert transition_reward(HIDDEN_BOARD, after, GameStatus.RUNNING) == 1.0


def test_safe_cascade_is_plus_k() -> None:
    after = board([[0, 1], [2, 3]])
    assert transition_reward(HIDDEN_BOARD, after, GameStatus.RUNNING) == 4.0


def test_no_new_cells_is_zero() -> None:
    assert transition_reward(HIDDEN_BOARD, HIDDEN_BOARD, GameStatus.RUNNING) == 0.0


# ---------------------------------------------------------------------------
# C - hidden cells are never counted as revealed
# ---------------------------------------------------------------------------


def test_hidden_cells_are_not_counted_as_revealed() -> None:
    mixed = board([[HIDDEN, HIDDEN, 0, 1]])
    assert revealed_cells(mixed) == 2
    before = board([[HIDDEN, HIDDEN]])
    after = board([[0, HIDDEN]])
    assert transition_reward(before, after, GameStatus.RUNNING) == 1.0


# ---------------------------------------------------------------------------
# J/K - terminal rewards
# ---------------------------------------------------------------------------


def test_mine_hit_scores_terminal_penalty_without_progress() -> None:
    before = board([[HIDDEN]])
    # The exploded mine is reported as HIDDEN, so there is no fake progress.
    after = board([[HIDDEN]])
    assert transition_reward(before, after, GameStatus.LOST) == MINE_REWARD
    assert MINE_REWARD == -10.0


def test_loss_with_revealed_safe_cells_keeps_progress() -> None:
    before = board([[HIDDEN, HIDDEN, HIDDEN]])
    after = board([[0, 1, HIDDEN]])  # two safe cells opened, then a mine
    assert transition_reward(before, after, GameStatus.LOST) == 2.0 + MINE_REWARD


def test_win_adds_progress_to_bonus() -> None:
    before = board([[HIDDEN, HIDDEN]])
    after = board([[0, 1]])
    assert transition_reward(before, after, GameStatus.WON) == WIN_REWARD + 2.0


def test_unknown_status_is_non_terminal() -> None:
    before = board([[HIDDEN]])
    after = board([[0]])
    assert transition_reward(before, after, None) == 1.0


# ---------------------------------------------------------------------------
# O - outcome detection uses status, not reward equality
# ---------------------------------------------------------------------------


def test_classify_outcome_uses_status() -> None:
    assert classify_outcome(True, GameStatus.WON) == "won"
    assert classify_outcome(True, GameStatus.LOST) == "lost"
    assert classify_outcome(False, GameStatus.RUNNING) == "ended"
    assert classify_outcome(True, None) == "ended"


@pytest.mark.parametrize(
    ("status", "expected"),
    [(GameStatus.WON, "won"), (GameStatus.LOST, "lost")],
)
def test_episode_outcome_survives_progress_shaping(
    status: GameStatus, expected: str
) -> None:
    # Terminal reward now includes progress, so it is no longer exactly
    # +/- the terminal constant; the outcome must still be read from status.
    buffer = SimpleNamespace(terminated=[False, True], rewards=[1.0, WIN_REWARD + 7.0])
    trainer = SimpleNamespace(env=SimpleNamespace(last_info={"game_status": status}))
    assert _episode_outcome(trainer, buffer) == expected


def test_episode_outcome_ended_without_terminal() -> None:
    buffer = SimpleNamespace(terminated=[False], rewards=[1.0])
    trainer = SimpleNamespace(env=SimpleNamespace(last_info={}))
    assert _episode_outcome(trainer, buffer) == "ended"
