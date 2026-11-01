from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from minesweeper_rl.actions import InvalidActionError
from minesweeper_rl.dom import FLAGGED, HIDDEN, GameStatus
from minesweeper_rl.metrics import EnvironmentMetrics, Latency
from minesweeper_rl.stress import (
    CheckReport,
    EpisodeReport,
    StressReport,
    check_board_parsing,
    check_dom_sync,
    check_invalid_actions,
    play_stress_episode,
    summarize_stress,
)


def test_latency_percentiles_and_mean() -> None:
    latency = Latency()
    for value in [0.1, 0.2, 0.3, 0.4, 0.5]:
        latency.add(value)

    assert latency.count == 5
    assert latency.mean_ms == pytest.approx(300.0)
    assert latency.median_ms == pytest.approx(300.0)
    assert latency.max_ms == pytest.approx(500.0)
    assert latency.p95_ms == pytest.approx(500.0)
    assert Latency().mean_ms == 0.0


class FakeEnv:
    """Env double whose step reveals one cell until a mine is hit."""

    def __init__(self, width: int = 2, height: int = 2, mine: tuple[int, int] = (0, 0)) -> None:
        self.config = SimpleNamespace(width=width, height=height)
        self.mine = mine
        self.board = np.full((height, width), HIDDEN, dtype=np.int8)
        self.status = GameStatus.RUNNING

    def reset(self) -> np.ndarray:
        self.board = np.full((self.config.height, self.config.width), HIDDEN, dtype=np.int8)
        self.status = GameStatus.RUNNING
        return self.board.copy()

    def get_game_status(self) -> GameStatus:
        return self.status

    def observe(self) -> np.ndarray:
        return self.board.copy()

    def step(self, action: tuple[int, int]):
        x, y = action
        if self.board[y, x] != HIDDEN:
            raise InvalidActionError(f"cell ({x}, {y}) is already revealed")
        if (x, y) == self.mine:
            status = GameStatus.LOST
            reward = -10.0
            hit = True
        else:
            self.board[y, x] = 0
            status = GameStatus.RUNNING
            reward = 1.0
            hit = False
        self.status = status
        return self.board.copy(), reward, status in (GameStatus.WON, GameStatus.LOST), False, {
            "game_status": status,
            "hit_mine": hit,
        }


class FailingEnv(FakeEnv):
    def step(self, action: tuple[int, int]):
        raise RuntimeError("boom")


def test_play_stress_episode_records_success() -> None:
    result = play_stress_episode(FakeEnv(2, 2, mine=(0, 0)), 1, "main", random.Random(0))

    assert result.error is None
    assert result.status is GameStatus.LOST
    assert result.steps >= 1
    assert result.duration_s >= 0.0


def test_play_stress_episode_records_failure() -> None:
    result = play_stress_episode(FailingEnv(), 3, "main", random.Random(0))

    assert result.episode == 3
    assert result.status is None
    assert result.error_type == "RuntimeError"
    assert "boom" in (result.error or "")


def test_check_invalid_actions_passes_for_a_well_behaved_env() -> None:
    checks = check_invalid_actions(_CheckEnv())

    assert {check.name: check.passed for check in checks} == {
        "invalid:out_of_bounds": True,
        "invalid:non_integer": True,
        "invalid:malformed_action": True,
        "invalid:already_revealed": True,
        "invalid:no_state_corruption": True,
    }


class _CheckEnv(FakeEnv):
    def __init__(self) -> None:
        super().__init__(2, 2, mine=(0, 0))
        self.board[1, 1] = 0

    def step(self, action: object):
        if not isinstance(action, tuple) or len(action) != 2:
            raise TypeError("action must be an (x, y) pair")
        x, y = action
        if not isinstance(x, int) or not isinstance(y, int):
            raise TypeError("coordinates must be integers")
        if not (0 <= x < 2 and 0 <= y < 2):
            raise IndexError("outside board")
        if self.board[y, x] != HIDDEN:
            raise InvalidActionError("already revealed")
        self.board[y, x] = 0
        return self.board.copy(), 1.0, False, False, {"game_status": GameStatus.RUNNING}


def test_check_dom_sync_detects_instability() -> None:
    class Stable(FakeEnv):
        def observe(self) -> np.ndarray:
            return self.board.copy()

    stable = check_dom_sync(Stable())
    assert stable.passed is True

    class Flapping(FakeEnv):
        def __init__(self) -> None:
            super().__init__()
            self.toggle = 0

        def observe(self) -> np.ndarray:
            self.toggle += 1
            board = self.board.copy()
            if self.toggle % 2 == 0:
                board[0, 0] = 0
            return board

    assert check_dom_sync(Flapping()).passed is False


def test_check_board_parsing_reports_shape() -> None:
    check = check_board_parsing(FakeEnv(3, 2))
    assert check.passed is True
    assert "shape=(2, 3)" in check.detail


def test_check_board_parsing_accepts_flagged_cells() -> None:
    env = FakeEnv(3, 2)
    env.board[0, 0] = FLAGGED
    check = check_board_parsing(env)
    assert check.passed is True


def test_summarize_stress_reports_reliability_and_latency() -> None:
    metrics = EnvironmentMetrics()
    for value in [0.01, 0.02, 0.03]:
        metrics.step.add(value)
    metrics.reset.add(0.5)
    metrics.clicks = 3

    report = StressReport(
        episodes=[
            EpisodeReport(1, "main", GameStatus.LOST, 4, -7.0, 1.0),
            EpisodeReport(2, "main", None, 0, 0.0, 30.0, "timeout", "ActionTimeoutError"),
        ],
        checks=[CheckReport("dom_sync:stable_idle_observations", True)],
        metrics=metrics,
        recovery_attempts=1,
        recovery_succeeded=1,
    )

    assert report.reliability_percent == 50.0
    summary = summarize_stress(report)

    assert "reliability        : 50.0%" in summary
    assert "episodes failed    : 1" in summary
    assert "recovery           : 1/1" in summary
    assert "step" in summary and "reset" in summary and "parse" in summary
    assert "[PASS] dom_sync:stable_idle_observations" in summary
    assert "ActionTimeoutError" in summary
