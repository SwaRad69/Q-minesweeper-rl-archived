from __future__ import annotations

import random

import numpy as np
import pytest

from minesweeper_rl.agents.random_agent import (
    EpisodeResult,
    format_episode,
    hidden_cells,
    play_episode,
    run_random_agent,
    summarize,
)
from minesweeper_rl.dom import HIDDEN, GameStatus


class FakeEnv:
    """Minimal deterministic Minesweeper used to test the agent loop."""

    def __init__(self, width: int, height: int, mines: set[tuple[int, int]]) -> None:
        self.width = width
        self.height = height
        self.mines = set(mines)
        self.resets = 0
        self.board = np.full((height, width), HIDDEN, dtype=np.int8)
        self.opened: set[tuple[int, int]] = set()

    def reset(self) -> np.ndarray:
        self.resets += 1
        self.board = np.full((self.height, self.width), HIDDEN, dtype=np.int8)
        self.opened = set()
        return self.board.copy()

    def step(self, action: tuple[int, int]) -> tuple[np.ndarray, float, bool, bool, dict]:
        x, y = action
        if self.board[y, x] != HIDDEN:
            raise AssertionError(f"agent selected already revealed cell ({x}, {y})")
        self.opened.add((x, y))
        if (x, y) in self.mines:
            self.board[y, x] = HIDDEN
            status, reward, hit_mine = GameStatus.LOST, -10.0, True
        else:
            self.board[y, x] = 0
            safe_total = self.width * self.height - len(self.mines)
            safe_opened = len(self.opened - self.mines)
            if safe_opened >= safe_total:
                status, reward, hit_mine = GameStatus.WON, 100.0, False
            else:
                status, reward, hit_mine = GameStatus.RUNNING, 1.0, False
        terminated = status in (GameStatus.WON, GameStatus.LOST)
        info = {
            "action": (x, y),
            "game_status": status,
            "hit_mine": hit_mine,
            "newly_revealed": 1,
            "opened_cells": len(self.opened),
        }
        return self.board.copy(), reward, terminated, False, info


def test_hidden_cells_lists_only_hidden_coordinates() -> None:
    board = np.array([[-1, 0, -1], [2, -1, 3]], dtype=np.int8)

    assert hidden_cells(board) == [(0, 0), (2, 0), (1, 1)]


def test_play_episode_terminates_and_is_consistent() -> None:
    env = FakeEnv(2, 2, mines={(0, 0), (1, 0), (0, 1)})

    result = play_episode(env, 1, random.Random(0))

    assert result.episode == 1
    assert result.status in (GameStatus.WON, GameStatus.LOST)
    assert result.steps >= 1
    assert result.duration_s >= 0.0
    assert result.hit_mine is (result.status is GameStatus.LOST)
    if result.status is GameStatus.LOST:
        assert result.cells_revealed == result.steps - 1
    else:
        assert result.cells_revealed == result.steps


def test_run_random_agent_plays_requested_number_of_games() -> None:
    env = FakeEnv(3, 3, mines={(0, 0)})
    seen: list[EpisodeResult] = []

    results = run_random_agent(env, 5, rng=random.Random(1), on_episode=seen.append)

    assert len(results) == 5
    assert seen == results
    assert [result.episode for result in results] == [1, 2, 3, 4, 5]
    assert env.resets == 5
    assert all(result.steps >= 1 for result in results)


def test_run_random_agent_rejects_non_positive_games() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_random_agent(FakeEnv(2, 2, mines=set()), 0)


def test_summary_reports_wins_losses_and_averages() -> None:
    results = [
        EpisodeResult(1, GameStatus.LOST, 4, -7.0, 3, True, 1.0),
        EpisodeResult(2, GameStatus.WON, 8, 106.0, 7, False, 2.0),
        EpisodeResult(3, GameStatus.LOST, 2, -9.0, 1, True, 0.5),
    ]

    summary = summarize(results)

    assert "games=3" in summary
    assert "wins=1" in summary
    assert "losses=2" in summary
    assert "win_rate=33.3%" in summary
    assert "avg_steps=4.7" in summary
    assert "total=3.5s" in summary


def test_format_episode_contains_required_fields() -> None:
    line = format_episode(
        EpisodeResult(7, GameStatus.LOST, 12, -8.0, 11, True, 3.25)
    )

    assert "episode 007" in line
    assert "LOST" in line
    assert "steps= 12" in line
    assert "reward=   -8.0" in line
    assert "revealed= 11" in line
    assert "mine=yes" in line
    assert "3.25s" in line
