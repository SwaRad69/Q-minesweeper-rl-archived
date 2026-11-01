from __future__ import annotations

import os

import pytest

from minesweeper_rl.agents.random_agent import run_random_agent
from minesweeper_rl.browser import BrowserController
from minesweeper_rl.config import EnvironmentConfig
from minesweeper_rl.dom import GameStatus
from minesweeper_rl.environment import MinesweeperEnvironment


@pytest.mark.live
def test_live_random_agent_plays_complete_games() -> None:
    url = os.environ.get("MINESWEEPER_URL")
    if not url:
        pytest.skip("set MINESWEEPER_URL to enable the live random-agent test")

    controller = BrowserController(
        url,
        os.environ.get("MINESWEEPER_PROFILE", "browser_profile"),
        wait_for_board=False,
    )
    try:
        page = controller.start()
        env = MinesweeperEnvironment(page, EnvironmentConfig())
        results = run_random_agent(env, 2)

        assert len(results) == 2
        assert [result.episode for result in results] == [1, 2]
        assert all(result.status in (GameStatus.WON, GameStatus.LOST) for result in results)
        assert all(result.steps >= 1 for result in results)
        assert all(
            result.hit_mine is (result.status is GameStatus.LOST)
            for result in results
        )
        assert all(result.duration_s >= 0.0 for result in results)
    finally:
        controller.shutdown()
