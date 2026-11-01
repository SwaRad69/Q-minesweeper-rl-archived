from __future__ import annotations

import os

import pytest

from minesweeper_rl.browser import BrowserController
from minesweeper_rl.config import EnvironmentConfig
from minesweeper_rl.environment import MinesweeperEnvironment

EXPERT_CELL_COUNT = 16 * 30
BOARD_SELECTOR = ".cell[data-x][data-y]"


@pytest.mark.live
def test_live_board_prints_matrix() -> None:
    url = os.environ.get("MINESWEEPER_URL")
    if not url:
        pytest.skip("set MINESWEEPER_URL to enable the live board test")

    controller = BrowserController(
        url,
        os.environ.get("MINESWEEPER_PROFILE", "browser_profile"),
        wait_for_board=False,
    )
    try:
        page = controller.start()
        if page.locator(BOARD_SELECTOR).count() < EXPERT_CELL_COUNT:
            expert = page.locator("a[href='/start/3'], a.level3-link").first
            if expert.count():
                expert.click()
            page.wait_for_function(
                "([selector, count]) => document.querySelectorAll(selector).length >= count",
                arg=[BOARD_SELECTOR, EXPERT_CELL_COUNT],
                timeout=60_000,
            )

        board = MinesweeperEnvironment(page, EnvironmentConfig()).observe()
        print("Current Minesweeper board matrix:")
        for row in board.tolist():
            print(row)
    finally:
        controller.shutdown()
