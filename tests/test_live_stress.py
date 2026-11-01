from __future__ import annotations

import os

import pytest

from minesweeper_rl.browser import BrowserController
from minesweeper_rl.stress import run_stress

CHECK_NAMES = {
    "invalid:out_of_bounds",
    "invalid:non_integer",
    "invalid:malformed_action",
    "invalid:already_revealed",
    "invalid:no_state_corruption",
    "board_parsing:shape_dtype_values",
    "dom_sync:stable_idle_observations",
    "recovery:page_reload",
    "unexpected_state:boardless_page",
}


@pytest.mark.live
def test_live_stress_smoke() -> None:
    url = os.environ.get("MINESWEEPER_URL")
    if not url:
        pytest.skip("set MINESWEEPER_URL to enable the live stress test")

    controller = BrowserController(
        url,
        os.environ.get("MINESWEEPER_PROFILE", "browser_profile"),
        wait_for_board=False,
    )
    try:
        report = run_stress(controller, url, 2, seed=0, win_phase=False)
    finally:
        controller.shutdown()

    assert report.attempted == 2
    assert report.reliability_percent == 100.0
    assert report.metrics.step.count > 0
    assert CHECK_NAMES.issubset({check.name for check in report.checks})
    assert [check for check in report.checks if not check.passed] == []
