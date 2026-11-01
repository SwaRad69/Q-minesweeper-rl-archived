"""Opt-in live integration coverage for the human recorder lifecycle."""

from __future__ import annotations

import os
import time
from urllib.parse import urlsplit

import numpy as np
import pytest

from minesweeper_rl.browser import BrowserController
from minesweeper_rl.config import EnvironmentConfig
from minesweeper_rl.data import HumanDatasetStore, PersistentHumanRecorder
from minesweeper_rl.data.human_dataset import validate_dataset
from minesweeper_rl.data.record_human import HumanTrajectoryRecorder
from minesweeper_rl.dom import GameStatus, HIDDEN
from minesweeper_rl.environment import MinesweeperEnvironment


@pytest.mark.live
def test_live_long_active_human_capture_reaches_real_terminal_result(tmp_path) -> None:
    """A live game remains active beyond action_timeout and ends from the face.

    Playwright's low-level mouse input produces the same trusted browser pointer
    events as physical input, so this checks the production capture listener and
    DOM.  It is deliberately opt-in; the documented manual smoke test remains
    the final human-input check.
    """

    base_url = os.environ.get("MINESWEEPER_URL")
    if not base_url:
        pytest.skip("set MINESWEEPER_URL to enable the live human-recorder test")
    parts = urlsplit(base_url)
    url = f"{parts.scheme}://{parts.netloc}/start/9x9/10"
    profile = os.environ.get("MINESWEEPER_PROFILE", "browser_profile")
    config = EnvironmentConfig(width=9, height=9, mines=10, difficulty="custom")

    store = HumanDatasetStore(
        tmp_path / "human", width=9, height=9, mines=10, difficulty="custom"
    )
    sink = PersistentHumanRecorder(
        store, width=9, height=9, mines=10, difficulty="custom"
    )
    controller = BrowserController(url, profile, wait_for_board=False)
    try:
        page = controller.start()
        env = MinesweeperEnvironment(page, config)
        env.wait_until_ready(60_000)
        human = HumanTrajectoryRecorder(
            env, sink, poll_interval=0.02, action_timeout=1.0,
            quiescence_time=0.05, emit=lambda _message: None,
        )
        human.install_capture()
        initial = env.observe()
        assert np.all(initial == HIDDEN)
        human._handle_idle_board(initial)

        def click_and_record(x: int, y: int) -> str:
            page.locator(f'.cell[data-x="{x}"][data-y="{y}"]').click()
            events = human.drain_events()
            assert events, "the live capture listener missed a trusted click"
            result = "rejected"
            for event in events:
                result = human.handle_event(event)
            return result

        assert click_and_record(0, 0) == "recorded"
        # Poll the unchanged active board for longer than action_timeout.  This
        # is real wall time, not a patched clock.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            human._handle_idle_board(env.observe())
        assert sink.active_game_id is not None
        assert store.game_count == 0

        for _ in range(81):
            if sink.active_game_id is None:
                break
            board = env.observe()
            hidden = np.argwhere(board == HIDDEN)
            assert len(hidden), "active live game has no hidden cell"
            y, x = (int(value) for value in hidden[0])
            click_and_record(x, y)

        assert sink.active_game_id is None
        game = store.load_game("game_000001")
        assert game["result"] in {GameStatus.WON.value, GameStatus.LOST.value}
        assert game["complete"] is True
        assert game["termination_reason"] is None
        assert validate_dataset(store.root).ok
    finally:
        controller.shutdown()
