from __future__ import annotations

import os
from urllib.parse import urlsplit

import numpy as np
import pytest

from minesweeper_rl.actions import InvalidActionError
from minesweeper_rl.browser import BrowserController
from minesweeper_rl.config import EnvironmentConfig
from minesweeper_rl.dom import GameStatus
from minesweeper_rl.environment import (
    MINE_REWARD,
    SAFE_REVEAL_REWARD,
    WIN_REWARD,
    MinesweeperEnvironment,
)
from minesweeper_rl.logging_config import enable_diagnostics

EXPERT_CELL_COUNT = 16 * 30
BOARD_SELECTOR = ".cell[data-x][data-y]"


def ensure_expert_board(page) -> None:
    """Start an expert game when the current page does not already show one."""

    if page.locator(BOARD_SELECTOR).count() >= EXPERT_CELL_COUNT:
        return
    expert = page.locator("a[href='/start/3'], a.level3-link").first
    if expert.count():
        expert.click()
    page.wait_for_function(
        "([selector, count]) => document.querySelectorAll(selector).length >= count",
        arg=[BOARD_SELECTOR, EXPERT_CELL_COUNT],
        timeout=60_000,
    )


def _hidden_unflagged_cell(page, board) -> tuple[int, int]:
    for y in range(board.shape[0]):
        for x in range(board.shape[1]):
            if board[y, x] != -1:
                continue
            classes = page.locator(
                f'{BOARD_SELECTOR}[data-x="{x}"][data-y="{y}"]'
            ).get_attribute("class") or ""
            if "flag" not in classes:
                return x, y
    raise AssertionError("no hidden, unflagged cell found on the live board")


@pytest.mark.live
def test_live_expert_board_observation() -> None:
    url = os.environ.get("MINESWEEPER_URL")
    if not url:
        pytest.skip("set MINESWEEPER_URL to enable the live website test")

    diagnostics = os.environ.get("MINESWEEPER_DIAGNOSTICS", "").lower() in {"1", "true", "yes"}
    if diagnostics:
        enable_diagnostics()

    controller = BrowserController(
        url,
        os.environ.get("MINESWEEPER_PROFILE", "browser_profile"),
        wait_for_board=False,
    )
    try:
        page = controller.start()
        ensure_expert_board(page)

        env = MinesweeperEnvironment(page, EnvironmentConfig(diagnostics=diagnostics))
        env.wait_until_ready()
        board = env.observe()

        assert board.shape == (16, 30)
        assert board.dtype == np.int8
        assert np.isin(board, np.arange(-1, 9)).all()
    finally:
        controller.shutdown()


@pytest.mark.live
def test_live_reveal_hidden_cell() -> None:
    url = os.environ.get("MINESWEEPER_URL")
    if not url:
        pytest.skip("set MINESWEEPER_URL to enable the live action test")

    diagnostics = os.environ.get("MINESWEEPER_DIAGNOSTICS", "").lower() in {"1", "true", "yes"}
    if diagnostics:
        enable_diagnostics()

    controller = BrowserController(
        url,
        os.environ.get("MINESWEEPER_PROFILE", "browser_profile"),
        wait_for_board=False,
    )
    try:
        page = controller.start()
        ensure_expert_board(page)

        env = MinesweeperEnvironment(page, EnvironmentConfig(diagnostics=diagnostics))
        env.wait_until_ready()

        before = env.observe()
        x, y = _hidden_unflagged_cell(page, before)
        before_value = int(before[y, x])

        after = env.reveal(x, y)
        after_value = int(after[y, x])

        print(f"reveal ({x}, {y}): before={before_value} after={after_value}")
        print(f"revealed cells: {int((before == -1).sum())} -> {int((after == -1).sum())} hidden")

        assert before_value == -1
        assert after_value != -1
        assert not np.array_equal(before, after)
    finally:
        controller.shutdown()


@pytest.mark.live
def test_live_reset_starts_a_fresh_game_with_a_controlled_first_move() -> None:
    url = os.environ.get("MINESWEEPER_URL")
    if not url:
        pytest.skip("set MINESWEEPER_URL to enable the live reset test")

    diagnostics = os.environ.get("MINESWEEPER_DIAGNOSTICS", "").lower() in {"1", "true", "yes"}
    if diagnostics:
        enable_diagnostics()

    controller = BrowserController(
        url,
        os.environ.get("MINESWEEPER_PROFILE", "browser_profile"),
        wait_for_board=False,
    )
    try:
        page = controller.start()
        env = MinesweeperEnvironment(page, EnvironmentConfig(diagnostics=diagnostics))

        first = env.reset()
        assert first.shape == (16, 30)
        assert first.dtype == np.int8
        assert (first != -1).any()
        assert env.get_game_status() is GameStatus.RUNNING

        second = env.reset()
        assert second.shape == (16, 30)
        assert (second != -1).any()
        assert env.get_game_status() is GameStatus.RUNNING
    finally:
        controller.shutdown()


@pytest.mark.live
def test_live_step_reward_and_terminal_consistency() -> None:
    url = os.environ.get("MINESWEEPER_URL")
    if not url:
        pytest.skip("set MINESWEEPER_URL to enable the live step test")

    diagnostics = os.environ.get("MINESWEEPER_DIAGNOSTICS", "").lower() in {"1", "true", "yes"}
    if diagnostics:
        enable_diagnostics()

    controller = BrowserController(
        url,
        os.environ.get("MINESWEEPER_PROFILE", "browser_profile"),
        wait_for_board=False,
    )
    try:
        page = controller.start()
        env = MinesweeperEnvironment(page, EnvironmentConfig(diagnostics=diagnostics))
        env.reset()

        safe_steps = 0
        terminated = False
        for _ in range(60):
            state = env.observe()
            x, y = _hidden_unflagged_cell(page, state)
            previous = int(state[y, x])
            next_state, reward, terminated, truncated, info = env.step((x, y))

            assert next_state.shape == (16, 30)
            assert truncated is False
            assert info["action"] == (x, y)
            assert info["previous_value"] == previous == -1
            assert info["newly_revealed"] >= 1
            assert info["opened_cells"] >= info["newly_revealed"]
            assert info["elapsed_ms"] >= 0.0

            status = info["game_status"]
            if status is GameStatus.LOST:
                assert reward == MINE_REWARD
                assert info["hit_mine"] is True
                assert terminated is True
            elif status is GameStatus.WON:
                assert reward == WIN_REWARD
                assert info["hit_mine"] is False
                assert terminated is True
            else:
                assert reward == SAFE_REVEAL_REWARD
                assert info["hit_mine"] is False
                assert terminated is False
                safe_steps += 1

            if terminated:
                break

        assert safe_steps >= 1
    finally:
        controller.shutdown()


@pytest.mark.live
def test_live_step_rejects_already_revealed_and_out_of_bounds() -> None:
    url = os.environ.get("MINESWEEPER_URL")
    if not url:
        pytest.skip("set MINESWEEPER_URL to enable the live step test")

    controller = BrowserController(
        url,
        os.environ.get("MINESWEEPER_PROFILE", "browser_profile"),
        wait_for_board=False,
    )
    try:
        page = controller.start()
        env = MinesweeperEnvironment(page, EnvironmentConfig())
        state = env.reset()

        revealed = [
            (x, y)
            for y in range(state.shape[0])
            for x in range(state.shape[1])
            if state[y, x] != -1
        ]
        x, y = revealed[0]
        with pytest.raises(InvalidActionError, match="already revealed"):
            env.step((x, y))
        with pytest.raises(IndexError, match="outside board"):
            env.step((state.shape[1], 0))
    finally:
        controller.shutdown()


@pytest.mark.live
def test_live_game_status_ready_running_lost() -> None:
    url = os.environ.get("MINESWEEPER_URL")
    if not url:
        pytest.skip("set MINESWEEPER_URL to enable the live game-status test")

    diagnostics = os.environ.get("MINESWEEPER_DIAGNOSTICS", "").lower() in {"1", "true", "yes"}
    if diagnostics:
        enable_diagnostics()

    controller = BrowserController(
        url,
        os.environ.get("MINESWEEPER_PROFILE", "browser_profile"),
        wait_for_board=False,
    )
    try:
        page = controller.start()
        ensure_expert_board(page)

        env = MinesweeperEnvironment(page, EnvironmentConfig(diagnostics=diagnostics))
        env.wait_until_ready()

        assert env.get_game_status() is GameStatus.READY

        x, y = _hidden_unflagged_cell(page, env.observe())
        env.reveal(x, y)
        assert env.get_game_status() is GameStatus.RUNNING

        for _ in range(480):
            if env.get_game_status() is GameStatus.LOST:
                break
            x, y = _hidden_unflagged_cell(page, env.observe())
            env.reveal(x, y)

        assert env.get_game_status() is GameStatus.LOST
    finally:
        controller.shutdown()


@pytest.mark.live
def test_live_game_status_won() -> None:
    url = os.environ.get("MINESWEEPER_URL")
    if not url:
        pytest.skip("set MINESWEEPER_URL to enable the live game-status test")

    diagnostics = os.environ.get("MINESWEEPER_DIAGNOSTICS", "").lower() in {"1", "true", "yes"}
    if diagnostics:
        enable_diagnostics()

    parts = urlsplit(url)
    won_url = f"{parts.scheme}://{parts.netloc}/start/9x9/0"
    controller = BrowserController(
        won_url,
        os.environ.get("MINESWEEPER_PROFILE", "browser_profile"),
        wait_for_board=False,
    )
    try:
        page = controller.start()
        page.wait_for_function(
            "() => document.querySelectorAll('.cell[data-x][data-y]').length >= 81",
            timeout=30_000,
        )
        env = MinesweeperEnvironment(
            page, EnvironmentConfig(width=9, height=9, diagnostics=diagnostics)
        )
        env.wait_until_ready()

        assert env.get_game_status() is GameStatus.READY

        # A mine-free board is safe to clear completely; reveal hidden cells
        # until the site reports the win.
        for _ in range(81):
            if env.get_game_status() is GameStatus.WON:
                break
            x, y = _hidden_unflagged_cell(page, env.observe())
            env.reveal(x, y)

        assert env.get_game_status() is GameStatus.WON
    finally:
        controller.shutdown()
