from __future__ import annotations

import numpy as np
import pytest

from minesweeper_rl.config import EnvironmentConfig
from minesweeper_rl.environment import MinesweeperEnvironment

CELL_SELECTOR = ".cell[data-x][data-y]"


def make_cell(x: int, y: int, *classes: str) -> dict[str, object]:
    return {"x": str(x), "y": str(y), "classes": ["cell", "size24", *classes]}


class FakeFace:
    def __init__(self, page: "FakePage") -> None:
        self.page = page
        self.first = self

    def wait_for(self, *, state: str, timeout: float) -> None:
        assert state == "attached"

    def click(self) -> None:
        self.page.restart()


class FakeAllCells:
    def __init__(self, page: "FakePage") -> None:
        self.page = page

    def evaluate_all(self, _: str) -> list[dict[str, object]]:
        return [
            {"x": cell["x"], "y": cell["y"], "classes": list(cell["classes"])}
            for cell in self.page.cells
        ]

    def count(self) -> int:
        return len(self.page.cells)


class FakeCell:
    def __init__(self, page: "FakePage", x: int, y: int) -> None:
        self.page = page
        self.x = x
        self.y = y

    def count(self) -> int:
        return 1

    def scroll_into_view_if_needed(self) -> None:
        return None

    def click(self) -> None:
        self.page.click_cell(self.x, self.y)

    def dispatch_event(self, _name: str, _payload: object) -> None:
        self.page.click_cell(self.x, self.y)


class FakePage:
    def __init__(self, *, width: int = 2, height: int = 2) -> None:
        self.width = width
        self.height = height
        self.game_id = 100
        self.url = f"https://minesweeper.online/game/{self.game_id}"
        self.face_classes = ["top-area-face", "hd_top-area-face-unpressed"]
        self.cells: list[dict[str, object]] = []
        self.lose_moves = 0
        self.wait_timeouts: list[float] = []
        self.ready = True
        self._fresh()

    def _fresh(self) -> None:
        self.cells = [
            make_cell(x, y, "hd_closed")
            for y in range(self.height)
            for x in range(self.width)
        ]
        self.face_classes = ["top-area-face", "hd_top-area-face-unpressed"]

    def restart(self) -> None:
        self.game_id += 1
        self.url = f"https://minesweeper.online/game/{self.game_id}"
        self._fresh()

    def click_cell(self, x: int, y: int) -> None:
        if self.lose_moves > 0:
            self.lose_moves -= 1
            self._set(x, y, "hd_opened", "hd_type10")
            self.face_classes = ["top-area-face", "hd_top-area-face-lose"]
            return
        self._set(x, y, "hd_opened", "hd_type3")

    def _set(self, x: int, y: int, *classes: str) -> None:
        for cell in self.cells:
            if cell["x"] == str(x) and cell["y"] == str(y):
                cell["classes"] = ["cell", "size24", *classes]
                return
        raise AssertionError(f"cell ({x}, {y}) not found")

    def locator(self, selector: str) -> object:
        if selector == CELL_SELECTOR:
            return FakeAllCells(self)
        if selector.startswith(CELL_SELECTOR):
            x = int(selector.split('data-x="')[1].split('"')[0])
            y = int(selector.split('data-y="')[1].split('"')[0])
            return FakeCell(self, x, y)
        if selector == "#top_area_face":
            return FakeFace(self)
        raise AssertionError(f"unexpected selector: {selector}")

    def wait_for_function(
        self,
        _: str,
        *,
        arg: list[object] | None = None,
        timeout: float | None = None,
    ) -> None:
        if not self.ready:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

            raise PlaywrightTimeoutError("board never appeared")
        if timeout is not None:
            self.wait_timeouts.append(timeout)

    def evaluate(self, _: str) -> list[str]:
        return list(self.face_classes)


def environment(page: FakePage, **overrides: object) -> MinesweeperEnvironment:
    settings: dict[str, object] = {
        "width": page.width,
        "height": page.height,
        "cell_selector": CELL_SELECTOR,
        "observation_timeout_ms": 100,
        "observation_poll_ms": 0,
        "action_timeout_ms": 50,
        "action_poll_ms": 0,
        "new_game_timeout_ms": 100,
    }
    settings.update(overrides)
    return MinesweeperEnvironment(page, EnvironmentConfig(**settings))


def test_reset_starts_a_new_game_and_returns_the_first_move_board() -> None:
    page = FakePage()
    env = environment(page)

    board = env.reset()

    assert isinstance(board, np.ndarray)
    assert board.dtype == np.int8
    assert board.shape == (2, 2)
    assert board[1, 1] == 3
    assert board[0, 0] == -1
    assert page.game_id == 101


def test_reset_uses_the_configured_first_move() -> None:
    page = FakePage()
    env = environment(page, first_move=(0, 1))

    board = env.reset()

    assert board[1, 0] == 3
    assert board[1, 1] == -1


def test_reset_retries_the_first_move_on_a_fresh_board_after_a_loss() -> None:
    page = FakePage()
    page.lose_moves = 1
    env = environment(page)

    board = env.reset()

    assert page.game_id == 102
    assert board[1, 1] == 3


def test_reset_raises_when_the_first_move_always_loses() -> None:
    page = FakePage()
    page.lose_moves = 10
    env = environment(page)

    with pytest.raises(RuntimeError, match="first-click guarantee"):
        env.reset()
    assert page.game_id == 103


def test_reset_raises_when_no_board_is_available() -> None:
    page = FakePage()
    page.ready = False
    env = environment(page)

    with pytest.raises(RuntimeError, match="no Minesweeper board"):
        env.reset()


def test_reset_uses_the_new_game_timeout_for_readiness_waits() -> None:
    page = FakePage()
    env = environment(page, ready_timeout_ms=7, new_game_timeout_ms=123)

    env.reset()

    assert page.wait_timeouts
    assert set(page.wait_timeouts) == {123.0}
