from __future__ import annotations

import re

import numpy as np
import pytest

from minesweeper_rl.actions import ActionTimeoutError, InvalidActionError
from minesweeper_rl.config import EnvironmentConfig
from minesweeper_rl.environment import MinesweeperEnvironment


def make_cell(x: int, y: int, *classes: str) -> dict[str, object]:
    return {"x": str(x), "y": str(y), "classes": ["cell", "size24", *classes]}


def closed_board() -> list[dict[str, object]]:
    return [make_cell(x, y, "hd_closed") for y in range(2) for x in range(2)]


def set_cell(cells: list[dict[str, object]], x: int, y: int, *classes: str) -> None:
    for cell in cells:
        if cell["x"] == str(x) and cell["y"] == str(y):
            cell["classes"] = ["cell", "size24", *classes]
            return
    raise AssertionError(f"cell ({x}, {y}) not found")


def reveal_cell(cells: list[dict[str, object]], x: int, y: int, value: int = 0) -> None:
    set_cell(cells, x, y, "hd_opened", f"hd_type{value}")


class FakeLocator:
    def __init__(self, page: "FakePage", x: int | None, y: int | None) -> None:
        self.page = page
        self.x = x
        self.y = y

    def evaluate_all(self, _: str) -> list[dict[str, object]]:
        return [
            {"x": c["x"], "y": c["y"], "classes": list(c["classes"])}
            for c in self.page.cells
        ]

    def count(self) -> int:
        return 1

    def scroll_into_view_if_needed(self) -> None:
        return None

    def click(self) -> None:
        self.page.click_calls.append((self.x, self.y))
        if self.page.click_hook is not None:
            self.page.click_hook(self.x, self.y)

    def dispatch_event(self, name: str, _: object) -> None:
        self.page.dispatch_calls.append((self.x, self.y, name))
        if name == "click" and self.page.dispatch_hook is not None:
            self.page.dispatch_hook(self.x, self.y)


class FakePage:
    def __init__(self, cells: list[dict[str, object]]) -> None:
        self.cells = cells
        self.click_calls: list[tuple[int | None, int | None]] = []
        self.dispatch_calls: list[tuple[int | None, int | None, str]] = []
        self.click_hook = None
        self.dispatch_hook = None

    def locator(self, selector: str) -> FakeLocator:
        x = re.search(r'data-x="(\d+)"', selector)
        y = re.search(r'data-y="(\d+)"', selector)
        return FakeLocator(self, int(x.group(1)) if x else None, int(y.group(1)) if y else None)


def environment(page: FakePage) -> MinesweeperEnvironment:
    config = EnvironmentConfig(
        width=2,
        height=2,
        observation_timeout_ms=50,
        observation_poll_ms=0,
        action_timeout_ms=40,
        action_poll_ms=0,
    )
    return MinesweeperEnvironment(page, config)


def test_reveal_validates_coordinates_before_touching_the_board() -> None:
    page = FakePage(closed_board())
    env = environment(page)

    with pytest.raises(IndexError, match="outside board"):
        env.reveal(2, 0)
    with pytest.raises(TypeError, match="integers"):
        env.reveal(0.5, 0)
    assert page.click_calls == []


def test_reveal_rejects_already_revealed_cell() -> None:
    cells = closed_board()
    reveal_cell(cells, 0, 0, 5)
    page = FakePage(cells)
    env = environment(page)

    with pytest.raises(InvalidActionError, match="already revealed"):
        env.reveal(0, 0)
    assert page.click_calls == []


def test_reveal_rejects_flagged_cell() -> None:
    cells = closed_board()
    set_cell(cells, 1, 1, "hd_closed", "hd_flag")
    page = FakePage(cells)
    env = environment(page)

    with pytest.raises(InvalidActionError, match="not hidden"):
        env.reveal(1, 1)
    assert page.click_calls == []


def test_reveal_clicks_and_returns_changed_board() -> None:
    page = FakePage(closed_board())
    page.click_hook = lambda x, y: reveal_cell(page.cells, x, y, 3)
    env = environment(page)

    board = env.reveal(0, 1)

    assert isinstance(board, np.ndarray)
    assert board[1, 0] == 3
    assert page.click_calls == [(0, 1)]
    assert page.dispatch_calls == []


def test_reveal_accepts_action_tuple() -> None:
    page = FakePage(closed_board())
    page.click_hook = lambda x, y: reveal_cell(page.cells, x, y, 1)
    env = environment(page)

    board = env.reveal((1, 0))

    assert board[0, 1] == 1


def test_reveal_falls_back_to_event_sequence_when_click_has_no_effect() -> None:
    page = FakePage(closed_board())
    page.dispatch_hook = lambda x, y: reveal_cell(page.cells, x, y, 2)
    env = environment(page)

    board = env.reveal(0, 0)

    assert board[0, 0] == 2
    assert page.click_calls == [(0, 0)]
    assert [call[2] for call in page.dispatch_calls] == ["mousedown", "mouseup", "click"]


def test_reveal_falls_back_when_playwright_click_raises() -> None:
    page = FakePage(closed_board())

    def boom(_x: int, _y: int) -> None:
        raise RuntimeError("element is not clickable")

    page.click_hook = boom
    page.dispatch_hook = lambda x, y: reveal_cell(page.cells, x, y, 4)
    env = environment(page)

    board = env.reveal(1, 1)

    assert board[1, 1] == 4
    assert page.dispatch_calls


def test_reveal_raises_timeout_when_board_never_changes() -> None:
    page = FakePage(closed_board())
    env = environment(page)

    with pytest.raises(ActionTimeoutError, match="no observable board change"):
        env.reveal(0, 0)
