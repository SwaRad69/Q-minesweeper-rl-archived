from __future__ import annotations

import numpy as np
import pytest
from playwright.sync_api import sync_playwright

from minesweeper_rl.config import EnvironmentConfig
from minesweeper_rl.dom import (
    FLAGGED,
    BoardShapeError,
    CellFormatError,
    parse_board,
    parse_board_snapshot,
)
from minesweeper_rl.environment import MinesweeperEnvironment


class FakeLocator:
    def __init__(self, cells: list[dict[str, object]]) -> None:
        self.cells = cells

    def evaluate_all(self, _: str) -> list[dict[str, object]]:
        return self.cells


class FakePage:
    def __init__(self, cells: list[dict[str, object]]) -> None:
        self.locator_instance = FakeLocator(cells)

    def locator(self, _: str) -> FakeLocator:
        return self.locator_instance


def cell(x: object, y: object, *classes: str) -> dict[str, object]:
    return {"x": x, "y": y, "classes": list(classes)}


def test_observe_parses_real_dom_classes() -> None:
    markup = """
        <div class="cell size24 hd_closed" data-x="0" data-y="0"></div>
        <div class="cell size24 hd_opened hd_type0" data-x="1" data-y="0"></div>
        <div class="cell size24 hd_opened hd_type8" data-x="0" data-y="1"></div>
        <div class="cell size24 hd_closed hd_flag" data-x="1" data-y="1"></div>
    """
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except Exception as error:
            pytest.skip(f"Chromium is not installed: {error}")
        page = browser.new_page()
        page.set_content(markup)

        config = EnvironmentConfig(width=2, height=2)
        board = MinesweeperEnvironment(page, config).observe()

        browser.close()

    np.testing.assert_array_equal(board, [[-1, 0], [8, FLAGGED]])


def test_left_click_dispatches_mouse_sequence_to_coordinate() -> None:
    markup = """
        <div class="cell hd_closed" data-x="0" data-y="0"></div>
        <div class="cell hd_closed" data-x="1" data-y="0"></div>
        <output id="events"></output>
        <script>
            const events = [];
            const cell = document.querySelector('[data-x="1"]');
            for (const name of ['mousedown', 'mouseup', 'click']) {
                cell.addEventListener(name, event => events.push(`${name}:${event.button}`));
            }
            window.getEvents = () => events;
        </script>
    """
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(markup)

        config = EnvironmentConfig(width=2, height=1)
        MinesweeperEnvironment(page, config).left_click(x=1, y=0)
        events = page.evaluate("window.getEvents()")

        browser.close()

    assert events == ["mousedown:0", "mouseup:0", "click:0"]


def test_parse_board_reads_mocked_dom_into_y_x_matrix() -> None:
    page = FakePage([
        cell("0", "0", "cell", "size24", "hd_closed"),
        cell("1", "0", "cell", "size24", "hd_opened", "hd_type3"),
        cell("0", "1", "cell", "size24", "hd_opened", "hd_type0"),
        cell("1", "1", "cell", "size24", "hd_opened", "hd_type8"),
    ])

    board = parse_board(page, selector=".cell", shape=(2, 2))

    np.testing.assert_array_equal(board, [[-1, 3], [0, 8]])


def test_parse_board_marks_flagged_cells() -> None:
    raw = [
        cell("0", "0", "cell", "size24", "hd_closed", "hd_flag"),
        cell("1", "0", "cell", "size24", "hd_closed_flag"),
        cell("2", "0", "cell", "size24", "hd_closed"),
    ]

    board = parse_board_snapshot(raw, expected_shape=(1, 3))

    np.testing.assert_array_equal(board, [[FLAGGED, FLAGGED, -1]])


def test_parse_board_accepts_decorations_and_no_guessing_prefix() -> None:
    raw = [
        cell("0", "0", "cell", "size24", "hd_closed", "hd_pressed", "blackout"),
        cell("1", "0", "cell", "size24", "nnhd_opened", "nnhd_type2", "checking"),
    ]

    board = parse_board_snapshot(raw, expected_shape=(1, 2))

    np.testing.assert_array_equal(board, [[-1, 2]])


def test_parse_board_accepts_hdd_prefix() -> None:
    """The site switched from ``hd_*`` to ``hdd_*`` classes in 2026-09."""

    raw = [
        cell("0", "0", "cell", "size28", "hdd_closed"),
        cell("1", "0", "cell", "size28", "hdd_opened", "hdd_type3"),
        cell("0", "1", "cell", "size28", "hdd_closed", "hdd_flag"),
        cell("1", "1", "cell", "size28", "hdd_opened", "hdd_type0"),
    ]

    board = parse_board_snapshot(raw, expected_shape=(2, 2))

    np.testing.assert_array_equal(board, [[-1, 3], [FLAGGED, 0]])


def test_parse_board_maps_revealed_mines_to_hidden() -> None:
    raw = [cell("0", "0", "cell", "size24", "hd_opened", "hd_type10")]

    board = parse_board_snapshot(raw, expected_shape=(1, 1))

    np.testing.assert_array_equal(board, [[-1]])


@pytest.mark.parametrize(
    ("cells", "error", "message"),
    [
        ([cell("0", "0", "cell", "hd_closed")], BoardShapeError, "expected"),
        (
            [cell("2", "0", "cell", "hd_closed"), cell("1", "0", "cell", "hd_closed")],
            BoardShapeError,
            "outside board",
        ),
        (
            [cell("0", "0", "cell", "hd_closed"), cell("0", "0", "cell", "hd_closed")],
            BoardShapeError,
            "duplicate",
        ),
        (
            [cell("x", "0", "cell", "hd_closed"), cell("1", "0", "cell", "hd_closed")],
            CellFormatError,
            "invalid x",
        ),
        ([cell("0", "0", "cell"), cell("1", "0", "cell", "hd_closed")], CellFormatError, "exactly one"),
        (
            [cell("0", "0", "cell", "hd_closed", "hd_opened"), cell("1", "0", "cell", "hd_closed")],
            CellFormatError,
            "both opened and covered",
        ),
        (
            [cell("0", "0", "cell", "hd_closed", "hd_type1"), cell("1", "0", "cell", "hd_closed")],
            CellFormatError,
            "number class",
        ),
        (
            [cell("0", "0", "cell", "hd_opened", "hd_type1", "hd_type2"), cell("1", "0", "cell", "hd_closed")],
            CellFormatError,
            "multiple",
        ),
        (
            [cell("0", "0", "cell", "hd_opened"), cell("1", "0", "cell", "hd_closed")],
            CellFormatError,
            "no number class",
        ),
        (
            [cell("0", "0", "cell", "hd_opened", "hd_type9"), cell("1", "0", "cell", "hd_closed")],
            CellFormatError,
            "invalid number",
        ),
        (
            [cell("0", "0", "cell", "hd_opened", "hd_type1", "hd_unknown"), cell("1", "0", "cell", "hd_closed")],
            CellFormatError,
            "unexpected class",
        ),
    ],
)
def test_parse_board_rejects_malformed_mocked_dom(
    cells: list[dict[str, object]], error: type[ValueError], message: str
) -> None:
    with pytest.raises(error, match=message):
        parse_board_snapshot(cells, expected_shape=(1, 2))
