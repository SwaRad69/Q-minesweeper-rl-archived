"""Convert Minesweeper.com cell elements into a numeric board.

The live site renders each cell as a ``div`` whose classes encode the game
state.  Three class-name prefixes are recognised:

- ``hd_``   — standard games (original site scheme)
- ``hdd_``  — standard games (updated site scheme, observed 2026-09)
- ``nnhd_`` — "no guessing" mode

The relevant class suffixes are:

``*_closed``
    A covered, unflagged cell (:data:`HIDDEN`).
``*_flag`` / ``*_closed_flag``
    A covered, flagged cell (:data:`FLAGGED`).  ``*_closed_flag`` is used by
    some modes instead of the separate ``*_flag`` decoration.
``*_opened`` + ``*_typeN``
    A revealed cell.  ``N`` is the adjacent mine count (``0``..``8``).  The
    site adds ``*_opened`` first and then the ``*_typeN`` number; the two are
    set synchronously, so an ``*_opened`` cell without a number is treated as
    malformed.  Terminal reveal states use ``*_type10``..``*_type12`` (mine,
    wrong flag); those carry no playable number and are reported as hidden.
``*_pressed`` / ``*_check``
    Transient decorations that may accompany a covered cell.

Any other prefixed class is rejected as unexpected.  Classes without a
recognised prefix (``size24``, ``size28``, ``blackout``, ``hint-*`` ...) are
layout and hint decorations and are ignored.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from playwright.sync_api import Page

HIDDEN = -1
FLAGGED = -2
MAX_NUMBER = 8

_CLOSED = frozenset({"hd_closed", "hdd_closed", "nnhd_closed"})
_CLOSED_FLAG = frozenset({"hd_closed_flag", "hdd_closed_flag", "nnhd_closed_flag"})
_OPENED = frozenset({"hd_opened", "hdd_opened", "nnhd_opened"})
_FLAG = frozenset({"hd_flag", "hdd_flag", "nnhd_flag"})
_DECORATIONS = frozenset({
    "hd_pressed", "hdd_pressed", "nnhd_pressed",
    "hd_check", "hdd_check", "nnhd_check",
})
_STATE_CLASSES = _CLOSED | _CLOSED_FLAG | _OPENED | _FLAG | _DECORATIONS

_TYPE = re.compile(r"^(?:nn)?hdd?_type(?P<number>\d+)$")
_PREFIXED = re.compile(r"^(?:nn)?hdd?_")

_NEIGHBOUR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def chord_is_legal(board: npt.ArrayLike, x: int, y: int) -> bool:
    """Return whether OPEN on an opened numbered cell is a legal chord.

    A chord is legal when the cell is an opened number ``n`` (``1..8``), exactly
    ``n`` of its eight neighbours are FLAGGED, and at least one neighbour is
    still hidden.  This uses only observable information -- it never inspects
    the hidden mine layout -- and mirrors the site's chord condition.
    """

    array = np.asarray(board)
    height, width = array.shape
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"coordinate outside board: ({x}, {y})")
    number = int(array[y, x])
    if not 1 <= number <= MAX_NUMBER:
        return False
    flagged = hidden = 0
    for dx, dy in _NEIGHBOUR_OFFSETS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            value = int(array[ny, nx])
            if value == FLAGGED:
                flagged += 1
            elif value == HIDDEN:
                hidden += 1
    return flagged == number and hidden > 0


class BoardShapeError(ValueError):
    """Raised when the DOM does not describe one complete rectangular board."""


class BoardMissingError(BoardShapeError):
    """Raised when the DOM contains no board cells at all."""


class CellFormatError(ValueError):
    """Raised when a DOM cell has invalid coordinates or state classes."""


@dataclass(frozen=True, slots=True)
class CellState:
    """A parsed cell, using website coordinates and the experiment value."""

    x: int
    y: int
    value: int


def parse_board(
    page: Page,
    *,
    selector: str = ".cell[data-x][data-y]",
    shape: tuple[int, int] = (16, 30),
) -> npt.NDArray[np.int8]:
    """Read cell classes and coordinates from ``page`` into ``[y, x]`` order."""

    return parse_board_snapshot(
        read_cell_snapshot(page, selector=selector), expected_shape=shape
    )


def read_cell_snapshot(page: Page, *, selector: str = ".cell[data-x][data-y]") -> list[dict[str, Any]]:
    """Return the relevant DOM data for one atomic locator evaluation.

    ``evaluate_all`` runs a single synchronous pass over the matched elements,
    so a snapshot is never interleaved with another read; callers still compare
    consecutive snapshots to detect updates between passes.
    """

    return page.locator(selector).evaluate_all(
        """cells => cells.map(cell => {
            return {
                x: cell.dataset.x,
                y: cell.dataset.y,
                classes: [...cell.classList]
            };
        })"""
    )


def parse_board_snapshot(
    raw: Sequence[dict[str, Any]], *, expected_shape: tuple[int, int] = (16, 30)
) -> npt.NDArray[np.int8]:
    """Validate and convert a previously captured DOM snapshot."""

    if not raw:
        raise BoardMissingError("no board cells present in the DOM")

    cells: list[CellState] = []
    for item in raw:
        cells.append(_parse_cell(item))
    return board_from_cells(cells, expected_shape=expected_shape)


def _parse_cell(item: dict[str, Any]) -> CellState:
    classes = set(item["classes"])
    x = _parse_coordinate(item["x"], "x")
    y = _parse_coordinate(item["y"], "y")
    location = f"({x}, {y})"

    for name in classes:
        if _PREFIXED.match(name) and name not in _STATE_CLASSES and not _TYPE.match(name):
            raise CellFormatError(f"cell {location} has unexpected class: {name!r}")

    opened = bool(classes & _OPENED)
    covered = bool(classes & (_CLOSED | _CLOSED_FLAG))
    if opened and covered:
        raise CellFormatError(
            f"cell {location} cannot be both opened and covered"
        )
    if not opened and not covered:
        raise CellFormatError(
            f"cell {location} must have exactly one of hd_opened or hd_closed"
        )

    type_classes = [name for name in classes if _TYPE.match(name)]
    numbers: list[int] = []
    for name in type_classes:
        match = _TYPE.match(name)
        assert match is not None
        numbers.append(int(match.group("number")))
    if len(numbers) > 1:
        raise CellFormatError(f"cell {location} has multiple number classes")

    if opened:
        if not numbers:
            raise CellFormatError(f"opened cell {location} has no number class")
        number = numbers[0]
        if 0 <= number <= MAX_NUMBER:
            return CellState(x=x, y=y, value=number)
        if number in (10, 11, 12):
            # Revealed mine / wrong flag at game over carries no playable value.
            return CellState(x=x, y=y, value=HIDDEN)
        raise CellFormatError(f"cell {location} has invalid number class: hd_type{number}")

    if numbers:
        raise CellFormatError(f"closed cell {location} has a number class")
    flagged = bool(classes & (_FLAG | _CLOSED_FLAG))
    return CellState(x=x, y=y, value=FLAGGED if flagged else HIDDEN)


def _parse_coordinate(raw: Any, axis: str) -> int:
    if not isinstance(raw, str) or not raw.isdigit():
        raise CellFormatError(f"cell has invalid {axis} coordinate: {raw!r}")
    return int(raw)


def board_from_cells(
    cells: Sequence[CellState], *, expected_shape: tuple[int, int]
) -> npt.NDArray[np.int8]:
    """Validate parsed cells and return a ``numpy.int8`` board."""

    if len(cells) != expected_shape[0] * expected_shape[1]:
        raise BoardShapeError(
            f"DOM contains {len(cells)} cells but expected {expected_shape[0] * expected_shape[1]}"
        )
    coordinates: set[tuple[int, int]] = set()
    board = np.full(expected_shape, HIDDEN, dtype=np.int8)
    for cell in cells:
        if not (0 <= cell.x < expected_shape[1] and 0 <= cell.y < expected_shape[0]):
            raise BoardShapeError(f"cell coordinate outside board: ({cell.x}, {cell.y})")
        if cell.value not in range(FLAGGED, MAX_NUMBER + 1):
            raise CellFormatError(f"invalid cell value: {cell.value}")
        if (cell.x, cell.y) in coordinates:
            raise BoardShapeError(f"duplicate cell coordinate: ({cell.x}, {cell.y})")
        coordinates.add((cell.x, cell.y))
        board[cell.y, cell.x] = cell.value
    expected_coordinates = {
        (x, y)
        for y in range(expected_shape[0])
        for x in range(expected_shape[1])
    }
    missing = expected_coordinates - coordinates
    if missing:
        raise BoardShapeError(f"missing cell coordinates: {sorted(missing)}")
    return board
