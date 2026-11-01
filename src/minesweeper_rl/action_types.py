"""Model-independent Minesweeper action vocabulary.

The three human-like operations are OPEN, FLAG and CHORD.  Their integer codes
define the compound action encoding used by the flat RL action space::

    action_id = action_type * cells + cell_index

where ``cell_index = y * width + x`` in row-major order.  This module has no
dependencies so it can be imported by the browser environment and the RL layer
alike without creating an import cycle.
"""

from __future__ import annotations

#: Open a hidden cell.  Code ``0`` so the OPEN block is ``0 .. cells - 1`` and
#: the historical OPEN-only action ids keep their meaning.
ACTION_OPEN = 0
#: Toggle a flag on a covered cell (place or remove).  Code ``1``.
ACTION_FLAG = 1
#: Chord an opened numbered cell whose flagged neighbours match its number.
#: Code ``2``.
ACTION_CHORD = 2

#: Canonical order of the three action types.
ACTION_TYPES: tuple[int, ...] = (ACTION_OPEN, ACTION_FLAG, ACTION_CHORD)

#: Human-readable operation name for each action type.
ACTION_NAMES: dict[int, str] = {
    ACTION_OPEN: "open",
    ACTION_FLAG: "flag",
    ACTION_CHORD: "chord",
}


def action_type_name(action_type: int) -> str:
    """Return the lowercase operation name for ``action_type``."""

    try:
        return ACTION_NAMES[int(action_type)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"unknown action type: {action_type!r}") from error


def compound_action_id(action_type: int, cell: int, *, cells: int) -> int:
    """Return ``action_type * cells + cell`` (the project action encoding)."""

    return int(action_type) * int(cells) + int(cell)


def split_compound_action(action_id: int, *, cells: int) -> tuple[int, int]:
    """Split ``action_type * cells + cell`` back into ``(action_type, cell)``."""

    return divmod(int(action_id), int(cells))

