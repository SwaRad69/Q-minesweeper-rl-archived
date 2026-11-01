"""Deterministic action indexing and legality masks for Minesweeper.

Two action spaces are supported:

``OPEN_ONLY``
    The completed baseline experiment: exactly one OPEN action per cell, so a
    flat action id *is* a row-major cell index (``0 .. cells - 1``).

``FLAGS``
    The human-like experiment: OPEN, FLAG and CHORD.  A flat action id packs an
    action type and a cell::

        action_id = action_type * cells + cell_index
        cell_index = y * width + x
        action_type = action_id // cells
        cell_index = action_id % cells

    With the default 30 x 16 Expert board (480 cells) that is::

        0    .. 479   OPEN
        480  .. 959   FLAG
        960  .. 1439  CHORD

The mapping is fully deterministic and independent of the network, so the actor
output is interpreted without any learned layout.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ..action_types import (
    ACTION_CHORD,
    ACTION_FLAG,
    ACTION_OPEN,
    ACTION_TYPES,
    compound_action_id,
    split_compound_action,
)
from ..config import EnvironmentConfig
from ..dom import FLAGGED, HIDDEN, chord_is_legal

_DEFAULTS = EnvironmentConfig()
WIDTH = _DEFAULTS.width
HEIGHT = _DEFAULTS.height
CELLS = WIDTH * HEIGHT

#: OPEN-only action space (the verified baseline).
OPEN_ONLY = "open_only"
#: OPEN + FLAG + CHORD action space (the human-like experiment).
FLAGS = "flags"
ACTION_SPACES: tuple[str, ...] = (OPEN_ONLY, FLAGS)

#: The action space used when a caller does not request one explicitly.  The
#: flag-aware experiment is the active primary, so FLAGS is the default; the
#: OPEN-only baseline is still fully reproducible by selecting ``OPEN_ONLY``.
DEFAULT_ACTION_SPACE = FLAGS

#: OPEN-only: one action per cell (480 for the default Expert board).
#: The OPEN-only space is a legacy baseline retained only for reading the
#: completed v4 checkpoints; the active pipeline is FLAGS.
NUM_ACTIONS_OPEN_ONLY = CELLS
#: Flag-aware: OPEN + FLAG + CHORD per cell (1440 for the default board).
NUM_ACTIONS_FLAGS = len(ACTION_TYPES) * CELLS


def validate_action_space(action_space: str) -> str:
    """Return a validated action-space name or raise :class:`ValueError`."""

    if action_space not in ACTION_SPACES:
        raise ValueError(
            f"unknown action space {action_space!r}; expected one of {ACTION_SPACES}"
        )
    return str(action_space)


def action_count(
    action_space: str = DEFAULT_ACTION_SPACE, *, cells: int = CELLS
) -> int:
    """Return the number of flat actions for ``action_space`` and ``cells``."""

    if action_space == OPEN_ONLY:
        return int(cells)
    if action_space == FLAGS:
        return len(ACTION_TYPES) * int(cells)
    raise ValueError(
        f"unknown action space {action_space!r}; expected one of {ACTION_SPACES}"
    )


def _validate_action_type(action_type: Any) -> None:
    if isinstance(action_type, bool) or action_type not in ACTION_TYPES:
        raise ValueError(
            f"action_type must be one of {ACTION_TYPES}, got {action_type!r}"
        )


def _validate_coordinate(x: Any, y: Any, width: int, height: int) -> None:
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not isinstance(x, int)
        or not isinstance(y, int)
    ):
        raise TypeError("coordinates must be integers")
    if not (0 <= x < width and 0 <= y < height):
        raise IndexError(f"coordinate outside board: ({x}, {y})")


def _validate_cell_index(cell: Any, width: int, height: int) -> None:
    if isinstance(cell, bool) or not isinstance(cell, int):
        raise TypeError("cell index must be an integer")
    if not 0 <= cell < width * height:
        raise IndexError(f"cell index outside board: {cell}")


def coordinate_to_action(
    x: int, y: int, *, width: int = WIDTH, height: int = HEIGHT
) -> int:
    """Map a board ``(x, y)`` cell to its row-major OPEN action id."""

    _validate_coordinate(x, y, width, height)
    return y * width + x


def action_to_coordinate(
    action: int, *, width: int = WIDTH, height: int = HEIGHT
) -> tuple[int, int]:
    """Map a row-major OPEN action id back to its ``(x, y)`` coordinate."""

    _validate_cell_index(action, width, height)
    return action % width, action // width


def encode_action(
    cell: int,
    *,
    action_type: int = ACTION_OPEN,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> int:
    """Return the flat action id for ``(action_type, cell)``.

    ``cell`` is the row-major cell index ``y * width + x``.  The default
    ``action_type=ACTION_OPEN`` makes this the identity for the OPEN block, so
    the historical OPEN-only mapping is preserved.
    """

    _validate_action_type(action_type)
    _validate_cell_index(cell, width, height)
    return compound_action_id(action_type, cell, cells=width * height)


def decode_action(
    action: int,
    *,
    action_type: int = ACTION_OPEN,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> int:
    """Return the cell index addressed by ``action`` inside its type block.

    ``action_type`` selects the block to validate against: ``ACTION_OPEN``
    accepts ``0 .. cells - 1`` (the historical behaviour), ``ACTION_FLAG``
    accepts ``cells .. 2 * cells - 1``, and ``ACTION_CHORD`` accepts
    ``2 * cells .. 3 * cells - 1``.
    """

    _validate_action_type(action_type)
    if isinstance(action, bool) or not isinstance(action, int):
        raise TypeError("action id must be an integer")
    cells = width * height
    low = action_type * cells
    if not low <= action < low + cells:
        raise IndexError(
            f"action {action} is outside the {action_type!r} block "
            f"[{low}, {low + cells})"
        )
    return action - low


def split_action(
    action: int, *, width: int = WIDTH, height: int = HEIGHT
) -> tuple[int, int]:
    """Split a compound flat action into ``(action_type, cell)``."""

    if isinstance(action, bool) or not isinstance(action, int):
        raise TypeError("action id must be an integer")
    cells = width * height
    if not 0 <= action < len(ACTION_TYPES) * cells:
        raise IndexError(
            f"action id outside the {len(ACTION_TYPES) * cells}-action space: {action}"
        )
    return split_compound_action(action, cells=cells)


def join_action(
    action_type: int,
    cell: int,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> int:
    """Combine ``(action_type, cell)`` into a compound flat action id."""

    return encode_action(cell, action_type=action_type, width=width, height=height)


def action_type_of(
    action: int, *, width: int = WIDTH, height: int = HEIGHT
) -> int:
    """Return the action type of a compound flat action id."""

    return split_action(action, width=width, height=height)[0]


def cell_of(action: int, *, width: int = WIDTH, height: int = HEIGHT) -> int:
    """Return the cell index of a compound flat action id."""

    return split_action(action, width=width, height=height)[1]


def _flatten(
    board: npt.ArrayLike, width: int, height: int
) -> tuple[npt.NDArray[np.int8], tuple[int, ...]]:
    array = np.asarray(board)
    if array.ndim < 2 or tuple(array.shape[-2:]) != (height, width):
        raise ValueError(
            f"board must have trailing shape ({height}, {width}), got {tuple(array.shape)}"
        )
    leading = array.shape[:-2]
    return array.reshape(-1, height, width), leading


def open_mask(
    board: npt.ArrayLike, *, width: int = WIDTH, height: int = HEIGHT
) -> npt.NDArray[np.bool_]:
    """OPEN is legal exactly on a currently hidden, unflagged cell."""

    flat, leading = _flatten(board, width, height)
    mask = (flat == HIDDEN).reshape(flat.shape[0], width * height)
    return mask.reshape(*leading, width * height) if leading else mask[0]


def flag_mask(
    board: npt.ArrayLike, *, width: int = WIDTH, height: int = HEIGHT
) -> npt.NDArray[np.bool_]:
    """FLAG (toggle) is legal on a covered cell: hidden or already flagged."""

    flat, leading = _flatten(board, width, height)
    mask = ((flat == HIDDEN) | (flat == FLAGGED)).reshape(
        flat.shape[0], width * height
    )
    return mask.reshape(*leading, width * height) if leading else mask[0]


def chord_mask(
    board: npt.ArrayLike, *, width: int = WIDTH, height: int = HEIGHT
) -> npt.NDArray[np.bool_]:
    """CHORD is legal only where a chord can actually be attempted.

    That is an opened number whose flagged neighbours already equal the number
    and which still has at least one hidden neighbour (see
    :func:`minesweeper_rl.dom.chord_is_legal`).  Only observable board state is
    consulted; the hidden mine layout is never used.
    """

    flat, leading = _flatten(board, width, height)
    mask = np.zeros((flat.shape[0], width * height), dtype=bool)
    for index, single in enumerate(flat):
        for y in range(height):
            for x in range(width):
                mask[index, y * width + x] = chord_is_legal(single, x, y)
    return mask.reshape(*leading, width * height) if leading else mask[0]


def action_type_mask(
    board: npt.ArrayLike,
    action_type: int,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> npt.NDArray[np.bool_]:
    """Return the legality mask for one action type over all cells."""

    if action_type == ACTION_OPEN:
        return open_mask(board, width=width, height=height)
    if action_type == ACTION_FLAG:
        return flag_mask(board, width=width, height=height)
    if action_type == ACTION_CHORD:
        return chord_mask(board, width=width, height=height)
    raise ValueError(f"unknown action type: {action_type!r}")


def action_mask(
    board: npt.ArrayLike,
    *,
    action_space: str = DEFAULT_ACTION_SPACE,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> npt.NDArray[np.bool_]:
    """Return the boolean legality mask over the selected action space.

    ``OPEN_ONLY`` returns ``[cells]`` (or ``[batch, cells]``) as before.
    ``FLAGS`` returns ``[3 * cells]`` (or ``[batch, 3 * cells]``) as the
    concatenation ``[OPEN | FLAG | CHORD]``, matching the compound encoding
    ``action_type * cells + cell``.
    """

    if action_space == OPEN_ONLY:
        return open_mask(board, width=width, height=height)
    if action_space != FLAGS:
        raise ValueError(
            f"unknown action space {action_space!r}; expected one of {ACTION_SPACES}"
        )
    flat, leading = _flatten(board, width, height)
    open_part = (flat == HIDDEN).reshape(flat.shape[0], width * height)
    flag_part = ((flat == HIDDEN) | (flat == FLAGGED)).reshape(
        flat.shape[0], width * height
    )
    chord_part = chord_mask(flat, width=width, height=height)
    chord_part = np.asarray(chord_part).reshape(flat.shape[0], width * height)
    mask = np.concatenate([open_part, flag_part, chord_part], axis=1)
    return mask.reshape(*leading, 3 * width * height) if leading else mask[0]
