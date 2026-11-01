"""Map observable Minesweeper DOM state to a game status.

The website's status face is the authoritative, DOM-only indicator for a
finished game.  It is the element ``#top_area_face``; the standard skin uses
the ``hd_`` prefix and no-guessing games use ``nnhd_``:

``*_top-area-face-win``
    The board is solved.
``*_top-area-face-lose``
    A mine was revealed.
``*_top-area-face-unpressed`` / ``*_top-area-face-pressed``
    The game is still in progress (or has not started yet).

Because the face does not expose a separate "new game" flag, a still-live board
with no revealed cells is reported as :attr:`GameStatus.READY`; once at least
one cell is revealed it becomes :attr:`GameStatus.RUNNING`.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

FACE_SELECTOR = "#top_area_face"
_FACE_WIN = "top-area-face-win"
_FACE_LOSE = "top-area-face-lose"


class GameStatus(str, Enum):
    """Playable state of a Minesweeper board."""

    READY = "ready"
    RUNNING = "running"
    WON = "won"
    LOST = "lost"


def _face_has_state(classes: Iterable[str], state: str) -> bool:
    return any(str(name).endswith(state) for name in classes)


def classify_game_status(
    face_classes: Iterable[str], *, opened_cells: int
) -> GameStatus:
    """Classify a game from the status-face classes and opened-cell count.

    Win/loss come from the face and take precedence.  Otherwise a board with no
    revealed cells is ``READY`` and any board with revealed cells is
    ``RUNNING``.
    """

    classes = list(face_classes)
    if _face_has_state(classes, _FACE_WIN):
        return GameStatus.WON
    if _face_has_state(classes, _FACE_LOSE):
        return GameStatus.LOST
    if opened_cells <= 0:
        return GameStatus.READY
    return GameStatus.RUNNING
