from __future__ import annotations

import pytest

from minesweeper_rl.config import EnvironmentConfig
from minesweeper_rl.dom import GameStatus, classify_game_status
from minesweeper_rl.environment import MinesweeperEnvironment

UNPRESSED = ["top-area-face", "zoomable", "hd_top-area-face-unpressed"]
WIN = ["top-area-face", "zoomable", "hd_top-area-face-win"]
LOSE = ["top-area-face", "zoomable", "hd_top-area-face-lose"]
NNHD_WIN = ["top-area-face", "zoomable", "nnhd_top-area-face-win"]


@pytest.mark.parametrize(
    ("face_classes", "opened_cells", "expected"),
    [
        (UNPRESSED, 0, GameStatus.READY),
        (UNPRESSED, 8, GameStatus.RUNNING),
        (WIN, 480, GameStatus.WON),
        (NNHD_WIN, 81, GameStatus.WON),
        (LOSE, 480, GameStatus.LOST),
        # The face wins over a zero opened count.
        (WIN, 0, GameStatus.WON),
    ],
)
def test_classify_game_status(face_classes: list[str], opened_cells: int, expected: GameStatus) -> None:
    assert classify_game_status(face_classes, opened_cells=opened_cells) is expected


def make_cell(x: int, y: int, *classes: str) -> dict[str, object]:
    return {"x": str(x), "y": str(y), "classes": ["cell", "size24", *classes]}


class FakeLocator:
    def __init__(self, snapshots: list[list[dict[str, object]]]) -> None:
        self.snapshots = snapshots

    def evaluate_all(self, _: str) -> list[dict[str, object]]:
        return self.snapshots.pop(0) if len(self.snapshots) > 1 else self.snapshots[0]


class FakePage:
    def __init__(self, cells: list[dict[str, object]], face_classes: list[str]) -> None:
        self.locator_instance = FakeLocator([cells])
        self.face_classes = face_classes

    def locator(self, _: str) -> FakeLocator:
        return self.locator_instance

    def evaluate(self, _: str) -> list[str]:
        return self.face_classes


def environment(cells: list[dict[str, object]], face_classes: list[str]) -> MinesweeperEnvironment:
    config = EnvironmentConfig(
        width=2, height=2, observation_timeout_ms=50, observation_poll_ms=0
    )
    return MinesweeperEnvironment(FakePage(cells, face_classes), config)


def closed_board() -> list[dict[str, object]]:
    return [make_cell(x, y, "hd_closed") for y in range(2) for x in range(2)]


def test_get_game_status_ready() -> None:
    assert environment(closed_board(), UNPRESSED).get_game_status() is GameStatus.READY


def test_missing_face_is_unknown_not_a_playable_status() -> None:
    with pytest.raises(RuntimeError, match="status face"):
        environment(closed_board(), []).get_game_status()


def test_get_game_status_running() -> None:
    cells = closed_board()
    cells[0] = make_cell(0, 0, "hd_opened", "hd_type1")

    assert environment(cells, UNPRESSED).get_game_status() is GameStatus.RUNNING


def test_get_game_status_won() -> None:
    cells = [make_cell(x, y, "hd_opened", "hd_type0") for y in range(2) for x in range(2)]

    assert environment(cells, WIN).get_game_status() is GameStatus.WON


def test_get_game_status_lost() -> None:
    cells = closed_board()
    cells[0] = make_cell(0, 0, "hd_opened", "hd_type10")

    assert environment(cells, LOSE).get_game_status() is GameStatus.LOST
