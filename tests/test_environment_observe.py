from __future__ import annotations

import numpy as np
import pytest

from minesweeper_rl.config import EnvironmentConfig
from minesweeper_rl.dom import BoardMissingError, BoardShapeError, CellFormatError
from minesweeper_rl.environment import MinesweeperEnvironment


class FakeLocator:
    def __init__(self, snapshots: list[list[dict[str, object]]]) -> None:
        self.snapshots = snapshots

    def evaluate_all(self, _: str) -> list[dict[str, object]]:
        return self.snapshots.pop(0) if len(self.snapshots) > 1 else self.snapshots[0]


class FakePage:
    def __init__(self, snapshots: list[list[dict[str, object]]]) -> None:
        self.locator_instance = FakeLocator(snapshots)

    def locator(self, _: str) -> FakeLocator:
        return self.locator_instance


def make_cell(x: int, y: int, state: str = "hd_closed") -> dict[str, object]:
    return {"x": str(x), "y": str(y), "classes": ["cell", "size24", state]}


def full_board() -> list[dict[str, object]]:
    return [make_cell(x, y) for y in range(16) for x in range(30)]


def environment(snapshots: list[list[dict[str, object]]]) -> MinesweeperEnvironment:
    config = EnvironmentConfig(observation_timeout_ms=100, observation_poll_ms=0)
    return MinesweeperEnvironment(FakePage(snapshots), config)


def test_observe_returns_a_stable_16_by_30_board() -> None:
    board = environment([full_board(), full_board()]).observe()

    assert board.shape == (16, 30)
    assert board.dtype == np.int8
    assert np.all(board == -1)


def test_observe_waits_for_partially_loaded_board() -> None:
    loaded = full_board()
    loaded[479] = {
        "x": "29", "y": "15",
        "classes": ["cell", "size24", "hd_opened", "hd_type8"],
    }

    board = environment([loaded[:-1], loaded, loaded]).observe()

    assert board[15, 29] == 8


@pytest.mark.parametrize(
    "bad_cell",
    [
        {"x": "29", "y": "15", "classes": ["cell"]},
        {"x": "29", "y": "15", "classes": ["cell", "hd_closed", "hd_opened"]},
        {"x": "29", "y": "15", "classes": ["cell", "hd_closed", "hd_type1"]},
        {"x": "29", "y": "15", "classes": ["cell", "hd_opened", "hd_type1", "hd_type2"]},
        {"x": "29", "y": "15", "classes": ["cell", "hd_opened"]},
        {"x": "29", "y": "15", "classes": ["cell", "hd_opened", "hd_type9"]},
    ],
)
def test_observe_rejects_malformed_cell(bad_cell: dict[str, object]) -> None:
    cells = full_board()
    cells[-1] = bad_cell

    with pytest.raises(CellFormatError, match=r"cell \(29, 15\)"):
        environment([cells]).observe()


def test_observe_rejects_missing_cell_after_timeout() -> None:
    cells = full_board()[:-1]

    with pytest.raises(BoardShapeError, match="expected"):
        environment([cells]).observe()


def test_observe_rejects_duplicate_coordinate() -> None:
    cells = full_board()
    cells[-1] = make_cell(0, 0)

    with pytest.raises(BoardShapeError, match="duplicate"):
        environment([cells]).observe()


def test_observe_rejects_unexpected_class() -> None:
    cells = full_board()
    cells[-1] = {
        "x": "29", "y": "15",
        "classes": ["cell", "hd_opened", "hd_type1", "hd_unknown"],
    }

    with pytest.raises(CellFormatError, match="unexpected class"):
        environment([cells]).observe()


def test_observe_raises_when_board_does_not_exist() -> None:
    with pytest.raises(BoardMissingError, match="no board cells"):
        environment([[]]).observe()


def test_board_exists_reflects_dom_state() -> None:
    assert environment([full_board()]).board_exists() is True
    assert environment([[]]).board_exists() is False
