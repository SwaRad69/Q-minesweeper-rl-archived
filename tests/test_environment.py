from __future__ import annotations

import numpy as np
import pytest

from minesweeper_rl.config import EnvironmentConfig
from minesweeper_rl.dom.parser import (
    BoardShapeError,
    CellState,
    board_from_cells,
)


def test_board_from_cells_uses_y_x_axis_order() -> None:
    cells = [
        CellState(x=0, y=0, value=-1),
        CellState(x=1, y=0, value=2),
        CellState(x=0, y=1, value=0),
        CellState(x=1, y=1, value=8),
    ]

    board = board_from_cells(cells, expected_shape=(2, 2))

    np.testing.assert_array_equal(board, [[-1, 2], [0, 8]])
    assert board.dtype == np.int8


def test_invalid_board_is_rejected() -> None:
    cells = [CellState(0, 0, -1), CellState(0, 0, 0)]

    with pytest.raises(BoardShapeError, match="duplicate cell"):
        board_from_cells(cells, expected_shape=(1, 2))


def test_invalid_cell_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid cell value"):
        board_from_cells([CellState(0, 0, 9)], expected_shape=(1, 1))


def test_default_config_matches_current_board() -> None:
    config = EnvironmentConfig()

    assert (config.height, config.width) == (16, 30)
