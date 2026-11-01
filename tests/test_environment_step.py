from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import pytest

from minesweeper_rl.actions import ActionTimeoutError, InvalidActionError
from minesweeper_rl.action_types import ACTION_CHORD, ACTION_FLAG
from minesweeper_rl.config import EnvironmentConfig
from minesweeper_rl.dom import FLAGGED, HIDDEN, GameStatus
from minesweeper_rl.environment import (
    MINE_REWARD,
    SAFE_REVEAL_REWARD,
    WIN_REWARD,
    MinesweeperEnvironment,
)

CELL_SELECTOR = ".cell[data-x][data-y]"


def make_cell(x: int, y: int, *classes: str) -> dict[str, object]:
    return {"x": str(x), "y": str(y), "classes": ["cell", "size24", *classes]}


class FakeCell:
    def __init__(self, page: "FakePage", x: int, y: int) -> None:
        self.page = page
        self.x = x
        self.y = y

    def count(self) -> int:
        return 1

    def scroll_into_view_if_needed(self) -> None:
        return None

    def click(self, button: str = "left") -> None:
        self.page.click_cell(self.x, self.y, button=button)

    def dispatch_event(self, _name: str, _payload: object) -> None:
        self.page.click_cell(self.x, self.y)


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


class FakePage:
    def __init__(self, *, width: int = 2, height: int = 2) -> None:
        self.width = width
        self.height = height
        self.cells: list[dict[str, object]] = []
        self.face_classes = ["top-area-face", "hd_top-area-face-unpressed"]
        self.ready = True
        self.click_calls: list[tuple[int, int]] = []
        self.click_hook: Callable[[int, int], None] | None = None
        self.right_click_hook: Callable[[int, int], None] | None = None
        self._reset_cells()

    def _reset_cells(self) -> None:
        self.cells = [
            make_cell(x, y, "hd_closed")
            for y in range(self.height)
            for x in range(self.width)
        ]

    def set_cell(self, x: int, y: int, *classes: str) -> None:
        for cell in self.cells:
            if cell["x"] == str(x) and cell["y"] == str(y):
                cell["classes"] = ["cell", "size24", *classes]
                return
        raise AssertionError(f"cell ({x}, {y}) not found")

    def open_cell(self, x: int, y: int, value: int = 3) -> None:
        self.set_cell(x, y, "hd_opened", f"hd_type{value}")

    def click_cell(self, x: int, y: int, button: str = "left") -> None:
        self.click_calls.append((x, y))
        if button == "right" and self.right_click_hook is not None:
            self.right_click_hook(x, y)
            return
        if self.click_hook is not None:
            self.click_hook(x, y)
        else:
            self.open_cell(x, y)

    def locator(self, selector: str) -> object:
        if selector == CELL_SELECTOR:
            return FakeAllCells(self)
        if selector.startswith(CELL_SELECTOR):
            x = int(re.search(r'data-x="(\d+)"', selector).group(1))
            y = int(re.search(r'data-y="(\d+)"', selector).group(1))
            return FakeCell(self, x, y)
        raise AssertionError(f"unexpected selector: {selector}")

    def wait_for_function(self, _: str, *, arg: object = None, timeout: object = None) -> None:
        if not self.ready:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

            raise PlaywrightTimeoutError("board never appeared")

    def evaluate(self, _: str) -> list[str]:
        return list(self.face_classes)


def environment(page: FakePage, **overrides: object) -> MinesweeperEnvironment:
    settings: dict[str, object] = {
        "width": page.width,
        "height": page.height,
        "cell_selector": CELL_SELECTOR,
        "observation_timeout_ms": 100,
        "observation_poll_ms": 0,
        "action_timeout_ms": 60,
        "action_poll_ms": 0,
        "new_game_timeout_ms": 100,
    }
    settings.update(overrides)
    return MinesweeperEnvironment(page, EnvironmentConfig(**settings))


def test_step_safe_reveal_rewards_and_reports_info() -> None:
    page = FakePage()
    env = environment(page)

    next_state, reward, terminated, truncated, info = env.step((0, 1))

    assert isinstance(next_state, np.ndarray)
    assert next_state.dtype == np.int8
    assert next_state[1, 0] == 3
    assert reward == SAFE_REVEAL_REWARD
    assert terminated is False
    assert truncated is False
    assert info["action"] == (0, 1)
    assert info["previous_value"] == -1
    assert info["new_value"] == 3
    assert info["game_status"] is GameStatus.RUNNING
    assert info["hit_mine"] is False
    assert info["newly_revealed"] == 1
    assert info["opened_cells"] == 1
    assert isinstance(info["elapsed_ms"], float)
    assert info["elapsed_ms"] >= 0.0


def test_step_flood_fill_counts_all_new_cells() -> None:
    page = FakePage()

    def flood(x: int, y: int) -> None:
        page.open_cell(0, 0, 0)
        page.open_cell(1, 0, 1)
        page.open_cell(0, 1, 2)

    page.click_hook = flood
    env = environment(page)

    _, reward, terminated, _, info = env.step((1, 1))

    assert reward == 3.0, "a cascade of 3 revealed cells is worth +3"
    assert terminated is False
    assert info["newly_revealed"] == 3
    assert info["opened_cells"] == 3


def test_step_mine_loses_with_negative_reward() -> None:
    page = FakePage()

    def hit_mine(x: int, y: int) -> None:
        page.set_cell(x, y, "hd_opened", "hd_type10")
        page.face_classes = ["top-area-face", "hd_top-area-face-lose"]

    page.click_hook = hit_mine
    env = environment(page)

    next_state, reward, terminated, truncated, info = env.step((1, 0))

    assert reward == MINE_REWARD
    assert terminated is True
    assert truncated is False
    assert info["game_status"] is GameStatus.LOST
    assert info["hit_mine"] is True
    assert info["previous_value"] == -1
    assert info["new_value"] == -1
    assert next_state[0, 1] == -1


def test_step_win_has_terminal_positive_reward() -> None:
    page = FakePage()

    def win(x: int, y: int) -> None:
        page.open_cell(x, y, 0)
        page.face_classes = ["top-area-face", "hd_top-area-face-win"]

    page.click_hook = win
    env = environment(page)

    _, reward, terminated, truncated, info = env.step((0, 0))

    assert reward == WIN_REWARD + 1, "win bonus plus one newly revealed cell"
    assert terminated is True
    assert truncated is False
    assert info["game_status"] is GameStatus.WON
    assert info["hit_mine"] is False
    assert info["newly_revealed"] == 1


def test_step_rejects_already_revealed_cell_without_clicking() -> None:
    page = FakePage()
    page.open_cell(1, 1, 5)
    env = environment(page)

    with pytest.raises(InvalidActionError, match="already revealed"):
        env.step((1, 1))
    assert page.click_calls == []


def test_step_rejects_flagged_cell_without_clicking() -> None:
    page = FakePage()
    page.set_cell(0, 1, "hd_closed", "hd_flag")
    env = environment(page)

    with pytest.raises(InvalidActionError, match="not hidden"):
        env.step((0, 1))
    assert page.click_calls == []


def test_step_rejects_action_outside_the_board() -> None:
    page = FakePage()
    env = environment(page)

    with pytest.raises(IndexError, match="outside board"):
        env.step((2, 0))
    assert page.click_calls == []


def test_step_rejects_non_integer_coordinates() -> None:
    page = FakePage()
    env = environment(page)

    with pytest.raises(TypeError, match="integers"):
        env.step((0.5, 0))
    assert page.click_calls == []


@pytest.mark.parametrize("action", [5, (0,), (0, 1, 2)])
def test_step_rejects_malformed_action(action: object) -> None:
    page = FakePage()
    env = environment(page)

    with pytest.raises(TypeError, match=r"\(x, y\) pair"):
        env.step(action)  # type: ignore[arg-type]
    assert page.click_calls == []


def test_step_rejects_when_episode_is_already_over() -> None:
    page = FakePage()
    page.face_classes = ["top-area-face", "hd_top-area-face-lose"]
    env = environment(page)

    with pytest.raises(InvalidActionError, match="already over"):
        env.step((0, 0))
    assert page.click_calls == []


def test_step_propagates_when_the_board_never_changes() -> None:
    page = FakePage()
    page.click_hook = lambda _x, _y: None
    env = environment(page)

    with pytest.raises(ActionTimeoutError, match="no observable board change"):
        env.step((0, 0))


def test_environment_exposes_flag_and_chord_api() -> None:
    page = FakePage()
    env = environment(page)

    assert hasattr(env, "apply_action")
    assert hasattr(env, "flag")
    assert hasattr(env, "chord")
    assert hasattr(env, "step")


def test_apply_action_flag_places_flag_with_neutral_reward() -> None:
    page = FakePage()

    def place_flag(x: int, y: int) -> None:
        page.set_cell(x, y, "hd_closed", "hd_flag")

    page.right_click_hook = place_flag
    env = environment(page)

    board, reward, terminated, truncated, info = env.apply_action(ACTION_FLAG, 0, 1)

    assert board[1, 0] == FLAGGED
    assert reward == 0.0
    assert terminated is False and truncated is False
    assert info["operation"] == "flag"
    assert info["action_type"] == ACTION_FLAG
    assert info["newly_revealed"] == 0
    assert page.click_calls == [(0, 1)]


def test_apply_action_chord_opens_neighbours() -> None:
    page = FakePage()
    page.open_cell(0, 0, 1)
    page.set_cell(1, 0, "hd_closed", "hd_flag")

    def chord(x: int, y: int) -> None:
        page.open_cell(0, 1, 0)

    page.click_hook = chord
    env = environment(page)

    board, reward, terminated, truncated, info = env.apply_action(ACTION_CHORD, 0, 0)

    assert board[1, 0] == 0
    assert reward == 1.0
    assert terminated is False
    assert info["operation"] == "chord"
    assert info["action_type"] == ACTION_CHORD
    assert info["newly_revealed"] == 1


def test_apply_action_flag_rejects_opened_cell() -> None:
    page = FakePage()
    page.open_cell(0, 0, 3)
    env = environment(page)

    with pytest.raises(InvalidActionError, match="cannot be flagged"):
        env.apply_action(ACTION_FLAG, 0, 0)
    assert page.click_calls == []


def test_apply_action_flag_removes_existing_flag() -> None:
    page = FakePage()
    page.set_cell(0, 0, "hd_closed", "hd_flag")

    def remove_flag(x: int, y: int) -> None:
        page.set_cell(x, y, "hd_closed")

    page.right_click_hook = remove_flag
    env = environment(page)

    board, reward, _, _, info = env.apply_action(ACTION_FLAG, 0, 0)

    assert board[0, 0] == HIDDEN
    assert reward == 0.0
    assert info["previous_value"] == FLAGGED
    assert info["new_value"] == HIDDEN


def test_flag_change_is_visible_in_the_next_observation() -> None:
    # Board synchronization: the stable observation after a FLAG reflects it.
    page = FakePage()

    def place_flag(x: int, y: int) -> None:
        page.set_cell(x, y, "hd_closed", "hd_flag")

    page.right_click_hook = place_flag
    env = environment(page)

    env.apply_action(ACTION_FLAG, 1, 0)
    observed = env.observe()

    assert observed[0, 1] == FLAGGED
    assert observed[0, 0] == HIDDEN


def test_apply_action_chord_rejects_when_flags_do_not_match() -> None:
    page = FakePage()
    page.open_cell(0, 0, 2)  # a 2 with no flagged neighbours
    env = environment(page)

    with pytest.raises(InvalidActionError, match="not currently legal"):
        env.apply_action(ACTION_CHORD, 0, 0)
    assert page.click_calls == []


def test_apply_action_chord_rejects_hidden_cell() -> None:
    page = FakePage()
    env = environment(page)

    with pytest.raises(InvalidActionError, match="opened numbered"):
        env.apply_action(ACTION_CHORD, 0, 0)
    assert page.click_calls == []


def test_apply_action_rejects_unknown_action_type() -> None:
    page = FakePage()
    env = environment(page)

    with pytest.raises(InvalidActionError, match="unknown action type"):
        env.apply_action(99, 0, 0)
    assert page.click_calls == []


def test_open_on_opened_cell_is_rejected_not_treated_as_chord() -> None:
    page = FakePage()
    page.open_cell(0, 0, 1)
    env = environment(page)

    with pytest.raises(InvalidActionError, match="already revealed"):
        env.step((0, 0))
    assert page.click_calls == []
