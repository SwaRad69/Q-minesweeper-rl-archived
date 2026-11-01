"""Unit tests for the human observer recorder (no browser involved).

A fake page/observer stands in for Playwright so the recording logic can be
driven deterministically.  The fake observer can emit a *sequence* of board
observations, which lets the tests simulate delayed and multi-stage DOM updates.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from minesweeper_rl.data import (
    ACTION_CHORD,
    ACTION_FLAG,
    ACTION_OPEN,
    ACTION_REVEAL,
    GROUND_TRUTH_KEY,
    TrajectoryRecorder,
    TrajectoryStore,
)
from minesweeper_rl.data.record_human import HumanTrajectoryRecorder
from minesweeper_rl.dom import FLAGGED, HIDDEN, GameStatus
from minesweeper_rl.environment import WIN_REWARD

WIDTH, HEIGHT = 4, 2


def empty_board() -> np.ndarray:
    return np.full((HEIGHT, WIDTH), -1, dtype=np.int8)


def snapshot(board: np.ndarray) -> list[dict]:
    cells = []
    for y in range(board.shape[0]):
        for x in range(board.shape[1]):
            value = int(board[y, x])
            if value == FLAGGED:
                classes = ["hd_closed", "hd_flag"]
            elif value == HIDDEN:
                classes = ["hd_closed"]
            else:
                classes = ["hd_opened", f"hd_type{value}"]
            cells.append({"x": str(x), "y": str(y), "classes": classes})
    return cells


def click_event(
    board: np.ndarray,
    x: int,
    y: int,
    *,
    button: int = 0,
    trusted: bool = True,
    event_id: int | None = None,
) -> dict:
    return {
        "id": event_id,
        "token": "testdoc",
        "type": "pointerdown",
        "isTrusted": trusted,
        "button": button,
        "x": str(x),
        "y": str(y),
        "clientX": 10,
        "clientY": 20,
        "tag": "DIV",
        "targetClasses": ["cell"],
        "cells": snapshot(board),
    }


class FakePage:
    def __init__(self) -> None:
        self.init_scripts: list[str] = []
        self.evaluations: list[str] = []
        self.batches: list[list[dict]] = []

    def add_init_script(self, script: str) -> None:
        self.init_scripts.append(script)

    def evaluate(self, expression: str, *args: object):
        self.evaluations.append(expression)
        if "return events.slice()" in expression:
            return self.batches.pop(0) if self.batches else []
        return None


class FakeObserver:
    """Stable board source; ``observations`` scripts a sequence of reads."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(width=WIDTH, height=HEIGHT)
        self.page = FakePage()
        self.board = empty_board()
        self.status = GameStatus.READY
        self.observations: list[np.ndarray] = []
        self.statuses: list[GameStatus] = []
        self.ready = False

    def wait_until_ready(self) -> None:
        self.ready = True

    def observe(self) -> np.ndarray:
        if self.observations:
            self.board = self.observations.pop(0)
        return self.board.copy()

    def get_game_status(self) -> GameStatus:
        if self.statuses:
            self.status = self.statuses.pop(0)
        return self.status


def make_recorder(tmp_path):
    store = TrajectoryStore(tmp_path / "dataset")
    recorder = TrajectoryRecorder(
        store, source="human", width=WIDTH, height=HEIGHT, mines=3
    )
    return store, recorder


def make_human(tmp_path):
    store, recorder = make_recorder(tmp_path)
    observer = FakeObserver()
    emitted: list[str] = []
    human = HumanTrajectoryRecorder(
        observer,
        recorder,
        poll_interval=0.0,
        action_timeout=0.5,
        quiescence_time=0.0,
        emit=emitted.append,
    )
    # Sync the initial (fresh) board so the lifecycle starts in READY.
    human._handle_idle_board(observer.board.copy())
    return human, observer, recorder, store, emitted


def _reveal(human, observer, *, x=0, y=0):
    before = human._stable_board.copy()
    after = before.copy()
    after[y, x] = 1
    observer.board, observer.status = after, GameStatus.RUNNING
    return human.handle_event(click_event(before, x, y))


def _flag(human, observer, *, x=0, y=0, unflag=False):
    before = human._stable_board.copy()
    after = before.copy()
    after[y, x] = HIDDEN if unflag else FLAGGED
    observer.board, observer.status = after, GameStatus.RUNNING
    return human.handle_event(click_event(before, x, y, button=2))


# ---------------------------------------------------------------------------
# capture + authenticity
# ---------------------------------------------------------------------------


def test_install_capture_injects_listener(tmp_path) -> None:
    human, observer, *_ = make_human(tmp_path)
    human.install_capture()
    assert observer.page.init_scripts
    assert any("addEventListener" in script for script in observer.page.init_scripts)
    assert any("addEventListener" in expression for expression in observer.page.evaluations)


def test_untrusted_event_is_rejected(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    observer.board, observer.status = empty_board(), GameStatus.READY
    assert human.handle_event(click_event(empty_board(), 0, 0, trusted=False)) == "rejected"
    assert list(store.iter_games()) == []
    assert recorder.active_game_id is None


def test_middle_click_is_counted_not_recorded(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    observer.board = empty_board()
    assert human.handle_event(click_event(empty_board(), 0, 0, button=1)) == "other"
    assert human.report.other_clicks == 1
    assert list(store.iter_games()) == []


def test_rejected_events_create_no_game(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    observer.board, observer.status = empty_board(), GameStatus.READY
    for event in (
        click_event(empty_board(), 0, 0, trusted=False),
        click_event(empty_board(), 0, 0, button=1),
        click_event(empty_board(), 9, 9),
    ):
        human.handle_event(event)
    assert list(store.iter_games()) == []
    assert recorder.active_game_id is None


def test_phantom_board_change_is_rejected(tmp_path) -> None:
    """A press whose cell did not open must not become a transition."""

    human, observer, recorder, store, _ = make_human(tmp_path)
    before = empty_board()
    after = empty_board()
    after[1, 1] = 0  # unrelated board activity, not the clicked cell
    observer.board, observer.status = after, GameStatus.RUNNING
    assert human.handle_event(click_event(before, 0, 0)) == "rejected"
    assert list(store.iter_games()) == []
    assert recorder.active_game_id is None


# ---------------------------------------------------------------------------
# one click -> one transition
# ---------------------------------------------------------------------------


def test_one_click_one_transition(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    assert _reveal(human, observer) == "recorded"
    assert recorder.active_game_id is not None
    assert recorder.steps_recorded == 1


def test_two_clicks_two_transitions(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    assert _reveal(human, observer, x=0, y=0) == "recorded"
    assert _reveal(human, observer, x=1, y=0) == "recorded"
    assert recorder.steps_recorded == 2
    assert np.array_equal(recorder._steps[0].board_after, recorder._steps[1].board_before)


def test_duplicate_event_id_is_ignored(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    before = empty_board()
    after = before.copy()
    after[0, 0] = 1
    observer.board, observer.status = after, GameStatus.RUNNING
    assert human.handle_event(click_event(before, 0, 0, event_id=1)) == "recorded"
    observer.board, observer.status = after, GameStatus.RUNNING
    assert human.handle_event(click_event(before, 0, 0, event_id=1)) == "duplicate"
    assert recorder.steps_recorded == 1


def test_stale_snapshot_does_not_break_chain(tmp_path) -> None:
    """A queued press with a stale snapshot still chains from the stable board."""

    human, observer, recorder, store, _ = make_human(tmp_path)
    s0 = empty_board()
    s1 = s0.copy()
    s1[0, 0] = 1
    s2 = s1.copy()
    s2[0, 1] = 1

    observer.board, observer.status = s1, GameStatus.RUNNING
    assert human.handle_event(click_event(s0, 0, 0, event_id=1)) == "recorded"

    observer.board, observer.status = s2, GameStatus.RUNNING
    stale = click_event(s0, 1, 0, event_id=2)  # snapshot from before the first click
    assert human.handle_event(stale) == "recorded"
    assert recorder.steps_recorded == 2
    assert np.array_equal(recorder._steps[0].board_after, s1)
    assert np.array_equal(recorder._steps[1].board_before, s1)
    assert np.array_equal(recorder._steps[1].board_after, s2)


# ---------------------------------------------------------------------------
# flag / unflag actions
# ---------------------------------------------------------------------------


def test_right_click_on_hidden_cell_records_flag(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    assert _reveal(human, observer, x=1, y=0) == "recorded"
    assert _flag(human, observer, x=0, y=0) == "recorded"
    assert recorder.steps_recorded == 2
    transition = recorder._steps[-1]
    assert transition.action_type == ACTION_FLAG
    assert int(transition.board_before[0, 0]) == HIDDEN
    assert int(transition.board_after[0, 0]) == FLAGGED
    assert store.validate().ok


def test_right_click_on_flagged_cell_records_unflag(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    assert _reveal(human, observer, x=0, y=0) == "recorded"
    _flag(human, observer, x=1, y=0)
    assert _flag(human, observer, x=1, y=0, unflag=True) == "recorded"
    assert recorder.steps_recorded == 3
    place, remove = recorder._steps[1], recorder._steps[2]
    assert place.action_type == ACTION_FLAG and remove.action_type == ACTION_FLAG
    assert int(place.board_after[0, 1]) == FLAGGED
    assert int(remove.board_after[0, 1]) == HIDDEN
    assert np.array_equal(place.board_after, remove.board_before)
    assert place.reward == 0.0 and remove.reward == 0.0
    assert store.validate().ok


def test_untrusted_right_click_is_rejected(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    observer.board, observer.status = empty_board(), GameStatus.READY
    event = click_event(empty_board(), 0, 0, button=2, trusted=False)
    assert human.handle_event(event) == "rejected"
    assert list(store.iter_games()) == []
    assert recorder.active_game_id is None


def test_non_cell_right_click_is_rejected(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    assert human.handle_event(click_event(empty_board(), 9, 9, button=2)) == "rejected"
    missing = click_event(empty_board(), 0, 0, button=2)
    missing.pop("x")
    assert human.handle_event(missing) == "rejected"
    assert list(store.iter_games()) == []
    assert recorder.active_game_id is None


def test_duplicate_right_click_event_is_ignored(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    assert _reveal(human, observer, x=1, y=0) == "recorded"
    before = human._stable_board.copy()
    after = before.copy()
    after[0, 0] = FLAGGED
    observer.board, observer.status = after, GameStatus.RUNNING
    event = click_event(before, 0, 0, button=2, event_id=11)
    assert human.handle_event(event) == "recorded"
    observer.board, observer.status = after, GameStatus.RUNNING
    assert human.handle_event(event) == "duplicate"
    assert recorder.steps_recorded == 2


def test_flag_dom_update_with_transient_state_waits_for_stable(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    assert _reveal(human, observer, x=1, y=0) == "recorded"
    before = human._stable_board.copy()
    flagged = before.copy()
    flagged[0, 0] = FLAGGED
    observer.observations = [before, before, flagged, flagged]
    observer.status = GameStatus.RUNNING
    assert human.handle_event(click_event(before, 0, 0, button=2)) == "recorded"
    assert recorder.steps_recorded == 2
    assert int(recorder._steps[-1].board_after[0, 0]) == FLAGGED


def test_flag_then_open_forms_a_continuous_chain(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    assert _reveal(human, observer, x=0, y=0) == "recorded"
    assert _flag(human, observer, x=1, y=0) == "recorded"
    before = human._stable_board.copy()
    after = before.copy()
    after[0, 2] = 1
    observer.board, observer.status = after, GameStatus.RUNNING
    assert human.handle_event(click_event(before, 2, 0)) == "recorded"
    steps = recorder._steps
    assert [step.action_type for step in steps] == [
        ACTION_OPEN, ACTION_FLAG, ACTION_OPEN,
    ]
    for first, second in zip(steps, steps[1:]):
        assert np.array_equal(first.board_after, second.board_before)
    assert store.validate().ok


def test_open_then_flag_forms_a_continuous_chain(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    assert _reveal(human, observer, x=0, y=0) == "recorded"
    assert _flag(human, observer, x=1, y=0) == "recorded"
    steps = recorder._steps
    assert [step.action_type for step in steps] == [ACTION_OPEN, ACTION_FLAG]
    assert np.array_equal(steps[0].board_after, steps[1].board_before)
    assert store.validate().ok


def test_left_click_on_flagged_cell_is_rejected(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    assert _reveal(human, observer, x=1, y=0) == "recorded"
    _flag(human, observer, x=0, y=0)
    before = human._stable_board.copy()
    observer.board, observer.status = before, GameStatus.RUNNING
    assert human.handle_event(click_event(before, 0, 0)) == "rejected"
    assert recorder.steps_recorded == 2


def test_flagging_does_not_trigger_board_discontinuity(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    assert _reveal(human, observer, x=1, y=0) == "recorded"
    _flag(human, observer, x=0, y=0)
    human._handle_idle_board(human._stable_board.copy())
    human._handle_idle_board(human._stable_board.copy())
    assert list(store.iter_games()) == []
    assert recorder.active_game_id is not None


def test_chord_on_open_cell_is_recorded_as_open(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    _reveal(human, observer, x=0, y=0)
    before = human._stable_board.copy()
    after = before.copy()
    after[0, 1] = 2
    observer.board, observer.status = after, GameStatus.RUNNING
    assert human.handle_event(click_event(before, 0, 0)) == "recorded"
    assert recorder.steps_recorded == 2
    assert recorder._steps[-1].action_type == ACTION_OPEN
    assert store.validate().ok


def test_flag_reward_is_neutral(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    assert _reveal(human, observer, x=1, y=0) == "recorded"
    _flag(human, observer, x=0, y=0)
    assert recorder._steps[-1].reward == 0.0


def test_human_open_reward_is_newly_revealed(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    assert _reveal(human, observer, x=0, y=0) == "recorded"
    assert recorder._steps[0].reward == 1.0


def test_human_cascade_reward_counts_all_revealed_cells(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    before = human._stable_board.copy()
    after = before.copy()
    after[0, 0] = 1
    after[0, 1] = 0
    after[0, 2] = 2
    observer.board, observer.status = after, GameStatus.RUNNING
    assert human.handle_event(click_event(before, 0, 0)) == "recorded"
    assert recorder.steps_recorded == 1
    assert recorder._steps[0].reward == 3.0


def test_human_open_then_flag_rewards_are_progress_then_neutral(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    assert _reveal(human, observer, x=0, y=0) == "recorded"
    assert _flag(human, observer, x=1, y=0) == "recorded"
    assert [step.reward for step in recorder._steps] == [1.0, 0.0]


def test_new_press_during_wait_is_deferred_not_folded(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    before = empty_board()
    opened = before.copy()
    opened[0, 0] = 0
    folded = opened.copy()
    folded[0, 1] = FLAGGED
    flag_event = click_event(opened, 1, 0, button=2, event_id=9)

    # The reveal settles at `opened`, then a FLAG press is captured before the
    # quiescence window closes; its board effect (the flag) is already visible.
    observer.board = folded
    observer.observations = [opened]
    observer.page.batches = [[], [flag_event]]

    board_after, _status = human._wait_for_stable_change(before)

    assert int(board_after[0, 1]) == HIDDEN, "the newer flag must not be folded in"
    assert human._deferred_events == [flag_event]


def test_flag_clicked_during_open_wait_is_recorded_separately(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    start = empty_board()
    human._stable_board = start.copy()
    opened = start.copy()
    opened[0, 0] = 0
    folded = opened.copy()
    folded[0, 1] = FLAGGED
    flag_event = click_event(opened, 1, 0, button=2, event_id=3)

    observer.status = GameStatus.RUNNING
    observer.observations = [opened]
    observer.page.batches = [[], [flag_event]]
    open_event = click_event(start, 0, 0, event_id=2)
    assert human.handle_event(open_event) == "recorded"

    # The open transition must not contain the later flag.
    assert int(recorder._steps[0].board_after[0, 1]) == HIDDEN
    assert human.drain_events() == [flag_event]

    observer.status = GameStatus.RUNNING
    observer.observations = [folded, folded]
    observer.page.batches = []
    assert human.handle_event(flag_event) == "recorded"

    steps = recorder._steps
    assert [s.action_type for s in steps] == [ACTION_OPEN, ACTION_FLAG]
    assert np.array_equal(steps[0].board_after, steps[1].board_before)
    assert int(steps[1].board_after[0, 1]) == FLAGGED
    assert store.validate().ok


# ---------------------------------------------------------------------------
# synchronization: transients, cascades, delays
# ---------------------------------------------------------------------------


def test_cascade_is_one_transition(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    before = empty_board()
    final = before.copy()
    final[0, 0] = 1
    final[0, 1] = final[0, 2] = final[0, 3] = 0  # one click reveals many cells
    observer.board, observer.status = final, GameStatus.RUNNING
    assert human.handle_event(click_event(before, 0, 0)) == "recorded"
    assert recorder.steps_recorded == 1
    assert np.array_equal(recorder._steps[0].board_after, final)


def test_transient_intermediate_board_is_waited_out(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    before = empty_board()
    transient = before.copy()
    transient[0, 0] = 1
    final = transient.copy()
    final[0, 1] = 1
    observer.observations = [transient, final, final]
    observer.status = GameStatus.RUNNING
    assert human.handle_event(click_event(before, 0, 0)) == "recorded"
    assert recorder.steps_recorded == 1
    assert np.array_equal(recorder._steps[0].board_after, final)


def test_multiple_unstable_observations_before_stable(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    before = empty_board()
    t1 = before.copy()
    t1[0, 0] = 1
    t2 = t1.copy()
    t2[0, 1] = 1
    final = t2.copy()
    final[0, 2] = 1
    observer.observations = [t1, t2, final, final]
    observer.status = GameStatus.RUNNING
    assert human.handle_event(click_event(before, 0, 0)) == "recorded"
    assert recorder.steps_recorded == 1
    assert np.array_equal(recorder._steps[0].board_after, final)


def test_delayed_dom_update_is_waited_out(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    before = empty_board()
    final = before.copy()
    final[0, 0] = 1
    observer.observations = [before, before, final, final]
    observer.status = GameStatus.RUNNING
    assert human.handle_event(click_event(before, 0, 0)) == "recorded"
    assert recorder.steps_recorded == 1
    assert np.array_equal(recorder._steps[0].board_after, final)


def test_chord_with_no_board_change_is_not_recorded(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    _reveal(human, observer, x=0, y=0)
    observer.board, observer.status = human._stable_board.copy(), GameStatus.RUNNING
    result = human.handle_event(click_event(human._stable_board, 0, 0))  # chord, no change
    assert result in {"ignored", "rejected"}
    assert recorder.steps_recorded == 1


# ---------------------------------------------------------------------------
# terminal states
# ---------------------------------------------------------------------------


def test_terminal_loss_allows_hidden_clicked_cell(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    _reveal(human, observer, x=0, y=0)
    before = human._stable_board.copy()
    observer.board = before
    observer.status = GameStatus.LOST  # mine stays hidden
    observer.statuses = [GameStatus.RUNNING, GameStatus.LOST]
    assert human.handle_event(click_event(before, 1, 0)) == "game_end"
    game = list(store.iter_games())[0]
    assert game.result == "lost"
    final = store.load_game(game.game_id)[1][-1]
    assert final.terminated and final.game_status == "lost" and final.reward == -10.0


def test_terminal_win_records_once(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    before = human._stable_board.copy()
    final = before.copy()
    final[0, 0] = 0
    observer.board = final
    observer.status = GameStatus.WON
    observer.statuses = [GameStatus.RUNNING, GameStatus.WON]  # running then terminal
    assert human.handle_event(click_event(before, 0, 0)) == "game_end"
    game = list(store.iter_games())[0]
    assert game.result == "won" and game.total_steps == 1
    final = store.load_game(game.game_id)[1][-1]
    assert final.reward == WIN_REWARD + 1.0, "win bonus plus the revealed cell"
    assert store.validate().ok


def test_game_ids_unique_across_games(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    for _ in range(2):
        before = empty_board()
        final = before.copy()
        final[0, 0] = 0
        # A genuine reset must be observed before the next game may start.
        observer.status = GameStatus.READY
        human._handle_idle_board(before.copy())
        observer.board = final
        observer.status = GameStatus.WON
        observer.statuses = [GameStatus.RUNNING, GameStatus.WON]
        human.handle_event(click_event(before, 0, 0))
    ids = [game.game_id for game in store.iter_games()]
    assert len(ids) == 2 and len(set(ids)) == 2
    assert human.report.wins == 2


# ---------------------------------------------------------------------------
# game boundaries: reset vs unattributable change
# ---------------------------------------------------------------------------


def test_single_transient_idle_change_is_not_a_boundary(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    _reveal(human, observer)
    strange = human._stable_board.copy()
    strange[1, 1] = 0  # changed, but not fresh
    human._handle_idle_board(strange)  # one non-fresh observation only
    assert list(store.iter_games()) == []
    assert recorder.active_game_id is not None


def test_fresh_board_idle_abandons_previous_game(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    _reveal(human, observer)

    human._handle_idle_board(empty_board())  # a fresh board is a genuine reset
    games = list(store.iter_games())
    assert len(games) == 1 and games[0].result == "abandoned"
    assert np.array_equal(human._stable_board, empty_board())


def test_unattributable_idle_change_suspends_recording_not_game(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    _reveal(human, observer)

    strange = empty_board()
    strange[1, 1] = 0  # changed, but not a fresh board
    human._handle_idle_board(strange)
    human._handle_idle_board(strange)
    games = list(store.iter_games())
    assert games == []
    assert recorder.active_game_id is not None
    assert human._capture_error == "unattributed_board_change"
    human._handle_idle_board(empty_board())
    assert list(store.iter_games())[0].result == "incomplete"


def test_idle_board_equal_to_stable_is_ignored(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    _reveal(human, observer)
    human._handle_idle_board(human._stable_board.copy())
    human._handle_idle_board(human._stable_board.copy())
    assert list(store.iter_games()) == []
    assert recorder.active_game_id is not None


# ---------------------------------------------------------------------------
# run loop, no-leak and CLI
# ---------------------------------------------------------------------------


def test_run_honours_stop_event_and_installs_capture(tmp_path) -> None:
    human, observer, recorder, store, emitted = make_human(tmp_path)
    stop_event = threading.Event()
    stop_event.set()
    report = human.run(max_games=5, stop_event=stop_event)
    assert observer.ready is True
    assert any("addEventListener" in script for script in observer.page.init_scripts)
    assert report.games == 0
    assert list(store.iter_games()) == []


def test_no_hidden_mine_information_in_observations(tmp_path) -> None:
    human, observer, recorder, store, _ = make_human(tmp_path)
    before = empty_board()
    final = before.copy()
    final[0, 0] = 0
    observer.board = final
    observer.status = GameStatus.WON
    observer.statuses = [GameStatus.RUNNING, GameStatus.WON]
    human.handle_event(click_event(before, 0, 0))

    game = list(store.iter_games())[0]
    assert GROUND_TRUTH_KEY not in game.safe_view()
    for transition in store.iter_transitions():
        assert GROUND_TRUTH_KEY not in transition.to_dict()
        assert int(transition.board_before.min()) >= HIDDEN
        assert int(transition.board_after.max()) <= 8


def test_discard_game_writes_nothing(tmp_path) -> None:
    store, recorder = make_recorder(tmp_path)
    recorder.start_game()
    recorder.discard_game()
    assert recorder.active_game_id is None
    assert list(store.iter_games()) == []
    assert list(store.iter_transitions()) == []


def test_legacy_recorder_cli_is_removed() -> None:
    # The deprecated append-only JSONL entry point (chord-as-open) was removed;
    # the canonical recorder is minesweeper_rl.data.record_human_dataset.
    import minesweeper_rl.data.record_human as record_human
    assert not hasattr(record_human, "main")
    assert not hasattr(record_human, "build_parser")
