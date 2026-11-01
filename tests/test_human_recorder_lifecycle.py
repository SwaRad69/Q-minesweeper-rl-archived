"""Semantic/recording-correctness tests for the human recorder (no browser).

Covers the game-lifecycle state machine (fresh-board requirement, first-action
rules, terminal handling, reset/resync) and the semantic validator/loader that
keeps corrupted or fragment games out of training.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from minesweeper_rl.action_types import (
    ACTION_CHORD,
    ACTION_FLAG,
    ACTION_NAMES,
    ACTION_OPEN,
    compound_action_id,
)
from minesweeper_rl.data import (
    HUMAN_RECORDER_VERSION,
    HumanDatasetError,
    HumanDatasetStore,
    HumanDemonstrationDataset,
    PersistentHumanRecorder,
    canonical_identity,
    semantic_game_result,
    validate_dataset,
)
from minesweeper_rl.data import schema
from minesweeper_rl.data.record_human import HumanTrajectoryRecorder
from minesweeper_rl.dom import FLAGGED, HIDDEN, GameStatus

WIDTH, HEIGHT = 4, 2
CELLS = WIDTH * HEIGHT


# ---------------------------------------------------------------------------
# Synthetic observers / events
# ---------------------------------------------------------------------------


def empty_board() -> np.ndarray:
    return np.full((HEIGHT, WIDTH), HIDDEN, dtype=np.int8)


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


#: Face classes captured in the pre-action snapshot by the live capture script.
FACE_RUNNING = ["hd_top-area-face-unpressed"]
FACE_WON = ["hd_top-area-face-win"]
FACE_LOST = ["hd_top-area-face-lose"]


def click_event(board: np.ndarray, x: int, y: int, *, button: int = 0,
                event_id: int | None = None,
                face_classes: list[str] | None = None) -> dict:
    return {
        "id": event_id, "token": "t", "type": "pointerdown", "isTrusted": True,
        "button": button, "x": str(x), "y": str(y), "clientX": 1, "clientY": 2,
        "tag": "DIV", "targetClasses": ["cell"],
        "faceClasses": list(face_classes or []),
        "cells": snapshot(board),
    }


class FakePage:
    def __init__(self) -> None:
        self.closed = False

    def add_init_script(self, script: str) -> None:
        return None

    def evaluate(self, expression: str, *args: object):
        return []

    def is_closed(self) -> bool:
        return self.closed


class FakeObserver:
    def __init__(self) -> None:
        self.config = SimpleNamespace(width=WIDTH, height=HEIGHT)
        self.page = FakePage()
        self.board = empty_board()
        self.status = GameStatus.READY
        self.statuses: list[GameStatus] = []

    def wait_until_ready(self) -> None:
        return None

    def observe(self) -> np.ndarray:
        return self.board.copy()

    def get_game_status(self) -> GameStatus:
        if self.statuses:
            self.status = self.statuses.pop(0)
        return self.status


def make_observer_recorder(tmp_path):
    store = HumanDatasetStore(
        tmp_path / "human", width=WIDTH, height=HEIGHT, mines=1, difficulty="expert"
    )
    sink = PersistentHumanRecorder(
        store, width=WIDTH, height=HEIGHT, mines=1, difficulty="expert"
    )
    observer = FakeObserver()
    human = HumanTrajectoryRecorder(
        observer, sink, poll_interval=0.0, action_timeout=0.2,
        quiescence_time=0.0, emit=lambda _: None,
    )
    human._handle_idle_board(observer.board.copy())  # promote fresh -> READY
    return human, observer, sink, store


def play_open(human, observer, *, x=0, y=0, opened_value=1):
    before = human._stable_board.copy()
    after = before.copy()
    after[y, x] = opened_value
    observer.board = after
    observer.status = GameStatus.RUNNING
    return human.handle_event(click_event(before, x, y))


def play_win(human, observer):
    before = human._stable_board.copy()
    after = before.copy()
    after[0, 0] = 0
    observer.board = after
    observer.status = GameStatus.WON
    observer.statuses = [GameStatus.RUNNING, GameStatus.WON]
    return human.handle_event(click_event(before, 0, 0))


# ---------------------------------------------------------------------------
# Fresh board requirement / first action
# ---------------------------------------------------------------------------


def test_first_action_flag_is_rejected(tmp_path) -> None:
    human, observer, sink, store = make_observer_recorder(tmp_path)
    assert human.handle_event(click_event(empty_board(), 0, 0, button=2)) == "rejected"
    assert sink.active_game_id is None
    assert store.game_count == 0


def test_first_action_open_is_accepted(tmp_path) -> None:
    human, observer, sink, store = make_observer_recorder(tmp_path)
    assert play_open(human, observer) == "recorded"
    assert sink.steps_recorded == 1
    first = sink._transitions[0]
    assert first["action"]["action_type"] == ACTION_OPEN


def test_chord_first_is_rejected(tmp_path) -> None:
    """A chord needs an opened number; a board with one is not a fresh board."""

    human, observer, sink, store = make_observer_recorder(tmp_path)
    midgame = empty_board()
    midgame[0, 0] = 1  # already opened number
    human._handle_idle_board(midgame)  # not fresh -> not READY
    observer.board = midgame
    observer.status = GameStatus.RUNNING
    assert human.handle_event(click_event(midgame, 0, 0)) in {
        "waiting", "ignored_terminal", "ignored", "rejected",
    }
    assert sink.active_game_id is None
    assert store.game_count == 0


def test_midgame_click_does_not_start_a_game(tmp_path) -> None:
    human, observer, sink, store = make_observer_recorder(tmp_path)
    midgame = empty_board()
    midgame[1, 1] = 2
    human._handle_idle_board(midgame)
    observer.board = midgame
    observer.status = GameStatus.RUNNING
    assert human.handle_event(click_event(midgame, 0, 0)) in {
        "waiting", "ignored_terminal", "ignored", "rejected",
    }
    assert store.game_count == 0


# ---------------------------------------------------------------------------
# Terminal handling / reset lifecycle
# ---------------------------------------------------------------------------


def test_terminal_click_does_not_create_a_new_game(tmp_path) -> None:
    human, observer, sink, store = make_observer_recorder(tmp_path)
    assert play_win(human, observer) == "game_end"
    assert store.game_count == 1

    # A stray click on the finished board (still showing the terminal status)
    # must never begin a new game.
    ended = human._stable_board.copy()
    observer.board = ended
    observer.status = GameStatus.WON
    observer.statuses = []
    assert human.handle_event(click_event(ended, 1, 0)) == "ignored_terminal"
    assert store.game_count == 1


def test_terminal_noop_is_not_recorded(tmp_path) -> None:
    human, observer, sink, store = make_observer_recorder(tmp_path)
    play_open(human, observer, x=1, y=0)
    before = human._stable_board.copy()
    # The face is already terminal before the click: a no-op chord must never be
    # recorded as a new transition for the finished game.
    observer.board = before
    observer.status = GameStatus.LOST
    observer.statuses = []  # already lost before the action
    result = human.handle_event(click_event(before, 1, 0))
    assert result == "ignored_terminal"
    assert sink.steps_recorded == 0
    assert store.game_count == 1
    assert next(iter(store.iter_games())).outcome == "incomplete"


def test_losing_chord_with_unchanged_board_terminates(tmp_path) -> None:
    """A chord that detonates a mine ends the game even if the board looks
    unchanged (detonated mines are normalized to HIDDEN)."""

    human, observer, sink, store = make_observer_recorder(tmp_path)
    assert play_open(human, observer, x=0, y=0, opened_value=1) == "recorded"

    before = human._stable_board.copy()
    flagged = before.copy()
    flagged[0, 1] = FLAGGED
    observer.board = flagged
    observer.status = GameStatus.RUNNING
    assert human.handle_event(click_event(before, 1, 0, button=2)) == "recorded"

    before = human._stable_board.copy()  # unchanged by the losing chord
    observer.board = before
    observer.status = GameStatus.LOST
    observer.statuses = [GameStatus.RUNNING, GameStatus.LOST]
    assert human.handle_event(click_event(before, 0, 0)) == "game_end"

    game = store.load_game("game_000001")
    assert game["result"] == "lost" and game["complete"] is True
    final = game["transitions"][-1]
    assert final["action"]["action_type"] == ACTION_CHORD
    assert final["terminated"] is True


# ---------------------------------------------------------------------------
# Fast terminal DOM: the press that caused a win/loss must not be mistaken for
# a stray click after game over (live regression: genuine LOSS -> ABANDONED).
# ---------------------------------------------------------------------------


def test_fast_loss_is_recorded_not_abandoned(tmp_path) -> None:
    """A loss whose DOM update beats the press is still a terminal game."""

    human, observer, sink, store = make_observer_recorder(tmp_path)
    assert play_open(human, observer, x=0, y=0, opened_value=1) == "recorded"

    before = human._stable_board.copy()  # the detonated mine stays HIDDEN
    observer.board = before
    observer.status = GameStatus.LOST
    observer.statuses = []  # the loss face is already visible when drained

    event = click_event(before, 1, 0, face_classes=FACE_RUNNING)
    assert human.handle_event(event) == "game_end"

    game = store.load_game("game_000001")
    assert game["result"] == "lost" and game["complete"] is True
    final = game["transitions"][-1]
    assert final["terminated"] is True
    assert final["game_status"] == "lost"
    assert sink.active_game_id is None
    assert validate_dataset(tmp_path / "human").ok


def test_fast_win_is_recorded_not_abandoned(tmp_path) -> None:
    """A win whose DOM update beats the press is still a terminal game."""

    human, observer, sink, store = make_observer_recorder(tmp_path)
    before = human._stable_board.copy()
    final = before.copy()
    final[0, 0] = 0
    observer.board = final
    observer.status = GameStatus.WON
    observer.statuses = []

    event = click_event(before, 0, 0, face_classes=FACE_RUNNING)
    assert human.handle_event(event) == "game_end"

    game = store.load_game("game_000001")
    assert game["result"] == "won" and game["complete"] is True
    assert game["transitions"][-1]["terminated"] is True
    assert validate_dataset(tmp_path / "human").ok


def test_stray_click_on_finished_board_is_abandoned_not_recorded(tmp_path) -> None:
    """A press whose pre-action face was already terminal stays a stray click."""

    human, observer, sink, store = make_observer_recorder(tmp_path)
    assert play_open(human, observer, x=0, y=0, opened_value=1) == "recorded"

    before = human._stable_board.copy()
    observer.board = before
    observer.status = GameStatus.LOST
    observer.statuses = []

    event = click_event(before, 1, 0, face_classes=FACE_LOST)
    assert human.handle_event(event) == "ignored_terminal"

    games = list(store.iter_games())
    assert len(games) == 1 and games[0].outcome == "incomplete"
    stored = store.load_game("game_000001")
    assert len(stored["transitions"]) == 1  # the stray press was not recorded
    assert stored["transitions"][-1]["terminated"] is False


def test_loss_with_delayed_status_and_mine_reveal_is_recorded(tmp_path) -> None:
    """The post-loss board may change (wrong flag revealed) after the status."""

    human, observer, sink, store = make_observer_recorder(tmp_path)
    assert play_open(human, observer, x=0, y=0, opened_value=1) == "recorded"

    flagged = human._stable_board.copy()
    flagged[0, 1] = FLAGGED
    observer.board = flagged
    observer.status = GameStatus.RUNNING
    assert human.handle_event(click_event(human._stable_board, 1, 0, button=2)) == "recorded"

    before = human._stable_board.copy()
    after = before.copy()
    after[0, 1] = HIDDEN  # the wrongly-flagged cell is revealed on loss
    observer.board = after
    observer.status = GameStatus.LOST
    observer.statuses = [GameStatus.RUNNING, GameStatus.LOST]

    event = click_event(before, 2, 0, face_classes=FACE_RUNNING)
    assert human.handle_event(event) == "game_end"
    final = store.load_game("game_000001")["transitions"][-1]
    assert final["terminated"] is True and final["game_status"] == "lost"
    assert validate_dataset(tmp_path / "human").ok


def test_losing_chord_with_board_change_terminates(tmp_path) -> None:
    """A losing chord that first opens safe neighbours still terminates."""

    human, observer, sink, store = make_observer_recorder(tmp_path)
    assert play_open(human, observer, x=0, y=0, opened_value=1) == "recorded"
    flagged = human._stable_board.copy()
    flagged[0, 1] = FLAGGED
    observer.board = flagged
    observer.status = GameStatus.RUNNING
    assert human.handle_event(click_event(human._stable_board, 1, 0, button=2)) == "recorded"

    before = human._stable_board.copy()  # (0,0) is an opened number
    after = before.copy()
    after[0, 2] = 3  # a safe neighbour opens before the mine detonates
    observer.board = after
    observer.status = GameStatus.LOST
    observer.statuses = [GameStatus.RUNNING, GameStatus.LOST]

    event = click_event(before, 0, 0, face_classes=FACE_RUNNING)
    assert human.handle_event(event) == "game_end"
    final = store.load_game("game_000001")["transitions"][-1]
    assert final["action"]["action_type"] == ACTION_CHORD
    assert final["terminated"] is True and final["game_status"] == "lost"
    assert validate_dataset(tmp_path / "human").ok


def test_interrupted_game_without_terminal_press_is_abandoned(tmp_path) -> None:
    """No terminal press captured: still an incomplete/abandoned game."""

    human, observer, sink, store = make_observer_recorder(tmp_path)
    assert play_open(human, observer, x=0, y=0, opened_value=1) == "recorded"
    human._handle_idle_board(empty_board())  # player quit / started over

    games = list(store.iter_games())
    assert len(games) == 1 and games[0].outcome == "incomplete"
    report = validate_dataset(tmp_path / "human")
    assert report.ok and report.complete_games == 0


def test_phantom_press_with_running_face_is_rejected(tmp_path) -> None:
    """Unrelated board activity is still never recorded as a human action."""

    human, observer, sink, store = make_observer_recorder(tmp_path)
    assert play_open(human, observer, x=0, y=0, opened_value=1) == "recorded"
    before = human._stable_board.copy()
    phantom = before.copy()
    phantom[1, 1] = 0  # some other cell changes, not the clicked one

    observer.board = phantom
    observer.status = GameStatus.RUNNING
    observer.statuses = []
    event = click_event(before, 1, 0, face_classes=FACE_RUNNING)
    assert human.handle_event(event) == "rejected"
    assert sink.steps_recorded == 1  # only the earlier OPEN
    assert store.game_count == 0  # still the active, uncommitted game


def test_reset_lifecycle_allows_a_second_valid_game(tmp_path) -> None:
    human, observer, sink, store = make_observer_recorder(tmp_path)
    assert play_win(human, observer) == "game_end"
    # Observe a genuine fresh board (face reset) after the reset.
    observer.status = GameStatus.READY
    human._handle_idle_board(empty_board())
    assert human._phase == "ready"

    assert play_open(human, observer, x=1, y=0) == "recorded"
    assert sink.active_game_id == "game_000002"
    # Finish the second game so both are persisted, then validate.
    observer.status = GameStatus.READY
    human._handle_idle_board(empty_board())
    report = validate_dataset(tmp_path / "human")
    assert report.games == 2
    assert report.games_invalid == 0, report.errors


def test_interrupted_game_recovers_for_the_next_game(tmp_path) -> None:
    human, observer, sink, store = make_observer_recorder(tmp_path)
    assert play_open(human, observer) == "recorded"
    # A fresh board appears without the game ever ending: the partial game is
    # saved as incomplete and the recorder must be ready for the next one.
    human._handle_idle_board(empty_board())
    assert store.game_count == 1
    assert next(iter(store.iter_games())).outcome == "incomplete"

    observer.status = GameStatus.RUNNING
    assert play_open(human, observer, x=1, y=0) == "recorded"
    assert sink.active_game_id == "game_000002"
    assert sink.steps_recorded == 1


# ---------------------------------------------------------------------------
# ABANDONED semantics / long-game and transient-failure regressions
# ---------------------------------------------------------------------------


def test_normal_short_win_is_won(tmp_path) -> None:
    human, observer, _sink, store = make_observer_recorder(tmp_path)
    assert play_win(human, observer) == "game_end"
    game = store.load_game("game_000001")
    assert game["result"] == "won" and game["termination_reason"] is None


def test_normal_short_loss_is_lost(tmp_path) -> None:
    human, observer, _sink, store = make_observer_recorder(tmp_path)
    assert play_open(human, observer, x=0, y=0) == "recorded"
    before = human._stable_board.copy()
    observer.board = before
    observer.status = GameStatus.LOST
    observer.statuses = [GameStatus.RUNNING, GameStatus.LOST]
    assert human.handle_event(
        click_event(before, 1, 0, face_classes=FACE_RUNNING)
    ) == "game_end"
    assert store.load_game("game_000001")["result"] == "lost"


@pytest.mark.parametrize("terminal", [GameStatus.WON, GameStatus.LOST])
def test_long_active_game_waits_for_actual_terminal_result(tmp_path, terminal) -> None:
    """Total duration and a long player think-time never imply ABANDONED."""

    human, observer, sink, store = make_observer_recorder(tmp_path)
    assert play_open(human, observer, x=0, y=0) == "recorded"
    sink._started_at -= 24 * 60 * 60  # simulate a day-long active game

    # No event or board change for an arbitrarily long time: it stays active.
    human._handle_idle_board(human._stable_board.copy())
    human._handle_idle_board(human._stable_board.copy())
    assert sink.active_game_id is not None and store.game_count == 0

    before = human._stable_board.copy()
    after = before.copy()
    x = 1
    if terminal is GameStatus.WON:
        after[0, x] = 0
    observer.board = after
    observer.status = terminal
    observer.statuses = [GameStatus.RUNNING, terminal]
    result = human.handle_event(
        click_event(before, x, 0, face_classes=FACE_RUNNING)
    )
    assert result == "game_end"
    game = store.load_game("game_000001")
    assert game["result"] == terminal.value
    assert game["duration_s"] >= 24 * 60 * 60


def test_temporary_board_observation_failure_recovers_action(tmp_path) -> None:
    human, observer, sink, store = make_observer_recorder(tmp_path)
    human.action_timeout = 0.001
    before = human._stable_board.copy()
    after = before.copy()
    after[0, 0] = 1

    original_observe = observer.observe
    observer.observe = lambda: (_ for _ in ()).throw(RuntimeError("transient DOM"))
    observer.status = GameStatus.RUNNING
    assert human.handle_event(click_event(before, 0, 0)) == "rejected"
    assert human._pending_action is not None
    assert sink.active_game_id is None and store.game_count == 0

    observer.observe = original_observe
    observer.board = after
    human._handle_idle_board(after)
    assert human._pending_action is None
    assert sink.active_game_id is not None and sink.steps_recorded == 1
    assert store.game_count == 0


def test_temporary_status_dom_failure_recovers_action(tmp_path) -> None:
    human, observer, sink, store = make_observer_recorder(tmp_path)
    before = human._stable_board.copy()
    after = before.copy()
    after[0, 0] = 1
    observer.board = after

    original_status = observer.get_game_status
    observer.get_game_status = lambda: (_ for _ in ()).throw(
        RuntimeError("face temporarily missing")
    )
    assert human.handle_event(click_event(before, 0, 0)) == "rejected"
    assert human._pending_action is not None
    assert sink.active_game_id is None and store.game_count == 0

    observer.get_game_status = original_status
    observer.status = GameStatus.RUNNING
    human._handle_idle_board(after)
    assert human._pending_action is None
    assert sink.active_game_id is not None and sink.steps_recorded == 1


def test_temporary_playwright_event_queue_failure_is_unknown(tmp_path) -> None:
    human, observer, sink, store = make_observer_recorder(tmp_path)
    assert play_open(human, observer) == "recorded"
    original_evaluate = observer.page.evaluate
    observer.page.evaluate = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("temporary Playwright failure")
    )
    assert human._safe_drain_events() is None
    assert sink.active_game_id is not None and store.game_count == 0
    observer.page.evaluate = original_evaluate
    assert human._safe_drain_events() == []


def test_page_closure_abandons_with_explicit_reason(tmp_path) -> None:
    human, observer, _sink, store = make_observer_recorder(tmp_path)
    assert play_open(human, observer) == "recorded"
    observer.page.closed = True
    human.run(max_games=1)
    game = store.load_game("game_000001")
    assert game["result"] == "abandoned"
    assert game["termination_reason"] == "page_closed"


def test_explicit_cancellation_abandons_with_explicit_reason(tmp_path) -> None:
    human, observer, _sink, store = make_observer_recorder(tmp_path)
    assert play_open(human, observer) == "recorded"
    stop = threading.Event()
    stop.set()
    human.run(max_games=1, stop_event=stop)
    game = store.load_game("game_000001")
    assert game["result"] == "abandoned"
    assert game["termination_reason"] == "explicit_session_cancellation"


def test_persistence_failure_does_not_convert_win_to_abandoned(
    tmp_path, monkeypatch
) -> None:
    human, observer, sink, store = make_observer_recorder(tmp_path)

    def fail_write(_game):
        raise PermissionError("simulated OneDrive lock")

    original_write = store.write_game
    monkeypatch.setattr(store, "write_game", fail_write)
    before = human._stable_board.copy()
    after = before.copy()
    after[0, 0] = 0
    observer.board = after
    observer.status = GameStatus.WON
    observer.statuses = [GameStatus.RUNNING, GameStatus.WON]
    assert human.handle_event(click_event(before, 0, 0, face_classes=FACE_RUNNING)) == "game_end"

    # The terminal fact remains in memory for retry/diagnosis; no false
    # ABANDONED artifact was written.
    assert sink.active_game_id is not None
    assert sink.game_terminated is True
    assert store.game_count == 0
    assert list(store.iter_games()) == []
    monkeypatch.setattr(store, "write_game", original_write)
    assert human._retry_finalization()
    assert store.load_game("game_000001")["result"] == "won"
    assert validate_dataset(store.root).ok


# ---------------------------------------------------------------------------
# Semantic validation (handcrafted transcripts)
# ---------------------------------------------------------------------------


def transition(before, after, action_type, x, y, *, status="running",
               terminated=False):
    cell = y * WIDTH + x
    return {
        "step": 0,
        "timestamp": 0.0,
        "observation_before": schema.encode_board(before),
        "observation_after": schema.encode_board(after),
        "action": {
            "type": ACTION_NAMES[action_type],
            "action_type": action_type,
            "x": x,
            "y": y,
            "cell_index": cell,
            "action_id": compound_action_id(action_type, cell, cells=CELLS),
        },
        "game_status": status,
        "terminated": terminated,
        "truncated": False,
        "reward": None,
        "newly_revealed": None,
        "opened_cells": None,
    }


def make_game(transitions, *, complete, result,
              recorder_version=HUMAN_RECORDER_VERSION, outcome=None):
    for index, step in enumerate(transitions):
        step["step"] = index
    game = {
        "format": "minesweeper_human_game",
        "format_version": 2,
        "game_id": "game_000001",
        "game_index": 1,
        "source": "human",
        "width": WIDTH,
        "height": HEIGHT,
        "mines": 1,
        "difficulty": "expert",
        "outcome": outcome or (result if complete else "incomplete"),
        "complete": complete,
        "result": result,
        "total_steps": len(transitions),
        "transitions": transitions,
    }
    game.update(canonical_identity(WIDTH, HEIGHT))
    game["recorder_version"] = recorder_version
    return game


def valid_won_game():
    before = empty_board()
    after = before.copy()
    after[0, 0] = 0
    return make_game(
        [transition(before, after, ACTION_OPEN, 0, 0, status="won", terminated=True)],
        complete=True, result="won",
    )


def valid_flag_game():
    before = empty_board()
    opened = before.copy()
    opened[0, 0] = 1
    flagged = opened.copy()
    flagged[1, 0] = FLAGGED
    return make_game(
        [
            transition(before, opened, ACTION_OPEN, 0, 0),
            transition(opened, flagged, ACTION_FLAG, 0, 1),
        ],
        complete=False, result="abandoned",
    )


def valid_chord_game():
    before = empty_board()
    opened = before.copy()
    opened[0, 0] = 1
    after = opened.copy()
    after[1, 1] = 0
    return make_game(
        [
            transition(before, opened, ACTION_OPEN, 0, 0),
            transition(opened, after, ACTION_CHORD, 0, 0),
        ],
        complete=False, result="abandoned",
    )


def test_semantic_accepts_valid_games() -> None:
    for game in (valid_won_game(), valid_flag_game(), valid_chord_game()):
        result = semantic_game_result(game)
        assert result.valid, result.reasons


def test_semantic_accepts_valid_unflag() -> None:
    before = empty_board()
    opened = before.copy()
    opened[0, 0] = 1
    flagged = opened.copy()
    flagged[1, 0] = FLAGGED
    unflagged = flagged.copy()
    unflagged[1, 0] = HIDDEN
    game = make_game(
        [
            transition(before, opened, ACTION_OPEN, 0, 0),
            transition(opened, flagged, ACTION_FLAG, 0, 1),
            transition(flagged, unflagged, ACTION_FLAG, 0, 1),
        ],
        complete=False, result="abandoned",
    )
    assert semantic_game_result(game).valid


def test_semantic_accepts_losing_chord_with_unchanged_board() -> None:
    before = empty_board()
    opened = before.copy()
    opened[0, 0] = 1
    game = make_game(
        [
            transition(before, opened, ACTION_OPEN, 0, 0),
            transition(opened, opened.copy(), ACTION_CHORD, 0, 0,
                       status="lost", terminated=True),
        ],
        complete=True, result="lost",
    )
    result = semantic_game_result(game)
    assert result.valid, result.reasons


def test_semantic_rejects_non_fresh_initial_board() -> None:
    midgame = empty_board()
    midgame[0, 0] = 1
    after = midgame.copy()
    after[0, 1] = 1
    game = make_game(
        [transition(midgame, after, ACTION_OPEN, 1, 0)],
        complete=False, result="abandoned",
    )
    result = semantic_game_result(game)
    assert not result.valid
    assert any("not a fresh board" in reason for reason in result.reasons)


def test_semantic_rejects_first_action_not_open() -> None:
    before = empty_board()
    after = before.copy()
    after[0, 0] = FLAGGED
    game = make_game(
        [transition(before, after, ACTION_FLAG, 0, 0)],
        complete=False, result="abandoned",
    )
    result = semantic_game_result(game)
    assert not result.valid
    assert any("first action is not OPEN" in reason for reason in result.reasons)


def test_semantic_rejects_safe_open_ending_in_lost() -> None:
    before = empty_board()
    after = before.copy()
    after[0, 0] = 1  # safe reveal
    game = make_game(
        [transition(before, after, ACTION_OPEN, 0, 0, status="lost", terminated=True)],
        complete=True, result="lost",
    )
    result = semantic_game_result(game)
    assert not result.valid
    assert any("safe OPEN cannot end in LOST" in reason for reason in result.reasons)


def test_semantic_rejects_phantom_continuation() -> None:
    previous_final = empty_board()
    previous_final[0, 0] = 1
    # The next game begins exactly on the previous game's final board.
    after = previous_final.copy()
    after[0, 1] = 1
    game = make_game(
        [transition(previous_final, after, ACTION_OPEN, 1, 0)],
        complete=False, result="abandoned",
    )
    result = semantic_game_result(game, previous_final_board=previous_final)
    assert not result.valid
    assert any("phantom continuation" in reason for reason in result.reasons)


def test_semantic_rejects_pre_fix_recorder_version() -> None:
    game = valid_won_game()
    game["recorder_version"] = 1
    result = semantic_game_result(game)
    assert not result.valid
    assert any("predates the capture fix" in reason for reason in result.reasons)


# ---------------------------------------------------------------------------
# Loader training safety
# ---------------------------------------------------------------------------


def write_raw_game(root: Path, index: int, game: dict) -> None:
    game = dict(game)
    game["game_id"] = f"game_{index:06d}"
    game["game_index"] = index
    (root / "games" / f"{game['game_id']}.json").write_text(
        json.dumps(game), encoding="utf-8"
    )


def test_loader_excludes_invalid_games_by_default(tmp_path) -> None:
    root = tmp_path / "human"
    HumanDatasetStore(root, width=WIDTH, height=HEIGHT, mines=1, difficulty="expert")
    write_raw_game(root, 1, valid_won_game())

    midgame = empty_board()
    midgame[0, 0] = 1
    after = midgame.copy()
    after[0, 1] = 1
    write_raw_game(
        root, 2,
        make_game([transition(midgame, after, ACTION_OPEN, 1, 0)],
                  complete=False, result="abandoned"),
    )

    dataset = HumanDemonstrationDataset(root)
    assert dataset.game_ids() == ["game_000001"]
    assert len(dataset.invalid_games) == 1

    included = HumanDemonstrationDataset(root, include_invalid=True)
    assert len(included.games) == 2


def test_loader_refuses_all_invalid_collection(tmp_path) -> None:
    root = tmp_path / "human"
    HumanDatasetStore(root, width=WIDTH, height=HEIGHT, mines=1, difficulty="expert")
    midgame = empty_board()
    midgame[0, 0] = 1
    after = midgame.copy()
    after[0, 1] = 1
    write_raw_game(
        root, 1,
        make_game([transition(midgame, after, ACTION_OPEN, 1, 0)],
                  complete=False, result="abandoned"),
    )
    with pytest.raises(schema.DataError, match="semantically"):
        HumanDemonstrationDataset(root)


def test_validate_dataset_reports_valid_and_invalid_counts(tmp_path) -> None:
    root = tmp_path / "human"
    HumanDatasetStore(root, width=WIDTH, height=HEIGHT, mines=1, difficulty="expert")
    write_raw_game(root, 1, valid_won_game())

    midgame = empty_board()
    midgame[0, 0] = 1
    after = midgame.copy()
    after[0, 1] = 1
    write_raw_game(
        root, 2,
        make_game([transition(midgame, after, ACTION_OPEN, 1, 0)],
                  complete=False, result="abandoned"),
    )

    report = validate_dataset(root)
    assert report.games == 2
    assert report.games_valid == 1
    assert report.games_invalid == 1
    assert "game_000002" in report.invalid_games
    assert report.invalid_reasons
