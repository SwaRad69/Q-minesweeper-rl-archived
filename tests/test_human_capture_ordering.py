"""Production Game 69: browser input must stay ordered across blocking reads."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from minesweeper_rl.data import schema, validate_dataset
from minesweeper_rl.dom import FLAGGED, GameStatus
from minesweeper_rl.data import human_dataset as dataset_module
from minesweeper_rl.data import record_human as recorder_module
from test_human_recorder_lifecycle import (
    FACE_RUNNING, click_event, make_observer_recorder, play_open,
)


def test_press_arriving_inside_observation_does_not_contaminate_action(tmp_path):
    human, observer, sink, _store = make_observer_recorder(tmp_path)
    play_open(human, observer)
    before = human._stable_board.copy()
    flagged = before.copy()
    flagged[0, 1] = FLAGGED
    later = flagged.copy()
    later[1, 0] = 1
    next_press = click_event(flagged, 0, 0, event_id=3, face_classes=FACE_RUNNING)
    queue = []

    def observe():
        queue.append(next_press)
        return later.copy()

    def drain():
        events = list(queue)
        queue.clear()
        return events

    observer.observe = observe
    human.drain_events = drain
    human.handle_event(click_event(before, 1, 0, button=2, event_id=2))
    after = schema.decode_board(sink._transitions[-1]["observation_after"], width=4, height=2)
    assert np.array_equal(after, flagged), "later chord was folded into FLAG"


def test_run_batch_preserves_each_action_and_its_historical_status(tmp_path):
    human, observer, _sink, store = make_observer_recorder(tmp_path)
    play_open(human, observer)
    before = human._stable_board.copy()
    flagged = before.copy()
    flagged[0, 1] = FLAGGED
    opened = flagged.copy()
    opened[1, 0] = 1
    batch = [
        click_event(before, 1, 0, button=2, event_id=2, face_classes=FACE_RUNNING),
        click_event(flagged, 0, 0, event_id=3, face_classes=FACE_RUNNING),
        click_event(opened, 2, 0, event_id=4, face_classes=FACE_RUNNING),
    ]
    observer.board = opened
    observer.status = GameStatus.LOST
    stop = threading.Event()
    empty_reads = 0

    def evaluate(expression, *args):
        nonlocal empty_reads
        if "addEventListener" in expression:
            return []
        if batch:
            result = list(batch)
            batch.clear()
            return result
        empty_reads += 1
        if empty_reads > 30:
            stop.set()  # finite guard if the recorder loses the terminal action
        return []

    observer.page.evaluate = evaluate
    human.run(max_games=1, stop_event=stop)
    game = store.load_game("game_000001")
    assert game["result"] == "lost"
    assert [t["action"]["type"] for t in game["transitions"]] == ["open", "flag", "chord", "open"]
    assert game["transitions"][1]["newly_revealed"] == 0
    assert all(t["game_status"] == "running" for t in game["transitions"][:-1])
    assert validate_dataset(store.root).ok


def test_unknown_status_cannot_abandon_pending_action_on_idle_reads(tmp_path):
    human, observer, sink, store = make_observer_recorder(tmp_path)
    play_open(human, observer)
    before = human._stable_board.copy()
    after = before.copy()
    after[0, 1] = FLAGGED
    human._defer_action(click_event(before, 1, 0, button=2), 1, 0, "flag", before, "DOM")
    observer.get_game_status = lambda: (_ for _ in ()).throw(RuntimeError("face detached"))
    for _ in range(4):
        human._handle_idle_board(after)
    assert sink.active_game_id is not None
    assert store.game_count == 0
    observer.get_game_status = lambda: GameStatus.RUNNING
    human._handle_idle_board(after)
    assert sink.steps_recorded == 2


@pytest.mark.parametrize("terminal", [GameStatus.WON, GameStatus.LOST])
def test_many_moves_and_days_of_thinking_with_transient_failures(tmp_path, monkeypatch, terminal):
    human, observer, sink, store = make_observer_recorder(tmp_path)
    clock = [1_800_000_000.0]

    def sleep(seconds):
        clock[0] += max(seconds, 0.01)

    monkeypatch.setattr(recorder_module, "time", SimpleNamespace(
        time=lambda: clock[0], monotonic=lambda: clock[0], sleep=sleep,
    ))
    play_open(human, observer)
    started = clock[0]
    for index in range(240):
        clock[0] += 3600  # ten days total; one hour between moves
        human._handle_idle_board(human._stable_board.copy())
        assert sink.active_game_id is not None and store.game_count == 0
        before = human._stable_board.copy()
        after = before.copy()
        after[0, 1] = -1 if before[0, 1] == FLAGGED else FLAGGED
        event = click_event(before, 1, 0, button=2, event_id=index + 2, face_classes=FACE_RUNNING)
        observer.board = after
        original = observer.observe
        if index % 40 == 0:
            observer.observe = lambda: (_ for _ in ()).throw(RuntimeError("DOM replaced"))
            human.handle_event(event)  # action observation budget expires
            assert human._pending_action is not None
            observer.observe = original
            original_status = observer.get_game_status
            observer.get_game_status = lambda: (_ for _ in ()).throw(RuntimeError("face missing"))
            for _ in range(3):
                human._handle_idle_board(after)
            observer.get_game_status = original_status
            human._handle_idle_board(after)
        else:
            assert human.handle_event(event) == "recorded"
        assert sink.steps_recorded == index + 2
        assert store.game_count == 0
    before = human._stable_board.copy()
    if terminal is GameStatus.WON:
        observer.board[0, 2] = 1
    observer.status = terminal
    assert human.handle_event(click_event(before, 2, 0, event_id=999, face_classes=FACE_RUNNING)) == "game_end"
    game = store.load_game("game_000001")
    assert game["result"] == terminal.value
    assert game["total_steps"] == 242
    assert game["duration_s"] >= clock[0] - started
    assert validate_dataset(store.root).ok


@pytest.mark.parametrize("stage", ["game", "metadata_before", "metadata_after", "manifest"])
def test_partial_finalization_retries_once_without_changing_terminal_fact(tmp_path, monkeypatch, stage):
    human, observer, sink, store = make_observer_recorder(tmp_path)
    stop = threading.Event()
    play_open(human, observer)
    before = human._stable_board.copy()
    observer.status = GameStatus.LOST
    failed = [False]
    real_write = store.write_game
    real_append = dataset_module.append_jsonl
    real_save = store._save_manifest

    def fail_once():
        if not failed[0]:
            failed[0] = True
            stop.set()  # cancellation races with a terminal save
            raise PermissionError("OneDrive sharing violation")

    if stage == "game":
        def write(game):
            fail_once()
            return real_write(game)
        monkeypatch.setattr(store, "write_game", write)
    elif stage.startswith("metadata"):
        def append(path, records):
            if stage == "metadata_before":
                fail_once()
            real_append(path, records)
            fail_once()  # simulate an fsync error after the row reached disk
        monkeypatch.setattr(dataset_module, "append_jsonl", append)
    else:
        def save():
            fail_once()
            return real_save()
        monkeypatch.setattr(store, "_save_manifest", save)

    assert human.handle_event(click_event(before, 1, 0, face_classes=FACE_RUNNING)) == "game_end"
    assert human._pending_finalization is not None
    assert sink.game_terminated
    ended = human._pending_finalization[1]
    human.run(max_games=1, stop_event=stop)
    assert human.report.losses == 1 and human.report.abandoned == 0
    assert human._retry_finalization()  # repeated finalizer is idempotent
    game = store.load_game("game_000001")
    assert game["result"] == "lost" and game["ended_at"] == ended
    assert len(list(store.iter_games())) == 1
    assert store.transition_count == 2
    assert validate_dataset(store.root).ok


def test_pending_unchanged_mine_click_recovers_loss(tmp_path):
    human, observer, _sink, store = make_observer_recorder(tmp_path)
    play_open(human, observer)
    before = human._stable_board.copy()
    human._defer_action(click_event(before, 1, 0), 1, 0, "open", before, "DOM timeout")
    observer.status = GameStatus.LOST
    human._handle_idle_board(before)
    assert store.load_game("game_000001")["result"] == "lost"


def test_capture_gap_suspends_until_real_boundary_and_is_not_abandonment(tmp_path):
    human, observer, sink, store = make_observer_recorder(tmp_path)
    play_open(human, observer)
    observed = human._stable_board.copy()
    observed[1, 0] = 2
    for _ in range(4):
        human._handle_idle_board(observed)
    assert sink.active_game_id is not None and store.game_count == 0
    assert human._capture_error == "unattributed_board_change"
    observer.status = GameStatus.LOST
    human._handle_idle_board(observed)
    game = store.load_game("game_000001")
    assert game["result"] == "incomplete"
    assert game["termination_reason"] == "unattributed_board_change"
    assert human.report.abandoned == 0


@pytest.mark.parametrize("signal,reason", [("close", "context_closed"), ("disconnected", "browser_disconnected")])
def test_browser_lifecycle_signals_terminate_explicitly(tmp_path, signal, reason):
    human, observer, _sink, store = make_observer_recorder(tmp_path)
    play_open(human, observer)
    callbacks = {}
    browser = SimpleNamespace(on=lambda event, callback: callbacks.update({event: callback}))
    observer.page.context = SimpleNamespace(browser=browser, on=browser.on)
    human.install_capture()
    callbacks[signal]()
    human.run(max_games=1)
    assert store.load_game("game_000001")["termination_reason"] == reason


def test_input_during_idle_status_read_is_not_a_capture_gap(tmp_path):
    human, observer, sink, store = make_observer_recorder(tmp_path)
    play_open(human, observer)
    before = human._stable_board.copy()
    after = before.copy()
    after[0, 1] = FLAGGED
    queue = []

    def status():
        queue.append(click_event(before, 1, 0, button=2, face_classes=FACE_RUNNING))
        return GameStatus.RUNNING

    def drain():
        result = list(queue)
        queue.clear()
        return result

    human.drain_events = drain
    observer.get_game_status = status
    human._handle_idle_board(after)
    assert human._deferred_events
    assert sink.active_game_id is not None and store.game_count == 0


def test_programming_errors_are_not_swallowed_as_transient_dom(tmp_path):
    human, observer, _sink, _store = make_observer_recorder(tmp_path)
    observer.observe = lambda: (_ for _ in ()).throw(AssertionError("programming bug"))
    with pytest.raises(AssertionError, match="programming bug"):
        human._safe_observe()


def test_pending_press_recovers_when_next_press_was_captured_without_face(tmp_path):
    human, observer, sink, store = make_observer_recorder(tmp_path)
    play_open(human, observer)
    before = human._stable_board.copy()
    flagged = before.copy()
    flagged[0, 1] = FLAGGED
    human._defer_action(click_event(before, 1, 0, button=2), 1, 0, "flag", before, "face missing")
    # B's face snapshot remains empty even after the live DOM has recovered.
    event_b = click_event(flagged, 0, 0, event_id=3)
    original_status = observer.get_game_status
    observer.get_game_status = lambda: (_ for _ in ()).throw(RuntimeError("face missing"))
    assert human.handle_event(event_b) == "pending"
    assert human._pending_action is not None
    observer.get_game_status = original_status
    observer.board = flagged.copy()
    observer.board[1, 0] = 1
    human._process_events(human.drain_events(), 1)
    assert human._pending_action is None
    assert sink.steps_recorded == 3
    assert store.game_count == 0


def test_unrecoverable_historical_face_does_not_loop_after_actual_game_end(tmp_path):
    human, observer, _sink, store = make_observer_recorder(tmp_path)
    play_open(human, observer)
    before = human._stable_board.copy()
    flagged = before.copy()
    flagged[0, 1] = FLAGGED
    human._defer_action(click_event(before, 1, 0, button=2), 1, 0, "flag", before, "missing face")
    observer.board = flagged
    observer.status = GameStatus.LOST
    assert human.handle_event(click_event(flagged, 2, 0)) == "ignored_terminal"
    assert store.load_game("game_000001")["result"] == "incomplete"
    assert human.report.abandoned == 0
