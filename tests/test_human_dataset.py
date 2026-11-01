"""Tests for the persistent human demonstration dataset (no browser).

Covers persistence and numbering, save/load, interrupted recording, transition
integrity, the flag-aware action encoding, validation, duplicate detection and
statistics.  A local fake observer drives the real ``HumanTrajectoryRecorder``
for one end-to-end recording test.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from minesweeper_rl.agents.action_space import encode_action
from minesweeper_rl.data import (
    HUMAN_GAME_FORMAT,
    HumanDatasetError,
    HumanDatasetStore,
    PersistentHumanRecorder,
    encode_board,
    validate_dataset,
)
from minesweeper_rl.data.record_human import HumanTrajectoryRecorder
from minesweeper_rl.dom import FLAGGED, HIDDEN, GameStatus

WIDTH, HEIGHT = 4, 2


def empty_board(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    return np.full((height, width), HIDDEN, dtype=np.int8)


def make_store(tmp_path) -> HumanDatasetStore:
    return HumanDatasetStore(tmp_path / "human", width=WIDTH, height=HEIGHT, mines=3,
                             difficulty="expert")


def make_recorder(store: HumanDatasetStore, **kwargs) -> PersistentHumanRecorder:
    return PersistentHumanRecorder(
        store, width=WIDTH, height=HEIGHT, mines=3, difficulty="expert", **kwargs
    )


def write_won_game(store: HumanDatasetStore):
    """Record and persist a minimal complete (won) game."""

    recorder = make_recorder(store)
    recorder.start_game()
    before = empty_board()
    after = before.copy()
    after[0, 0] = 0
    recorder.record_step(
        board_before=before,
        action=(0, 0),
        board_after=after,
        game_status=GameStatus.WON,
        terminated=True,
    )
    return recorder.end_game("won")


# ---------------------------------------------------------------------------
# Persistent numbering
# ---------------------------------------------------------------------------


def test_persistent_game_numbering_across_restarts(tmp_path) -> None:
    store = make_store(tmp_path)
    assert store.next_game_index == 1

    first = write_won_game(store)
    assert first.game_id == "game_000001"

    # A brand-new store over the same directory must continue, not restart.
    reopened = make_store(tmp_path)
    assert reopened.next_game_index == 2
    assert reopened.game_count == 1
    second = write_won_game(reopened)
    assert second.game_id == "game_000002"
    assert reopened.next_game_index == 3


def test_dataset_grows_across_sessions(tmp_path) -> None:
    # Day 1: 5 games.
    store = make_store(tmp_path)
    for _ in range(5):
        write_won_game(store)
    assert store.game_count == 5

    # Day 2: the recorder is restarted and 12 more games are added.
    store = make_store(tmp_path)
    for _ in range(12):
        write_won_game(store)

    assert store.game_count == 17
    ids = [summary.game_id for summary in store.iter_games()]
    assert ids == [f"game_{index:06d}" for index in range(1, 18)]
    assert validate_dataset(tmp_path / "human").games == 17


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------


def test_save_and_load_game_round_trip(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store)
    recorder.start_game()

    before = empty_board()
    after = before.copy()
    after[0, 0] = 1
    recorder.record_step(
        board_before=before, action=(0, 0), board_after=after,
        game_status=GameStatus.RUNNING, terminated=False,
    )
    flagged = after.copy()
    flagged[0, 1] = FLAGGED
    recorder.record_step(
        board_before=after, action=(1, 0), board_after=flagged,
        game_status=GameStatus.RUNNING, terminated=False, action_type="flag",
    )
    summary = recorder.end_game("abandoned")

    assert (store.games_dir / f"{summary.game_id}.json").exists()
    assert store.game_count == 1
    assert store.transition_count == 2

    game = store.load_game(summary.game_id)
    assert game["format"] == HUMAN_GAME_FORMAT
    assert game["width"] == WIDTH and game["height"] == HEIGHT
    assert game["outcome"] == "incomplete" and game["complete"] is False
    assert len(game["transitions"]) == 2

    # A fresh store can stream the summary and load the game.
    reopened = make_store(tmp_path)
    loaded = list(reopened.iter_games())
    assert len(loaded) == 1
    assert loaded[0].game_id == summary.game_id
    assert loaded[0].transition_count == 2
    assert reopened.load_game(summary.game_id)["game_id"] == summary.game_id


def test_transitions_record_observable_state_and_timestamp(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store)
    recorder.start_game()
    before = empty_board()
    after = before.copy()
    after[1, 2] = 3
    recorder.record_step(
        board_before=before, action=(2, 1), board_after=after,
        game_status=GameStatus.RUNNING, terminated=False,
        newly_revealed=1, opened_cells=1, reward=1.0, timestamp=123.5,
    )
    recorder.end_game("abandoned")

    game = store.load_game("game_000001")
    step = game["transitions"][0]
    assert step["timestamp"] == 123.5
    assert step["observation_before"] == encode_board(before)
    assert step["observation_after"] == encode_board(after)
    assert step["newly_revealed"] == 1
    assert step["reward"] == 1.0
    # Only observable state is stored; ground truth is absent everywhere.
    assert "ground_truth" not in json.dumps(game).lower()


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------


def test_action_encoding_matches_flag_aware_action_space(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store)
    recorder.start_game()

    hidden = empty_board()
    opened = hidden.copy()
    opened[0, 0] = 1
    recorder.record_step(
        board_before=hidden, action=(0, 0), board_after=opened,
        game_status=GameStatus.RUNNING, terminated=False,
    )
    flagged = opened.copy()
    flagged[0, 1] = FLAGGED
    recorder.record_step(
        board_before=opened, action=(1, 0), board_after=flagged,
        game_status=GameStatus.RUNNING, terminated=False, action_type="flag",
    )
    chorded = flagged.copy()
    chorded[0, 1] = 0
    chorded[1, 1] = 0
    recorder.record_step(
        board_before=flagged, action=(0, 0), board_after=chorded,
        game_status=GameStatus.RUNNING, terminated=False, action_type="open",
    )
    recorder.end_game("abandoned")

    game = store.load_game("game_000001")
    actions = [step["action"] for step in game["transitions"]]
    assert [a["action_type"] for a in actions] == [0, 1, 2]
    assert [a["type"] for a in actions] == ["open", "flag", "chord"]

    expected = [
        encode_action(0, action_type=0, width=WIDTH, height=HEIGHT),
        encode_action(1, action_type=1, width=WIDTH, height=HEIGHT),
        encode_action(0, action_type=2, width=WIDTH, height=HEIGHT),
    ]
    assert [a["action_id"] for a in actions] == expected
    for action in actions:
        assert action["action_id"] == action["action_type"] * (WIDTH * HEIGHT) + (
            action["y"] * WIDTH + action["x"]
        )
    # Sanity: FLAG and CHORD are not in the OPEN block.
    assert actions[1]["action_id"] >= WIDTH * HEIGHT
    assert actions[2]["action_id"] >= 2 * WIDTH * HEIGHT


# ---------------------------------------------------------------------------
# Interrupted / incomplete games
# ---------------------------------------------------------------------------


def test_interrupted_game_is_marked_incomplete(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store)
    recorder.start_game()
    before = empty_board()
    after = before.copy()
    after[0, 0] = 0
    recorder.record_step(
        board_before=before, action=(0, 0), board_after=after,
        game_status=GameStatus.RUNNING, terminated=False,
    )
    summary = recorder.end_game("abandoned")

    assert summary.complete is False
    assert summary.outcome == "incomplete"
    game = store.load_game(summary.game_id)
    assert game["complete"] is False and game["outcome"] == "incomplete"
    assert validate_dataset(tmp_path / "human").incomplete_games == 1


def test_drop_incomplete_does_not_write(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store, store_incomplete=False)
    recorder.start_game()
    before = empty_board()
    after = before.copy()
    after[0, 0] = 0
    recorder.record_step(
        board_before=before, action=(0, 0), board_after=after,
        game_status=GameStatus.RUNNING, terminated=False,
    )
    recorder.end_game("abandoned")

    assert store.game_count == 0
    assert list((tmp_path / "human" / "games").glob("*.json")) == []


def test_interrupted_recording_with_no_steps_is_discarded(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store)
    recorder.start_game()
    recorder.discard_game()
    assert store.game_count == 0


# ---------------------------------------------------------------------------
# Transition integrity
# ---------------------------------------------------------------------------


def test_record_step_rejects_board_discontinuity(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store)
    recorder.start_game()
    first = empty_board()
    after = first.copy()
    after[0, 0] = 0
    recorder.record_step(
        board_before=first, action=(0, 0), board_after=after,
        game_status=GameStatus.RUNNING, terminated=False,
    )
    unrelated = empty_board()
    unrelated[1, 1] = 1
    with pytest.raises(HumanDatasetError, match="does not continue"):
        recorder.record_step(
            board_before=unrelated, action=(1, 1), board_after=unrelated,
            game_status=GameStatus.RUNNING, terminated=False,
        )


def test_record_step_rejects_impossible_effects(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store)

    recorder.start_game()
    opened = empty_board()
    opened[0, 0] = 2
    with pytest.raises(HumanDatasetError, match="hidden"):
        recorder.record_step(
            board_before=opened, action=(0, 0), board_after=opened,
            game_status=GameStatus.RUNNING, terminated=False, action_type=0,
        )
    recorder.discard_game()

    recorder.start_game()
    with pytest.raises(HumanDatasetError, match="covered"):
        recorder.record_step(
            board_before=opened, action=(0, 0), board_after=opened,
            game_status=GameStatus.RUNNING, terminated=False, action_type="flag",
        )
    recorder.discard_game()

    recorder.start_game()
    hidden = empty_board()
    with pytest.raises(HumanDatasetError, match="numbered"):
        recorder.record_step(
            board_before=hidden, action=(0, 0), board_after=hidden,
            game_status=GameStatus.RUNNING, terminated=False, action_type="chord",
        )
    recorder.discard_game()


def test_end_game_requires_terminal_step_for_win(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store)
    recorder.start_game()
    before = empty_board()
    after = before.copy()
    after[0, 0] = 0
    recorder.record_step(
        board_before=before, action=(0, 0), board_after=after,
        game_status=GameStatus.RUNNING, terminated=False,
    )
    with pytest.raises(HumanDatasetError, match="terminal"):
        recorder.end_game("won")


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def test_write_game_rejects_duplicate_id(tmp_path) -> None:
    store = make_store(tmp_path)
    summary = write_won_game(store)
    game = store.load_game(summary.game_id)
    with pytest.raises(HumanDatasetError, match="duplicate"):
        store.write_game(game)


def test_validation_detects_duplicate_metadata(tmp_path) -> None:
    store = make_store(tmp_path)
    write_won_game(store)
    with (tmp_path / "human" / "metadata.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"game_id": "game_000001"}) + "\n")

    report = validate_dataset(tmp_path / "human")
    assert not report.ok
    assert any("duplicate metadata entry" in error for error in report.errors)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_reports_statistics(tmp_path) -> None:
    store = make_store(tmp_path)
    write_won_game(store)
    write_won_game(store)

    # One lost game with two transitions.
    recorder = make_recorder(store)
    recorder.start_game()
    before = empty_board()
    opened = before.copy()
    opened[0, 0] = 1
    recorder.record_step(
        board_before=before, action=(0, 0), board_after=opened,
        game_status=GameStatus.RUNNING, terminated=False,
    )
    detonated = opened.copy()
    recorder.record_step(
        board_before=opened, action=(1, 0), board_after=detonated,
        game_status=GameStatus.LOST, terminated=True,
    )
    recorder.end_game("lost")

    # One incomplete game.
    recorder = make_recorder(store)
    recorder.start_game()
    recorder.record_step(
        board_before=before, action=(0, 0), board_after=opened,
        game_status=GameStatus.RUNNING, terminated=False,
    )
    recorder.end_game("abandoned")

    report = validate_dataset(tmp_path / "human")
    assert report.ok, report.errors
    assert report.games == 4
    assert report.winning_games == 2
    assert report.lost_games == 1
    assert report.incomplete_games == 1
    assert report.transitions == 5
    assert report.open_count >= 3
    assert report.average_game_length == pytest.approx(5 / 4)


def record_incomplete(store: HumanDatasetStore, *, x: int = 0, y: int = 0) -> str:
    """Persist a valid one-step incomplete game and return its file path."""

    recorder = make_recorder(store)
    recorder.start_game()
    before = empty_board()
    after = before.copy()
    after[y, x] = 0
    recorder.record_step(
        board_before=before, action=(x, y), board_after=after,
        game_status=GameStatus.RUNNING, terminated=False,
    )
    return recorder.end_game("abandoned").game_id


def mutate_game(store: HumanDatasetStore, game_id: str, **changes) -> None:
    """Overwrite fields of a stored game file for negative testing.

    Supported keys: ``action_id``, ``x``, ``action_type``,
    ``observation_before`` and ``total_steps``.
    """

    path = store.games_dir / f"{game_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    action = data["transitions"][0]["action"]
    for key in ("action_id", "x", "action_type"):
        if key in changes:
            action[key] = changes[key]
    if "observation_before" in changes:
        data["transitions"][0]["observation_before"] = changes["observation_before"]
    if "total_steps" in changes:
        data["total_steps"] = changes["total_steps"]
    path.write_text(json.dumps(data), encoding="utf-8")


def test_validate_reports_corruption_and_integrity_problems(tmp_path) -> None:
    store = make_store(tmp_path)

    # Corrupted file with a game-looking name.
    (store.games_dir / "game_000099.json").write_text(
        "{not valid json", encoding="utf-8"
    )

    mutate_game(store, record_incomplete(store), action_id=999999)
    mutate_game(store, record_incomplete(store), x=99)
    mutate_game(store, record_incomplete(store), action_type=7)
    mutate_game(store, record_incomplete(store), observation_before="AAAA")
    mutate_game(store, record_incomplete(store), total_steps=5)

    report = validate_dataset(tmp_path / "human")
    assert not report.ok
    joined = "\n".join(report.errors)
    assert "corrupted" in joined
    assert "action_id" in joined
    assert "outside board" in joined
    assert "impossible action type" in joined
    assert "board payload" in joined
    assert "total_steps" in joined


def test_validate_empty_dataset_reports_no_games(tmp_path) -> None:
    make_store(tmp_path)
    report = validate_dataset(tmp_path / "human")
    assert report.ok
    assert report.games == 0
    assert report.transitions == 0
    assert report.average_game_length == 0.0


# ---------------------------------------------------------------------------
# Integration with the existing human observer
# ---------------------------------------------------------------------------


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


def click_event(board: np.ndarray, x: int, y: int, *, button: int = 0,
                event_id: int | None = None) -> dict:
    return {
        "id": event_id, "token": "t", "type": "pointerdown", "isTrusted": True,
        "button": button, "x": str(x), "y": str(y), "clientX": 1, "clientY": 2,
        "tag": "DIV", "targetClasses": ["cell"], "cells": snapshot(board),
    }


class FakePage:
    def add_init_script(self, script: str) -> None:  # noqa: ARG002
        return None

    def evaluate(self, expression: str, *args: object):
        if "return events.slice()" in expression:
            return []
        return None


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


def test_observer_records_open_flag_chord_and_completes(tmp_path) -> None:
    store = make_store(tmp_path)
    sink = make_recorder(store)
    observer = FakeObserver()
    human = HumanTrajectoryRecorder(
        observer, sink, poll_interval=0.0, action_timeout=0.5,
        quiescence_time=0.0, emit=lambda _: None,
    )
    human._stable_board = observer.board.copy()

    def apply(event: dict) -> str:
        return human.handle_event(event)

    before = empty_board()
    opened = before.copy()
    opened[0, 0] = 1
    observer.board, observer.status = opened, GameStatus.RUNNING
    assert apply(click_event(before, 0, 0, event_id=1)) == "recorded"

    flagged = opened.copy()
    flagged[0, 1] = FLAGGED
    observer.board, observer.status = flagged, GameStatus.RUNNING
    assert apply(click_event(opened, 1, 0, button=2, event_id=2)) == "recorded"

    chorded = flagged.copy()
    chorded[1, 1] = 0
    observer.board, observer.status = chorded, GameStatus.RUNNING
    assert apply(click_event(flagged, 0, 0, event_id=3)) == "recorded"

    final = chorded.copy()
    final[1, 0] = 0
    observer.board = final
    observer.status = GameStatus.WON
    observer.statuses = [GameStatus.RUNNING, GameStatus.WON]
    assert apply(click_event(chorded, 0, 1, event_id=4)) == "game_end"

    game = store.load_game("game_000001")
    assert game["outcome"] == "won" and game["complete"] is True
    assert [s["action"]["type"] for s in game["transitions"]] == [
        "open", "flag", "chord", "open",
    ]
    assert validate_dataset(tmp_path / "human").ok
