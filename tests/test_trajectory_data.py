"""Tests for the canonical trajectory schema, recorder and storage.

All boards are synthetic; no browser or website is involved.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from minesweeper_rl.data import (
    ACTION_CHORD,
    ACTION_FLAG,
    ACTION_OPEN,
    ACTION_REVEAL,
    GROUND_TRUTH_KEY,
    AgentMetadata,
    DataError,
    GameRecord,
    RecordingEnvironment,
    TrajectoryRecorder,
    TrajectoryStore,
    TransitionRecord,
    decode_board,
    encode_board,
)
from minesweeper_rl.dom import FLAGGED, HIDDEN, GameStatus

WIDTH, HEIGHT = 4, 2


def empty_board() -> np.ndarray:
    return np.full((HEIGHT, WIDTH), -1, dtype=np.int8)


def make_store(tmp_path) -> TrajectoryStore:
    return TrajectoryStore(tmp_path / "dataset")


def make_recorder(store, *, source: str = "human", agent: AgentMetadata | None = None):
    return TrajectoryRecorder(
        store, source=source, width=WIDTH, height=HEIGHT, mines=3, agent=agent
    )


def play_game(recorder: TrajectoryRecorder, *, result: str = "won") -> GameRecord:
    recorder.start_game(first_move=(0, 0), started_at=1000.0)
    board0 = empty_board()
    board1 = board0.copy()
    board1[0, 0] = 1
    recorder.record_step(
        board_before=board0, action=(0, 0), board_after=board1,
        game_status="running", terminated=False, reward=1.0,
        opened_cells=1, newly_revealed=1, timestamp=1000.5,
    )
    board2 = board1.copy()
    board2[0, 1] = 0
    status = "running" if result == "abandoned" else result
    terminated = result in ("won", "lost")
    reward = {"won": 100.0, "lost": -10.0}.get(result, 1.0)
    recorder.record_step(
        board_before=board1, action=(1, 0), board_after=board2,
        game_status=status, terminated=terminated, reward=reward,
        opened_cells=2, newly_revealed=1, timestamp=1001.0,
    )
    return recorder.end_game(result, ended_at=1002.0)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_board_codec_round_trip() -> None:
    board = np.array([[HIDDEN, 0, 8, 3], [1, 2, HIDDEN, 5]], dtype=np.int8)
    assert np.array_equal(decode_board(encode_board(board), width=4, height=2), board)


def test_board_codec_rejects_leaked_or_wrong_data() -> None:
    leaked = np.array([[10, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int8)
    with pytest.raises(DataError, match="values"):
        encode_board(leaked)  # a mine marker can never enter an observation
    with pytest.raises(DataError, match="payload"):
        decode_board(encode_board(empty_board()), width=5, height=2)
    with pytest.raises(DataError):
        decode_board("not base64!!", width=4, height=2)
    with pytest.raises(DataError, match="dtype"):
        encode_board(empty_board().astype(float))


def test_game_record_validation(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store)
    game = play_game(recorder)
    game.validate()
    assert game.source == "human" and game.agent is None
    assert game.total_steps == 2 and game.result == "won"

    with pytest.raises(DataError, match="source"):
        replace(game, source="robot").validate()
    with pytest.raises(DataError, match="result"):
        replace(game, result="maybe").validate()
    with pytest.raises(DataError, match="agent metadata"):
        replace(game, agent=AgentMetadata("ppo")).validate()


def test_transition_record_validation() -> None:
    before = empty_board()
    after = before.copy()
    after[0, 0] = 1
    valid = TransitionRecord(
        schema_version=1, game_id="g", step_id=0, action_type=ACTION_REVEAL,
        x=0, y=0, action_index=0, board_before=before, board_after=after,
        game_status="running", terminated=False, reward=1.0,
    )
    assert valid.action_index == 0

    with pytest.raises(DataError, match="action_index"):
        replace(valid, action_index=99)
    # A left-click on an already-open numbered cell is a chord and must change
    # at least one cell; an unchanged OPEN on an open cell is rejected.
    with pytest.raises(DataError, match="chord"):
        TransitionRecord(
            schema_version=1, game_id="g", step_id=0, action_type=ACTION_OPEN,
            x=0, y=0, action_index=0, board_before=after, board_after=after,
            game_status="running", terminated=False,
        )
    opened_more = after.copy()
    opened_more[1, 0] = 0
    TransitionRecord(
        schema_version=1, game_id="g", step_id=0, action_type=ACTION_OPEN,
        x=0, y=0, action_index=0, board_before=after, board_after=opened_more,
        game_status="running", terminated=False,
    )  # chording is an OPEN action
    # Legacy CHORD / REVEAL records from older datasets remain readable.
    TransitionRecord(
        schema_version=1, game_id="g", step_id=0, action_type=ACTION_CHORD,
        x=0, y=0, action_index=0, board_before=after, board_after=after,
        game_status="running", terminated=False,
    )
    TransitionRecord(
        schema_version=1, game_id="g", step_id=0, action_type=ACTION_REVEAL,
        x=0, y=0, action_index=0, board_before=before, board_after=after,
        game_status="running", terminated=False,
    )
    with pytest.raises(DataError, match="post-action"):
        replace(valid, game_status="ready")
    with pytest.raises(DataError, match="terminated"):
        replace(valid, terminated=True, game_status="running")
    with pytest.raises(DataError, match="won/lost"):
        replace(valid, terminated=False, game_status="won")
    with pytest.raises(DataError, match="both"):
        replace(valid, terminated=True, truncated=True, game_status="won")


def _transition(action_type, before, after, *, status="running", terminated=False):
    return TransitionRecord(
        schema_version=1, game_id="g", step_id=0, action_type=action_type,
        x=0, y=0, action_index=0, board_before=before, board_after=after,
        game_status=status, terminated=terminated,
    )


def test_flag_transition_place_and_remove() -> None:
    hidden = empty_board()
    flagged = hidden.copy()
    flagged[0, 0] = FLAGGED

    place = _transition(ACTION_FLAG, hidden, flagged)
    remove = _transition(ACTION_FLAG, flagged, hidden)

    assert int(place.board_before[0, 0]) == HIDDEN
    assert int(place.board_after[0, 0]) == FLAGGED
    assert int(remove.board_before[0, 0]) == FLAGGED
    assert int(remove.board_after[0, 0]) == HIDDEN


def test_flag_transition_validation_rejects_bad_targets() -> None:
    hidden = empty_board()
    flagged = hidden.copy()
    flagged[0, 0] = FLAGGED
    revealed = hidden.copy()
    revealed[0, 0] = 3

    with pytest.raises(DataError, match="flag action must change"):
        _transition(ACTION_FLAG, hidden, hidden)
    with pytest.raises(DataError, match="hidden or flagged"):
        _transition(ACTION_FLAG, revealed, revealed)


def test_open_transition_rejects_flagged_and_unopened_cells() -> None:
    hidden = empty_board()
    flagged = hidden.copy()
    flagged[0, 0] = FLAGGED
    with pytest.raises(DataError, match="flagged"):
        _transition(ACTION_OPEN, flagged, hidden)
    with pytest.raises(DataError, match="must open"):
        _transition(ACTION_OPEN, hidden, hidden)


def test_board_codec_round_trips_flagged_cells() -> None:
    board = np.array([[HIDDEN, FLAGGED, 0], [3, FLAGGED, 8]], dtype=np.int8)
    assert np.array_equal(decode_board(encode_board(board), width=3, height=2), board)


def test_legacy_action_types_still_deserialize() -> None:
    before = empty_board()
    after = before.copy()
    after[0, 0] = 1
    common = {
        "schema_version": 1, "game_id": "old", "step_id": 0,
        "action": {"x": 0, "y": 0}, "action_index": 0,
        "board_before": encode_board(before), "board_after": encode_board(after),
        "game_status": "running", "terminated": False, "truncated": False,
    }
    for legacy in (ACTION_REVEAL, ACTION_CHORD):
        record = TransitionRecord.from_dict(
            {**common, "action_type": legacy}, board_shape=(HEIGHT, WIDTH)
        )
        assert record.action_type == legacy


def test_legacy_record_without_action_type_defaults_to_open() -> None:
    before = empty_board()
    after = before.copy()
    after[0, 0] = 1
    record = TransitionRecord.from_dict(
        {
            "schema_version": 1, "game_id": "old", "step_id": 0,
            "action": {"x": 0, "y": 0}, "action_index": 0,
            "board_before": encode_board(before), "board_after": encode_board(after),
            "game_status": "running", "terminated": False, "truncated": False,
        },
        board_shape=(HEIGHT, WIDTH),
    )
    assert record.action_type == ACTION_OPEN


# ---------------------------------------------------------------------------
# recorder + storage
# ---------------------------------------------------------------------------


def test_recorder_persists_and_reloads(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store)
    game = play_game(recorder)

    assert store.games_path.exists() and store.transitions_path.exists()
    games = list(store.iter_games())
    assert games == [game]

    reloaded, transitions = store.load_game(game.game_id)
    assert reloaded is not None and reloaded.game_id == game.game_id
    assert [t.step_id for t in transitions] == [0, 1]
    assert np.array_equal(transitions[0].board_before, empty_board())
    assert transitions[1].terminated and transitions[1].game_status == "won"
    assert transitions[1].reward == 100.0
    report = store.validate()
    assert report.ok, report.errors


def test_agent_source_and_metadata(tmp_path) -> None:
    store = make_store(tmp_path)
    agent = AgentMetadata(agent_type="ppo", model_version="v1", checkpoint="c.pt", seed=7)
    recorder = make_recorder(store, source="agent", agent=agent)
    game = play_game(recorder)
    assert game.source == "agent" and game.agent == agent

    reloaded = list(store.iter_games())[0]
    assert reloaded.agent == agent


def test_human_recorder_rejects_agent_metadata(tmp_path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(DataError, match="human"):
        make_recorder(store, source="human", agent=AgentMetadata("ppo"))


def test_recorder_validation_failures(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store)
    with pytest.raises(DataError, match="start_game"):
        recorder.record_step(
            board_before=empty_board(), action=(0, 0),
            board_after=empty_board(), game_status="running", terminated=False,
        )
    recorder.start_game()
    with pytest.raises(DataError, match="values"):
        recorder.record_step(
            board_before=empty_board() + 20, action=(0, 0),
            board_after=empty_board(), game_status="running", terminated=False,
        )
    with pytest.raises(DataError, match="outside"):
        recorder.record_step(
            board_before=empty_board(), action=(9, 9),
            board_after=empty_board(), game_status="running", terminated=False,
        )
    recorder.end_game("abandoned")
    with pytest.raises(DataError, match="active game"):
        recorder.record_other_click()


def test_end_game_result_must_match_terminal_transition(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store)
    game = play_game(recorder, result="won")
    assert game.result == "won"

    other = make_recorder(store)
    with pytest.raises(DataError, match="terminal"):
        other.start_game()
        other.end_game("won")

    terminal = make_recorder(store)
    terminal.start_game()
    before = empty_board()
    after = before.copy()
    after[0, 0] = 1
    terminal.record_step(
        board_before=before, action=(0, 0), board_after=after,
        game_status="won", terminated=True, reward=100.0,
    )
    with pytest.raises(DataError, match="contradicts"):
        terminal.end_game("lost")


def test_multiple_games_have_unique_ids(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store)
    ids = [play_game(recorder).game_id for _ in range(3)]
    assert len(set(ids)) == 3
    assert store.game_count == 3
    assert store.transition_count == 6
    assert store.validate().ok


def test_final_opened_cells_excludes_flags(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store)
    recorder.start_game(started_at=0.0)

    board0 = empty_board()
    board1 = board0.copy()
    board1[0, 0] = 0
    recorder.record_step(
        board_before=board0, action=(0, 0), board_after=board1,
        game_status="running", terminated=False, reward=1.0,
        action_type=ACTION_OPEN,
    )
    board2 = board1.copy()
    board2[0, 1] = FLAGGED
    recorder.record_step(
        board_before=board1, action=(1, 0), board_after=board2,
        game_status="running", terminated=False, reward=0.0,
        action_type=ACTION_FLAG,
    )
    board3 = board2.copy()
    board3[0, 2] = 0
    recorder.record_step(
        board_before=board2, action=(2, 0), board_after=board3,
        game_status="won", terminated=True, reward=100.0,
        action_type=ACTION_OPEN,
    )
    game = recorder.end_game("won", ended_at=1.0)

    assert int((board3 >= 0).sum()) == 2
    assert game.final_opened_cells == 2, "the flag must not count as opened"


def test_mixed_open_flag_trajectory_validates(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store)
    recorder.start_game(started_at=0.0)

    board0 = empty_board()
    board1 = board0.copy()
    board1[0, 0] = 0
    recorder.record_step(
        board_before=board0, action=(0, 0), board_after=board1,
        game_status="running", terminated=False, reward=1.0,
        opened_cells=1, newly_revealed=1, action_type=ACTION_OPEN,
    )

    board2 = board1.copy()
    board2[0, 1] = FLAGGED
    recorder.record_step(
        board_before=board1, action=(1, 0), board_after=board2,
        game_status="running", terminated=False, reward=0.0,
        opened_cells=1, newly_revealed=0, action_type=ACTION_FLAG,
    )  # place a flag

    recorder.record_step(
        board_before=board2, action=(1, 0), board_after=board1,
        game_status="running", terminated=False, reward=0.0,
        opened_cells=1, newly_revealed=0, action_type=ACTION_FLAG,
    )  # remove it again

    board3 = board1.copy()
    board3[0, 1] = 2
    recorder.record_step(
        board_before=board1, action=(0, 0), board_after=board3,
        game_status="running", terminated=False, reward=1.0,
        opened_cells=2, newly_revealed=1, action_type=ACTION_OPEN,
    )  # chord on the open (0, 0) cell, recorded as OPEN

    board4 = board3.copy()
    board4[0, 3] = 0
    recorder.record_step(
        board_before=board3, action=(3, 0), board_after=board4,
        game_status="won", terminated=True, reward=100.0,
        opened_cells=3, newly_revealed=1, action_type=ACTION_OPEN,
    )
    game = recorder.end_game("won", ended_at=1.0)

    transitions = list(store.iter_transitions(game_ids=[game.game_id]))
    assert [t.action_type for t in transitions] == [
        ACTION_OPEN, ACTION_FLAG, ACTION_FLAG, ACTION_OPEN, ACTION_OPEN,
    ]
    assert store.validate().ok, store.validate().errors


def test_dataset_append_across_reopen(tmp_path) -> None:
    store = make_store(tmp_path)
    play_game(make_recorder(store))
    play_game(make_recorder(store))

    reopened = TrajectoryStore(tmp_path / "dataset")
    assert reopened.game_count == 2
    play_game(make_recorder(reopened))

    assert reopened.game_count == 3
    assert [g.game_id for g in reopened.iter_games()] == sorted(
        {g.game_id for g in reopened.iter_games()}
    )
    assert reopened.validate().ok


def test_duplicate_game_id_rejected(tmp_path) -> None:
    store = make_store(tmp_path)
    game = play_game(make_recorder(store))
    with pytest.raises(DataError, match="duplicate"):
        store.append_game(game)


# ---------------------------------------------------------------------------
# crash safety + validation
# ---------------------------------------------------------------------------


def test_crash_safe_trailing_line_is_repaired(tmp_path) -> None:
    store = make_store(tmp_path)
    play_game(make_recorder(store))
    with store.transitions_path.open("ab") as handle:
        handle.write(b'{"game_id": "agent_XX')

    reopened = TrajectoryStore(tmp_path / "dataset")
    assert len(list(reopened.iter_transitions())) == 2
    assert reopened.validate().ok
    assert not reopened.transitions_path.read_bytes().endswith(b'agent_XX')


def test_mid_file_corruption_raises(tmp_path) -> None:
    store = make_store(tmp_path)
    play_game(make_recorder(store))
    lines = store.transitions_path.read_text(encoding="utf-8").splitlines()
    lines.insert(1, "{not json")
    store.transitions_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(DataError, match="not valid JSON"):
        list(store.iter_transitions())


def test_validate_detects_missing_transition(tmp_path) -> None:
    store = make_store(tmp_path)
    play_game(make_recorder(store))
    lines = store.transitions_path.read_text(encoding="utf-8").splitlines()
    store.transitions_path.write_text("\n".join(lines[:1]) + "\n", encoding="utf-8")
    report = store.validate()
    assert not report.ok
    assert any("expected step_id" in error or "total_steps" in error for error in report.errors)


def test_validate_detects_orphan_transition(tmp_path) -> None:
    store = make_store(tmp_path)
    play_game(make_recorder(store))
    line = json.loads(store.transitions_path.read_text(encoding="utf-8").splitlines()[0])
    line["game_id"] = "human_ghost"
    with store.transitions_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line) + "\n")
    report = store.validate()
    assert any("unknown game" in error for error in report.errors)


def test_ground_truth_is_never_in_observations(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store)
    recorder.start_game(started_at=0.0)
    before = empty_board()
    after = before.copy()
    after[0, 0] = 1
    recorder.record_step(
        board_before=before, action=(0, 0), board_after=after,
        game_status="won", terminated=True, reward=100.0,
    )
    game = recorder.end_game(
        "won", ground_truth={"captured": True, "capture_method": "post_game_dom",
                             "mines": [[3, 1]]},
    )
    assert game.to_dict()[GROUND_TRUTH_KEY]["mines"] == [[3, 1]]
    assert GROUND_TRUTH_KEY not in game.safe_view()
    _, transitions = store.load_game(game.game_id)
    for transition in transitions:
        assert GROUND_TRUTH_KEY not in transition.to_dict()


def test_invalid_ground_truth_rejected(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store)
    recorder.start_game()
    with pytest.raises(DataError, match="outside"):
        recorder.end_game("abandoned", ground_truth={"captured": True, "mines": [[9, 9]]})


# ---------------------------------------------------------------------------
# agent-path wrapper
# ---------------------------------------------------------------------------


class FakeEnv:
    def __init__(self, width=WIDTH, height=HEIGHT, first_move=(0, 0)) -> None:
        self.config = SimpleNamespace(width=width, height=height, first_move=first_move)
        self._board = empty_board()

    def reset(self):
        self._board = empty_board()
        x, y = self.config.first_move
        self._board[y, x] = 0
        return self._board.copy()

    def step(self, action):
        x, y = action
        if self._board[y, x] != HIDDEN:
            raise ValueError("already revealed")
        self._board = self._board.copy()
        self._board[y, x] = 0
        done = (self._board == HIDDEN).sum() == 0
        status = GameStatus.WON if done else GameStatus.RUNNING
        reward = 100.0 if done else 1.0
        info = {
            "game_status": status,
            "opened_cells": int((self._board != HIDDEN).sum()),
            "newly_revealed": 1,
        }
        return self._board.copy(), reward, done, False, info


def test_recording_environment_records_full_game(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store, source="agent", agent=AgentMetadata("test"))
    env = RecordingEnvironment(FakeEnv(), recorder)

    board = env.reset()
    done = False
    while not done:
        mask = board.reshape(-1) == HIDDEN
        action = int(np.argmax(mask))
        x, y = action % WIDTH, action // WIDTH
        board, _, done, _, _ = env.step((x, y))

    games = list(store.iter_games())
    assert len(games) == 1
    game = games[0]
    assert game.result == "won" and game.source == "agent"
    assert game.total_steps == 7  # 8 cells, one opened by reset
    assert game.first_move == (0, 0)
    assert store.validate().ok


def test_recording_environment_abandons_on_early_reset(tmp_path) -> None:
    store = make_store(tmp_path)
    recorder = make_recorder(store, source="agent", agent=AgentMetadata("test"))
    env = RecordingEnvironment(FakeEnv(), recorder)
    env.reset()
    env.step((1, 0))
    env.reset()  # abandon the first game, start a second

    games = list(store.iter_games())
    assert len(games) == 1
    assert games[0].result == "abandoned"
    assert games[0].total_steps == 1
    assert recorder.active_game_id is not None  # second game is open


def test_safe_view_and_round_trip(tmp_path) -> None:
    store = make_store(tmp_path)
    game = play_game(make_recorder(store))
    reloaded = GameRecord.from_dict(game.to_dict())
    assert reloaded == game
    assert reloaded.safe_view() == game.safe_view()
