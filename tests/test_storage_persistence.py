"""Reliability tests for the trajectory dataset persistence layer.

These cover the Windows/OneDrive failure mode where a filter driver briefly
holds a handle on ``meta.json`` and ``os.replace`` raises a transient
``PermissionError``.  They also assert that the atomic-write contract is kept:
the previous valid metadata survives a failed replacement, temporary files are
always cleaned up and a permanent lock fails with a clear error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from minesweeper_rl.data import (
    StorageError,
    TrajectoryRecorder,
    TrajectoryStore,
)
from minesweeper_rl.data import storage

WIDTH, HEIGHT = 4, 2


def empty_board() -> np.ndarray:
    return np.full((HEIGHT, WIDTH), -1, dtype=np.int8)


def dataset_dir(tmp_path) -> Path:
    return tmp_path / "dataset"


def make_store(tmp_path) -> TrajectoryStore:
    return TrajectoryStore(dataset_dir(tmp_path))


def read_meta(tmp_path) -> dict:
    return json.loads((dataset_dir(tmp_path) / "meta.json").read_text(encoding="utf-8"))


def temporary_files(tmp_path) -> list[Path]:
    return list(dataset_dir(tmp_path).glob("*.tmp"))


def play_game(store: TrajectoryStore):
    recorder = TrajectoryRecorder(
        store, source="human", width=WIDTH, height=HEIGHT, mines=3
    )
    recorder.start_game(started_at=1.0)
    before = empty_board()
    after = before.copy()
    after[0, 0] = 0
    recorder.record_step(
        board_before=before, action=(0, 0), board_after=after,
        game_status="won", terminated=True, reward=100.0,
        opened_cells=1, newly_revealed=1, timestamp=1.5,
    )
    return recorder.end_game("won", ended_at=2.0)


def deny_replace(*_args, **_kwargs):
    raise PermissionError(13, "Access is denied")


def no_sleep(monkeypatch) -> None:
    monkeypatch.setattr(storage.time, "sleep", lambda _seconds: None)


# ---------------------------------------------------------------------------
# 1 + 2 - normal atomic write and successful replacement
# ---------------------------------------------------------------------------


def test_atomic_meta_write_creates_valid_json_and_no_temp(tmp_path) -> None:
    store = make_store(tmp_path)
    assert (dataset_dir(tmp_path) / "meta.json").exists()
    meta = read_meta(tmp_path)
    assert meta["schema_version"] == 1
    assert meta["game_seq"] == 0
    assert temporary_files(tmp_path) == []
    assert store.validate().ok


def test_successful_replacement_updates_destination_once(tmp_path, monkeypatch) -> None:
    store = make_store(tmp_path)
    real_replace = storage.os.replace
    calls: list[tuple[Path, Path]] = []

    def recording_replace(source, destination):
        calls.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", recording_replace)
    store.next_game_id("human")

    assert len(calls) == 1
    source, destination = calls[0]
    assert source.suffix == ".tmp"
    assert destination == dataset_dir(tmp_path) / "meta.json"
    assert read_meta(tmp_path)["game_seq"] == 1
    assert temporary_files(tmp_path) == []


# ---------------------------------------------------------------------------
# 3 - transient PermissionError followed by successful replacement
# ---------------------------------------------------------------------------


def test_transient_permission_error_is_retried(tmp_path, monkeypatch) -> None:
    store = make_store(tmp_path)
    real_replace = storage.os.replace
    attempts = {"count": 0}

    def flaky_replace(source, destination):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError(13, "Access is denied")
        return real_replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", flaky_replace)
    no_sleep(monkeypatch)

    game_id = store.next_game_id("human")

    assert attempts["count"] == 3
    assert game_id.endswith("000001")
    assert store.meta["game_seq"] == 1
    assert read_meta(tmp_path)["game_seq"] == 1
    assert temporary_files(tmp_path) == []


def test_backoff_delays_increase_and_are_bounded(tmp_path, monkeypatch) -> None:
    make_store(tmp_path)
    delays: list[float] = []
    monkeypatch.setattr(storage.time, "sleep", delays.append)

    storage._backoff(1)
    storage._backoff(2)
    storage._backoff(3)

    assert delays == [pytest.approx(0.05), pytest.approx(0.1), pytest.approx(0.2)]
    assert max(delays) <= storage._STORAGE_MAX_DELAY
    assert storage._STORAGE_ATTEMPTS >= 2


# ---------------------------------------------------------------------------
# 4 + 6 - permanent PermissionError preserves metadata and fails clearly
# ---------------------------------------------------------------------------


def test_permanent_permission_error_raises_clear_error(tmp_path, monkeypatch) -> None:
    store = make_store(tmp_path)
    monkeypatch.setattr(storage.os, "replace", deny_replace)
    no_sleep(monkeypatch)

    with pytest.raises(StorageError) as info:
        store.next_game_id("human")

    message = str(info.value)
    assert "could not replace" in message
    assert "meta.json" in message
    assert "OneDrive" in message
    assert str(storage._STORAGE_ATTEMPTS) in message


def test_existing_meta_remains_valid_after_failed_replacement(
    tmp_path, monkeypatch
) -> None:
    store = make_store(tmp_path)
    store.next_game_id("human")
    before = (dataset_dir(tmp_path) / "meta.json").read_bytes()

    monkeypatch.setattr(storage.os, "replace", deny_replace)
    no_sleep(monkeypatch)
    with pytest.raises(StorageError):
        store.next_game_id("human")

    after = (dataset_dir(tmp_path) / "meta.json").read_bytes()
    assert after == before
    meta = json.loads(after.decode("utf-8"))  # still valid JSON
    assert meta["game_seq"] == 1


# ---------------------------------------------------------------------------
# 5 - temporary file cleanup
# ---------------------------------------------------------------------------


def test_temporary_file_is_cleaned_up_after_failed_replacement(
    tmp_path, monkeypatch
) -> None:
    store = make_store(tmp_path)
    monkeypatch.setattr(storage.os, "replace", deny_replace)
    no_sleep(monkeypatch)

    with pytest.raises(StorageError):
        store.next_game_id("human")

    assert temporary_files(tmp_path) == []


def test_temporary_file_is_cleaned_up_on_json_serialisation_error(
    tmp_path, monkeypatch
) -> None:
    store = make_store(tmp_path)

    def explode(*_args, **_kwargs):
        raise TypeError("not JSON serialisable")

    monkeypatch.setattr(storage.json, "dump", explode)
    with pytest.raises(TypeError):
        store.next_game_id("human")

    assert temporary_files(tmp_path) == []


# ---------------------------------------------------------------------------
# permanent, non-lock errors are not hidden by retries
# ---------------------------------------------------------------------------


def test_non_transient_error_is_not_retried(tmp_path, monkeypatch) -> None:
    store = make_store(tmp_path)
    attempts = {"count": 0}

    def missing(source, destination):  # pragma: no cover - body raises
        attempts["count"] += 1
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(storage.os, "replace", missing)
    no_sleep(monkeypatch)

    with pytest.raises(FileNotFoundError):
        store.next_game_id("human")

    assert attempts["count"] == 1, "a permanent error must fail immediately"


# ---------------------------------------------------------------------------
# 7 - repeated metadata updates
# ---------------------------------------------------------------------------


def test_repeated_metadata_updates_stay_consistent(tmp_path) -> None:
    store = make_store(tmp_path)
    for expected in range(1, 21):
        store.next_game_id("human")
        assert store.meta["game_seq"] == expected
        assert read_meta(tmp_path)["game_seq"] == expected
    assert temporary_files(tmp_path) == []


# ---------------------------------------------------------------------------
# 8 - game recording followed by metadata update
# ---------------------------------------------------------------------------


def test_game_recording_followed_by_metadata_update(tmp_path) -> None:
    store = make_store(tmp_path)
    game = play_game(store)

    assert store.game_count == 1
    assert store.transition_count == 1
    meta = read_meta(tmp_path)
    assert meta["game_count"] == 1
    assert meta["transition_count"] == 1

    reloaded, transitions = store.load_game(game.game_id)
    assert reloaded is not None and reloaded.game_id == game.game_id
    assert len(transitions) == 1
    assert store.validate().ok
    assert temporary_files(tmp_path) == []


def test_append_retries_transient_open_lock(tmp_path, monkeypatch) -> None:
    target = tmp_path / "data.jsonl"
    real_open = Path.open
    attempts = {"count": 0}

    def flaky_open(self, mode="r", *args, **kwargs):
        if self == target and "a" in mode:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise PermissionError(13, "Access is denied")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky_open)
    no_sleep(monkeypatch)

    storage.append_jsonl(target, [{"a": 1}])

    assert attempts["count"] == 3
    assert json.loads(target.read_text(encoding="utf-8").strip()) == {"a": 1}


# ---------------------------------------------------------------------------
# 9 - Windows-style (nested, spaced) paths
# ---------------------------------------------------------------------------


def test_windows_style_nested_path_with_spaces(tmp_path) -> None:
    root = tmp_path / "rl minesweeper" / "datasets" / "human_expert"
    store = TrajectoryStore(root)
    game = play_game(store)

    assert (root / "meta.json").exists()
    assert (root / "games.jsonl").exists()
    assert (root / "transitions.jsonl").exists()
    assert store.validate().ok
    assert list(root.glob("*.tmp")) == []
    assert list(store.iter_games())[0].game_id == game.game_id
