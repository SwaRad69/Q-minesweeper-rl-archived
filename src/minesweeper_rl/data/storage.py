"""Append-only JSONL storage for canonical trajectory datasets.

Layout of one dataset directory::

    <dataset>/
        meta.json          # identity + counters, replaced atomically
        games.jsonl        # one GameRecord per line, written at game commit
        transitions.jsonl  # TransitionRecord lines, appended per game

A game is committed by appending its transitions first and its game record
last, so the game record acts as a commit marker.  Writes are flushed when the
append call returns; a crash can therefore only truncate the final line, which
the streaming readers detect and skip.  Previously committed games are never
rewritten.  Nothing here holds the whole dataset in memory.

Windows / OneDrive reliability
-----------------------------
Metadata is replaced atomically (temp file in the same directory, then
``os.replace``).  On Windows, file-system filter drivers -- OneDrive's sync
engine, antivirus scanners and the search indexer -- can briefly hold a handle
on a file while it is uploaded, scanned or indexed.  A rename or append during
that window fails with a sharing/access-denied error even though the file is
healthy.  Persistence therefore retries a bounded number of times with a short
increasing delay, always cleans up the temporary file, and never overwrites the
last valid ``meta.json`` if the replacement cannot be completed.  After the
retries are exhausted a clear :class:`StorageError` is raised instead of
pretending the update was saved.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import (
    RESULT_LOST,
    RESULT_WON,
    SCHEMA_VERSION,
    SOURCES,
    DataError,
    GameRecord,
    TransitionRecord,
    validate_transition,
)

logger = logging.getLogger(__name__)

GAMES_FILENAME = "games.jsonl"
TRANSITIONS_FILENAME = "transitions.jsonl"
META_FILENAME = "meta.json"
_TAIL_BYTES = 65536

#: Windows errors a filter driver (OneDrive, antivirus, indexer) returns when it
#: holds a handle on a healthy file: access denied, sharing violation and lock
#: violation.  These are transient and worth retrying.
_TRANSIENT_LOCK_WINERRORS = frozenset({5, 32, 33})
_STORAGE_ATTEMPTS = 6
_STORAGE_INITIAL_DELAY = 0.05
_STORAGE_MAX_DELAY = 0.5


class StorageError(RuntimeError):
    """Raised when a dataset file cannot be persisted safely."""


def _is_transient_lock_error(error: OSError) -> bool:
    """Return whether ``error`` is a transient Windows/OneDrive file lock."""

    winerror = getattr(error, "winerror", None)
    if winerror is not None:
        return winerror in _TRANSIENT_LOCK_WINERRORS
    # Non-Windows platforms report sharing/permission problems as PermissionError.
    return isinstance(error, PermissionError)


def _backoff(attempt: int) -> None:
    """Sleep for a short, bounded, increasing delay between retries."""

    delay = min(_STORAGE_INITIAL_DELAY * (2 ** (attempt - 1)), _STORAGE_MAX_DELAY)
    time.sleep(delay)


def _persistence_error(action: str, path: Path, last_error: OSError | None) -> StorageError:
    return StorageError(
        f"could not {action} {path} after {_STORAGE_ATTEMPTS} attempts; the "
        f"previous file was left intact. Last error: {last_error}. Another "
        f"process is holding the file open - pause or close any folder syncing "
        f"or antivirus scanning (e.g. OneDrive), or move the dataset outside the "
        f"synced folder, then retry."
    )


def _append_text(path: Path, text: str) -> None:
    """Append ``text``, retrying a transient open-time file lock.

    Only the ``open`` is retried: a failure while writing the buffer could have
    written part of the line, and re-appending would duplicate data.
    """

    last_error: OSError | None = None
    for attempt in range(1, _STORAGE_ATTEMPTS + 1):
        try:
            handle = path.open("a", encoding="utf-8")
        except OSError as error:
            if not _is_transient_lock_error(error):
                raise
            last_error = error
            if attempt < _STORAGE_ATTEMPTS:
                logger.warning(
                    "could not open %s for append (attempt %d/%d): %s; retrying",
                    path, attempt, _STORAGE_ATTEMPTS, error,
                )
                _backoff(attempt)
            continue
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        return
    raise _persistence_error("append to", path, last_error) from last_error


def _replace_with_retry(temporary: Path, destination: Path) -> None:
    """Atomically move ``temporary`` onto ``destination``, retrying briefly."""

    last_error: OSError | None = None
    for attempt in range(1, _STORAGE_ATTEMPTS + 1):
        try:
            os.replace(temporary, destination)
            return
        except OSError as error:
            if not _is_transient_lock_error(error):
                raise
            last_error = error
            if attempt < _STORAGE_ATTEMPTS:
                logger.warning(
                    "atomic replace of %s failed (attempt %d/%d): %s; retrying",
                    destination, attempt, _STORAGE_ATTEMPTS, error,
                )
                _backoff(attempt)
    raise _persistence_error("replace", destination, last_error) from last_error


def append_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Append ``records`` as JSON lines and flush before returning."""

    lines = [json.dumps(dict(record), sort_keys=True) for record in records]
    if not lines:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _append_text(path, "\n".join(lines) + "\n")


def _parse_line(path: Path, number: int, text: str, *, final: bool) -> dict | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        record = json.loads(stripped)
    except json.JSONDecodeError as error:
        if final:
            logger.warning(
                "ignoring truncated trailing record (line %d) in %s", number, path
            )
            return None
        raise DataError(f"{path} line {number} is not valid JSON: {error}") from error
    if not isinstance(record, dict):
        raise DataError(f"{path} line {number} is not a JSON object")
    return record


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Stream JSON objects, tolerating a truncated final line."""

    path = Path(path)
    if not path.exists():
        return
    pending: tuple[int, str] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for number, text in enumerate(handle, start=1):
            if pending is not None:
                record = _parse_line(path, *pending, final=False)
                if record is not None:
                    yield record
            pending = (number, text)
        if pending is not None:
            record = _parse_line(path, *pending, final=True)
            if record is not None:
                yield record


def repair_jsonl_tail(path: Path) -> bool:
    """Drop a truncated trailing record and ensure the file ends in a newline.

    Returns ``True`` when the file was modified.  Only the last 64 KiB are read,
    so this stays cheap for large files.
    """

    path = Path(path)
    if not path.exists():
        return False
    size = path.stat().st_size
    if size == 0:
        return False
    with path.open("rb") as handle:
        start = max(0, size - _TAIL_BYTES)
        handle.seek(start)
        chunk = handle.read()
    newline = chunk.rfind(b"\n")
    tail = chunk[newline + 1 :]
    if not tail:
        return False
    if newline == -1 and start > 0:
        logger.warning("could not repair %s: trailing record exceeds %d bytes", path, _TAIL_BYTES)
        return False
    keep_from = start + newline + 1
    complete = False
    if tail.strip():
        try:
            json.loads(tail.decode("utf-8"))
            complete = True
        except (json.JSONDecodeError, UnicodeDecodeError):
            complete = False
    if complete:
        with path.open("ab") as handle:
            handle.write(b"\n")
    else:
        with path.open("r+b") as handle:
            handle.truncate(keep_from)
        logger.warning("repaired truncated trailing record in %s", path)
    return True


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically.

    The JSON is written and flushed to a temporary file in the same directory
    (so the final rename stays on one volume), then ``os.replace``-d over the
    destination.  If the replacement cannot be completed the previous file is
    untouched and the temporary file is always removed, so a failed write can
    never leave ``meta.json`` truncated or make the temp file the only copy.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically (public wrapper)."""

    _write_json_atomic(path, payload)


@dataclass(slots=True)
class ValidationReport:
    """Outcome of a full-dataset integrity check."""

    games: int
    transitions: int
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        status = "OK" if self.ok else f"{len(self.errors)} problem(s)"
        return f"games={self.games} transitions={self.transitions} status={status}"


class TrajectoryStore:
    """Load/create a dataset directory and append records to it."""

    def __init__(self, dataset_dir: str | Path, *, name: str | None = None) -> None:
        self.dir = Path(dataset_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.games_path = self.dir / GAMES_FILENAME
        self.transitions_path = self.dir / TRANSITIONS_FILENAME
        self.meta_path = self.dir / META_FILENAME
        if self.meta_path.exists():
            self.meta = self._load_meta()
        else:
            self._create_meta(name or self.dir.name)
        if repair_jsonl_tail(self.games_path):
            logger.info("repaired games file %s", self.games_path)
        if repair_jsonl_tail(self.transitions_path):
            logger.info("repaired transitions file %s", self.transitions_path)
        self._known_ids: set[str] | None = None
        self._games_cache: dict[str, GameRecord] | None = None

    def _create_meta(self, name: str) -> None:
        now = time.time()
        self.meta = {
            "name": name,
            "schema_version": SCHEMA_VERSION,
            "created_at": now,
            "updated_at": now,
            "game_seq": 0,
            "game_count": 0,
            "transition_count": 0,
        }
        _write_json_atomic(self.meta_path, self.meta)

    def _load_meta(self) -> dict[str, Any]:
        try:
            with self.meta_path.open("r", encoding="utf-8") as stream:
                meta = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise DataError(f"could not read dataset meta {self.meta_path}: {error}") from error
        if not isinstance(meta, dict):
            raise DataError(f"dataset meta {self.meta_path} is not an object")
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise DataError(
                f"dataset {self.dir} uses schema_version {meta.get('schema_version')!r}; "
                f"this build supports {SCHEMA_VERSION}"
            )
        meta.setdefault("game_seq", 0)
        meta.setdefault("game_count", 0)
        meta.setdefault("transition_count", 0)
        meta.setdefault("updated_at", meta.get("created_at", time.time()))
        return meta

    def _save_meta(self) -> None:
        self.meta["updated_at"] = time.time()
        _write_json_atomic(self.meta_path, self.meta)

    def configure_board(
        self, *, width: int, height: int, mines: int | None = None
    ) -> None:
        """Record the board dimensions shared by every game in this dataset."""

        board = self.meta.get("board")
        if board is None:
            self.meta["board"] = {
                "width": int(width),
                "height": int(height),
                "mines": None if mines is None else int(mines),
            }
            self._save_meta()
            return
        if int(board.get("width", width)) != int(width) or int(
            board.get("height", height)
        ) != int(height):
            raise DataError(
                f"dataset {self.dir} board is {board.get('width')}x{board.get('height')} "
                f"but the recorder is {width}x{height}"
            )

    @property
    def name(self) -> str:
        return str(self.meta.get("name", self.dir.name))

    @property
    def game_count(self) -> int:
        return int(self.meta.get("game_count", 0))

    @property
    def transition_count(self) -> int:
        return int(self.meta.get("transition_count", 0))

    def next_game_id(self, source: str) -> str:
        if source not in SOURCES:
            raise DataError(f"invalid source: {source!r}")
        self.meta["game_seq"] = int(self.meta.get("game_seq", 0)) + 1
        self._save_meta()
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        return f"{source}_{stamp}_{self.meta['game_seq']:06d}"

    def _known_game_ids(self) -> set[str]:
        if self._known_ids is None:
            self._known_ids = {game.game_id for game in self.iter_games()}
        return self._known_ids

    def append_transitions(
        self, transitions: Iterable[TransitionRecord], *, width: int, height: int
    ) -> int:
        records = list(transitions)
        if not records:
            return 0
        game_ids = {record.game_id for record in records}
        if len(game_ids) != 1:
            raise DataError("a batch of transitions must belong to exactly one game")
        previous = None
        for expected, record in enumerate(records):
            if record.step_id != expected:
                raise DataError(
                    f"transition step_id {record.step_id} out of order; expected {expected}"
                )
            validate_transition(
                record, width=width, height=height, previous_board=previous
            )
            previous = record.board_after
        append_jsonl(self.transitions_path, (record.to_dict() for record in records))
        self.meta["transition_count"] = self.transition_count + len(records)
        self._save_meta()
        return len(records)

    def append_game(self, record: GameRecord) -> None:
        record.validate()
        if record.game_id in self._known_game_ids():
            raise DataError(f"duplicate game id: {record.game_id}")
        append_jsonl(self.games_path, [record.to_dict()])
        self._known_game_ids().add(record.game_id)
        self.meta["game_count"] = self.game_count + 1
        self._save_meta()

    def iter_games(self, *, source: str | None = None) -> Iterator[GameRecord]:
        for data in iter_jsonl(self.games_path):
            game = GameRecord.from_dict(data)
            if source is None or game.source == source:
                yield game

    def iter_transitions(
        self, *, game_ids: Iterable[str] | None = None
    ) -> Iterator[TransitionRecord]:
        selected = None if game_ids is None else set(game_ids)
        for data in iter_jsonl(self.transitions_path):
            game_id = data.get("game_id")
            if selected is not None and game_id not in selected:
                continue
            yield TransitionRecord.from_dict(data, board_shape=self._board_shape(data))

    def _board_shape(self, data: Mapping[str, Any]) -> tuple[int, int]:
        game_id = data.get("game_id")
        game = self._game_lookup().get(game_id) if isinstance(game_id, str) else None
        if game is not None:
            return game.height, game.width
        return self._fallback_shape()

    def _fallback_shape(self) -> tuple[int, int]:
        shape = self.meta.get("board")
        if isinstance(shape, Mapping) and "height" in shape and "width" in shape:
            return int(shape["height"]), int(shape["width"])
        raise DataError(
            "cannot decode transitions without board dimensions; "
            "the game record or meta.json must describe the board"
        )

    def _game_lookup(self) -> dict[str, GameRecord]:
        if self._games_cache is None:
            self._games_cache = {game.game_id: game for game in self.iter_games()}
        return self._games_cache

    def load_game(self, game_id: str) -> tuple[GameRecord | None, list[TransitionRecord]]:
        game = self._game_lookup().get(game_id)
        transitions = list(self.iter_transitions(game_ids=[game_id]))
        return game, transitions

    def game_ids(self) -> set[str]:
        return {game.game_id for game in self.iter_games()}

    def refresh_counts(self) -> None:
        games = sum(1 for _ in self.iter_games())
        transitions = sum(1 for _ in self.iter_transitions())
        self.meta["game_count"] = games
        self.meta["transition_count"] = transitions
        self._save_meta()

    def validate(self) -> ValidationReport:
        errors: list[str] = []
        game_count = 0
        transition_count = 0
        dims: dict[str, tuple[int, int]] = {}
        results: dict[str, str] = {}
        declared_steps: dict[str, int] = {}
        seen: set[str] = set()
        for game in self.iter_games():
            game_count += 1
            if game.game_id in seen:
                errors.append(f"game {game.game_id}: duplicate game id")
            seen.add(game.game_id)
            try:
                game.validate()
            except DataError as error:
                errors.append(f"game {game.game_id}: {error}")
            dims[game.game_id] = (game.width, game.height)
            results[game.game_id] = game.result
            declared_steps[game.game_id] = game.total_steps

        current: str | None = None
        expected_step = 0
        previous_board = None
        steps_per_game: dict[str, int] = {}
        last_status: dict[str, str] = {}
        terminated_seen: dict[str, bool] = {}
        for record in self.iter_transitions():
            transition_count += 1
            game_id = record.game_id
            if game_id not in dims:
                errors.append(
                    f"transition {game_id}/{record.step_id}: references unknown game"
                )
                current = None
                continue
            width, height = dims[game_id]
            if game_id != current:
                current = game_id
                expected_step = 0
                previous_board = None
            if terminated_seen.get(game_id):
                errors.append(
                    f"transition {game_id}/{record.step_id}: step after a terminal transition"
                )
            if record.step_id != expected_step:
                errors.append(
                    f"transition {game_id}/{record.step_id}: expected step_id {expected_step}"
                )
                expected_step = record.step_id + 1
            else:
                expected_step += 1
            try:
                validate_transition(
                    record, width=width, height=height, previous_board=previous_board
                )
            except DataError as error:
                errors.append(f"transition {game_id}/{record.step_id}: {error}")
            terminated_seen[game_id] = record.terminated
            steps_per_game[game_id] = record.step_id + 1
            last_status[game_id] = record.game_status
            previous_board = record.board_after

        for game_id, declared in declared_steps.items():
            actual = steps_per_game.get(game_id, 0)
            if actual != declared:
                errors.append(
                    f"game {game_id}: total_steps {declared} but {actual} transitions found"
                )
            result = results[game_id]
            if result in (RESULT_WON, RESULT_LOST):
                if last_status.get(game_id) != result:
                    errors.append(
                        f"game {game_id}: result {result} but last transition "
                        f"status is {last_status.get(game_id)!r}"
                    )
            elif result == "abandoned" and terminated_seen.get(game_id):
                errors.append(
                    f"game {game_id}: abandoned game has a terminal transition"
                )
        if game_count != self.game_count:
            errors.append(
                f"meta game_count {self.game_count} but {game_count} games on disk"
            )
        if transition_count != self.transition_count:
            errors.append(
                f"meta transition_count {self.transition_count} but "
                f"{transition_count} transitions on disk"
            )
        return ValidationReport(games=game_count, transitions=transition_count, errors=errors)
