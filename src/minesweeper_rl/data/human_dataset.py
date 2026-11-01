"""Persistent, append-only human demonstration dataset.

This module implements a durable per-game dataset of **human** Minesweeper
demonstrations that can be accumulated over many separate play sessions without
ever rewriting or losing earlier data.  It is intentionally independent from the
PPO training stack and is designed to be usable later for imitation learning or
for other RL agents.

Directory layout
----------------
::

    data/human/
        games/
            game_000001.json
            game_000002.json
            ...
        metadata.jsonl          # one lightweight summary line per game
        dataset_manifest.json   # format identity, board, counters, next id
        README.md

Each game file is immutable once written.  ``metadata.jsonl`` is appended to and
the small manifest is replaced atomically, so recording a new game never rewrites
the dataset.  Game ids are allocated from the manifest (and reconciled against
the files that actually exist) so numbering continues across restarts.

Observations and actions
------------------------
Boards use the project's flag-aware representation (``FLAGGED = -2``,
``HIDDEN = -1``, ``OPEN = 0..8``) and are stored base64-encoded ``int8`` in
row-major ``[y, x]`` order, exactly like :mod:`minesweeper_rl.data.schema`.
Actions use the flag-aware encoding::

    action_type: 0 = OPEN, 1 = FLAG, 2 = CHORD
    cell_index = y * width + x
    action_id = action_type * (width * height) + cell_index

Only information an agent could observe at decision time is stored.  Hidden mine
positions are never recorded; a post-game ground-truth argument is accepted for
signature compatibility with the existing recorder but is deliberately ignored.

Canonical and model-independent
-------------------------------
``data/human/`` is the SINGLE canonical human demonstration dataset.  The raw
game files keep only canonical observable facts; every neural representation
(CNN one-hot channels, GNN nodes/edges, quantum feature maps) is a later,
model-specific adapter.  Each manifest and game file is self-describing: it
records the frozen action identity (``action_space``, ``action_types``,
``num_actions``, ``cells``) and observation identity (``observation_version``)
so a future loader can interpret it without importing this module.

The legacy append-only JSONL dataset written by the older ``TrajectoryStore``
(``datasets/human_expert/``) is NOT canonical and is explicitly rejected by this
module: its schema folds a chord into an OPEN action, which must never be read
as this dataset's 1440-action semantics.

Derived fields
--------------
``reward``, ``newly_revealed`` and ``opened_cells`` are retained for validation
and offline experiments but are DERIVED: they can be recomputed from the
observable boards and the game status.  The ``reward_spec`` identifier versions
their definition so the raw transcript stays usable if reward shaping changes.

The recorder reuses the existing human event capture and board synchronisation
through :class:`~minesweeper_rl.data.record_human.HumanTrajectoryRecorder`: the
:class:`PersistentHumanRecorder` below is a drop-in sink for that observer, so a
chord (a left-click on an opened number) is classified from the pre-action board.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..action_types import (
    ACTION_CHORD,
    ACTION_FLAG,
    ACTION_NAMES,
    ACTION_OPEN,
    compound_action_id,
    split_compound_action,
)
from ..dom import FLAGGED, HIDDEN, MAX_NUMBER
from . import schema
from .storage import append_jsonl, iter_jsonl, repair_jsonl_tail, write_json_atomic

logger = logging.getLogger(__name__)

#: Dataset identity recorded in ``dataset_manifest.json``.
HUMAN_DATASET_FORMAT = "minesweeper_human_dataset"
#: v1 = initial per-game JSON dataset.  v2 adds the machine-readable action and
#: observation identity fields (``action_space``, ``action_types``,
#: ``num_actions``, ``cells``, ``observation_version``, ``reward_spec``).
#: ``format_version`` versions the file schema; ``recorder_version`` versions the
#: capture semantics (bumped to 2 with the game-lifecycle fix).
HUMAN_DATASET_VERSION = 2
#: Per-game file identity recorded inside each ``game_XXXXXX.json``.
HUMAN_GAME_FORMAT = "minesweeper_human_game"
HUMAN_GAME_VERSION = 2
#: Recorder version (bump only when the capture semantics change).
#: v2 adds the explicit game-lifecycle state machine, fresh-board gating and
#: semantic transition validation.  v3 fixes terminal handling for losing
#: chords (a detonated chord can leave the canonical board unchanged).  The raw
#: file schema is unchanged (``format_version`` stays 2).
HUMAN_RECORDER_VERSION = 3
#: Observation representation and action encoding identities for reproducibility.
OBSERVATION_REPRESENTATION = "flag_aware_v1"  # FLAGGED=-2, HIDDEN=-1, OPEN 0..8
#: Explicit integer version of the observation representation above.
OBSERVATION_VERSION = 1
ACTION_ENCODING = "action_type*cells+cell"
#: Frozen, machine-readable action space for the canonical human dataset.
ACTION_SPACE = "flags"
#: Canonical action-type code -> name.  Derived from the single source of truth
#: in :mod:`minesweeper_rl.action_types` (0 = OPEN, 1 = FLAG, 2 = CHORD).
ACTION_TYPES_MAPPING: dict[str, str] = {
    str(code): name for code, name in sorted(ACTION_NAMES.items())
}
#: Reward definition version for the derived reward/coverage fields.
REWARD_SPEC = "reveal_progress_terminal_v1"
#: Transition fields that are DERIVED (recomputable from raw observable facts).
DERIVED_TRANSITION_FIELDS: tuple[str, ...] = (
    "reward",
    "newly_revealed",
    "opened_cells",
)
#: Site / environment identity.
ENVIRONMENT = "minesweeper.online"
ENVIRONMENT_VERSION = "minesweeper.online/expert"

#: Keys that must never appear in a canonical game or transition: they would
#: leak the hidden mine layout into what must remain observable-only data.
FORBIDDEN_LAYOUT_KEYS: tuple[str, ...] = (
    "post_game_ground_truth",
    "ground_truth",
    "mine_positions",
    "hidden_mines",
    "mine_layout",
    "solution",
)

GAMES_DIRNAME = "games"
METADATA_FILENAME = "metadata.jsonl"
MANIFEST_FILENAME = "dataset_manifest.json"
GAME_ID_TEMPLATE = "game_{index:06d}"

#: Files written by the legacy append-only ``TrajectoryStore``.  Their presence
#: means the directory is NOT a canonical human demonstration dataset.
LEGACY_FILENAMES: tuple[str, ...] = ("games.jsonl", "transitions.jsonl")
LEGACY_META_FILENAME = "meta.json"


def legacy_dataset_reason(root: str | "Path") -> str | None:
    """Return why ``root`` is a legacy non-canonical dataset, or ``None``.

    The legacy JSONL store folds a chord into an OPEN action and uses string
    action names with no ``action_id``.  If such a directory were silently read
    as a canonical dataset every CHORD demonstration would be misinterpreted as
    an OPEN, so callers must reject it explicitly.
    """

    root = Path(root)
    if not root.exists() or not root.is_dir():
        return None
    for name in LEGACY_FILENAMES:
        if (root / name).exists():
            return (
                f"legacy append-only JSONL dataset detected ({name}); it uses "
                f"the old chord-as-open schema and is not the canonical "
                f"{HUMAN_DATASET_FORMAT} format"
            )
    meta = root / LEGACY_META_FILENAME
    if meta.exists():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if (
            isinstance(data, dict)
            and data.get("format") != HUMAN_DATASET_FORMAT
            and "schema_version" in data
        ):
            return (
                f"legacy TrajectoryStore {LEGACY_META_FILENAME} detected; it is "
                f"not the canonical {HUMAN_DATASET_FORMAT} format"
            )
    return None


def canonical_identity(width: int, height: int) -> dict[str, Any]:
    """Return the frozen, model-independent identity fields for a board size."""

    cells = int(width) * int(height)
    return {
        "observation_representation": OBSERVATION_REPRESENTATION,
        "observation_version": OBSERVATION_VERSION,
        "action_space": ACTION_SPACE,
        "action_encoding": ACTION_ENCODING,
        "action_types": dict(ACTION_TYPES_MAPPING),
        "num_actions": len(ACTION_NAMES) * cells,
        "cells": cells,
        "reward_spec": REWARD_SPEC,
        "derived_fields": list(DERIVED_TRANSITION_FIELDS),
        "environment": ENVIRONMENT,
        "environment_version": ENVIRONMENT_VERSION,
        "recorder_version": HUMAN_RECORDER_VERSION,
    }


@dataclass(frozen=True, slots=True)
class HumanGameSummary:
    """Lightweight per-game view used by the recorder's live totals."""

    game_id: str
    game_index: int
    source: str
    result: str
    outcome: str
    complete: bool
    total_steps: int
    transition_count: int
    width: int
    height: int
    mines: int | None
    difficulty: str | None
    started_at: float
    ended_at: float
    duration_s: float
    open_count: int = 0
    flag_count: int = 0
    chord_count: int = 0
    path: str | None = None
    termination_reason: str | None = None


@dataclass(slots=True)
class HumanDatasetReport:
    """Result of a full human-dataset integrity check."""

    games: int = 0
    complete_games: int = 0
    winning_games: int = 0
    lost_games: int = 0
    incomplete_games: int = 0
    transitions: int = 0
    open_count: int = 0
    flag_count: int = 0
    chord_count: int = 0
    games_valid: int = 0
    games_invalid: int = 0
    transitions_valid: int = 0
    transitions_invalid: int = 0
    invalid_games: list[str] = field(default_factory=list)
    invalid_reasons: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def average_game_length(self) -> float:
        if self.games == 0:
            return 0.0
        return self.transitions / self.games

    def format(self) -> str:
        status = "OK" if self.ok else f"{len(self.errors)} problem(s)"
        return (
            f"status={status} games={self.games} "
            f"(won={self.winning_games} lost={self.lost_games} "
            f"incomplete={self.incomplete_games}) "
            f"valid={self.games_valid} invalid={self.games_invalid} "
            f"transitions={self.transitions} "
            f"(valid={self.transitions_valid} invalid={self.transitions_invalid}) "
            f"open={self.open_count} flag={self.flag_count} "
            f"chord={self.chord_count} "
            f"avg_length={self.average_game_length:.1f}"
        )


class HumanDatasetError(schema.DataError):
    """Raised when the human dataset is malformed or incompatible."""


def _now() -> float:
    return time.time()


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HumanDatasetError(f"expected an integer, got {value!r}")
    return int(value)


def _game_index(game_id: str) -> int:
    suffix = game_id.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        raise HumanDatasetError(f"invalid game id: {game_id!r}")
    return int(suffix)


def apply_canonical_identity(game: dict[str, Any]) -> dict[str, Any]:
    """Stamp the frozen, model-independent identity onto one game dict.

    This is authoritative: it guarantees every game file is self-describing and
    cannot drift from the canonical action/observation identity, while leaving
    the observable raw facts (boards, actions, metadata) untouched.
    """

    width = _as_int(game.get("width", 0))
    height = _as_int(game.get("height", 0))
    if width < 1 or height < 1:
        raise HumanDatasetError("game must declare a positive width and height")
    fmt = game.get("format")
    if fmt not in (None, HUMAN_GAME_FORMAT):
        raise HumanDatasetError(
            f"game format {fmt!r} is not the canonical {HUMAN_GAME_FORMAT!r}"
        )
    game["format"] = HUMAN_GAME_FORMAT
    game["format_version"] = HUMAN_GAME_VERSION
    game.update(canonical_identity(width, height))
    return game


class HumanDatasetStore:
    """Own one persistent human dataset directory."""

    def __init__(
        self,
        root: str | Path,
        *,
        width: int,
        height: int,
        mines: int | None = None,
        difficulty: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.games_dir = self.root / GAMES_DIRNAME
        self.metadata_path = self.root / METADATA_FILENAME
        self.manifest_path = self.root / MANIFEST_FILENAME
        self.width = int(width)
        self.height = int(height)
        self.mines = None if mines is None else int(mines)
        self.difficulty = difficulty

        reason = legacy_dataset_reason(self.root)
        if reason is not None:
            raise HumanDatasetError(
                f"{self.root} is not a canonical human dataset: {reason}. The "
                f"canonical dataset is data/human/ with dataset_manifest.json "
                f"and games/game_XXXXXX.json."
            )

        self.games_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_or_create_manifest()
        self._reconcile()
        self._known_ids: set[str] = set(self._existing_ids())

    # -- manifest ---------------------------------------------------------

    def _load_or_create_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise HumanDatasetError(
                    f"could not read manifest {self.manifest_path}: {error}"
                ) from error
            if not isinstance(data, dict):
                raise HumanDatasetError(
                    f"manifest {self.manifest_path} is not a JSON object"
                )
            if data.get("format") != HUMAN_DATASET_FORMAT:
                raise HumanDatasetError(
                    f"manifest {self.manifest_path} has format "
                    f"{data.get('format')!r}; expected {HUMAN_DATASET_FORMAT!r}"
                )
            self._check_board(data.get("board"))
            if self._ensure_identity_fields(data):
                write_json_atomic(self.manifest_path, data)
            return data
        manifest = {
            "format": HUMAN_DATASET_FORMAT,
            "format_version": HUMAN_DATASET_VERSION,
            "created_at": _now(),
            "updated_at": _now(),
            "source": schema.SOURCE_HUMAN,
            "board": {
                "width": self.width,
                "height": self.height,
                "mines": self.mines,
                "difficulty": self.difficulty,
            },
            "game_count": 0,
            "transition_count": 0,
            "next_game_index": 1,
        }
        manifest.update(canonical_identity(self.width, self.height))
        write_json_atomic(self.manifest_path, manifest)
        return manifest

    def _ensure_identity_fields(self, manifest: dict[str, Any]) -> bool:
        """Backfill missing identity fields; return whether anything changed.

        A manifest written before the identity fields existed (``format_version``
        1) is upgraded in place to the current schema.  The observable data in
        any existing game files is untouched, and each game is self-describing.
        """

        board = manifest.get("board")
        width = self.width
        height = self.height
        if isinstance(board, Mapping):
            width = int(board.get("width", width))
            height = int(board.get("height", height))
        changed = False
        for key, value in canonical_identity(width, height).items():
            if key not in manifest:
                manifest[key] = value
                changed = True
        if not manifest.get("source"):
            manifest["source"] = schema.SOURCE_HUMAN
            changed = True
        if int(manifest.get("format_version", 0)) < HUMAN_DATASET_VERSION:
            manifest["format_version"] = HUMAN_DATASET_VERSION
            changed = True
        return changed

    def _check_board(self, board: Any) -> None:
        if board is None:
            return
        if not isinstance(board, Mapping):
            raise HumanDatasetError("manifest board must be an object")
        if int(board.get("width", self.width)) != self.width or int(
            board.get("height", self.height)
        ) != self.height:
            raise HumanDatasetError(
                f"dataset board is {board.get('width')}x{board.get('height')} "
                f"but the recorder is {self.width}x{self.height}"
            )

    def _save_manifest(self) -> None:
        self.manifest["updated_at"] = _now()
        write_json_atomic(self.manifest_path, self.manifest)

    def _reconcile(self) -> None:
        """Make the manifest consistent with the game files on disk."""

        indices = self._existing_indices()
        changed = False
        if indices:
            disk_count = len(indices)
            if int(self.manifest.get("game_count", 0)) != disk_count:
                self.manifest["game_count"] = disk_count
                changed = True
            next_index = int(self.manifest.get("next_game_index", 1))
            if next_index <= max(indices):
                self.manifest["next_game_index"] = max(indices) + 1
                changed = True
        if changed:
            self._save_manifest()

    # -- disk scans -------------------------------------------------------

    def _game_files(self) -> list[Path]:
        return sorted(self.games_dir.glob("game_*.json"))

    def _existing_indices(self) -> list[int]:
        indices: list[int] = []
        for path in self._game_files():
            suffix = path.stem.rsplit("_", 1)[-1]
            if suffix.isdigit():
                indices.append(int(suffix))
        return indices

    def _existing_ids(self) -> list[str]:
        return [path.stem for path in self._game_files()]

    # -- counters / identity ---------------------------------------------

    @property
    def game_count(self) -> int:
        return int(self.manifest.get("game_count", 0))

    @property
    def transition_count(self) -> int:
        return int(self.manifest.get("transition_count", 0))

    @property
    def next_game_index(self) -> int:
        return int(self.manifest.get("next_game_index", 1))

    def game_ids(self) -> set[str]:
        return set(self._existing_ids())

    def next_game_id(self, source: str = schema.SOURCE_HUMAN) -> str:
        """Allocate and persist the next unique, monotonic game id."""

        if source not in schema.SOURCES:
            raise HumanDatasetError(f"invalid source: {source!r}")
        index = self.next_game_index
        self.manifest["next_game_index"] = index + 1
        self._save_manifest()
        return GAME_ID_TEMPLATE.format(index=index)

    # -- writing ----------------------------------------------------------

    def write_game(self, game: Mapping[str, Any]) -> HumanGameSummary:
        """Write one immutable game file, append its metadata, update counters."""

        game_id = str(game.get("game_id", ""))
        if not game_id:
            raise HumanDatasetError("game is missing a game_id")
        if game_id in self._known_ids or (self.games_dir / f"{game_id}.json").exists():
            raise HumanDatasetError(f"duplicate game id: {game_id}")

        transitions = game.get("transitions")
        if not isinstance(transitions, list):
            raise HumanDatasetError("game transitions must be a list")
        payload = apply_canonical_identity(dict(game))
        if int(payload["width"]) != self.width or int(payload["height"]) != self.height:
            raise HumanDatasetError(
                f"game board {payload['width']}x{payload['height']} does not match "
                f"the dataset board {self.width}x{self.height}"
            )
        path = self.games_dir / f"{game_id}.json"
        write_json_atomic(path, payload)

        self._known_ids.add(game_id)
        summary = _summary_from_game(payload, path)
        append_jsonl(self.metadata_path, [_summary_dict(summary)])
        self.manifest["game_count"] = self.game_count + 1
        self.manifest["transition_count"] = self.transition_count + len(transitions)
        self._save_manifest()
        return summary

    def resume_game_write(self, game: Mapping[str, Any]) -> HumanGameSummary:
        """Retry this exact commit after a failed write, without duplicating it.

        The immutable game file is the source of truth. Metadata append and
        manifest replace may fail *after* it was committed (including fsync).
        Only an identical payload may resume; ordinary duplicate writes remain
        errors through write_game().
        """
        game_id = str(game["game_id"])
        path = self.games_dir / f"{game_id}.json"
        if not path.exists():
            return self.write_game(game)
        if self.load_game(game_id) != game:
            raise HumanDatasetError(f"conflicting retry for game id: {game_id}")
        summary = _summary_from_game(game, path)
        repair_jsonl_tail(self.metadata_path)
        if not any(row.get("game_id") == game_id for row in iter_jsonl(self.metadata_path)):
            append_jsonl(self.metadata_path, [_summary_dict(summary)])
        self._known_ids.add(game_id)
        summaries = [_summary_from_game(self.load_game(p.stem), p) for p in self._game_files()]
        self.manifest["game_count"] = len(summaries)
        self.manifest["transition_count"] = sum(row.total_steps for row in summaries)
        self._save_manifest()
        return summary

    # -- reading ----------------------------------------------------------

    def iter_games(self, *, source: str | None = None) -> Iterator[HumanGameSummary]:
        """Stream per-game summaries from ``metadata.jsonl`` (fallback: files)."""

        if self.metadata_path.exists() and self.metadata_path.stat().st_size > 0:
            for data in iter_jsonl(self.metadata_path):
                summary = _summary_from_metadata(data)
                if source is None or summary.source == source:
                    yield summary
            return
        for path in self._game_files():
            data = json.loads(path.read_text(encoding="utf-8"))
            summary = _summary_from_game(data, path)
            if source is None or summary.source == source:
                yield summary

    def load_game(self, game_id: str) -> dict[str, Any]:
        """Return the full per-game payload (metadata + transitions)."""

        path = self.games_dir / f"{game_id}.json"
        if not path.exists():
            raise HumanDatasetError(f"game not found: {game_id}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HumanDatasetError(f"could not read {path}: {error}") from error
        if not isinstance(data, dict):
            raise HumanDatasetError(f"{path} is not a JSON object")
        return data

    def validate(self) -> HumanDatasetReport:
        """Validate every game file; see :func:`validate_dataset`."""

        return validate_dataset(self.root)


def _action_counts_from_transitions(transitions: Any) -> dict[str, int]:
    counts = {ACTION_NAMES[ACTION_OPEN]: 0, ACTION_NAMES[ACTION_FLAG]: 0,
              ACTION_NAMES[ACTION_CHORD]: 0}
    if isinstance(transitions, list):
        for transition in transitions:
            action = transition.get("action", {}) if isinstance(transition, Mapping) else {}
            name = ACTION_NAMES.get(action.get("action_type"))
            if name is not None:
                counts[name] += 1
    return counts


def _summary_from_game(game: Mapping[str, Any], path: Path | None) -> HumanGameSummary:
    counts = _action_counts_from_transitions(game.get("transitions"))
    return HumanGameSummary(
        game_id=str(game.get("game_id")),
        game_index=_as_int(game.get("game_index", _game_index(str(game.get("game_id"))))),
        source=str(game.get("source", schema.SOURCE_HUMAN)),
        result=str(game.get("result", "abandoned")),
        outcome=str(game.get("outcome", "incomplete")),
        complete=bool(game.get("complete", False)),
        total_steps=_as_int(game.get("total_steps", 0)),
        transition_count=_as_int(game.get("total_steps", 0)),
        width=_as_int(game.get("width", 0)),
        height=_as_int(game.get("height", 0)),
        mines=game.get("mines"),
        difficulty=game.get("difficulty"),
        started_at=float(game.get("started_at", 0.0)),
        ended_at=float(game.get("ended_at", 0.0)),
        duration_s=float(game.get("duration_s", 0.0)),
        open_count=counts[ACTION_NAMES[ACTION_OPEN]],
        flag_count=counts[ACTION_NAMES[ACTION_FLAG]],
        chord_count=counts[ACTION_NAMES[ACTION_CHORD]],
        path=None if path is None else path.name,
        termination_reason=game.get("termination_reason"),
    )


def _summary_from_metadata(data: Mapping[str, Any]) -> HumanGameSummary:
    return HumanGameSummary(
        game_id=str(data.get("game_id")),
        game_index=_as_int(data.get("game_index", 0)),
        source=str(data.get("source", schema.SOURCE_HUMAN)),
        result=str(data.get("result", "abandoned")),
        outcome=str(data.get("outcome", "incomplete")),
        complete=bool(data.get("complete", False)),
        total_steps=_as_int(data.get("total_steps", 0)),
        transition_count=_as_int(data.get("transition_count", data.get("total_steps", 0))),
        width=_as_int(data.get("width", 0)),
        height=_as_int(data.get("height", 0)),
        mines=data.get("mines"),
        difficulty=data.get("difficulty"),
        started_at=float(data.get("started_at", 0.0)),
        ended_at=float(data.get("ended_at", 0.0)),
        duration_s=float(data.get("duration_s", 0.0)),
        open_count=_as_int(data.get("open_count", 0)),
        flag_count=_as_int(data.get("flag_count", 0)),
        chord_count=_as_int(data.get("chord_count", 0)),
        path=data.get("file"),
        termination_reason=data.get("termination_reason"),
    )


def _summary_dict(summary: HumanGameSummary) -> dict[str, Any]:
    return {
        "game_id": summary.game_id,
        "game_index": summary.game_index,
        "source": summary.source,
        "result": summary.result,
        "outcome": summary.outcome,
        "complete": summary.complete,
        "total_steps": summary.total_steps,
        "transition_count": summary.transition_count,
        "open_count": summary.open_count,
        "flag_count": summary.flag_count,
        "chord_count": summary.chord_count,
        "width": summary.width,
        "height": summary.height,
        "mines": summary.mines,
        "difficulty": summary.difficulty,
        "started_at": summary.started_at,
        "ended_at": summary.ended_at,
        "duration_s": summary.duration_s,
        "file": summary.path,
        "termination_reason": summary.termination_reason,
        "recorded_at": _now(),
    }


class PersistentHumanRecorder:
    """A drop-in recording sink for :class:`HumanTrajectoryRecorder`.

    It implements the same ``start_game``/``record_step``/``end_game`` surface
    as :class:`~minesweeper_rl.data.recorder.TrajectoryRecorder` but commits each
    completed game as an independent file in a :class:`HumanDatasetStore`.
    Transitions are buffered for the single active game only, so the dataset is
    never loaded into memory and never rewritten.
    """

    def __init__(
        self,
        store: HumanDatasetStore,
        *,
        width: int,
        height: int,
        mines: int | None = None,
        difficulty: str | None = None,
        source: str = schema.SOURCE_HUMAN,
        store_incomplete: bool = True,
    ) -> None:
        if source not in schema.SOURCES:
            raise HumanDatasetError(f"invalid source: {source!r}")
        if width < 1 or height < 1:
            raise HumanDatasetError("board dimensions must be positive")
        self.store = store
        self.width = int(width)
        self.height = int(height)
        self.mines = None if mines is None else int(mines)
        self.difficulty = difficulty
        self.source = source
        self.store_incomplete = bool(store_incomplete)
        self._reset()

    def _reset(self) -> None:
        self._game_id: str | None = None
        self._transitions: list[dict[str, Any]] = []
        self._started_at: float | None = None
        self._first_move: tuple[int, int] | None = None
        self._other_clicks = 0
        self._last_board = None
        self._pending_game: dict[str, Any] | None = None

    @property
    def active_game_id(self) -> str | None:
        return self._game_id

    @property
    def steps_recorded(self) -> int:
        return len(self._transitions)

    @property
    def last_board(self):
        return self._last_board

    @property
    def game_terminated(self) -> bool:
        """Whether the active game already recorded a terminal transition."""

        return bool(self._transitions and self._transitions[-1].get("terminated"))

    def discard_game(self) -> None:
        self._reset()

    def start_game(
        self,
        *,
        first_move: tuple[int, int] | None = None,
        started_at: float | None = None,
        board: Any = None,
    ) -> str:
        if self._game_id is not None:
            raise HumanDatasetError(
                f"game {self._game_id} is still being recorded; call end_game() first"
            )
        if first_move is not None:
            if len(first_move) != 2:
                raise HumanDatasetError("first_move must be an (x, y) pair")
            x, y = int(first_move[0]), int(first_move[1])
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise HumanDatasetError(f"first_move outside board: ({x}, {y})")
            first_move = (x, y)
        if board is not None:
            self._last_board = schema.validate_board(
                board, width=self.width, height=self.height
            )
        self._game_id = self.store.next_game_id(self.source)
        self._started_at = _now() if started_at is None else float(started_at)
        self._first_move = first_move
        return self._game_id

    @staticmethod
    def _resolve_action_type(action_type: Any, before: Any, y: int, x: int) -> int:
        """Map a schema action name (or numeric code) to the numeric type.

        The human observer reports a left-click as OPEN even when it is a chord;
        a chord is identified from the pre-action board: an OPEN on an already
        opened number is a CHORD.
        """

        if action_type == schema.ACTION_FLAG:
            return ACTION_FLAG
        if action_type == schema.ACTION_CHORD:
            return ACTION_CHORD
        if action_type in (schema.ACTION_OPEN, schema.ACTION_REVEAL, None):
            value = int(before[y, x])
            if value == HIDDEN:
                return ACTION_OPEN
            if 0 <= value <= MAX_NUMBER:
                return ACTION_CHORD
            raise HumanDatasetError(
                f"cannot classify an open action on cell ({x}, {y}) "
                f"with value {value}"
            )
        if action_type in ACTION_NAMES:
            return int(action_type)
        raise HumanDatasetError(f"invalid action_type: {action_type!r}")

    def record_step(
        self,
        *,
        board_before: Any,
        action: tuple[int, int],
        board_after: Any,
        game_status: Any,
        terminated: bool,
        truncated: bool = False,
        reward: float | None = None,
        opened_cells: int | None = None,
        newly_revealed: int | None = None,
        timestamp: float | None = None,
        action_type: Any = schema.ACTION_OPEN,
    ) -> dict[str, Any]:
        if self._game_id is None:
            raise HumanDatasetError("start_game() must be called before record_step()")
        if self._transitions and self._transitions[-1]["terminated"]:
            raise HumanDatasetError("cannot record a step after a terminal transition")
        try:
            x, y = action
            x, y = int(x), int(y)
        except (TypeError, ValueError) as error:
            raise HumanDatasetError("action must be an (x, y) pair") from error
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise HumanDatasetError(f"action outside board: ({x}, {y})")

        before = schema.validate_board(board_before, width=self.width, height=self.height)
        after = schema.validate_board(board_after, width=self.width, height=self.height)

        numeric_type = self._resolve_action_type(action_type, before, y, x)
        self._validate_effect(numeric_type, before, after, x, y)
        self._validate_terminal_semantics(
            numeric_type, before, after, x, y, game_status, terminated
        )

        previous = None
        if self._transitions:
            previous = schema.decode_board(
                self._transitions[-1]["observation_after"],
                width=self.width,
                height=self.height,
            )
            if previous is not None and not (previous == before).all():
                raise HumanDatasetError(
                    "board_before does not continue from the previous board_after"
                )

        cell = y * self.width + x
        action_id = compound_action_id(numeric_type, cell, cells=self.width * self.height)
        transition = {
            "step": len(self._transitions),
            "timestamp": _now() if timestamp is None else float(timestamp),
            "observation_before": schema.encode_board(before),
            "observation_after": schema.encode_board(after),
            "action": {
                "type": ACTION_NAMES[numeric_type],
                "action_type": numeric_type,
                "x": x,
                "y": y,
                "cell_index": cell,
                "action_id": action_id,
            },
            "game_status": schema.normalize_status(game_status),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "newly_revealed": newly_revealed,
            "opened_cells": opened_cells,
            "reward": reward,
        }
        self._transitions.append(transition)
        self._last_board = after
        return transition

    def _validate_effect(
        self, numeric_type: int, before: Any, after: Any, x: int, y: int
    ) -> None:
        before_cell = int(before[y, x])
        after_cell = int(after[y, x])
        if numeric_type == ACTION_OPEN:
            if before_cell != HIDDEN:
                raise HumanDatasetError("an OPEN action must target a hidden cell")
        elif numeric_type == ACTION_FLAG:
            if before_cell not in (HIDDEN, FLAGGED):
                raise HumanDatasetError("a FLAG action must target a covered cell")
            if before_cell == after_cell:
                raise HumanDatasetError(
                    "a FLAG action must toggle the clicked cell"
                )
        elif numeric_type == ACTION_CHORD:
            if not 1 <= before_cell <= MAX_NUMBER:
                raise HumanDatasetError(
                    "a CHORD action must target an opened numbered cell"
                )

    @staticmethod
    def _validate_terminal_semantics(
        numeric_type: int,
        before: Any,
        after: Any,
        x: int,
        y: int,
        game_status: Any,
        terminated: bool,
    ) -> None:
        """Reject transitions whose status cannot follow from the board change.

        This is the recorder-side guard that keeps impossible/phantom events out
        of the raw dataset in the first place (the semantic validator applies the
        same rules to already-written games).
        """

        status = schema.normalize_status(game_status)
        terminal = status in (schema.RESULT_WON, schema.RESULT_LOST)
        if terminal and not terminated:
            raise HumanDatasetError("a won/lost status requires terminated=True")
        if terminated and not terminal:
            raise HumanDatasetError(
                "a terminated transition must end in won or lost"
            )
        before_cell = int(before[y, x])
        after_cell = int(after[y, x])
        unchanged = bool((before == after).all())
        if numeric_type == ACTION_OPEN:
            if (
                before_cell == HIDDEN
                and after_cell >= 0
                and status == schema.RESULT_LOST
            ):
                raise HumanDatasetError("a safe OPEN cannot end in LOST")
            if before_cell == HIDDEN and after_cell == HIDDEN and not terminal:
                raise HumanDatasetError("an OPEN must reveal the clicked cell")
        if (
            terminated
            and unchanged
            and not (
                status == schema.RESULT_LOST
                and (
                    (numeric_type == ACTION_OPEN and before_cell == HIDDEN)
                    or numeric_type == ACTION_CHORD
                )
            )
        ):
            raise HumanDatasetError(
                "a terminal transition must change the board"
            )

    def record_other_click(self) -> None:
        if self._game_id is None:
            raise HumanDatasetError("no active game to attach the click to")
        self._other_clicks += 1

    def end_game(
        self,
        result: str,
        *,
        ground_truth: dict[str, Any] | None = None,
        ended_at: float | None = None,
        final_opened_cells: int | None = None,
        termination_reason: str | None = None,
    ) -> HumanGameSummary:
        if self._game_id is None:
            raise HumanDatasetError("no game is being recorded")
        if ground_truth:
            # Hidden mine positions are never part of the dataset.
            logger.warning("ignoring ground truth supplied to the human recorder")
        outcome, complete = _outcome_for_result(result)
        terminated_index = [
            index for index, step in enumerate(self._transitions) if step["terminated"]
        ]
        if terminated_index and terminated_index != [len(self._transitions) - 1]:
            raise HumanDatasetError(
                "a terminal transition must be the final step of a game"
            )
        if complete:
            if not self._transitions or not self._transitions[-1]["terminated"]:
                raise HumanDatasetError(f"result {result!r} requires a terminal step")
            if self._transitions[-1]["game_status"] != result:
                raise HumanDatasetError(
                    f"result {result!r} contradicts terminal status "
                    f"{self._transitions[-1]['game_status']!r}"
                )
        elif terminated_index:
            raise HumanDatasetError("an incomplete game cannot contain a terminal step")

        ended = _now() if ended_at is None else float(ended_at)
        board = self._last_board
        if final_opened_cells is None:
            final_opened_cells = 0 if board is None else int((board >= 0).sum())
        started = self._started_at if self._started_at is not None else ended
        counts = {}
        for numeric, name in ACTION_NAMES.items():
            counts[name] = sum(
                1
                for step in self._transitions
                if step["action"]["action_type"] == numeric
            )
        game = {
            "format": HUMAN_GAME_FORMAT,
            "format_version": HUMAN_GAME_VERSION,
            "game_id": self._game_id,
            "game_index": _game_index(self._game_id),
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "mines": self.mines,
            "difficulty": self.difficulty,
            "outcome": outcome,
            "complete": complete,
            "result": result,
            "termination_reason": termination_reason,
            "total_steps": len(self._transitions),
            "started_at": started,
            "ended_at": ended,
            "duration_s": ended - started,
            "first_move": list(self._first_move) if self._first_move else None,
            "other_clicks": self._other_clicks,
            "final_opened_cells": int(final_opened_cells),
            "action_counts": counts,
            "transitions": list(self._transitions),
        }
        apply_canonical_identity(game)
        summary = _summary_from_game(game, self.store.games_dir / f"{self._game_id}.json")
        if complete or self.store_incomplete:
            if self._pending_game is None:
                self._pending_game = game
                self.store.write_game(game)
            else:
                game = self._pending_game
                summary = self.store.resume_game_write(game)
        self._reset()
        return summary


def _outcome_for_result(result: str) -> tuple[str, bool]:
    if result == schema.RESULT_WON:
        return "won", True
    if result == schema.RESULT_LOST:
        return "lost", True
    if result in (schema.RESULT_ABANDONED, "incomplete"):
        return "incomplete", False
    raise HumanDatasetError(f"invalid result: {result!r}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_ACTION_NAMES_BY_CODE = dict(ACTION_NAMES)

#: Ordered (substring, category) pairs used to group invalidation reasons.
_REASON_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("first state has", "non-fresh initial board"),
    ("first state equals", "phantom continuation"),
    ("first action is not OPEN", "first action not OPEN"),
    ("first observation", "unreadable initial state"),
    ("OPEN on an already-open", "OPEN on an already-open cell"),
    ("OPEN on a flagged", "OPEN on a flagged cell"),
    ("safe OPEN cannot", "safe OPEN ending in LOST"),
    ("OPEN did not reveal", "OPEN with no reveal"),
    ("CHORD on a non-number", "CHORD on a non-number"),
    ("CHORD changed nothing", "CHORD no-op"),
    ("FLAG on an open cell", "FLAG on an open cell"),
    ("FLAG did not toggle", "FLAG no-op"),
    ("terminal no-op", "terminal no-op"),
    ("complete game does not", "complete without terminal"),
    ("incomplete game contains", "incomplete with terminal"),
    ("contradicts terminal status", "outcome/status mismatch"),
    ("predates the capture fix", "pre-fix recorder version"),
    ("action_id", "action id mismatch"),
    ("cell_index", "cell index mismatch"),
    ("continu", "board discontinuity"),
    ("impossible action type", "impossible action type"),
)


def reason_category(reason: str) -> str:
    """Return a short grouped category for one invalidation reason."""

    for token, label in _REASON_CATEGORIES:
        if token in reason:
            return label
    return "other"


@dataclass(slots=True)
class GameSemanticResult:
    """Semantic validity of one recorded game (observable-state rules only)."""

    valid: bool = True
    reasons: list[str] = field(default_factory=list)
    transitions_total: int = 0
    transitions_valid: int = 0
    transitions_invalid: int = 0
    invalid_steps: set[int] = field(default_factory=set)

    def fail(self, reason: str, *, step: int | None = None) -> None:
        self.valid = False
        self.reasons.append(reason if step is None else f"step {step}: {reason}")
        if step is not None:
            self.invalid_steps.add(step)


def semantic_game_result(
    game: Mapping[str, Any],
    *,
    previous_final_board: Any = None,
) -> GameSemanticResult:
    """Validate one game transcript against observable-state semantics.

    This is deliberately about *what a Minesweeper transcript can mean*, not
    about storage formatting: a fresh game must start all-hidden with an OPEN,
    OPEN/FLAG/CHORD must be legal for the pre-action board, a safe reveal can
    never end in LOST, terminal transitions must not be no-ops, and the game
    must not continue the previous game's final board.  It never inspects a
    hidden mine layout, so no rule depends on ground truth.
    """

    result = GameSemanticResult()
    try:
        width = _as_int(game.get("width", 0))
        height = _as_int(game.get("height", 0))
    except HumanDatasetError as error:
        result.fail(f"invalid dimensions ({error})")
        return result
    if width < 1 or height < 1:
        result.fail(f"invalid dimensions {width}x{height}")
        return result
    cells = width * height

    recorder_version = game.get("recorder_version")
    if (
        not isinstance(recorder_version, bool)
        and isinstance(recorder_version, int)
        and recorder_version < HUMAN_RECORDER_VERSION
    ):
        result.fail(
            f"recorder_version {recorder_version} predates the capture fix "
            f"(current {HUMAN_RECORDER_VERSION}); capture is unreliable"
        )

    transitions = game.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        result.fail("game has no transitions")
        return result
    result.transitions_total = len(transitions)

    first = transitions[0]
    try:
        first_board = schema.decode_board(
            first["observation_before"], width=width, height=height
        )
    except (KeyError, schema.DataError, TypeError) as error:
        result.fail(f"first observation is unreadable ({error})")
        return result
    opened = int((first_board >= 0).sum())
    flagged = int((first_board == FLAGGED).sum())
    if opened > 0:
        result.fail(
            f"first state has {opened} open cells (not a fresh board)"
        )
    if flagged > 0:
        result.fail(
            f"first state has {flagged} flagged cells (not a fresh board)"
        )
    if previous_final_board is not None and np.array_equal(
        first_board, np.asarray(previous_final_board)
    ):
        result.fail(
            "first state equals the previous game's final state "
            "(phantom continuation)"
        )
    first_action = first.get("action") if isinstance(first, Mapping) else None
    try:
        first_type = (
            _as_int(first_action.get("action_type", -1))
            if isinstance(first_action, Mapping)
            else -1
        )
    except HumanDatasetError:
        first_type = -1
    if first_type != ACTION_OPEN:
        result.fail("first action is not OPEN")

    previous_after = None
    last_status = None
    for index, transition in enumerate(transitions):
        if not isinstance(transition, Mapping):
            result.fail("transition is not an object", step=index)
            continue
        before_ok = True
        try:
            before = schema.decode_board(
                transition["observation_before"], width=width, height=height
            )
            after = schema.decode_board(
                transition["observation_after"], width=width, height=height
            )
        except (KeyError, schema.DataError, TypeError) as error:
            result.fail(f"unreadable observation ({error})", step=index)
            before_ok = False
        action = transition.get("action")
        status = str(transition.get("game_status", ""))
        terminated = bool(transition.get("terminated", False))
        last_status = status
        if not before_ok:
            continue
        if previous_after is not None and not np.array_equal(before, previous_after):
            result.fail("board_before does not continue from board_after", step=index)
        previous_after = after

        if not isinstance(action, Mapping):
            result.fail("missing action object", step=index)
            continue
        action_type = action.get("action_type")
        try:
            numeric = _as_int(action_type)
        except HumanDatasetError:
            result.fail("impossible action type", step=index)
            continue
        if numeric not in _ACTION_NAMES_BY_CODE:
            result.fail("impossible action type", step=index)
            continue
        try:
            x = _as_int(action.get("x", -1))
            y = _as_int(action.get("y", -1))
        except HumanDatasetError:
            result.fail("action coordinate is not an integer", step=index)
            continue
        if not (0 <= x < width and 0 <= y < height):
            result.fail("action coordinate outside board", step=index)
            continue
        cell = y * width + x
        if action.get("cell_index") is not None:
            try:
                if _as_int(action["cell_index"]) != cell:
                    result.fail("cell_index does not match x/y", step=index)
            except HumanDatasetError:
                result.fail("cell_index is not an integer", step=index)
        try:
            action_id = _as_int(action.get("action_id", -1))
        except HumanDatasetError:
            result.fail("action_id is not an integer", step=index)
            action_id = -1
        expected_id = compound_action_id(numeric, cell, cells=cells)
        if action_id != expected_id:
            result.fail(
                f"action_id {action_id} does not match {numeric}*{cells}+{cell}"
                f" -> {expected_id}",
                step=index,
            )
        name = _ACTION_NAMES_BY_CODE[numeric]
        if action.get("type") not in (None, name):
            result.fail("action type name does not match action_type", step=index)

        before_cell = int(before[y, x])
        after_cell = int(after[y, x])
        if numeric == ACTION_OPEN:
            if before_cell == FLAGGED:
                result.fail("OPEN on a flagged cell", step=index)
            elif before_cell >= 0:
                result.fail(
                    "OPEN on an already-open cell (should be CHORD)", step=index
                )
            elif after_cell >= 0 and status == "lost":
                result.fail("safe OPEN cannot end in LOST", step=index)
            elif after_cell == HIDDEN and status != "lost":
                result.fail("OPEN did not reveal the clicked cell", step=index)
        elif numeric == ACTION_CHORD:
            if not 1 <= before_cell <= MAX_NUMBER:
                result.fail("CHORD on a non-number cell", step=index)
            elif np.array_equal(before, after) and status != "lost":
                # A losing chord can be visually unchanged because detonated
                # mines are normalized to HIDDEN in the canonical board.
                result.fail("CHORD changed nothing", step=index)
        else:  # ACTION_FLAG
            if before_cell not in (HIDDEN, FLAGGED):
                result.fail("FLAG on an open cell", step=index)
            elif before_cell == after_cell:
                result.fail("FLAG did not toggle the cell", step=index)

        if (
            terminated
            and status in ("won", "lost")
            and np.array_equal(before, after)
            and not (
                status == "lost"
                and (
                    (numeric == ACTION_OPEN and before_cell == HIDDEN)
                    or numeric == ACTION_CHORD
                )
            )
        ):
            result.fail("terminal no-op transition", step=index)

    invalid = result.invalid_steps
    result.transitions_invalid = len(invalid)
    result.transitions_valid = result.transitions_total - result.transitions_invalid

    complete = bool(game.get("complete", False))
    outcome = str(game.get("outcome", ""))
    last = transitions[-1]
    last_terminated = bool(last.get("terminated", False)) if isinstance(
        last, Mapping
    ) else False
    if complete:
        if not last_terminated or last_status not in ("won", "lost"):
            result.fail("complete game does not end in a terminal transition")
        elif outcome != last_status:
            result.fail(
                f"outcome {outcome!r} contradicts terminal status {last_status!r}"
            )
    elif last_terminated:
        result.fail("incomplete game contains a terminal transition")
    return result


def _last_observation(game: Mapping[str, Any]) -> Any | None:
    """Decode a game's final observable board, or ``None`` if unusable."""

    transitions = game.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        return None
    last = transitions[-1]
    if not isinstance(last, Mapping):
        return None
    try:
        width = _as_int(game.get("width", 0))
        height = _as_int(game.get("height", 0))
        return schema.decode_board(
            last["observation_after"], width=width, height=height
        )
    except (KeyError, HumanDatasetError, schema.DataError, TypeError):
        return None


def validate_dataset(root: str | Path) -> HumanDatasetReport:
    """Validate a human dataset and return structural + semantic statistics.

    Structural: corrupted files, duplicate game ids, invalid action ids,
    coordinates, state shapes, transition continuity and game identity.

    Semantic: every game must begin on a fresh all-hidden board with an OPEN,
    must not continue the previous game's final board, must use legal actions
    for the pre-action board, must not contain terminal no-ops or a safe reveal
    that ends in LOST, and must agree with its terminal outcome.
    """

    root = Path(root)
    games_dir = root / GAMES_DIRNAME
    report = HumanDatasetReport()
    reason = legacy_dataset_reason(root)
    if reason is not None:
        report.errors.append(f"{root}: {reason}")
        return report
    _validate_manifest(root, report)
    if not games_dir.exists():
        report.errors.append(f"missing games directory: {games_dir}")
        return report

    seen_ids: set[str] = set()
    metadata_ids: list[str] = []
    previous_final = None
    for path in sorted(games_dir.glob("game_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            report.errors.append(f"{path.name}: corrupted file: {error}")
            continue
        if not isinstance(data, dict):
            report.errors.append(f"{path.name}: not a JSON object")
            continue
        game_id = data.get("game_id")
        if not isinstance(game_id, str) or not game_id:
            report.errors.append(f"{path.name}: missing game_id")
            game_id = path.stem
        if game_id in seen_ids:
            report.errors.append(f"{game_id}: duplicate game id")
        seen_ids.add(game_id)
        if path.stem != game_id:
            report.errors.append(
                f"{path.name}: file name does not match game_id {game_id!r}"
            )
        errors_before = len(report.errors)
        _validate_game_file(data, path.name, report)
        semantic = semantic_game_result(data, previous_final_board=previous_final)
        for reason in semantic.reasons:
            report.errors.append(f"{path.name}: {reason}")
            category = reason_category(reason)
            report.invalid_reasons[category] = (
                report.invalid_reasons.get(category, 0) + 1
            )
        if len(report.errors) == errors_before and semantic.valid:
            report.games_valid += 1
        else:
            report.games_invalid += 1
            report.invalid_games.append(str(game_id))
        report.transitions_valid += semantic.transitions_valid
        report.transitions_invalid += semantic.transitions_invalid
        previous_final = _last_observation(data)

    if (root / METADATA_FILENAME).exists():
        for data in iter_jsonl(root / METADATA_FILENAME):
            game_id = data.get("game_id")
            if isinstance(game_id, str):
                metadata_ids.append(game_id)
        duplicates = {gid for gid in metadata_ids if metadata_ids.count(gid) > 1}
        for game_id in sorted(duplicates):
            report.errors.append(f"{game_id}: duplicate metadata entry")
        missing = seen_ids - set(metadata_ids)
        for game_id in sorted(missing):
            report.errors.append(f"{game_id}: game file has no metadata entry")
        extra = set(metadata_ids) - seen_ids
        for game_id in sorted(extra):
            report.errors.append(f"{game_id}: metadata entry has no game file")

    return report


def _validate_manifest(root: Path, report: HumanDatasetReport) -> None:
    """Validate the dataset manifest identity without importing Python code."""

    path = root / MANIFEST_FILENAME
    if not path.exists():
        report.errors.append(f"missing manifest: {path}")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        report.errors.append(f"{path.name}: corrupted manifest: {error}")
        return
    if not isinstance(data, dict):
        report.errors.append(f"{path.name}: not a JSON object")
        return
    if data.get("format") != HUMAN_DATASET_FORMAT:
        report.errors.append(
            f"{path.name}: format {data.get('format')!r} != "
            f"{HUMAN_DATASET_FORMAT!r}"
        )
    board = data.get("board")
    if not isinstance(board, Mapping):
        report.errors.append(f"{path.name}: missing board object")
    else:
        try:
            width = _as_int(board["width"])
            height = _as_int(board["height"])
        except (KeyError, HumanDatasetError) as error:
            report.errors.append(f"{path.name}: invalid board ({error})")
        else:
            _validate_identity(data, width, height, f"{path.name}:", report)


def _validate_identity(
    data: Mapping[str, Any],
    width: int,
    height: int,
    prefix: str,
    report: HumanDatasetReport,
) -> None:
    """Validate the frozen action/observation identity of a manifest or game."""

    cells = width * height
    format_version = data.get("format_version", 1)
    if (
        isinstance(format_version, bool)
        or not isinstance(format_version, int)
        or format_version < 1
    ):
        report.errors.append(f"{prefix} invalid format_version {format_version!r}")
        format_version = 1
    if format_version >= 2:
        required = (
            "observation_representation",
            "observation_version",
            "action_space",
            "action_encoding",
            "action_types",
            "num_actions",
            "cells",
            "reward_spec",
            "environment_version",
        )
        for key in required:
            if data.get(key) is None:
                report.errors.append(
                    f"{prefix} missing required identity field {key!r}"
                )
    action_space = data.get("action_space")
    if action_space is not None and action_space != ACTION_SPACE:
        report.errors.append(
            f"{prefix} action_space {action_space!r} != {ACTION_SPACE!r} "
            f"(chord-as-open legacy data must not be used)"
        )
    action_types = data.get("action_types")
    if action_types is not None:
        if not isinstance(action_types, Mapping) or dict(action_types) != (
            ACTION_TYPES_MAPPING
        ):
            report.errors.append(
                f"{prefix} action_types {action_types!r} != {ACTION_TYPES_MAPPING!r}"
            )
    declared_cells = data.get("cells")
    if declared_cells is not None:
        try:
            if _as_int(declared_cells) != cells:
                report.errors.append(
                    f"{prefix} cells {declared_cells!r} != board cells {cells}"
                )
        except HumanDatasetError as error:
            report.errors.append(f"{prefix} cells {error}")
    declared_actions = data.get("num_actions")
    if declared_actions is not None:
        expected_actions = len(ACTION_NAMES) * cells
        try:
            if _as_int(declared_actions) != expected_actions:
                report.errors.append(
                    f"{prefix} num_actions {declared_actions!r} != "
                    f"{expected_actions}"
                )
        except HumanDatasetError as error:
            report.errors.append(f"{prefix} num_actions {error}")
    observation_version = data.get("observation_version")
    if observation_version is not None and (
        isinstance(observation_version, bool)
        or not isinstance(observation_version, int)
        or observation_version < 1
    ):
        report.errors.append(
            f"{prefix} invalid observation_version {observation_version!r}"
        )
    for key in FORBIDDEN_LAYOUT_KEYS:
        if key in data:
            report.errors.append(
                f"{prefix} forbidden hidden-mine field {key!r} present"
            )


def _validate_game_file(data: Mapping[str, Any], name: str, report: HumanDatasetReport) -> None:
    game_id = str(data.get("game_id", name))
    prefix = f"{name}:"
    report.games += 1
    complete = bool(data.get("complete", False))
    outcome = str(data.get("outcome", ""))
    if complete:
        report.complete_games += 1
        if outcome == "won":
            report.winning_games += 1
        elif outcome == "lost":
            report.lost_games += 1
        else:
            report.errors.append(
                f"{prefix} complete game has non-terminal outcome {outcome!r}"
            )
    else:
        report.incomplete_games += 1
        if outcome not in ("incomplete", "abandoned", ""):
            report.errors.append(
                f"{prefix} incomplete game has outcome {outcome!r}"
            )

    try:
        width = _as_int(data["width"])
        height = _as_int(data["height"])
    except (KeyError, HumanDatasetError) as error:
        report.errors.append(f"{prefix} invalid board dimensions ({error})")
        return
    if width < 1 or height < 1:
        report.errors.append(f"{prefix} invalid board dimensions {width}x{height}")
        return
    _validate_identity(data, width, height, prefix, report)

    transitions = data.get("transitions")
    if not isinstance(transitions, list):
        report.errors.append(f"{prefix} missing transitions list")
        return
    declared = data.get("total_steps")
    if declared is not None and _as_int(declared) != len(transitions):
        report.errors.append(
            f"{prefix} total_steps {declared} but {len(transitions)} transitions found"
        )

    previous_board = None
    for expected, transition in enumerate(transitions):
        try:
            board_before = _validate_transition(
                transition, expected, width, height, previous_board, prefix, report
            )
        except schema.DataError as error:
            report.errors.append(f"{prefix} step {expected}: {error}")
            board_before = None
        previous_board = board_before
        report.transitions += 1

    if complete and not transitions:
        report.errors.append(f"{prefix} complete game has no transitions")


def _validate_transition(
    transition: Mapping[str, Any],
    expected: int,
    width: int,
    height: int,
    previous_board: Any,
    prefix: str,
    report: HumanDatasetReport,
) -> Any | None:
    cells = width * height
    if not isinstance(transition, Mapping):
        raise schema.DataError("transition is not an object")
    for key in FORBIDDEN_LAYOUT_KEYS:
        if key in transition:
            raise schema.DataError(
                f"hidden-mine field {key!r} must not appear in a transition"
            )
    if _as_int(transition.get("step", -1)) != expected:
        raise schema.DataError(
            f"step index {transition.get('step')!r} does not match position {expected}"
        )
    before = schema.decode_board(
        transition["observation_before"], width=width, height=height
    )
    after = schema.decode_board(
        transition["observation_after"], width=width, height=height
    )
    if before.shape != (height, width) or after.shape != (height, width):
        raise schema.DataError("state shape does not match the board dimensions")
    if previous_board is not None and not (previous_board == before).all():
        raise schema.DataError("board_before does not continue from board_after")

    action = transition.get("action")
    if not isinstance(action, Mapping):
        raise schema.DataError("missing action object")
    action_type = _as_int(action.get("action_type", -1))
    if action_type not in _ACTION_NAMES_BY_CODE:
        raise schema.DataError(f"impossible action type: {action_type!r}")
    name = _ACTION_NAMES_BY_CODE[action_type]
    if action.get("type") not in (None, name):
        raise schema.DataError(
            f"action type name {action.get('type')!r} does not match code {action_type}"
        )
    x = _as_int(action.get("x", -1))
    y = _as_int(action.get("y", -1))
    if not (0 <= x < width and 0 <= y < height):
        raise schema.DataError(f"action coordinate outside board: ({x}, {y})")
    cell = y * width + x
    expected_id = compound_action_id(action_type, cell, cells=cells)
    action_id = _as_int(action.get("action_id", -1))
    if action_id != expected_id:
        raise schema.DataError(
            f"action_id {action_id} does not match {name} at ({x}, {y}) "
            f"-> {expected_id}"
        )
    if action.get("cell_index") is not None and _as_int(action["cell_index"]) != cell:
        raise schema.DataError("cell_index does not match x/y")
    if not 0 <= action_id < len(ACTION_NAMES) * cells:
        raise schema.DataError(f"action_id outside the action space: {action_id}")

    if action_type == ACTION_OPEN:
        report.open_count += 1
        if int(before[y, x]) != HIDDEN:
            raise schema.DataError("OPEN action targets a non-hidden cell")
    elif action_type == ACTION_FLAG:
        report.flag_count += 1
        if int(before[y, x]) not in (HIDDEN, FLAGGED):
            raise schema.DataError("FLAG action targets an open cell")
        if int(before[y, x]) == int(after[y, x]):
            raise schema.DataError("FLAG action did not toggle the cell")
    else:
        report.chord_count += 1
        if not 1 <= int(before[y, x]) <= MAX_NUMBER:
            raise schema.DataError("CHORD action targets a non-number cell")
    return after
