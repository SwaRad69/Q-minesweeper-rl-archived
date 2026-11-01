"""Canonical, model-independent trajectory records for Minesweeper games.

The dataset stores plain MDP facts -- the observable board before and after one
human action, the action, the reward and the terminal flags -- plus game-level
context.  Nothing here is specific to PPO, DDQN, GNN or quantum models; those
representations are derived later by model-specific adapters.

Board representation follows :mod:`minesweeper_rl.dom.parser`:

* ``-2`` (:data:`FLAGGED`) -- a covered cell carrying a flag
* ``-1`` (:data:`HIDDEN`) -- a covered, unflagged cell
* ``0..8`` -- a revealed cell's adjacent-mine count

A left-click is :data:`ACTION_OPEN`; on an already-open numbered cell it is a
chord and still :data:`ACTION_OPEN`.  A right-click is :data:`ACTION_FLAG`, and
``board_before``/``board_after`` distinguish placing a flag from removing one.

INFORMATION LEAKAGE
    ``board_before``/``board_after`` contain ONLY what the player could observe.
    The hidden mine layout is never part of any transition.  When it can be
    recovered after a game ends it is stored separately on the game record under
    :data:`GROUND_TRUTH_KEY` (``post_game_ground_truth``) and must never be fed
    to a model.  :meth:`GameRecord.safe_view` returns a projection with ground
    truth removed for training-side use, and :class:`TransitionRecord` has no
    ground-truth fields at all.
"""

from __future__ import annotations

import base64
import binascii
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import numpy.typing as npt

from ..dom import FLAGGED, HIDDEN, MAX_NUMBER, GameStatus

SCHEMA_VERSION = 1
RECORDER_VERSION = 2

SOURCE_HUMAN = "human"
SOURCE_AGENT = "agent"
SOURCES = (SOURCE_HUMAN, SOURCE_AGENT)

RESULT_WON = "won"
RESULT_LOST = "lost"
RESULT_ABANDONED = "abandoned"
# The persistent human format already supports incomplete recordings. Keep
# the legacy sink compatible when observation integrity (not play) is lost.
RESULTS = (RESULT_WON, RESULT_LOST, RESULT_ABANDONED, "incomplete")

# Canonical action vocabulary.  ``open`` is a left-click: it opens a hidden
# cell, or -- on an already-open numbered cell -- attempts a chord.  ``flag`` is
# a right-click that places or removes a flag (the board state says which).
ACTION_OPEN = "open"
ACTION_FLAG = "flag"
ACTION_TYPES = (ACTION_OPEN, ACTION_FLAG)

# Older revisions spelled an open action ``reveal`` and used a distinct
# ``chord`` action.  They are accepted when reading so existing datasets remain
# valid, but the recorder no longer writes them: a chord is now an OPEN action.
ACTION_REVEAL = "reveal"
ACTION_CHORD = "chord"
LEGACY_ACTION_TYPES = (ACTION_REVEAL, ACTION_CHORD)
ALL_ACTION_TYPES = ACTION_TYPES + LEGACY_ACTION_TYPES

GROUND_TRUTH_KEY = "post_game_ground_truth"
ENV_NAME = "minesweeper.online"

_POST_ACTION_STATUSES = frozenset(
    {GameStatus.RUNNING.value, GameStatus.WON.value, GameStatus.LOST.value}
)
_TERMINAL_STATUSES = frozenset({GameStatus.WON.value, GameStatus.LOST.value})


class DataError(ValueError):
    """Raised when trajectory data is malformed, inconsistent or incompatible."""


def _as_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise DataError(f"{name} must be an integer, got {type(value).__name__}")
    number = int(value)
    if minimum is not None and number < minimum:
        raise DataError(f"{name} must be >= {minimum}, got {number}")
    return number


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise DataError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise DataError(f"{name} must be finite")
    return number


def _as_optional_float(value: Any, name: str) -> float | None:
    return None if value is None else _as_float(value, name)


def validate_board(
    board: npt.ArrayLike, *, width: int | None = None, height: int | None = None
) -> npt.NDArray[np.int8]:
    """Return ``board`` as an integer 2D array or raise :class:`DataError`."""

    array = np.asarray(board)
    if array.ndim != 2:
        raise DataError(f"board must be a 2D [height, width] array, got {array.ndim}D")
    if array.dtype.kind not in "iu":
        raise DataError(f"board must have an integer dtype, got {array.dtype}")
    if width is not None and array.shape[1] != width:
        raise DataError(f"board width {array.shape[1]} does not match expected {width}")
    if height is not None and array.shape[0] != height:
        raise DataError(
            f"board height {array.shape[0]} does not match expected {height}"
        )
    if array.size:
        low, high = int(array.min()), int(array.max())
        if low < FLAGGED or high > MAX_NUMBER:
            raise DataError(
                f"board values must be in [{FLAGGED}, {MAX_NUMBER}], got [{low}, {high}]"
            )
    return array


def encode_board(board: npt.ArrayLike) -> str:
    """Encode an observable board as base64 ``int8`` in row-major ``[y, x]``."""

    array = validate_board(board)
    return base64.b64encode(array.astype(np.int8).tobytes(order="C")).decode("ascii")


def decode_board(encoded: str, *, width: int, height: int) -> npt.NDArray[np.int8]:
    """Decode a base64 board, validating length and observable values."""

    if not isinstance(encoded, str):
        raise DataError("encoded board must be a base64 string")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError) as error:
        raise DataError(f"invalid base64 board: {error}") from error
    expected = width * height
    if len(raw) != expected:
        raise DataError(
            f"board payload is {len(raw)} bytes but {width}x{height} needs {expected}"
        )
    board = np.frombuffer(raw, dtype=np.int8).reshape(height, width).copy()
    return validate_board(board, width=width, height=height)


def action_index(x: int, y: int, *, width: int, height: int) -> int:
    """Row-major action index ``y * width + x`` (matches action_space)."""

    x = _as_int(x, "action x")
    y = _as_int(y, "action y")
    if not (0 <= x < width and 0 <= y < height):
        raise DataError(f"action outside board: ({x}, {y})")
    return y * width + x


def normalize_status(status: Any) -> str:
    """Return a validated :class:`GameStatus` string value."""

    if isinstance(status, GameStatus):
        return status.value
    try:
        return GameStatus(str(status)).value
    except ValueError as error:
        raise DataError(f"invalid game status: {status!r}") from error


def validate_ground_truth(
    ground_truth: Mapping[str, Any] | None, *, width: int, height: int
) -> None:
    """Validate post-game ground truth without ever exposing it as observation."""

    if ground_truth is None:
        return
    if not isinstance(ground_truth, Mapping):
        raise DataError("post_game_ground_truth must be an object")
    if not isinstance(ground_truth.get("captured"), bool):
        raise DataError("post_game_ground_truth requires a boolean 'captured'")
    mines = ground_truth.get("mines")
    if mines is None:
        return
    if not isinstance(mines, Sequence) or isinstance(mines, (str, bytes)):
        raise DataError("post_game_ground_truth 'mines' must be a list of [x, y]")
    seen: set[tuple[int, int]] = set()
    for mine in mines:
        if not isinstance(mine, (list, tuple)) or len(mine) != 2:
            raise DataError("each mine must be an [x, y] pair")
        x, y = _as_int(mine[0], "mine x"), _as_int(mine[1], "mine y")
        action_index(x, y, width=width, height=height)
        if (x, y) in seen:
            raise DataError(f"duplicate mine coordinate: ({x}, {y})")
        seen.add((x, y))


@dataclass(slots=True)
class AgentMetadata:
    """Provenance for agent-generated games (optional for human games)."""

    agent_type: str
    model_version: str | None = None
    checkpoint: str | None = None
    seed: int | None = None

    def validate(self) -> None:
        if not isinstance(self.agent_type, str) or not self.agent_type:
            raise DataError("agent_type must be a non-empty string")
        for name in ("model_version", "checkpoint"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise DataError(f"agent {name} must be a string or null")
        if self.seed is not None:
            _as_int(self.seed, "agent seed")

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentMetadata":
        if not isinstance(data, Mapping):
            raise DataError("agent metadata must be an object")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise DataError(f"unknown agent metadata fields: {sorted(unknown)}")
        if "agent_type" not in data:
            raise DataError("agent metadata requires 'agent_type'")
        metadata = cls(
            agent_type=data["agent_type"],
            model_version=data.get("model_version"),
            checkpoint=data.get("checkpoint"),
            seed=data.get("seed"),
        )
        metadata.validate()
        return metadata


@dataclass(slots=True)
class GameRecord:
    """One complete (or abandoned) recorded game."""

    schema_version: int
    game_id: str
    source: str
    recorded_at: float
    width: int
    height: int
    mines: int | None
    result: str
    total_steps: int
    started_at: float
    ended_at: float
    duration_s: float
    final_opened_cells: int = 0
    other_clicks: int = 0
    first_move: tuple[int, int] | None = None
    difficulty: str | None = None
    agent: AgentMetadata | None = None
    recorder_version: int = RECORDER_VERSION
    env_name: str = ENV_NAME
    ground_truth: dict[str, Any] | None = None
    termination_reason: str | None = None

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise DataError(
                f"unsupported schema_version {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if not isinstance(self.game_id, str) or not self.game_id:
            raise DataError("game_id must be a non-empty string")
        if self.source not in SOURCES:
            raise DataError(f"invalid source: {self.source!r}")
        if self.source == SOURCE_HUMAN and self.agent is not None:
            raise DataError("human games must not carry agent metadata")
        width = _as_int(self.width, "width", minimum=1)
        height = _as_int(self.height, "height", minimum=1)
        if self.mines is not None:
            mines = _as_int(self.mines, "mines", minimum=0)
            if mines > width * height:
                raise DataError("mines cannot exceed the number of cells")
        if self.result not in RESULTS:
            raise DataError(f"invalid result: {self.result!r}")
        _as_int(self.total_steps, "total_steps", minimum=0)
        _as_int(self.final_opened_cells, "final_opened_cells", minimum=0)
        _as_int(self.other_clicks, "other_clicks", minimum=0)
        _as_float(self.recorded_at, "recorded_at")
        started = _as_float(self.started_at, "started_at")
        ended = _as_float(self.ended_at, "ended_at")
        if ended < started:
            raise DataError("ended_at must not precede started_at")
        _as_float(self.duration_s, "duration_s")
        if self.first_move is not None:
            if not isinstance(self.first_move, (list, tuple)) or len(self.first_move) != 2:
                raise DataError("first_move must be an [x, y] pair")
            action_index(self.first_move[0], self.first_move[1], width=width, height=height)
        if self.agent is not None:
            self.agent.validate()
        if self.termination_reason is not None and (
            not isinstance(self.termination_reason, str) or not self.termination_reason
        ):
            raise DataError("termination_reason must be a non-empty string or null")
        validate_ground_truth(self.ground_truth, width=width, height=height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "game_id": self.game_id,
            "source": self.source,
            "recorded_at": self.recorded_at,
            "board": {"width": self.width, "height": self.height, "mines": self.mines},
            "difficulty": self.difficulty,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "first_move": list(self.first_move) if self.first_move else None,
            "result": self.result,
            "total_steps": self.total_steps,
            "final_opened_cells": self.final_opened_cells,
            "other_clicks": self.other_clicks,
            "termination_reason": self.termination_reason,
            "agent": self.agent.to_dict() if self.agent else None,
            "recorder": {"version": self.recorder_version, "env": self.env_name},
            GROUND_TRUTH_KEY: self.ground_truth,
        }

    def safe_view(self) -> dict[str, Any]:
        """Observation-safe game projection: ground truth is always removed."""

        data = self.to_dict()
        data.pop(GROUND_TRUTH_KEY, None)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GameRecord":
        if not isinstance(data, Mapping):
            raise DataError("game record must be an object")
        known = {field.name for field in fields(cls)}
        unknown = set(data) - known - {"board", "recorder", GROUND_TRUTH_KEY}
        if unknown:
            raise DataError(f"unknown game record fields: {sorted(unknown)}")
        required = (
            "schema_version", "game_id", "source", "recorded_at", "board",
            "result", "total_steps", "started_at", "ended_at", "duration_s",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise DataError(f"missing game record fields: {missing}")
        board = data["board"]
        if not isinstance(board, Mapping) or "width" not in board or "height" not in board:
            raise DataError("game record 'board' requires width and height")
        first_move = data.get("first_move")
        if first_move is not None:
            if not isinstance(first_move, (list, tuple)) or len(first_move) != 2:
                raise DataError("first_move must be an [x, y] pair")
            first_move = (first_move[0], first_move[1])
        agent = data.get("agent")
        if agent is not None:
            agent = AgentMetadata.from_dict(agent)
        recorder = data.get("recorder", {})
        if not isinstance(recorder, Mapping):
            raise DataError("'recorder' must be an object")
        record = cls(
            schema_version=data["schema_version"],
            game_id=data["game_id"],
            source=data["source"],
            recorded_at=data["recorded_at"],
            width=board["width"],
            height=board["height"],
            mines=board.get("mines"),
            result=data["result"],
            total_steps=data["total_steps"],
            started_at=data["started_at"],
            ended_at=data["ended_at"],
            duration_s=data["duration_s"],
            final_opened_cells=data.get("final_opened_cells", 0),
            other_clicks=data.get("other_clicks", 0),
            first_move=first_move,
            difficulty=data.get("difficulty"),
            agent=agent,
            recorder_version=recorder.get("version", RECORDER_VERSION),
            env_name=recorder.get("env", ENV_NAME),
            ground_truth=data.get(GROUND_TRUTH_KEY),
            termination_reason=data.get("termination_reason"),
        )
        record.validate()
        return record


@dataclass(slots=True)
class TransitionRecord:
    """One observable state/action/next-state/reward transition."""

    schema_version: int
    game_id: str
    step_id: int
    action_type: str
    x: int
    y: int
    action_index: int
    board_before: npt.NDArray[np.int8]
    board_after: npt.NDArray[np.int8]
    game_status: str
    terminated: bool
    truncated: bool = False
    reward: float | None = None
    opened_cells: int | None = None
    newly_revealed: int | None = None
    timestamp: float | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise DataError(
                f"unsupported schema_version {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if not isinstance(self.game_id, str) or not self.game_id:
            raise DataError("game_id must be a non-empty string")
        self.board_before = np.array(
            validate_board(self.board_before), dtype=np.int8, copy=True
        )
        self.board_after = np.array(
            validate_board(self.board_after), dtype=np.int8, copy=True
        )
        if self.board_before.shape != self.board_after.shape:
            raise DataError("board_before and board_after must share a shape")
        height, width = self.board_before.shape
        self.step_id = _as_int(self.step_id, "step_id", minimum=0)
        self.x = _as_int(self.x, "action x")
        self.y = _as_int(self.y, "action y")
        expected = action_index(self.x, self.y, width=width, height=height)
        self.action_index = _as_int(self.action_index, "action_index")
        if self.action_index != expected:
            raise DataError(
                f"action_index {self.action_index} does not match ({self.x}, {self.y}) "
                f"-> {expected}"
            )
        if self.action_type not in ALL_ACTION_TYPES:
            raise DataError(f"invalid action_type: {self.action_type!r}")
        self.game_status = normalize_status(self.game_status)
        if self.game_status not in _POST_ACTION_STATUSES:
            raise DataError(f"invalid post-action game_status: {self.game_status!r}")
        before = int(self.board_before[self.y, self.x])
        after = int(self.board_after[self.y, self.x])
        if self.action_type in (ACTION_OPEN, ACTION_REVEAL):
            if before == FLAGGED:
                raise DataError("an open action cannot target a flagged cell")
            if before == HIDDEN:
                # A reveal must open the cell, unless it detonated a mine, which
                # the parser reports as a still-hidden cell with status lost.
                if after == HIDDEN and self.game_status != GameStatus.LOST.value:
                    raise DataError("an open action must open the clicked cell")
            elif (
                np.array_equal(self.board_before, self.board_after)
                and self.game_status not in _TERMINAL_STATUSES
            ):
                # An open action on an already-open cell is a chord; it must
                # change at least one neighbouring cell.
                raise DataError("a chord (open on an opened cell) must change a cell")
        elif self.action_type == ACTION_FLAG:
            if before not in (HIDDEN, FLAGGED):
                raise DataError("a flag action must target a hidden or flagged cell")
            if before == after:
                raise DataError(
                    "a flag action must change the clicked cell between hidden and flagged"
                )
        # Legacy ``chord`` records are accepted unchanged for backward
        # compatibility with datasets written before chording became an OPEN.
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise DataError("terminated and truncated must be booleans")
        if self.terminated and self.truncated:
            raise DataError("a transition cannot be both terminated and truncated")
        if self.terminated and self.game_status not in _TERMINAL_STATUSES:
            raise DataError("a terminated transition must end in won or lost")
        if self.game_status in _TERMINAL_STATUSES and not self.terminated:
            raise DataError("a won/lost status requires terminated=True")
        self.reward = _as_optional_float(self.reward, "reward")
        self.opened_cells = (
            None if self.opened_cells is None
            else _as_int(self.opened_cells, "opened_cells", minimum=0)
        )
        self.newly_revealed = (
            None if self.newly_revealed is None
            else _as_int(self.newly_revealed, "newly_revealed", minimum=0)
        )
        self.timestamp = _as_optional_float(self.timestamp, "timestamp")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "game_id": self.game_id,
            "step_id": self.step_id,
            "action_type": self.action_type,
            "action": {"x": self.x, "y": self.y},
            "action_index": self.action_index,
            "board_before": encode_board(self.board_before),
            "board_after": encode_board(self.board_after),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "game_status": self.game_status,
            "opened_cells": self.opened_cells,
            "newly_revealed": self.newly_revealed,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, board_shape: tuple[int, int]
    ) -> "TransitionRecord":
        if not isinstance(data, Mapping):
            raise DataError("transition record must be an object")
        known = {field.name for field in fields(cls)}
        unknown = set(data) - known - {"action"}
        if unknown:
            raise DataError(f"unknown transition record fields: {sorted(unknown)}")
        required = (
            "schema_version", "game_id", "step_id", "action", "action_index",
            "board_before", "board_after", "terminated", "truncated", "game_status",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise DataError(f"missing transition record fields: {missing}")
        height, width = board_shape
        action = data["action"]
        if not isinstance(action, Mapping) or "x" not in action or "y" not in action:
            raise DataError("transition 'action' requires x and y")
        return cls(
            schema_version=data["schema_version"],
            game_id=data["game_id"],
            step_id=data["step_id"],
            action_type=data.get("action_type", ACTION_OPEN),
            x=action["x"],
            y=action["y"],
            action_index=data["action_index"],
            board_before=decode_board(data["board_before"], width=width, height=height),
            board_after=decode_board(data["board_after"], width=width, height=height),
            game_status=data["game_status"],
            terminated=data["terminated"],
            truncated=data["truncated"],
            reward=data.get("reward"),
            opened_cells=data.get("opened_cells"),
            newly_revealed=data.get("newly_revealed"),
            timestamp=data.get("timestamp"),
        )


def validate_transition(
    record: TransitionRecord,
    *,
    width: int,
    height: int,
    previous_board: npt.NDArray[np.int8] | None = None,
) -> None:
    """Validate a transition against board dimensions and the prior board."""

    validate_board(record.board_before, width=width, height=height)
    validate_board(record.board_after, width=width, height=height)
    if previous_board is not None and not np.array_equal(
        record.board_before, previous_board
    ):
        raise DataError(
            "board_before does not continue from the previous board_after"
        )
