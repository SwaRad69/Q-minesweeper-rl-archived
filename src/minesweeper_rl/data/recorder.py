"""High-level trajectory recording for human and agent Minesweeper games.

:class:`TrajectoryRecorder` owns the lifecycle of one game at a time: allocate a
unique ``game_id``, buffer the game's transitions in memory, then commit them to
a :class:`~minesweeper_rl.data.storage.TrajectoryStore` when the game ends.  The
buffer is bounded by the length of a single game (well under a megabyte), so the
dataset itself is never loaded into memory.

:class:`RecordingEnvironment` is the transparent agent-path wrapper: it sits
between the raw browser environment and ``MinesweeperPPOEnv`` (or any other
driver) and records every ``reset``/``step`` without changing behaviour.  Human
play is captured separately by an observer recorder so this system never clicks
the board itself.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import numpy.typing as npt

from ..dom import GameStatus
from .schema import (
    ACTION_OPEN,
    ACTION_TYPES,
    ENV_NAME,
    RECORDER_VERSION,
    RESULT_ABANDONED,
    RESULT_LOST,
    RESULT_WON,
    RESULTS,
    SCHEMA_VERSION,
    SOURCE_AGENT,
    SOURCE_HUMAN,
    SOURCES,
    AgentMetadata,
    DataError,
    GameRecord,
    TransitionRecord,
    encode_board,
    normalize_status,
    validate_board,
    validate_ground_truth,
    validate_transition,
)
from .storage import TrajectoryStore

logger = logging.getLogger(__name__)


class TrajectoryRecorder:
    """Record complete games into a single canonical dataset."""

    def __init__(
        self,
        store: TrajectoryStore,
        *,
        source: str,
        width: int,
        height: int,
        mines: int | None = None,
        difficulty: str | None = None,
        agent: AgentMetadata | None = None,
        env_name: str = ENV_NAME,
    ) -> None:
        if source not in SOURCES:
            raise DataError(f"invalid source: {source!r}")
        if source == SOURCE_HUMAN and agent is not None:
            raise DataError("human recordings must not carry agent metadata")
        if agent is not None and not isinstance(agent, AgentMetadata):
            raise DataError("agent metadata must be an AgentMetadata instance")
        if agent is not None:
            agent.validate()
        width = int(width)
        height = int(height)
        if width < 1 or height < 1:
            raise DataError("board dimensions must be positive")
        if mines is not None and int(mines) < 0:
            raise DataError("mine count cannot be negative")
        self.store = store
        self.source = source
        self.width = width
        self.height = height
        self.mines = None if mines is None else int(mines)
        self.difficulty = difficulty
        self.agent = agent
        self.env_name = env_name
        self.store.configure_board(width=width, height=height, mines=self.mines)
        self._reset()

    def _reset(self) -> None:
        self._game_id: str | None = None
        self._steps: list[TransitionRecord] = []
        self._started_at: float | None = None
        self._first_move: tuple[int, int] | None = None
        self._other_clicks = 0
        self._start_board: npt.NDArray[np.int8] | None = None

    @property
    def active_game_id(self) -> str | None:
        return self._game_id

    @property
    def steps_recorded(self) -> int:
        return len(self._steps)

    @property
    def last_board(self) -> npt.NDArray[np.int8] | None:
        return self._steps[-1].board_after if self._steps else self._start_board

    @property
    def game_terminated(self) -> bool:
        """Whether the active game already recorded a terminal transition."""

        return bool(self._steps and self._steps[-1].terminated)

    def discard_game(self) -> None:
        """Drop the in-progress game without writing any record.

        Used when an action turns out to be invalid or not attributable to a
        real interaction, so a rejected observation can never create a game.
        """

        self._reset()

    def start_game(
        self,
        *,
        first_move: tuple[int, int] | None = None,
        started_at: float | None = None,
        board: npt.ArrayLike | None = None,
    ) -> str:
        if self._game_id is not None:
            raise DataError(
                f"game {self._game_id} is still being recorded; call end_game() first"
            )
        if first_move is not None:
            if len(first_move) != 2:
                raise DataError("first_move must be an (x, y) pair")
            x, y = int(first_move[0]), int(first_move[1])
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise DataError(f"first_move outside board: ({x}, {y})")
            first_move = (x, y)
        if board is not None:
            self._start_board = np.array(
                validate_board(board, width=self.width, height=self.height),
                dtype=np.int8,
                copy=True,
            )
        self._game_id = self.store.next_game_id(self.source)
        self._started_at = time.time() if started_at is None else float(started_at)
        self._first_move = first_move
        return self._game_id

    def record_step(
        self,
        *,
        board_before: npt.ArrayLike,
        action: tuple[int, int],
        board_after: npt.ArrayLike,
        game_status: Any,
        terminated: bool,
        truncated: bool = False,
        reward: float | None = None,
        opened_cells: int | None = None,
        newly_revealed: int | None = None,
        timestamp: float | None = None,
        action_type: str = ACTION_OPEN,
    ) -> TransitionRecord:
        if self._game_id is None:
            raise DataError("start_game() must be called before record_step()")
        if self._steps and self._steps[-1].terminated:
            raise DataError("cannot record a step after a terminal transition")
        if action_type not in ACTION_TYPES:
            raise DataError(f"invalid action_type: {action_type!r}")
        try:
            x, y = action
        except (TypeError, ValueError) as error:
            raise DataError("action must be an (x, y) pair") from error
        before = validate_board(board_before, width=self.width, height=self.height)
        after = validate_board(board_after, width=self.width, height=self.height)
        record = TransitionRecord(
            schema_version=SCHEMA_VERSION,
            game_id=self._game_id,
            step_id=len(self._steps),
            action_type=action_type,
            x=int(x),
            y=int(y),
            action_index=int(y) * self.width + int(x),
            board_before=before,
            board_after=after,
            game_status=game_status,
            terminated=bool(terminated),
            truncated=bool(truncated),
            reward=reward,
            opened_cells=opened_cells,
            newly_revealed=newly_revealed,
            timestamp=timestamp,
        )
        previous = self._steps[-1].board_after if self._steps else None
        validate_transition(
            record, width=self.width, height=self.height, previous_board=previous
        )
        self._steps.append(record)
        return record

    def record_other_click(self) -> None:
        if self._game_id is None:
            raise DataError("no active game to attach the click to")
        self._other_clicks += 1

    def end_game(
        self,
        result: str,
        *,
        ground_truth: dict[str, Any] | None = None,
        ended_at: float | None = None,
        final_opened_cells: int | None = None,
        termination_reason: str | None = None,
    ) -> GameRecord:
        if self._game_id is None:
            raise DataError("no game is being recorded")
        if result not in RESULTS:
            raise DataError(f"invalid result: {result!r}")
        validate_ground_truth(ground_truth, width=self.width, height=self.height)
        ended = time.time() if ended_at is None else float(ended_at)
        terminated_index = [
            index for index, step in enumerate(self._steps) if step.terminated
        ]
        if terminated_index and terminated_index != [len(self._steps) - 1]:
            raise DataError("a terminal transition must be the final step of a game")
        if result in (RESULT_WON, RESULT_LOST):
            if not self._steps or not self._steps[-1].terminated:
                raise DataError(f"result {result!r} requires a terminal final step")
            if self._steps[-1].game_status != result:
                raise DataError(
                    f"result {result!r} contradicts terminal status "
                    f"{self._steps[-1].game_status!r}"
                )
        elif result == RESULT_ABANDONED and terminated_index:
            raise DataError("an abandoned game cannot contain a terminal transition")
        if final_opened_cells is None:
            board = self.last_board
            # Only revealed numbers (0..8) are open; FLAGGED (-2) and HIDDEN
            # (-1) are both covered.
            final_opened_cells = 0 if board is None else int((board >= 0).sum())
        started = self._started_at if self._started_at is not None else ended
        record = GameRecord(
            schema_version=SCHEMA_VERSION,
            game_id=self._game_id,
            source=self.source,
            recorded_at=ended,
            width=self.width,
            height=self.height,
            mines=self.mines,
            result=result,
            total_steps=len(self._steps),
            started_at=started,
            ended_at=ended,
            duration_s=ended - started,
            final_opened_cells=int(final_opened_cells),
            other_clicks=self._other_clicks,
            first_move=self._first_move,
            difficulty=self.difficulty,
            agent=self.agent,
            recorder_version=RECORDER_VERSION,
            env_name=self.env_name,
            ground_truth=ground_truth,
            termination_reason=termination_reason,
        )
        record.validate()
        self.store.append_transitions(self._steps, width=self.width, height=self.height)
        self.store.append_game(record)
        self._reset()
        return record


class RecordingEnvironment:
    """Transparent wrapper that records trajectories from a coordinate environment.

    Insert it under the flat-action adapter::

        MinesweeperPPOEnv(RecordingEnvironment(environment, recorder))

    It observes only -- the underlying environment still performs the actions.
    """

    def __init__(self, environment: Any, recorder: TrajectoryRecorder) -> None:
        self.environment = environment
        self.recorder = recorder
        self.config = getattr(environment, "config", None)
        self._board: npt.NDArray[np.int8] | None = None

    def reset(self) -> npt.NDArray[np.int8]:
        if self.recorder.active_game_id is not None:
            self.recorder.end_game(RESULT_ABANDONED)
        self.recorder.start_game(first_move=self._first_move())
        board = self.environment.reset()
        self._board = np.array(board, dtype=np.int8, copy=True)
        return board

    def step(
        self, action: tuple[int, int]
    ) -> tuple[npt.NDArray[np.int8], float, bool, bool, dict[str, Any]]:
        if self._board is None:
            raise RuntimeError("reset() must be called before step()")
        x, y = action
        board_before = self._board
        board, reward, terminated, truncated, info = self.environment.step(
            (int(x), int(y))
        )
        status = info.get("game_status")
        self.recorder.record_step(
            board_before=board_before,
            action=(int(x), int(y)),
            board_after=board,
            game_status=status,
            terminated=terminated,
            truncated=truncated,
            reward=float(reward),
            opened_cells=info.get("opened_cells"),
            newly_revealed=info.get("newly_revealed"),
            timestamp=time.time(),
        )
        self._board = np.array(board, dtype=np.int8, copy=True)
        if terminated or truncated:
            status_value = normalize_status(status)
            if status_value == GameStatus.WON.value:
                result = RESULT_WON
            elif status_value == GameStatus.LOST.value:
                result = RESULT_LOST
            else:
                result = RESULT_ABANDONED
            self.recorder.end_game(result)
        return board, reward, terminated, truncated, info

    def _first_move(self) -> tuple[int, int] | None:
        config = self.config
        if config is None:
            return None
        move = getattr(config, "first_move", None)
        if move is not None:
            return (int(move[0]), int(move[1]))
        width = getattr(config, "width", None)
        height = getattr(config, "height", None)
        if width and height:
            return (int(width) // 2, int(height) // 2)
        return None

    def __getattr__(self, name: str) -> Any:
        if name == "environment":
            raise AttributeError(name)
        return getattr(self.environment, name)
