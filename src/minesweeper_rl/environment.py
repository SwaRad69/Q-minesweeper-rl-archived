"""Environment facade for the real Minesweeper website (OPEN, FLAG, CHORD).

``step((x, y))`` keeps the original OPEN-only behaviour.  ``apply_action``
dispatches the three human-like operations and is what the flag-aware RL
adapter uses.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

from .action_types import (
    ACTION_CHORD,
    ACTION_FLAG,
    ACTION_NAMES,
    ACTION_OPEN,
    ACTION_TYPES,
)
from .actions import ActionHandler, ActionTimeoutError, InvalidActionError
from .config import EnvironmentConfig
from .dom import (
    FACE_SELECTOR,
    FLAGGED,
    HIDDEN,
    MAX_NUMBER,
    BoardMissingError,
    BoardShapeError,
    CellFormatError,
    GameStatus,
    chord_is_legal,
    classify_game_status,
    parse_board_snapshot,
    read_cell_snapshot,
)
from .metrics import EnvironmentMetrics, timed

logger = logging.getLogger(__name__)


#: PRIMARY reward unit: one point per genuinely revealed (opened) cell.
SAFE_REVEAL_REWARD = 1.0
MINE_REWARD = -10.0
WIN_REWARD = 100.0
#: FLAG/UNFLAG is neutral: it neither reveals a cell nor depends on mine
#: correctness, so it never affects the reward.
FLAG_REWARD = 0.0


def revealed_cells(board: npt.ArrayLike) -> int:
    """Count genuinely revealed cells (``board >= 0``).

    A covered cell (``HIDDEN``, ``-1``) is never counted.
    """

    return int(np.count_nonzero(np.asarray(board) >= 0))


def transition_reward(
    board_before: npt.ArrayLike,
    board_after: npt.ArrayLike,
    status: GameStatus | None,
) -> float:
    """Return the approved PRIMARY reward for one environment transition.

    ``r = progress + terminal`` where ``progress`` is the change in the number
    of genuinely revealed cells and ``terminal`` is :data:`WIN_REWARD` on a win
    or :data:`MINE_REWARD` on a loss.

    The hidden mine layout is **never** consulted here: the only reward term
    that depends on it is the existing terminal loss signal, which is triggered
    by the game ending, not by any observable cell.
    """

    progress = SAFE_REVEAL_REWARD * (
        revealed_cells(board_after) - revealed_cells(board_before)
    )
    if status is GameStatus.LOST:
        return progress + MINE_REWARD
    if status is GameStatus.WON:
        return progress + WIN_REWARD
    return progress


class _ActionResult(NamedTuple):
    """Internal before/after DOM and board state for one action."""

    before_snapshot: list[dict[str, Any]]
    before_board: npt.NDArray[np.int8]
    after_snapshot: list[dict[str, Any]]
    after_board: npt.NDArray[np.int8]


def _opened_count(snapshot: Sequence[dict[str, Any]]) -> int:
    return sum(
        1
        for cell in snapshot
        if any(str(name).endswith("opened") for name in cell.get("classes", ()))
    )


def _fingerprint(snapshot: Sequence[dict[str, Any]]) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Return an order-independent identity for a DOM snapshot.

    Comparing fingerprints rather than raw lists keeps the stability check
    deterministic even if the browser reports attributes or classes in a
    different order between reads.
    """

    return tuple(
        sorted(
            (
                str(cell.get("x")),
                str(cell.get("y")),
                tuple(sorted(str(name) for name in cell.get("classes", ()))),
            )
            for cell in snapshot
        )
    )


class MinesweeperEnvironment:
    """Expose browser observation and OPEN actions without an RL algorithm."""

    def __init__(self, page: Page | object, config: EnvironmentConfig | None = None) -> None:
        self.config = config or EnvironmentConfig()
        self.page = page.get_page() if hasattr(page, "get_page") else page
        self._shape = (self.config.height, self.config.width)
        self._expected_cells = self.config.width * self.config.height
        self._actions = ActionHandler(
            self.page, selector=self.config.cell_selector,
            width=self.config.width, height=self.config.height,
        )
        self.metrics: EnvironmentMetrics | None = None

    def _timer(self, name: str) -> AbstractContextManager[None]:
        """Return a timing context manager for ``name`` or a no-op."""

        if self.metrics is None:
            return timed(None)
        return timed(getattr(self.metrics, name))

    def board_exists(self) -> bool:
        """Return whether the DOM currently contains at least one board cell."""

        return len(self._read_snapshot()) > 0

    def wait_until_ready(self, timeout_ms: float | None = None) -> None:
        """Wait until the configured number of DOM cells exists.

        ``timeout_ms`` overrides ``ready_timeout_ms``; reset passes the longer
        new-game timeout while a freshly navigated page loads its board.
        """

        timeout = self.config.ready_timeout_ms if timeout_ms is None else timeout_ms
        self.page.wait_for_function(
            "([selector, count]) => document.querySelectorAll(selector).length >= count",
            arg=[self.config.cell_selector, self._expected_cells],
            timeout=timeout,
        )

    def observe(self) -> npt.NDArray[np.int8]:
        """Return one complete, stable DOM board or raise a validation error."""

        with self._timer("observe"):
            return self._stable_state()[1]

    def reset(self) -> npt.NDArray[np.int8]:
        """Start a fresh game and return the board after a controlled first move.

        The restart control (``new_game_selector``) was verified on the live
        site to start a new game by navigating to a fresh ``/game/<id>`` URL, so
        this waits for that navigation (or an in-place reset), a fresh stable
        all-hidden board, and then performs one deterministic first reveal.

        The controlled first move exists so the agent never has to make an
        arbitrary opening click and can never lose during ``reset()``: the site
        guarantees the first reveal is never a mine.  The coordinate defaults
        to the board centre and can be pinned with ``EnvironmentConfig.first_move``.
        If the site's guarantee ever fails, the game is restarted and the same
        deterministic cell is retried up to ``first_move_attempts`` times.
        """

        with self._timer("reset"):
            self._start_new_game()
            return self._controlled_first_move()

    def step(
        self, action: tuple[int, int]
    ) -> tuple[npt.NDArray[np.int8], float, bool, bool, dict[str, Any]]:
        """Open one cell and return ``(next_state, reward, terminated, truncated, info)``.

        ``action`` is an ``(x, y)`` pair naming a currently hidden cell.  This is
        the OPEN operation; :meth:`apply_action` dispatches OPEN, FLAG and CHORD.

        Returns the revealed-cell progress change (one point per newly revealed
        cell, so a cascade of ``k`` cells scores ``+k``) plus the terminal bonus:
        :data:`WIN_REWARD` on a win and :data:`MINE_REWARD` on a loss.  Hidden
        mine locations are never used to score a reveal beyond the terminal
        loss already triggered by hitting one.  ``terminated`` is true only when
        the status is WON or LOST; ``truncated`` is currently always false
        because there is no step or time limit yet.

        ``info`` carries ``action``, ``action_type``, ``operation``,
        ``previous_value``, ``new_value``, ``game_status``, ``hit_mine``,
        ``newly_revealed``, ``opened_cells``, and ``elapsed_ms`` for debugging.
        """

        x, y = self._parse_action(action)
        return self.apply_action(ACTION_OPEN, x, y)

    def apply_action(
        self, action_type: int, x: int, y: int
    ) -> tuple[npt.NDArray[np.int8], float, bool, bool, dict[str, Any]]:
        """Dispatch one OPEN/FLAG/CHORD operation on cell ``(x, y)``.

        Legality is checked against the observable board before any click:

        * OPEN requires a hidden (unflagged) cell;
        * FLAG requires a covered cell (hidden or already flagged) to toggle;
        * CHORD requires an opened number whose flagged neighbours already
          equal the number and that still has a hidden neighbour.

        Invalid targets raise :class:`~minesweeper_rl.actions.InvalidActionError`
        without touching the browser.  FLAG is rewarded neutrally
        (:data:`FLAG_REWARD`); OPEN and CHORD share the revealed-cell progress
        reward plus the terminal bonus.
        """

        with self._timer("step"):
            return self._apply_action(action_type, x, y)

    def reveal(self, x: int | tuple[int, int], y: int | None = None) -> npt.NDArray[np.int8]:
        """Open one hidden cell and return the resulting stable board.

        ``action`` may be passed as ``reveal(x, y)`` or ``reveal((x, y))``.  A
        cell that is not currently hidden (already open or flagged) is rejected
        before any click so it can never be submitted as a valid RL action.  The
        cell is clicked with Playwright's trusted mouse interaction and, if that
        produces no board change, the verified DOM event sequence is used.
        """

        if y is None:
            x, y = self._parse_action(x)
        return self._execute_action(ACTION_OPEN, x, y).after_board

    def flag(self, x: int | tuple[int, int], y: int | None = None) -> npt.NDArray[np.int8]:
        """Toggle a flag on one covered cell and return the stable board.

        A hidden cell gains a flag and a flagged cell loses it.  Open cells are
        rejected.  The right-click is a browser-only interaction with no
        revealed-cell effect, so it carries no reward.
        """

        if y is None:
            x, y = self._parse_action(x)
        return self._execute_action(ACTION_FLAG, x, y).after_board

    def chord(self, x: int | tuple[int, int], y: int | None = None) -> npt.NDArray[np.int8]:
        """Chord an opened numbered cell and return the resulting stable board.

        A chord is a left-click on an opened number whose flagged neighbours
        equal the number; it opens the remaining hidden neighbours.  It is only
        accepted when :func:`minesweeper_rl.dom.chord_is_legal` holds, so the
        hidden mine layout is never consulted.
        """

        if y is None:
            x, y = self._parse_action(x)
        return self._execute_action(ACTION_CHORD, x, y).after_board

    def left_click(self, x: int, y: int) -> None:
        """Perform one left-click action without waiting for a board change."""

        self._actions.left_click(x, y)

    def _apply_action(
        self, action_type: int, x: int, y: int
    ) -> tuple[npt.NDArray[np.int8], float, bool, bool, dict[str, Any]]:
        if action_type not in ACTION_TYPES:
            raise InvalidActionError(f"unknown action type: {action_type!r}")
        started = time.perf_counter()
        result = self._execute_action(action_type, x, y, require_playable=True)
        return self._finish_transition(result, action_type, x, y, started)

    def _finish_transition(
        self,
        result: _ActionResult,
        action_type: int,
        x: int,
        y: int,
        started: float,
    ) -> tuple[npt.NDArray[np.int8], float, bool, bool, dict[str, Any]]:
        """Build the common ``(board, reward, terminated, truncated, info)``."""

        after_board = result.after_board
        after_opened = _opened_count(result.after_snapshot)
        status = classify_game_status(
            self._read_face_classes(), opened_cells=after_opened
        )

        # PRIMARY reward: revealed-cell progress plus any terminal bonus for
        # OPEN and CHORD.  FLAG/UNFLAG is neutral: it reveals nothing and never
        # depends on whether the flag matches a hidden mine.
        if action_type == ACTION_FLAG:
            reward = FLAG_REWARD
        else:
            reward = transition_reward(result.before_board, after_board, status)
        newly_revealed = max(
            0, revealed_cells(after_board) - revealed_cells(result.before_board)
        )

        terminated = status in (GameStatus.WON, GameStatus.LOST)
        operation = ACTION_NAMES[action_type]
        info: dict[str, Any] = {
            "action": (x, y),
            "action_type": action_type,
            "operation": operation,
            "previous_value": int(result.before_board[y, x]),
            "new_value": int(after_board[y, x]),
            "game_status": status,
            "hit_mine": status is GameStatus.LOST,
            "newly_revealed": newly_revealed,
            "opened_cells": after_opened,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }

        if self.config.diagnostics:
            logger.debug(
                "%s (%d, %d): reward=%.1f status=%s newly_revealed=%d terminated=%s",
                operation, x, y, reward, status.value,
                info["newly_revealed"], terminated,
            )
        return after_board, reward, terminated, False, info

    def get_game_status(self) -> GameStatus:
        """Return READY, RUNNING, WON, or LOST from observable DOM state only.

        A stable cell snapshot is read first, then win/loss is taken from the
        status face.  A win or loss always changes the revealed cells, so the
        face cannot be mid-change while the board is stable.
        """

        snapshot, _board = self._stable_state()
        face_classes = self._read_face_classes()
        if not face_classes:
            raise RuntimeError("Minesweeper status face is temporarily unavailable")
        status = classify_game_status(
            face_classes, opened_cells=_opened_count(snapshot)
        )
        if self.config.diagnostics:
            logger.debug(
                "game status: %s (%d cells open)", status.value, _opened_count(snapshot)
            )
        return status

    def _execute_action(
        self, action_type: int, x: int, y: int, *, require_playable: bool = False
    ) -> _ActionResult:
        """Validate, click, wait, and return the before/after board details."""

        self._actions.validate(x, y)
        before_snapshot, before_board = self._stable_state()
        if require_playable:
            before_status = classify_game_status(
                self._read_face_classes(), opened_cells=_opened_count(before_snapshot)
            )
            if before_status in (GameStatus.WON, GameStatus.LOST):
                raise InvalidActionError(
                    f"episode is already over ({before_status.value}); call reset()"
                )
        self._validate_target(action_type, before_board, x, y)

        before_opened = _opened_count(before_snapshot)
        if self.metrics is not None:
            self.metrics.clicks += 1
        if self.config.diagnostics:
            logger.debug(
                "%s (%d, %d); %d cells open before",
                ACTION_NAMES[action_type], x, y, before_opened,
            )

        try:
            with self._timer("click"):
                self._perform_click(action_type, x, y)
            after_snapshot, after_board = self._wait_for_change(before_board, before_opened)
        except ActionTimeoutError:
            if self.config.diagnostics:
                logger.debug("no change after Playwright click; using event sequence")
            if self.metrics is not None:
                self.metrics.event_fallbacks += 1
            with self._timer("click"):
                self._click_fallback(action_type, x, y)
            after_snapshot, after_board = self._wait_for_change(before_board, before_opened)
        except Exception as error:  # noqa: BLE001 - any click failure falls back
            if self.config.diagnostics:
                logger.debug("Playwright click failed (%s); using event sequence", error)
            if self.metrics is not None:
                self.metrics.event_fallbacks += 1
            with self._timer("click"):
                self._click_fallback(action_type, x, y)
            after_snapshot, after_board = self._wait_for_change(before_board, before_opened)

        return _ActionResult(
            before_snapshot, before_board, after_snapshot, after_board
        )

    @staticmethod
    def _validate_target(
        action_type: int, board: npt.NDArray[np.int8], x: int, y: int
    ) -> None:
        """Reject an action whose target is not legal on the current board."""

        value = int(board[y, x])
        if action_type == ACTION_OPEN:
            if value != HIDDEN:
                raise InvalidActionError(
                    f"cell ({x}, {y}) is not hidden (already revealed or flagged)"
                )
        elif action_type == ACTION_FLAG:
            if value not in (HIDDEN, FLAGGED):
                raise InvalidActionError(
                    f"cell ({x}, {y}) is opened and cannot be flagged"
                )
        elif action_type == ACTION_CHORD:
            if not 1 <= value <= MAX_NUMBER:
                raise InvalidActionError(
                    f"chord requires an opened numbered cell at ({x}, {y})"
                )
            if not chord_is_legal(board, x, y):
                raise InvalidActionError(
                    f"chord at ({x}, {y}) is not currently legal"
                )
        else:
            raise InvalidActionError(f"unknown action type: {action_type!r}")

    def _perform_click(self, action_type: int, x: int, y: int) -> None:
        """Trusted mouse interaction: left for OPEN/CHORD, right for FLAG."""

        if action_type == ACTION_FLAG:
            self._actions.right_click(x, y)
        else:
            self._actions.click(x, y)

    def _click_fallback(self, action_type: int, x: int, y: int) -> None:
        """Synthetic DOM event sequence for when the trusted click is ignored."""

        if action_type == ACTION_FLAG:
            self._actions.right_click_events(x, y)
        else:
            self._actions.left_click(x, y)

    @staticmethod
    def _parse_action(action: object) -> tuple[int, int]:
        try:
            x, y = action  # type: ignore[misc]
        except (TypeError, ValueError) as error:
            raise TypeError("action must be an (x, y) pair") from error
        return x, y

    def _start_new_game(self) -> None:
        """Click the verified restart control and wait for a fresh board."""

        try:
            self.wait_until_ready(self.config.new_game_timeout_ms)
        except PlaywrightTimeoutError as error:
            raise RuntimeError(
                "Cannot reset: no Minesweeper board is available to restart"
            ) from error

        control = self.page.locator(self.config.new_game_selector)
        try:
            control.first.wait_for(
                state="attached", timeout=self.config.new_game_timeout_ms
            )
        except PlaywrightTimeoutError as error:
            raise RuntimeError(
                f"new-game control {self.config.new_game_selector!r} did not appear"
            ) from error

        if self.config.diagnostics:
            logger.debug("restarting game from %s", self.page.url)
        control.first.click()
        self._wait_for_fresh_board()

    def _wait_for_fresh_board(self) -> None:
        """Wait for a stable board with no open cells and no win/loss state.

        Clicking the restart control resets in place when the board is already
        untouched but navigates to a new game once play has started; polling the
        board itself covers both without a fixed delay.  Reads racing with the
        navigation are treated as "not ready yet" and retried until the
        deadline.
        """

        deadline = time.monotonic() + self.config.new_game_timeout_ms / 1000
        last_error: Exception | None = None
        while time.monotonic() <= deadline:
            try:
                snapshot, _board = self._stable_state()
                opened = _opened_count(snapshot)
                status = classify_game_status(
                    self._read_face_classes(), opened_cells=opened
                )
            except Exception as error:  # noqa: BLE001 - page may be mid-navigation
                last_error = error
                self._pause()
                continue
            if opened == 0 and status is GameStatus.READY:
                if self.config.diagnostics:
                    logger.debug("fresh board ready with %d cells", len(snapshot))
                return
            self._pause()

        raise RuntimeError(
            "new game did not reach a fresh board within "
            f"{self.config.new_game_timeout_ms:g} ms"
            + (f" (last error: {last_error})" if last_error else "")
        )

    def _controlled_first_move(self) -> npt.NDArray[np.int8]:
        """Reveal one deterministic safe cell so reset never starts hidden.

        The live site guarantees the first reveal is never a mine; the same
        centre cell is retried on a fresh board if that guarantee ever fails.
        """

        x, y = self._first_move_coordinate()
        attempts = max(1, self.config.first_move_attempts)
        for attempt in range(1, attempts + 1):
            board = self.reveal(x, y)
            if self.get_game_status() is not GameStatus.LOST:
                if self.config.diagnostics:
                    logger.debug("controlled first move at (%d, %d)", x, y)
                return board
            if self.config.diagnostics:
                logger.debug(
                    "unexpected loss after controlled first move at (%d, %d), "
                    "attempt %d of %d", x, y, attempt, attempts,
                )
            if attempt < attempts:
                self._start_new_game()

        raise RuntimeError(
            f"controlled first move at ({x}, {y}) lost {attempts} time(s); "
            "the site's first-click guarantee appears to have changed"
        )

    def _first_move_coordinate(self) -> tuple[int, int]:
        if self.config.first_move is not None:
            return self.config.first_move
        return (self.config.width // 2, self.config.height // 2)

    def _stable_state(self) -> tuple[list[dict[str, Any]], npt.NDArray[np.int8]]:
        """Read a complete, stable DOM snapshot and its parsed board.

        Each pass performs a single atomic DOM read, validates it, and only
        returns after two consecutive reads match.  This rejects a board that
        is still being inserted or updated without relying on a fixed sleep.
        A snapshot that stays malformed once the full cell count is present is
        raised immediately; partial boards are retried until the timeout.
        """

        deadline = time.monotonic() + self.config.observation_timeout_ms / 1000
        previous: tuple[tuple[str, str, tuple[str, ...]], ...] | None = None
        last_error: ValueError | None = None
        seen_cells = 0
        attempt = 0

        while time.monotonic() <= deadline:
            attempt += 1
            snapshot = self._read_snapshot()
            seen_cells = max(seen_cells, len(snapshot))
            fingerprint = _fingerprint(snapshot)
            if self.config.diagnostics:
                logger.debug(
                    "observe attempt %d: %d of %d cells present",
                    attempt, len(snapshot), self._expected_cells,
                )

            try:
                with self._timer("parse"):
                    board = parse_board_snapshot(snapshot, expected_shape=self._shape)
            except BoardMissingError as error:
                last_error = error
                previous = None
                if self.config.diagnostics:
                    logger.debug("no board cells present yet")
                self._pause()
                continue
            except (BoardShapeError, CellFormatError) as error:
                last_error = error
                stable = fingerprint == previous and len(snapshot) == self._expected_cells
                previous = fingerprint
                if self.config.diagnostics:
                    logger.debug("rejected %d-cell snapshot: %s", len(snapshot), error)
                if stable:
                    raise
                self._pause()
                continue

            if fingerprint == previous:
                if self.config.diagnostics:
                    logger.debug("accepted stable board after %d attempt(s)", attempt)
                return snapshot, board

            previous = fingerprint
            self._pause()

        if last_error is not None:
            raise last_error
        raise RuntimeError(
            "Minesweeper board did not provide two identical complete DOM snapshots "
            f"(saw up to {seen_cells} of {self._expected_cells} cells)"
        )

    def _wait_for_change(
        self,
        before_board: npt.NDArray[np.int8],
        before_opened: int,
    ) -> tuple[list[dict[str, Any]], npt.NDArray[np.int8]]:
        """Wait until the board changes in a meaningful way after a click.

        Change means a different numeric board (revealed numbers) or a change
        in the number of revealed cells (which also captures an exploded mine,
        whose value is reported as hidden).  The wait polls stable snapshots
        until ``action_timeout_ms`` instead of sleeping for a fixed duration.
        Returns the stable snapshot and board that first showed the change.
        """

        deadline = time.monotonic() + self.config.action_timeout_ms / 1000
        last_error: Exception | None = None
        while time.monotonic() <= deadline:
            try:
                snapshot, board = self._stable_state()
            except (ValueError, RuntimeError) as error:
                last_error = error
                time.sleep(self.config.action_poll_ms / 1000)
                continue
            if not np.array_equal(board, before_board) or _opened_count(snapshot) != before_opened:
                if self.config.diagnostics:
                    logger.debug(
                        "board changed: %d cells open", _opened_count(snapshot)
                    )
                return snapshot, board
            time.sleep(self.config.action_poll_ms / 1000)

        message = (
            f"no observable board change within {self.config.action_timeout_ms:g} ms"
        )
        if last_error is not None:
            message = f"{message} (last observation error: {last_error})"
        raise ActionTimeoutError(message)

    def _read_snapshot(self) -> list[dict[str, Any]]:
        with self._timer("dom_read"):
            return read_cell_snapshot(self.page, selector=self.config.cell_selector)

    def _read_face_classes(self) -> list[str]:
        return self.page.evaluate(
            f"() => {{ const face = document.querySelector('{FACE_SELECTOR}');"
            " return face ? [...face.classList] : []; }"
        )

    def _pause(self) -> None:
        time.sleep(self.config.observation_poll_ms / 1000)
