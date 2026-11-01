"""Observe a human playing Minesweeper and produce canonical transitions.

This module provides :class:`HumanTrajectoryRecorder`, the **observer** used by
the canonical human recorder (:mod:`minesweeper_rl.data.record_human_dataset`,
writing the FLAGS-only recorder-v3 ``data/human/`` dataset).  It is a pure
OBSERVER: it never clicks the board itself.  A small
capture-phase DOM listener (injected with ``page.add_init_script`` so it
survives the site's new-game navigations) records, on the earliest pointer
event, which cell was pressed, which mouse button was used and a diagnostic
snapshot of the cell classes.  Python polls those events and appends one
canonical transition per press: a left-click is an OPEN action (a left-click on
an already-open numbered cell is a chord, still OPEN) and a right-click is a
FLAG action.  ``board_before``/``board_after`` distinguish placing a flag from
removing one, so a FLAG both toggles the clicked cell and is confirmed from the
stable board.

Synchronization model
---------------------
The recorder keeps an authoritative, continuously observed **stable** board
(``_stable_board``) using the environment's existing two-snapshot stability
logic (``MinesweeperEnvironment.observe``).  A human action is applied to that
stable board, not to the one-shot snapshot taken inside the browser event:

    stable board_before
            |  (one real human press)
            v
    wait until the board differs AND stops changing (quiescence)
            v
    stable board_after  -- becomes the board_before of the next action

The next captured press bounds the previous action with its atomic pre-click
snapshot. Unavailable DOM reads remain pending. Unattributable changes suspend
the transcript until a verified boundary; they do not declare abandonment.

Ground-truth mines are never written into transitions; the canonical game
record never carries a hidden mine layout.

Record with the canonical entry point instead::

    $env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
    python -m minesweeper_rl.data.record_human_dataset --dataset data/human
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from playwright.sync_api import Error as PlaywrightError

from ..dom import (
    FLAGGED,
    HIDDEN,
    GameStatus,
    classify_game_status,
    parse_board_snapshot,
)
from ..environment import transition_reward
from .schema import (
    ACTION_FLAG,
    ACTION_OPEN,
    RESULT_ABANDONED,
    RESULT_LOST,
    RESULT_WON,
    SOURCE_HUMAN,
    DataError,
)
from .storage import StorageError, _is_transient_lock_error

logger = logging.getLogger(__name__)

#: A flag does not move the player toward a win or a loss, so it is neutral.
FLAG_REWARD = 0.0

#: Explicit game lifecycle.  A new game may only be created from a verified
#: fresh board in the READY phase, never from an arbitrary click on a terminal
#: or mid-game board.
PHASE_WAITING = "waiting_for_fresh_board"
PHASE_READY = "ready"
PHASE_RECORDING = "recording"
PHASE_TERMINAL = "terminal"
PHASE_RESYNCING = "resyncing"


def _cell_state_name(value: int) -> str:
    if value == FLAGGED:
        return "flagged"
    if value == HIDDEN:
        return "hidden"
    return f"open:{int(value)}"

# ``pointerdown`` fires before ``mousedown``; registering on ``window`` in the
# capture phase (from an init script, which runs before page scripts) makes our
# snapshot as close to the true pre-click board as the browser allows.
_CLICK_CAPTURE_SCRIPT = r"""
(() => {
  if (window.__msrlInstalled) { return; }
  window.__msrlInstalled = true;
  window.__msrlEvents = window.__msrlEvents || [];
  window.__msrlEventSeq = window.__msrlEventSeq || 0;
  window.__msrlToken = window.__msrlToken
    || Math.random().toString(36).slice(2);
  const snapshot = () => Array.from(
    document.querySelectorAll('.cell[data-x][data-y]')
  ).map((cell) => ({
    x: cell.dataset.x,
    y: cell.dataset.y,
    classes: Array.from(cell.classList),
  }));
  const faceClasses = () => {
    const face = document.querySelector('#top_area_face');
    return face ? Array.from(face.classList) : [];
  };
  const handler = (event) => {
    const target = event.target;
    const cell = target && target.closest
      ? target.closest('.cell[data-x][data-y]')
      : null;
    if (!cell) { return; }
    window.__msrlEventSeq += 1;
    window.__msrlEvents.push({
      id: window.__msrlEventSeq,
      token: window.__msrlToken,
      capturedAt: Date.now() / 1000,
      type: event.type,
      isTrusted: event.isTrusted,
      button: event.button,
      x: cell.dataset.x,
      y: cell.dataset.y,
      clientX: event.clientX,
      clientY: event.clientY,
      tag: target && target.tagName ? target.tagName : null,
      targetClasses: target && target.classList ? Array.from(target.classList) : [],
      faceClasses: faceClasses(),
      cells: snapshot(),
    });
  };
  const eventName = window.PointerEvent ? 'pointerdown' : 'mousedown';
  window.addEventListener(eventName, handler, true);
})();
"""

_DRAIN_EVENTS = """ack => {
  // Delete only events Python confirmed receiving on an earlier call. If this
  // evaluation's reply is lost, these events are returned again on retry.
  const events = window.__msrlEvents || [];
  if (ack && ack.token === window.__msrlToken) {
    while (events.length && events[0].id <= ack.id) { events.shift(); }
  }
  return events.slice();
}"""


@dataclass(slots=True)
class HumanRecordingReport:
    """Final totals for one human recording session."""

    games: int = 0
    transitions: int = 0
    wins: int = 0
    losses: int = 0
    abandoned: int = 0
    incomplete: int = 0
    other_clicks: int = 0
    errors: list[str] = field(default_factory=list)

    def format(self) -> str:
        return (
            f"games={self.games} transitions={self.transitions} "
            f"wins={self.wins} losses={self.losses} abandoned={self.abandoned} "
            f"incomplete={self.incomplete}"
        )


@dataclass(slots=True)
class _PendingAction:
    """A captured press whose DOM effect was temporarily unavailable.

    The old recorder forgot the press after ``action_timeout``.  When the DOM
    became readable again, its (legitimate) late board change then looked like
    an unattributed game boundary and the active game was abandoned.  Keeping
    the press lets an idle observation or the next press's pre-action snapshot
    finish the original transition without inventing an action.
    """

    event: dict[str, Any]
    x: int
    y: int
    action_type: str
    board_before: npt.NDArray[np.int8]


class HumanTrajectoryRecorder:
    """Observe a human playing through an existing Minesweeper environment."""

    def __init__(
        self,
        environment: Any,
        recorder: TrajectoryRecorder,
        *,
        poll_interval: float = 0.05,
        action_timeout: float = 5.0,
        quiescence_time: float = 0.3,
        mine_grace: float = 1.0,
        diagnostics: bool = False,
        emit: Callable[[str], None] | None = None,
    ) -> None:
        if recorder.source != SOURCE_HUMAN:
            raise DataError("HumanTrajectoryRecorder requires a human-source recorder")
        self.environment = environment
        self.recorder = recorder
        self.poll_interval = float(poll_interval)
        self.action_timeout = float(action_timeout)
        self.quiescence_time = float(quiescence_time)
        self.mine_grace = float(mine_grace)
        self.diagnostics = bool(diagnostics)
        self.emit = emit or logger.info
        self.report = HumanRecordingReport()
        # Authoritative stable board; always the basis for the next action.
        self._stable_board: npt.NDArray[np.int8] | None = None
        # The verified all-hidden board a new game must start from.  A new game
        # is never created from an arbitrary click.
        self._fresh_board: npt.NDArray[np.int8] | None = None
        self._phase: str = PHASE_WAITING
        self._pending_idle: npt.NDArray[np.int8] | None = None
        self._seen_events: set[tuple[str | None, int]] = set()
        # Presses captured while an earlier action was still settling.  They are
        # handed back to the run loop in order so their board effects are never
        # folded into the earlier action.
        self._deferred_events: list[dict[str, Any]] = []
        self._pending_action: _PendingAction | None = None
        self._transient_warning_at: dict[str, float] = {}
        self._capture_error: str | None = None
        self._pending_finalization: tuple[str, float, str | None] | None = None
        self._stop_event: threading.Event | None = None
        self._closed_reason: str | None = None
        self._event_ack: dict[str, Any] | None = None
        self._last_observation_at: float | None = None

    @property
    def width(self) -> int:
        return self.recorder.width

    @property
    def height(self) -> int:
        return self.recorder.height

    def install_capture(self) -> None:
        page = self.environment.page
        def closed(reason):
            self._closed_reason = reason
        if hasattr(page, "on"):
            page.on("close", lambda *_: closed("page_closed"))
        context = getattr(page, "context", None)
        if context is not None and hasattr(context, "on"):
            context.on("close", lambda *_: closed("context_closed"))
            browser = getattr(context, "browser", None)
            if browser is not None:
                browser.on("disconnected", lambda *_: closed("browser_disconnected"))
        page.add_init_script(_CLICK_CAPTURE_SCRIPT)
        page.evaluate(_CLICK_CAPTURE_SCRIPT)

    def drain_events(self) -> list[dict[str, Any]]:
        events = list(self.environment.page.evaluate(_DRAIN_EVENTS, self._event_ack) or [])
        if events:
            self._event_ack = {"token": events[-1].get("token"), "id": events[-1].get("id")}
        if self._deferred_events:
            events = self._deferred_events + events
            self._deferred_events = []
        return events

    def _safe_drain_events(self) -> list[dict[str, Any]] | None:
        """Return queued presses, or ``None`` while Playwright is unavailable.

        An unavailable queue is UNKNOWN, not an empty queue.  Treating the two
        alike lets a board effect race ahead of its temporarily unreadable
        press and was one route to false abandonment.
        """

        try:
            return self.drain_events()
        except (PlaywrightError, RuntimeError, ValueError) as error:
            self._warn_transient("event queue", error)
            return None

    def _warn_transient(self, component: str, error: Exception) -> None:
        """Surface recovery without flooding a 20 Hz polling loop."""

        now = time.monotonic()
        last = self._transient_warning_at.get(component)
        if last is None or now - last >= 5.0:
            logger.warning(
                "%s temporarily unavailable; retrying game=%s phase=%s steps=%s error=%s: %s",
                component, self.recorder.active_game_id, self._phase,
                self.recorder.steps_recorded, type(error).__name__, error,
            )
            self._transient_warning_at[component] = now

    def _status(self) -> GameStatus:
        return self.environment.get_game_status()

    def _observe(self) -> npt.NDArray[np.int8]:
        return self.environment.observe()

    def _safe_status(self) -> GameStatus | None:
        try:
            return self._status()
        except (PlaywrightError, RuntimeError, ValueError) as error:
            self._warn_transient("game status", error)
            return None

    def _safe_observe(self) -> npt.NDArray[np.int8] | None:
        try:
            board = self._observe()
            self._last_observation_at = time.time()
            return board
        except (PlaywrightError, RuntimeError, ValueError) as error:
            self._warn_transient("board observation", error)
            return None

    def _parse_event_snapshot(
        self, event: dict[str, Any]
    ) -> npt.NDArray[np.int8] | None:
        """Parse the diagnostic pre-click snapshot, or ``None`` if unusable."""

        cells = event.get("cells")
        if not cells:
            return None
        try:
            return parse_board_snapshot(
                cells, expected_shape=(self.height, self.width)
            )
        except Exception:  # noqa: BLE001 - snapshot is diagnostic only
            return None

    @staticmethod
    def _pre_action_status(event: dict[str, Any]) -> GameStatus | None:
        """Return the status face captured at pointerdown, or ``None``.

        The capture listener snapshots ``#top_area_face`` before the site
        handles the press.  This is what distinguishes the press that *caused*
        a win/loss (the pre-action face is still running) from a stray click on
        an already-finished board (the pre-action face is already terminal).
        ``None`` means the event carries no face snapshot (older capture or a
        test double), in which case callers fall back to the safe behaviour.
        """

        classes = event.get("faceClasses")
        if not classes:
            return None
        return classify_game_status(classes, opened_cells=1)

    def _is_fresh_board(
        self, board: npt.NDArray[np.int8] | None, status: GameStatus | None
    ) -> bool:
        """Return whether ``board`` is a genuinely fresh Expert board.

        A fresh board is complete, all-hidden (no opened cells and no flags) and
        not already won/lost.  Checking this on a stable observation is what
        prevents fragments and phantom continuations.
        """

        if board is None:
            return False
        if status in (GameStatus.WON, GameStatus.LOST):
            return False
        if board.shape != (self.height, self.width):
            return False
        return bool(np.all(board == HIDDEN))

    def _enter_resyncing(self, board: npt.NDArray[np.int8] | None = None) -> None:
        """Discard any fresh window and require a new verified fresh board."""

        self._phase = PHASE_RESYNCING
        self._fresh_board = None
        self._pending_idle = None
        if board is not None:
            self._stable_board = np.array(board, dtype=np.int8, copy=True)

    def _wait_for_stable_change(
        self,
        before: npt.NDArray[np.int8],
        expect: Callable[[npt.NDArray[np.int8], GameStatus | None], bool] | None = None,
    ) -> tuple[npt.NDArray[np.int8], GameStatus | None]:
        """Wait until the clicked cell shows its expected effect and settles.

        ``expect`` decides whether an observed board reflects the action; when
        omitted any change from ``before`` counts (the historic behaviour).
        Requiring the clicked cell's own change keeps unrelated board activity
        from ending the wait early.  The returned board is the first one that
        has stayed unchanged for ``quiescence_time`` seconds, so a multi-stage
        cascade is never mistaken for a completed action.
        """

        deadline = time.monotonic() + self.action_timeout
        changed = False
        observed_any = False
        candidate = before
        quiescent_since: float | None = None
        while time.monotonic() < deadline:
            if self._page_is_closed() or (
                self._stop_event is not None and self._stop_event.is_set()
            ):
                raise TimeoutError("session stopped while observing an action")
            # If a new press was captured while this action was still settling,
            # its board effect must not be attributed to this action.  Finalize
            # with that press's pre-action snapshot and hand it back to the
            # run loop in order.
            new_events = self._safe_drain_events()
            if new_events is None:
                time.sleep(self.poll_interval)
                continue
            if new_events:
                return self._action_boundary(new_events, before)
            board = self._safe_observe()
            status = self._safe_status()
            # Both status and board reads yield to the browser. A later press
            # may already have changed this board, even on the final iteration.
            new_events = self._safe_drain_events()
            if new_events is None:
                time.sleep(self.poll_interval)
                continue
            if new_events:
                return self._action_boundary(new_events, before)
            if board is not None:
                observed_any = True
                matches = (
                    not np.array_equal(board, before)
                    if expect is None
                    else expect(board, status)
                )
                if not changed and matches:
                    changed = True
                    candidate = board
                    quiescent_since = time.monotonic()
                elif changed:
                    if np.array_equal(board, candidate):
                        if time.monotonic() - quiescent_since >= self.quiescence_time:
                            return candidate, status
                    else:
                        candidate = board
                        quiescent_since = time.monotonic()
                if status in (GameStatus.WON, GameStatus.LOST):
                    return (candidate if changed else before.copy()), status
            elif status in (GameStatus.WON, GameStatus.LOST):
                return (candidate if changed else before.copy()), status
            time.sleep(self.poll_interval)
        if not observed_any:
            raise TimeoutError("no stable board change after the click")
        # The expected change never appeared; hand the last board (or ``before``)
        # back so the caller can reject the action with a precise reason.
        status = self._safe_status()
        events = self._safe_drain_events()
        if events:
            return self._action_boundary(events, before)
        return (candidate if changed else before.copy()), (status if events is not None else None)

    def _action_boundary(
        self, events: list[dict[str, Any]], before: npt.NDArray[np.int8]
    ) -> tuple[npt.NDArray[np.int8], GameStatus | None]:
        """The next press's atomic pre-state bounds the current action.

        Never pair a historical board with today's face, or use a candidate
        sampled after that press. Missing boundary data stays pending.
        """
        self._deferred_events = list(events) + self._deferred_events
        snapshot = self._parse_event_snapshot(events[0])
        status = self._boundary_status(events[0])
        return (before.copy() if snapshot is None else snapshot), status

    def _boundary_status(self, event: dict[str, Any]) -> GameStatus | None:
        status = self._pre_action_status(event)
        # Older captures/test doubles have no face; a nonterminal current face
        # is safe as a fallback, but a later terminal face is not attributable.
        if status is None:
            current = self._safe_status()
            if current in (GameStatus.READY, GameStatus.RUNNING):
                status = current
        return status

    @staticmethod
    def _confirm_action(
        x: int,
        y: int,
        action_type: str,
        before: npt.NDArray[np.int8],
        after: npt.NDArray[np.int8],
        status: GameStatus | None,
    ) -> str | None:
        """Return a rejection reason unless the click had its expected effect.

        A reveal always opens the clicked hidden cell (a clicked mine stays
        ``HIDDEN`` but turns the game ``LOST``).  A left-click on an already
        open numbered cell is a chord and must change at least one neighbour.
        A flag toggles the clicked cell between hidden and flagged.  Anything
        else is unrelated board activity, not a human action, and must not be
        recorded.
        """

        before_cell = int(before[y, x])
        after_cell = int(after[y, x])
        if action_type == ACTION_FLAG:
            if before_cell not in (HIDDEN, FLAGGED):
                return "flag target is not a covered cell"
            if before_cell == after_cell:
                return "flag did not change the clicked cell"
            return None
        # ACTION_OPEN: either a reveal on a hidden cell, or a chord on an open
        # numbered cell.
        if before_cell == HIDDEN:
            if after_cell == HIDDEN:
                # The clicked cell stayed hidden: only a detonated mine is
                # legitimate, and it must have ended the game.
                if status is GameStatus.LOST:
                    return None
                return "reveal did not open the clicked cell"
            # The clicked cell became an open number: a genuine safe reveal.
            if status is GameStatus.LOST:
                return "a safe reveal cannot end in LOST"
            return None
        # A left-click on an already-open cell is a chord.
        if np.array_equal(before, after):
            # A chord that detonates a mine ends the game; the mine-reveal
            # artifacts are normalized to HIDDEN, so the canonical board can be
            # unchanged.  Any other no-op chord is not a real action.
            if status is GameStatus.LOST:
                return None
            return "chord changed nothing"
        return None

    def _log_event(
        self,
        event: dict[str, Any],
        *,
        accepted: bool,
        reason: str,
        action_type: str | None = None,
        before: npt.NDArray[np.int8] | None = None,
        after: npt.NDArray[np.int8] | None = None,
    ) -> None:
        before_state = after_state = None
        if before is not None and after is not None:
            try:
                x = int(event.get("x"))
                y = int(event.get("y"))
                before_state = _cell_state_name(int(before[y, x]))
                after_state = _cell_state_name(int(after[y, x]))
            except (TypeError, ValueError, IndexError):
                before_state = after_state = None
        logger.info(
            "recorder event | game=%s phase=%s id=%s token=%s captured_at=%s "
            "type=%s trusted=%s button=%s data=(%s,%s) "
            "action=%s before=%s after=%s accepted=%s reason=%s",
            self.recorder.active_game_id, self._phase, event.get("id"),
            event.get("token"), event.get("capturedAt"),
            event.get("type"), event.get("isTrusted"), event.get("button"),
            event.get("x"), event.get("y"),
            action_type.upper() if action_type else "-",
            before_state or "-", after_state or "-", accepted, reason,
        )

    def _count_other_click(self) -> None:
        self.report.other_clicks += 1
        if self.recorder.active_game_id is not None:
            self.recorder.record_other_click()

    def _abandon_active(self, reason: str) -> None:
        if self.recorder.active_game_id is None:
            return
        if self.recorder.steps_recorded == 0:
            self.recorder.discard_game()
            return
        self._finish_active(
            "incomplete" if self._capture_error else RESULT_ABANDONED,
            self._capture_error or reason,
        )

    def _capture_failed(self, reason: str) -> None:
        """Suspend an untrustworthy transcript without declaring game over."""
        if self._capture_error is None:
            logger.error(
                "capture suspended game=%s steps=%s phase=%s reason=%s pending=%s "
                "last_observation_at=%s; "
                "waiting for a verified game boundary",
                self.recorder.active_game_id, self.recorder.steps_recorded,
                self._phase, reason,
                None if self._pending_action is None else self._pending_action.event.get("id"),
                self._last_observation_at,
            )
            self.report.errors.append(reason)
        self._capture_error = reason
        self._pending_action = None
        self._enter_resyncing()

    def _finish_active(self, result: str, reason: str | None = None) -> bool:
        # A terminal result owns finalization even if stop/close arrives while
        # persistence is retrying. Freeze its timestamp and never append twice.
        if self._pending_finalization is None:
            self._pending_finalization = (result, time.time(), reason)
            logger.info(
                "finalizing game=%s result=%s reason=%s steps=%s ended_at=%s last_observation_at=%s",
                self.recorder.active_game_id, result, reason,
                self.recorder.steps_recorded, self._pending_finalization[1],
                self._last_observation_at,
            )
        return self._retry_finalization()

    def _retry_finalization(self) -> bool:
        if self._pending_finalization is None:
            return True
        result, ended, reason = self._pending_finalization
        try:
            game = self.recorder.end_game(result, ended_at=ended, termination_reason=reason)
        except (OSError, StorageError) as error:
            if isinstance(error, OSError) and not _is_transient_lock_error(error):
                raise
            self._warn_transient("game persistence", error)
            return False
        self._pending_finalization = None
        self._capture_error = None
        self._record_game(game)
        return True

    def _page_is_closed(self) -> bool:
        """Return a positive page-closure signal without guessing from errors."""

        checker = getattr(self.environment.page, "is_closed", None)
        if self._closed_reason is not None:
            return True
        if checker is None:
            return False
        try:
            return bool(checker())
        except Exception:  # noqa: BLE001 - inability to query is UNKNOWN
            return False

    @staticmethod
    def _action_effect_matches(
        pending: _PendingAction,
        board: npt.NDArray[np.int8],
        status: GameStatus | None,
    ) -> bool:
        before = pending.board_before
        before_cell = int(before[pending.y, pending.x])
        if pending.action_type == ACTION_FLAG:
            return int(board[pending.y, pending.x]) != before_cell
        if before_cell == HIDDEN:
            return (
                int(board[pending.y, pending.x]) != HIDDEN
                or status is GameStatus.LOST
            )
        return (
            not np.array_equal(board, before)
            or status in (GameStatus.WON, GameStatus.LOST)
        )

    def _defer_action(
        self,
        event: dict[str, Any],
        x: int,
        y: int,
        action_type: str,
        board_before: npt.NDArray[np.int8],
        reason: str,
    ) -> str:
        self._pending_action = _PendingAction(
            event=dict(event),
            x=x,
            y=y,
            action_type=action_type,
            board_before=board_before.copy(),
        )
        self._log_event(
            event,
            accepted=False,
            reason=f"pending recovery: {reason}",
            action_type=action_type,
            before=board_before,
            after=board_before,
        )
        # Public compatibility: callers historically received "rejected" for
        # an action whose effect was not yet observable.  Internally it remains
        # pending and may still be committed after recovery.
        return "rejected"

    def _commit_action(
        self,
        event: dict[str, Any],
        x: int,
        y: int,
        action_type: str,
        board_before: npt.NDArray[np.int8],
        board_after: npt.NDArray[np.int8],
        status: GameStatus | None,
    ) -> str:
        """Validate, append, and possibly finalize one recovered live action."""

        reason = self._confirm_action(
            x, y, action_type, board_before, board_after, status
        )
        if reason is not None:
            self._log_event(
                event, accepted=False, reason=reason, action_type=action_type,
                before=board_before, after=board_after,
            )
            return "rejected"

        terminated = status in (GameStatus.WON, GameStatus.LOST)
        if self.recorder.active_game_id is None:
            try:
                self.recorder.start_game(started_at=time.time(), board=board_before)
            except (OSError, StorageError) as error:
                if isinstance(error, OSError) and not _is_transient_lock_error(error):
                    raise
                self._warn_transient("game allocation", error)
                return self._defer_action(event, x, y, action_type, board_before, str(error))
        opened_before = int((board_before >= 0).sum())
        opened_after = int((board_after >= 0).sum())
        reward = (
            FLAG_REWARD
            if action_type == ACTION_FLAG
            else transition_reward(board_before, board_after, status)
        )
        try:
            self.recorder.record_step(
                board_before=board_before,
                action=(x, y),
                board_after=board_after,
                game_status=status,
                terminated=terminated,
                truncated=False,
                reward=reward,
                opened_cells=opened_after,
                newly_revealed=max(0, opened_after - opened_before),
                timestamp=time.time(),
                action_type=action_type,
            )
        except DataError as error:
            logger.error("recorder rejected a step: %s", error)
            self.report.errors.append(str(error))
            self._capture_failed("transition_validation_failure")
            self._log_event(
                event, accepted=False, reason=f"recorder rejected: {error}",
                action_type=action_type, before=board_before, after=board_after,
            )
            return "rejected"

        self._stable_board = np.array(board_after, dtype=np.int8, copy=True)
        self._pending_idle = None
        self._log_event(
            event, accepted=True, reason="recorded", action_type=action_type,
            before=board_before, after=board_after,
        )
        if terminated:
            self._finish_active(status.value)
            self._enter_resyncing(board_after)
            return "game_end"
        self._phase = PHASE_RECORDING
        return "recorded"

    def _recover_pending(
        self,
        board: npt.NDArray[np.int8],
        status: GameStatus | None,
    ) -> str | None:
        """Finish a timed-out press when a later stable board proves its effect."""

        # A board without a readable face is only a partial observation.  Keep
        # waiting so UNKNOWN can never be persisted or interpreted as an end.
        if status is None:
            return None
        pending = self._pending_action
        if pending is None:
            return None
        if np.array_equal(board, pending.board_before) and status not in (GameStatus.WON, GameStatus.LOST):
            return None
        if not self._action_effect_matches(pending, board, status):
            return "unrelated"
        self._pending_action = None
        return self._commit_action(
            pending.event,
            pending.x,
            pending.y,
            pending.action_type,
            pending.board_before,
            board,
            status,
        )

    def handle_event(self, event: dict[str, Any]) -> str:
        """Process one captured press, returning what happened to it.

        A trusted primary-button press is a candidate OPEN action and a trusted
        secondary-button (right-click) press is a candidate FLAG action.  Either
        becomes a transition only when the stable post-action board shows the
        clicked cell's expected change (opened for OPEN, toggled for FLAG).
        Every other event is logged and rejected without creating a game.
        """

        if self._pending_finalization is not None:
            self._deferred_events.insert(0, event)
            return "pending"
        if event.get("isTrusted") is False:
            self._log_event(event, accepted=False, reason="untrusted (synthetic) event")
            return "rejected"

        try:
            button = int(event.get("button", 0))
        except (TypeError, ValueError):
            button = -1
        if button not in (0, 2):
            self._count_other_click()
            self._log_event(event, accepted=False, reason="unsupported mouse button")
            return "other"
        action_type = ACTION_FLAG if button == 2 else ACTION_OPEN

        event_id = event.get("id")
        if event_id is not None:
            key = (event.get("token"), event_id)
            if key in self._seen_events:
                self._log_event(
                    event, accepted=False, reason="duplicate event id",
                    action_type=action_type,
                )
                return "duplicate"
            self._seen_events.add(key)

        try:
            x = int(event["x"])
            y = int(event["y"])
        except (KeyError, TypeError, ValueError):
            self._log_event(
                event, accepted=False, reason="missing cell coordinates",
                action_type=action_type,
            )
            return "rejected"
        if not (0 <= x < self.width and 0 <= y < self.height):
            self._log_event(
                event, accepted=False, reason="coordinates outside board",
                action_type=action_type,
            )
            return "rejected"

        # A later press is also an atomic snapshot of the board immediately
        # before that press.  Use it to finish an earlier action whose DOM read
        # timed out.  If it is still identical, the earlier press was a no-op
        # and can safely be forgotten; if it proves the earlier effect, commit
        # that transition before handling this press.
        if self._pending_action is not None:
            snapshot = self._parse_event_snapshot(event)
            if snapshot is not None:
                pending = self._pending_action
                pre_status = self._boundary_status(event)
                if np.array_equal(snapshot, pending.board_before) and pre_status not in (GameStatus.WON, GameStatus.LOST):
                    self._pending_action = None
                else:
                    recovered = self._recover_pending(snapshot, pre_status)
                    if recovered == "unrelated":
                        self._capture_failed("unrecoverable_capture_gap")
                    elif recovered == "game_end":
                        # This press happened after the recovered terminal
                        # action and is necessarily a terminal stray click.
                        return "ignored_terminal"
            if self._pending_action is not None:
                if self._safe_status() in (GameStatus.WON, GameStatus.LOST):
                    # A historical boundary with no face cannot acquire one
                    # retroactively from a later terminal board. Preserve the
                    # partial recording instead of looping on it forever.
                    self._capture_failed("terminal_action_not_captured")
                    self._abandon_active("terminal_action_not_captured")
                    return "ignored_terminal"
                # Keep both presses in order; never overwrite an unresolved
                # action merely because another read/face is unavailable.
                if event_id is not None:
                    self._seen_events.discard((event.get("token"), event_id))
                self._deferred_events.insert(0, event)
                return "pending"

        # --- lifecycle gate: never create a game from an arbitrary click ---
        if self._phase in (PHASE_TERMINAL, PHASE_RESYNCING):
            self._log_event(
                event, accepted=False,
                reason="previous game is over; waiting for a verified fresh board",
                action_type=action_type,
            )
            return "ignored_terminal"

        if self._phase == PHASE_WAITING:
            snapshot = self._parse_event_snapshot(event)
            if self._is_fresh_board(snapshot, self._safe_status()):
                self._stable_board = snapshot.copy()
                self._fresh_board = snapshot.copy()
                self._phase = PHASE_READY
                logger.info("event pre-click snapshot is a fresh board; ready")
            else:
                self._log_event(
                    event, accepted=False,
                    reason="no verified fresh board yet",
                    action_type=action_type,
                )
                return "waiting"

        status_before = self._safe_status()
        if status_before in (GameStatus.WON, GameStatus.LOST):
            pre_action_status = self._pre_action_status(event)
            if pre_action_status is None or pre_action_status in (
                GameStatus.WON, GameStatus.LOST
            ):
                # The face was already terminal *before* this press (or the
                # event carries no pre-action face): a genuine stray click on a
                # finished board. Never let it create or extend a game.
                if (
                    self.recorder.active_game_id is not None
                    and not self.recorder.game_terminated
                ):
                    self._capture_failed("terminal_action_not_captured")
                    self._abandon_active("terminal_action_not_captured")
                self._enter_resyncing()
                self._log_event(
                    event, accepted=False,
                    reason="terminal status before the action",
                    action_type=action_type,
                )
                return "ignored_terminal"
            # Otherwise the face was still running at pointerdown, so this press
            # is the action that ended the game. A fast loss/win updates the DOM
            # before the press is drained; without this branch that terminal
            # press is mistaken for a stray click and the game is wrongly
            # ABANDONED. Fall through and attribute/record it as terminal.

        if self._phase == PHASE_READY:
            # First action of a brand-new game: must be OPEN on a hidden cell.
            snapshot = self._parse_event_snapshot(event)
            if snapshot is not None and not np.array_equal(snapshot, self._fresh_board):
                self._enter_resyncing()
                self._log_event(
                    event, accepted=False,
                    reason="board changed before the first action",
                    action_type=action_type,
                )
                return "ignored"
            if action_type != ACTION_OPEN:
                self._log_event(
                    event, accepted=False, reason="first action must be OPEN",
                    action_type=action_type,
                )
                return "rejected"
            board_before = self._fresh_board.copy()
            if int(board_before[y, x]) != HIDDEN:
                self._log_event(
                    event, accepted=False,
                    reason="first action must target a hidden cell",
                    action_type=action_type, before=board_before, after=board_before,
                )
                return "rejected"
        else:
            if self._stable_board is None:
                self._log_event(
                    event, accepted=False, reason="no stable board available",
                    action_type=action_type,
                )
                return "rejected"
            board_before = self._stable_board.copy()
            if action_type == ACTION_OPEN and int(board_before[y, x]) == FLAGGED:
                self._log_event(
                    event, accepted=False, reason="left click on a flagged cell",
                    action_type=action_type, before=board_before, after=board_before,
                )
                return "rejected"
            snapshot = self._parse_event_snapshot(event)
            if snapshot is not None and not np.array_equal(snapshot, board_before):
                logger.debug(
                    "click (%d, %d): event snapshot differs from the stable board "
                    "(transient/in-flight DOM); using the stable board as board_before",
                    x, y,
                )

        before_cell = int(board_before[y, x])
        if action_type == ACTION_FLAG:
            def expect_change(board: npt.NDArray[np.int8], _status: GameStatus | None) -> bool:
                return int(board[y, x]) != before_cell
        elif before_cell == HIDDEN:
            def expect_change(board: npt.NDArray[np.int8], status: GameStatus | None) -> bool:
                return int(board[y, x]) != HIDDEN or status is GameStatus.LOST
        else:
            def expect_change(board: npt.NDArray[np.int8], status: GameStatus | None) -> bool:
                return (
                    not np.array_equal(board, board_before)
                    or status in (GameStatus.WON, GameStatus.LOST)
                )

        try:
            board_after, status = self._wait_for_stable_change(board_before, expect_change)
        except TimeoutError:
            return self._defer_action(
                event, x, y, action_type, board_before,
                "board/status temporarily unavailable",
            )

        if status is None:
            return self._defer_action(
                event, x, y, action_type, board_before,
                "game status temporarily unavailable",
            )

        terminated = status in (GameStatus.WON, GameStatus.LOST)
        if (
            action_type == ACTION_OPEN
            and before_cell == HIDDEN
            and int(board_after[y, x]) == HIDDEN
            and not terminated
        ):
            deadline = time.monotonic() + self.mine_grace
            while time.monotonic() < deadline and not terminated:
                status = self._safe_status()
                terminated = status in (GameStatus.WON, GameStatus.LOST)
                if not terminated:
                    time.sleep(self.poll_interval)

        reason = self._confirm_action(x, y, action_type, board_before, board_after, status)
        if reason is not None:
            if reason in {
                "flag did not change the clicked cell",
                "reveal did not open the clicked cell",
                "chord changed nothing",
            }:
                return self._defer_action(
                    event, x, y, action_type, board_before, reason
                )
            self._log_event(
                event, accepted=False, reason=reason, action_type=action_type,
                before=board_before, after=board_after,
            )
            return "rejected"
        return self._commit_action(
            event, x, y, action_type, board_before, board_after, status
        )

    def _handle_idle_board(self, observed: npt.NDArray[np.int8]) -> None:
        """Advance the game lifecycle from a stable idle board.

        A fresh all-hidden board promotes the recorder to READY. Unexplained
        changes suspend the recording, leaving the game active until a real
        terminal/reset/closure boundary. Unknown status cannot end a game.
        """

        status = self._safe_status()
        if status is None or self._pending_finalization is not None:
            return
        # Status detection itself performs blocking DOM reads. Do not call a
        # newly changed board "idle" if input arrived inside those reads.
        events = self._safe_drain_events()
        if events is None:
            return
        if events:
            self._deferred_events = list(events) + self._deferred_events
            return
        fresh = self._is_fresh_board(observed, status)
        if self._capture_error and not fresh and status not in (GameStatus.WON, GameStatus.LOST):
            return

        # A stable late observation can complete a press that exceeded the
        # action wait.  Crucially, elapsed time and temporary DOM failure do
        # not end the game; only a verified, unrelated boundary below can.
        if self._pending_action is not None and not fresh:
            recovered = self._recover_pending(observed, status)
            if recovered == "game_end":
                return
            if recovered == "recorded":
                return

        # A game whose terminal click was somehow missed must not linger: once
        # the face is terminal but no terminal transition was recorded, save the
        # partial game as incomplete and wait for a genuine reset.
        if (
            self.recorder.active_game_id is not None
            and status in (GameStatus.WON, GameStatus.LOST)
            and not self.recorder.game_terminated
        ):
            logger.warning(
                "game status is %s but no terminal transition was captured; "
                "saving the partial game as incomplete", status.value,
            )
            self._capture_failed(self._capture_error or "terminal_action_not_captured")
            self._abandon_active("terminal_action_not_captured")
            self._enter_resyncing(observed)
            return

        if fresh:
            self._pending_action = None
            if self.recorder.active_game_id is not None:
                logger.info("fresh board detected; abandoning the in-progress game")
                self._abandon_active("fresh_board_reset")
                if self._pending_finalization is not None:
                    return
            self._capture_error = None
            if self._phase != PHASE_READY:
                logger.info("fresh board detected; ready to record a new game")
            self._stable_board = observed.copy()
            self._fresh_board = observed.copy()
            self._pending_idle = None
            self._phase = PHASE_READY
            return

        # The board is not fresh: a previously verified fresh window is no
        # longer usable as board_before for a new game.
        if self._phase == PHASE_READY:
            self._fresh_board = None
            self._phase = PHASE_RESYNCING

        if self._stable_board is None:
            self._stable_board = observed.copy()
            return
        if np.array_equal(observed, self._stable_board):
            self._pending_idle = None
            return
        if self._pending_idle is None or not np.array_equal(observed, self._pending_idle):
            self._pending_idle = observed.copy()
            return
        self._pending_idle = None
        self._stable_board = observed.copy()
        if self.recorder.active_game_id is not None:
            logger.warning(
                "board changed without an attributable press; suspending capture"
            )
            reason = (
                "unrecoverable_capture_gap"
                if self._pending_action is not None
                else "unattributed_board_change"
            )
            self._capture_failed(reason)
        else:
            logger.info("board changed between games; resynchronizing")
        # A non-fresh change is never a valid new game.
        self._fresh_board = None
        if self._phase in (PHASE_READY, PHASE_RECORDING, PHASE_TERMINAL):
            self._phase = PHASE_RESYNCING

    def _record_game(self, game: Any) -> None:
        if game.result == RESULT_WON:
            self.report.wins += 1
        elif game.result == RESULT_LOST:
            self.report.losses += 1
        elif game.result == RESULT_ABANDONED:
            self.report.abandoned += 1
        else:
            self.report.incomplete += 1
        totals = self._dataset_totals()
        self.report.games = totals["games"]
        self.report.transitions = totals["transitions"]
        lines = [
            f"Game {totals['games']}",
            f"Source: {game.source}",
            f"Result: {game.result.upper()}",
            f"Steps: {game.total_steps}",
            f"Duration: {game.duration_s:.1f} seconds",
            f"Transitions recorded: {game.total_steps}",
            "",
            "Dataset total:",
            f"Games: {totals['games']}",
            f"Transitions: {totals['transitions']}",
            f"Wins: {totals['wins']}",
            f"Losses: {totals['losses']}",
            f"Abandoned: {totals['abandoned']}",
            f"Incomplete recording: {totals['incomplete']}",
        ]
        self.emit("\n".join(lines))

    def _dataset_totals(self) -> dict[str, int]:
        games = list(self.recorder.store.iter_games())
        return {
            "games": len(games),
            "transitions": self.recorder.store.transition_count,
            "wins": sum(1 for game in games if game.result == RESULT_WON),
            "losses": sum(1 for game in games if game.result == RESULT_LOST),
            "abandoned": sum(1 for game in games if game.result == RESULT_ABANDONED),
            "incomplete": sum(1 for game in games if game.result == "incomplete"),
        }

    def run(
        self, *, max_games: int, stop_event: threading.Event | None = None
    ) -> HumanRecordingReport:
        """Observe play until ``max_games`` completed games have been recorded."""

        self._stop_event = stop_event
        if not self._page_is_closed():
            self.install_capture()
            self.environment.wait_until_ready()
        if self.diagnostics:
            self.emit(
                "Recorder diagnostics enabled: every cell press will be logged "
                "with its target, coordinates, isTrusted flag and accept/reject reason."
            )
        self.emit(
            "Human recording started; play Minesweeper in the browser window. "
            f"Target: {max_games} completed games."
        )

        if self._stable_board is None:
            deadline = time.monotonic() + self.action_timeout
            while self._stable_board is None and time.monotonic() < deadline:
                self._stable_board = self._safe_observe()
                if self._stable_board is None:
                    time.sleep(self.poll_interval)
        if self._stable_board is None:
            raise DataError("could not read the initial stable board")

        termination_reason: str | None = None
        while self._completed() < max_games:
            if not self._retry_finalization():
                time.sleep(max(self.poll_interval, 0.05))
                continue
            if self._completed() >= max_games:
                break
            if stop_event is not None and stop_event.is_set():
                termination_reason = "explicit_session_cancellation"
                break
            if self._page_is_closed():
                termination_reason = self._closed_reason or "page_closed"
                break
            events = self._safe_drain_events()
            if events is None:
                time.sleep(self.poll_interval)
                continue
            if events:
                self._pending_idle = None
                self._process_events(events, max_games)
                continue
            observed = self._safe_observe()
            # A press may have arrived while we were observing; if so, process it
            # instead of mistaking the resulting board for an idle change.
            pending = self._safe_drain_events()
            if pending is None:
                # The event queue is temporarily unavailable.  Do not inspect
                # the board as idle because its matching press may be queued.
                time.sleep(self.poll_interval)
                continue
            if pending:
                self._pending_idle = None
                self._process_events(pending, max_games)
                continue
            if observed is not None:
                self._handle_idle_board(observed)
            time.sleep(self.poll_interval)

        if self.recorder.active_game_id is not None:
            self._abandon_active(termination_reason or "session_ended")
        while not self._retry_finalization():
            time.sleep(max(self.poll_interval, 0.05))
        return self.report

    def _process_events(self, events: list[dict[str, Any]], max_games: int) -> None:
        # Keep the whole batch visible to the action observer. In particular,
        # the next pre-click snapshot is needed even when JS's queue is empty.
        self._deferred_events = list(events) + self._deferred_events
        while self._deferred_events and self._completed() < max_games:
            event = self._deferred_events.pop(0)
            result = self.handle_event(event)
            if result == "pending" or self._pending_finalization is not None:
                time.sleep(self.poll_interval)
                break

    def _completed(self) -> int:
        return self.report.wins + self.report.losses
