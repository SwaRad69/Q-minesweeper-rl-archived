"""Transparent environment wrapper that records per-episode metrics.

PPO and evaluation drive the wrapped environment through the normal
``reset()`` / ``step(action)`` / ``action_mask()`` interface, so this wrapper
stays out of the algorithm.  It turns the environment's step ``info`` into an
:class:`~minesweeper_rl.experiment.metrics.EpisodeMetrics` record and counts
invalid actions and environment failures.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ..action_types import ACTION_CHORD, ACTION_FLAG, ACTION_OPEN
from .metrics import EpisodeMetrics


class TrackedEnvironment:
    """Record episode-level metrics for any Minesweeper-style environment."""

    def __init__(
        self,
        env: Any,
        *,
        phase: str = "train",
        episode_offset: int = 0,
        on_episode: Callable[[EpisodeMetrics], None] | None = None,
    ) -> None:
        self.env = env
        self.phase = phase
        self.episode_offset = int(episode_offset)
        self.on_episode = on_episode
        self.episodes: list[EpisodeMetrics] = []
        self.invalid_actions = 0
        self.failures = 0
        self.last_episode: EpisodeMetrics | None = None
        self._episode = 0
        self._reset_accumulators()

    @property
    def num_actions(self) -> int:
        return int(self.env.num_actions)

    def _reset_accumulators(self) -> None:
        self._reward = 0.0
        self._steps = 0
        self._cells_revealed = 0
        self._mine_hit = False
        self._episode_invalid = 0
        self._open_actions = 0
        self._flag_actions = 0
        self._chord_actions = 0
        self._started: float | None = None

    def reset(self) -> Any:
        try:
            observation = self.env.reset()
        except Exception:
            self.failures += 1
            raise
        self._reset_accumulators()
        return observation

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self._started is None:
            self._started = time.perf_counter()

        if not self._action_is_valid(action):
            self._episode_invalid += 1
            self.invalid_actions += 1

        try:
            result = self.env.step(action)
        except Exception:
            self.failures += 1
            raise

        observation, reward, terminated, truncated, info = result
        self._steps += 1
        self._reward += float(reward)
        self._cells_revealed += int(info.get("newly_revealed", 0))
        self._count_operation(info)
        if bool(info.get("hit_mine", False)):
            self._mine_hit = True
        if terminated or truncated:
            self._finish(info, truncated, observation)
        return result

    def _count_operation(self, info: dict[str, Any]) -> None:
        """Attribute one step to its OPEN/FLAG/CHORD counter from ``info``."""

        action_type = info.get("action_type")
        if action_type is None:
            operation = info.get("operation")
            if operation == "flag":
                action_type = ACTION_FLAG
            elif operation == "chord":
                action_type = ACTION_CHORD
            else:
                action_type = ACTION_OPEN
        if action_type == ACTION_FLAG:
            self._flag_actions += 1
        elif action_type == ACTION_CHORD:
            self._chord_actions += 1
        else:
            self._open_actions += 1

    def finish_truncated_episode(self) -> EpisodeMetrics | None:
        """Finalize the current episode as truncated.

        The trainer can end an episode internally (for example a stuck game
        that exhausts the step budget) without the environment ever reporting
        ``terminated``/``truncated``, so :meth:`step` never calls
        :meth:`_finish`.  The training loop calls this to close that episode
        using the accumulators built up by the real ``step`` calls, so exactly
        one episode record is produced and episode ids cannot repeat.

        Returns ``None`` when no episode is in progress.
        """

        if self._started is None:
            return None
        self._finish({}, True, None)
        return self.last_episode

    def action_mask(self, observation: Any = None):
        return self.env.action_mask(observation)

    def _action_is_valid(self, action: int) -> bool:
        mask = self.env.action_mask()
        if mask is None:
            return True
        return bool(mask[action])

    def _finish(
        self, info: dict[str, Any], truncated: bool, observation: Any = None
    ) -> None:
        self._episode += 1
        status = info.get("game_status")
        if status is not None:
            outcome = getattr(status, "value", str(status))
        else:
            outcome = "truncated" if truncated else "ended"
        duration = (
            time.perf_counter() - self._started if self._started is not None else 0.0
        )
        # Total genuinely revealed cells on the final board (``board >= 0``).
        # This is the board-coverage numerator and, unlike ``cells_revealed``,
        # includes cells opened by the environment's controlled first move.
        revealed_total = (
            int(np.count_nonzero(np.asarray(observation) >= 0))
            if observation is not None
            else self._cells_revealed
        )
        episode = EpisodeMetrics(
            episode=self.episode_offset + self._episode,
            phase=self.phase,
            outcome=outcome,
            won=outcome == "won",
            reward=self._reward,
            steps=self._steps,
            mine_hit=self._mine_hit,
            cells_revealed=self._cells_revealed,
            duration_s=duration,
            invalid_actions=self._episode_invalid,
            open_actions=self._open_actions,
            flag_actions=self._flag_actions,
            chord_actions=self._chord_actions,
            revealed_total=revealed_total,
        )
        self.episodes.append(episode)
        self.last_episode = episode
        if self.on_episode is not None:
            self.on_episode(episode)
