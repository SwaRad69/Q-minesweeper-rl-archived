"""Tiny deterministic environment used to verify PPO before the browser.

``TargetCellEnv`` is a one-step, board-shaped contextual bandit.  Each episode
shows a single revealed clue cell at ``x = 0`` holding a number ``v`` (``1 ..
width - 1``); every other cell is hidden.  The agent must reveal the hidden
cell at ``x = v``.  A correct reveal pays ``+1`` and any other reveal pays
``-1``.  Clues cycle deterministically across resets, so training and
evaluation are fully reproducible.

It exercises the exact interfaces PPO relies on -- ``reset``/``step`` and an
``action_mask`` that forbids the already-revealed clue cell -- without any
browser, network or external state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ..dom import HIDDEN

CORRECT_REWARD = 1.0
WRONG_REWARD = -1.0


class TargetCellEnv:
    """Deterministic one-step board environment with action masking."""

    def __init__(self, width: int = 4) -> None:
        if width < 3:
            raise ValueError("width must be at least 3")
        self.width = width
        self.height = 1
        self.num_actions = width
        self._episode = 0
        self._target = 1
        self._board = np.full((self.height, self.width), HIDDEN, dtype=np.int8)
        self.reset()

    def reset(self) -> npt.NDArray[np.int8]:
        """Start a new episode and return the ``[1, width]`` clue board."""

        clue = 1 + (self._episode % (self.width - 1))
        self._episode += 1
        self._target = clue
        self._board = np.full((self.height, self.width), HIDDEN, dtype=np.int8)
        self._board[0, 0] = np.int8(clue)
        return self._board.copy()

    def step(
        self, action: int
    ) -> tuple[npt.NDArray[np.int8], float, bool, bool, dict[str, Any]]:
        """Reveal ``action``; the episode always terminates after one step."""

        if not 0 <= action < self.num_actions:
            raise ValueError(f"action outside board: {action}")
        correct = action == self._target
        reward = CORRECT_REWARD if correct else WRONG_REWARD
        info = {"target": self._target, "correct": correct}
        return self._board.copy(), reward, True, False, info

    def action_mask(self, observation: Any = None) -> npt.NDArray[np.bool_]:
        """Mark hidden cells (the legal reveals) in action-index order."""

        board = self._board if observation is None else np.asarray(observation)
        return (board.reshape(-1) == HIDDEN).astype(bool)

    @property
    def target(self) -> int:
        """The clue value naming the rewarded cell for the current episode."""

        return self._target
