"""Adapter exposing the browser environment through PPO's reset/step interface.

PPO works with flat integer action ids, while the browser environment accepts
``(x, y)`` coordinates.  Two action spaces are supported:

``OPEN_ONLY``
    A flat action id is exactly a row-major cell index (``y * width + x``), and
    the adapter forwards ``environment.step((x, y))`` as before.

``FLAGS``
    A flat action id packs ``action_type * cells + cell``.  The adapter splits
    it and forwards ``environment.apply_action(action_type, x, y)``, which
    dispatches OPEN, FLAG or CHORD.

The wrapped object must provide ``reset()`` and ``step((x, y))`` plus a
``config`` with ``width``/``height``; the flag-aware path additionally requires
``apply_action(action_type, x, y)`` (as :class:`MinesweeperEnvironment` does).
The adapter is not connected to a live browser by itself; it is unit tested
against fakes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ..action_types import ACTION_OPEN
from .action_space import (
    DEFAULT_ACTION_SPACE,
    OPEN_ONLY,
    action_count,
    action_mask as build_action_mask,
    action_to_coordinate,
    decode_action,
    split_action,
    validate_action_space,
)


class MinesweeperPPOEnv:
    """Present the browser environment as a flat-action PPO environment."""

    def __init__(
        self,
        environment: Any,
        *,
        action_space: str = DEFAULT_ACTION_SPACE,
    ) -> None:
        self.environment = environment
        config = environment.config
        self.width = int(config.width)
        self.height = int(config.height)
        #: Difficulty provenance for the run configuration (Expert = 99 mines).
        #: Read defensively so test doubles without these fields still work.
        self.mines = int(getattr(config, "mines", 99))
        self.difficulty = str(getattr(config, "difficulty", "expert"))
        self.cells = self.width * self.height
        self.action_space = validate_action_space(action_space)
        self.num_actions = action_count(self.action_space, cells=self.cells)
        self._board: npt.NDArray[np.int8] | None = None
        self.last_info: dict[str, Any] = {}

    def reset(self) -> npt.NDArray[np.int8]:
        """Start a fresh game and return the initial board observation."""

        self._board = self.environment.reset()
        self.last_info = {}
        return self._board

    def step(
        self, action: int
    ) -> tuple[npt.NDArray[np.int8], float, bool, bool, dict[str, Any]]:
        """Decode ``action`` and dispatch the corresponding board operation."""

        if self.action_space == OPEN_ONLY:
            cell = decode_action(action, width=self.width, height=self.height)
            action_type = ACTION_OPEN
        else:
            action_type, cell = split_action(
                action, width=self.width, height=self.height
            )
        x, y = action_to_coordinate(cell, width=self.width, height=self.height)

        if self.action_space == OPEN_ONLY:
            result = self.environment.step((x, y))
        else:
            result = self.environment.apply_action(action_type, x, y)
        board, reward, terminated, truncated, info = result
        self._board = board
        self.last_info = dict(info)
        return board, reward, terminated, truncated, info

    def action_mask(self, observation: Any = None) -> npt.NDArray[np.bool_]:
        """Return the legality mask over all flat actions for this action space."""

        if observation is None:
            if self._board is None:
                raise RuntimeError("reset() must be called before requesting a mask")
            board = self._board
        else:
            board = np.asarray(observation)
        return build_action_mask(
            board,
            action_space=self.action_space,
            width=self.width,
            height=self.height,
        )
