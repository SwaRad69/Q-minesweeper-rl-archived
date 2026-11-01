"""CNN actor-critic network for PPO on the Minesweeper board.

The network defines the policy/value architecture and the model-specific
observation preprocessing pipeline only::

    canonical board -> one-hot channels -> CNN encoder -> shared features
                    -> actor head  -> action logits
                    -> critic head -> scalar V(s)

Two observation encodings are supported, matching the two action spaces in
:mod:`minesweeper_rl.agents.action_space`:

``OPEN_ONLY`` (10 channels)
    channel 0 -> HIDDEN (-1), channels 1..9 -> OPEN 0..8.

``FLAGS`` (11 channels, the human-like experiment)
    channel 0 -> FLAGGED (-2), channel 1 -> HIDDEN (-1), channels 2..10 ->
    OPEN 0..8.

The actor head is a 1x1 convolution whose flattened output is already in the
row-major cell order used by the action encoding.  For the flag-aware action
space it has three output planes, so flattening yields ``action_type * cells +
cell`` directly.  Hidden mine locations are never part of the observation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from ..action_types import ACTION_TYPES
from ..dom import FLAGGED, HIDDEN, MAX_NUMBER
from .action_space import (
    DEFAULT_ACTION_SPACE,
    FLAGS,
    OPEN_ONLY,
    HEIGHT,
    WIDTH,
)

#: Canonical OPEN-only board values in the exact channel order the CNN consumes.
OBSERVATION_VALUES: tuple[int, ...] = (HIDDEN, *range(0, MAX_NUMBER + 1))

#: Explicit OPEN-only canonical-value -> channel mapping.
VALUE_TO_CHANNEL: dict[int, int] = {
    value: channel for channel, value in enumerate(OBSERVATION_VALUES)
}

#: Number of OPEN-only CNN input channels: HIDDEN + OPEN 0..8.
OBSERVATION_CHANNELS = len(OBSERVATION_VALUES)

#: Deprecated alias kept for callers that still import ``NUM_VALUES``.
NUM_VALUES = OBSERVATION_CHANNELS

#: Flag-aware board values: FLAGGED first, then HIDDEN, then OPEN 0..8.
OBSERVATION_VALUES_FLAGS: tuple[int, ...] = (FLAGGED, *OBSERVATION_VALUES)

#: Explicit flag-aware canonical-value -> channel mapping.
VALUE_TO_CHANNEL_FLAGS: dict[int, int] = {
    value: channel for channel, value in enumerate(OBSERVATION_VALUES_FLAGS)
}

#: Number of flag-aware CNN input channels: FLAGGED + HIDDEN + OPEN 0..8.
OBSERVATION_CHANNELS_FLAGS = len(OBSERVATION_VALUES_FLAGS)


def observation_values(action_space: str = DEFAULT_ACTION_SPACE) -> tuple[int, ...]:
    """Return the canonical board values for ``action_space`` in channel order."""

    if action_space == OPEN_ONLY:
        return OBSERVATION_VALUES
    if action_space == FLAGS:
        return OBSERVATION_VALUES_FLAGS
    raise ValueError(f"unknown action space: {action_space!r}")


def observation_channels(action_space: str = DEFAULT_ACTION_SPACE) -> int:
    """Return the number of observation channels for ``action_space``."""

    return len(observation_values(action_space))


def value_to_channel(action_space: str = DEFAULT_ACTION_SPACE) -> dict[int, int]:
    """Return the canonical-value -> channel mapping for ``action_space``."""

    return {
        value: channel
        for channel, value in enumerate(observation_values(action_space))
    }


def board_to_tensor(
    board: npt.ArrayLike,
    *,
    action_space: str = DEFAULT_ACTION_SPACE,
    width: int = WIDTH,
    height: int = HEIGHT,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """One-hot encode raw canonical boards into a CNN input tensor.

    Accepts a single ``[height, width]`` board or a batch ``[batch, height,
    width]``.  For ``OPEN_ONLY`` values must be :data:`HIDDEN` (``-1``) or a
    revealed number ``0..8``; for ``FLAGS`` :data:`FLAGGED` (``-2``) is also
    accepted.  Returns ``[channels, height, width]`` or ``[batch, channels,
    height, width]`` accordingly.
    """

    values = observation_values(action_space)
    channels_count = len(values)
    minimum = -2 if action_space == FLAGS else HIDDEN
    channel_for = value_to_channel(action_space)

    if isinstance(board, Tensor):
        tensor = board.detach().to(dtype=torch.long, device=board.device)
    else:
        tensor = torch.as_tensor(np.asarray(board), dtype=torch.long)
    if tensor.ndim not in (2, 3):
        raise ValueError(
            f"board must be 2D or 3D, got {tensor.ndim} dimensions"
        )
    if tuple(tensor.shape[-2:]) != (height, width):
        raise ValueError(
            f"board must have trailing shape ({height}, {width}), "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.numel():
        out_of_range = ((tensor < minimum) | (tensor > MAX_NUMBER)).any()
        if bool(out_of_range):
            raise ValueError(
                f"board values must be {minimum} or 0..8 for action_space "
                f"{action_space!r}"
            )

    channels = torch.zeros(
        *tensor.shape, channels_count, dtype=dtype, device=tensor.device
    )
    for value, channel in channel_for.items():
        channels[..., channel] = (tensor == value).to(dtype)
    return channels.movedim(-1, -3)


def apply_action_mask(
    logits: Tensor, mask: Any, *, fill_value: float = float("-inf")
) -> Tensor:
    """Set logits of invalid actions to ``fill_value``.

    ``mask`` is a boolean tensor/array where ``True`` marks a valid action; its
    trailing dimension must match ``logits``.  Works for unbatched
    ``[num_actions]`` and batched ``[batch, num_actions]`` logits.
    """

    boolean = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    return logits.masked_fill(~boolean, fill_value)


class PolicyValueNetwork(nn.Module):
    """Small CNN actor-critic over the selected Minesweeper action space."""

    def __init__(
        self,
        *,
        width: int = WIDTH,
        height: int = HEIGHT,
        channels: int = 16,
        action_space: str = DEFAULT_ACTION_SPACE,
    ) -> None:
        super().__init__()
        if action_space not in (OPEN_ONLY, FLAGS):
            raise ValueError(f"unknown action space: {action_space!r}")
        self.width = width
        self.height = height
        self.action_space = action_space
        cells = width * height
        #: OPEN-only: one action per cell.  FLAGS: OPEN + FLAG + CHORD per cell.
        self.num_actions = len(ACTION_TYPES) * cells if action_space == FLAGS else cells
        self.observation_channels = observation_channels(action_space)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.observation_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        actor_channels = len(ACTION_TYPES) if action_space == FLAGS else 1
        self.actor = nn.Conv2d(channels, actor_channels, kernel_size=1)
        self.critic = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
        )

    def encode(self, board: npt.ArrayLike) -> Tensor:
        """Run the observation preprocessing pipeline without the heads."""

        return board_to_tensor(
            board,
            action_space=self.action_space,
            width=self.width,
            height=self.height,
        )

    def forward(
        self, board: npt.ArrayLike, action_mask: Any | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return ``(action_logits, state_value)`` for a board or batch.

        A single ``[height, width]`` board yields logits of shape
        ``[num_actions]`` and a value of shape ``[1]``.  A batch
        ``[batch, height, width]`` yields ``[batch, num_actions]`` and
        ``[batch, 1]``.  When ``action_mask`` is given, invalid action logits
        are replaced with ``-inf`` before being returned.
        """

        batched = np.ndim(board) == 3
        encoded = self.encode(board)
        if encoded.ndim == 3:
            encoded = encoded.unsqueeze(0)

        features = self.encoder(encoded)
        logits = self.actor(features).flatten(1)
        if action_mask is not None:
            logits = apply_action_mask(logits, action_mask)
        values = self.critic(features)

        if not batched:
            return logits[0], values[0]
        return logits, values
