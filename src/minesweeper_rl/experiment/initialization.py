"""Initialize a PPO policy from a human behavioral-cloning checkpoint.

The imitation/BC experiment trains only the shared CNN encoder and the actor head
with cross-entropy; its critic is never optimized.  Initializing PPO from such a
checkpoint therefore copies the **encoder + actor** weights and deliberately
resets the **value head** to zero, so PPO starts with an unbiased ``V(s) = 0``
instead of a randomly (or accidentally) carried-over critic.

Compatibility is verified explicitly before any weight is touched: observation
channels, board dimensions, CNN architecture, hidden channel width, action space
and the 1440-action actor output must all match.  Any mismatch raises
:class:`PolicyCompatibilityError` and nothing is loaded -- weights are never
silently reshaped or partially applied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from ..agents.policy import PolicyValueNetwork
from .imitation_train import (
    IMITATION_CHECKPOINT_FORMAT,
    IMITATION_CHECKPOINT_VERSION,
    read_imitation_checkpoint,
)

logger = logging.getLogger(__name__)

#: Metadata keys that must match exactly between checkpoint and policy.
_COMPATIBILITY_KEYS = (
    "class",
    "width",
    "height",
    "channels",
    "observation_channels",
    "num_actions",
    "action_space",
)


class PolicyCompatibilityError(ValueError):
    """Raised when a pretrained checkpoint cannot initialize this policy."""


@dataclass(frozen=True, slots=True)
class PretrainedPolicyInfo:
    """Provenance of the imitation checkpoint used to initialize PPO."""

    path: str
    format: str
    version: int
    class_name: str
    width: int
    height: int
    channels: int
    observation_channels: int
    num_actions: int
    action_space: str
    epoch: int | None = None
    human_dataset: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format,
            "version": self.version,
            "class": self.class_name,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "observation_channels": self.observation_channels,
            "num_actions": self.num_actions,
            "action_space": self.action_space,
            "epoch": self.epoch,
            "human_dataset": self.human_dataset,
        }


def _policy_metadata(policy: nn.Module) -> dict[str, Any]:
    first = policy.encoder[0]
    return {
        "class": type(policy).__name__,
        "width": int(getattr(policy, "width", -1)),
        "height": int(getattr(policy, "height", -1)),
        "channels": int(first.out_channels),
        "observation_channels": int(getattr(policy, "observation_channels", -1)),
        "num_actions": int(getattr(policy, "num_actions", -1)),
        "action_space": str(getattr(policy, "action_space", "")),
    }


def _info_from_data(data: dict[str, Any], path: str) -> PretrainedPolicyInfo:
    model = data.get("model")
    if not isinstance(model, dict):
        raise PolicyCompatibilityError(
            f"{path} has no 'model' metadata; it cannot be used as a pretrained policy"
        )
    return PretrainedPolicyInfo(
        path=path,
        format=str(data.get("format")),
        version=int(data.get("version", -1)),
        class_name=str(model.get("class", "")),
        width=int(model.get("width", -1)),
        height=int(model.get("height", -1)),
        channels=int(model.get("channels", -1)),
        observation_channels=int(model.get("observation_channels", -1)),
        num_actions=int(model.get("num_actions", -1)),
        action_space=str(model.get("action_space", "")),
        epoch=int(data["epoch"]) if data.get("epoch") is not None else None,
        human_dataset=data.get("dataset") if isinstance(data.get("dataset"), dict) else None,
    )


def inspect_imitation_checkpoint(path: str | Path) -> PretrainedPolicyInfo:
    """Read and describe an imitation checkpoint without applying any weights."""

    data = read_imitation_checkpoint(path)
    return _info_from_data(data, str(path))


def check_policy_compatibility(
    info: PretrainedPolicyInfo, policy: nn.Module
) -> None:
    """Raise :class:`PolicyCompatibilityError` unless everything matches."""

    if info.format != IMITATION_CHECKPOINT_FORMAT:
        raise PolicyCompatibilityError(
            f"pretrained checkpoint format is {info.format!r}; expected "
            f"{IMITATION_CHECKPOINT_FORMAT!r}"
        )
    if info.version != IMITATION_CHECKPOINT_VERSION:
        raise PolicyCompatibilityError(
            f"pretrained checkpoint version {info.version!r} is not supported "
            f"(expected {IMITATION_CHECKPOINT_VERSION})"
        )
    current = _policy_metadata(policy)
    for key in _COMPATIBILITY_KEYS:
        saved = getattr(info, "class_name" if key == "class" else key)
        if saved != current[key]:
            raise PolicyCompatibilityError(
                f"pretrained checkpoint {key} {saved!r} does not match the PPO "
                f"policy {key} {current[key]!r}; refusing to load incompatible "
                f"weights"
            )


def _reset_value_head(state: dict[str, Any], policy: nn.Module) -> dict[str, Any]:
    """Zero the critic's final linear layer so PPO starts at ``V(s) = 0``."""

    reset: list[str] = []
    for name, module in policy.named_modules():
        if isinstance(module, nn.Linear) and name.startswith("critic"):
            state[f"{name}.weight"].zero_()
            state[f"{name}.bias"].zero_()
            reset.append(name)
    logger.info("reset value head layers: %s", ", ".join(reset) or "none")
    return state


def initialize_policy_from_imitation(
    policy: PolicyValueNetwork,
    checkpoint: str | Path,
    *,
    reset_value_head: bool = True,
) -> PretrainedPolicyInfo:
    """Copy encoder + actor weights from an imitation checkpoint into ``policy``.

    The value head is reset (see :func:`_reset_value_head`).  Compatibility is
    checked first; on any mismatch the policy is left untouched.
    """

    path = str(checkpoint)
    data = read_imitation_checkpoint(path)
    info = _info_from_data(data, path)
    check_policy_compatibility(info, policy)

    source = data["policy"]
    if not isinstance(source, dict):
        raise PolicyCompatibilityError(f"{path} has no policy state dict")

    current = policy.state_dict()
    encoder_actor = sorted(
        key for key in current if key.startswith(("encoder.", "actor."))
    )
    source_encoder_actor = sorted(
        key for key in source if key.startswith(("encoder.", "actor."))
    )
    if set(encoder_actor) != set(source_encoder_actor):
        raise PolicyCompatibilityError(
            "pretrained encoder/actor parameter names do not match the policy: "
            f"only-in-policy={sorted(set(encoder_actor) - set(source_encoder_actor))} "
            f"only-in-checkpoint={sorted(set(source_encoder_actor) - set(encoder_actor))}"
        )
    for key in encoder_actor:
        if tuple(source[key].shape) != tuple(current[key].shape):
            raise PolicyCompatibilityError(
                f"pretrained parameter {key} has shape {tuple(source[key].shape)} "
                f"but the policy expects {tuple(current[key].shape)}"
            )

    merged = dict(current)
    for key in encoder_actor:
        merged[key] = source[key].detach().clone()
    if reset_value_head:
        merged = _reset_value_head(merged, policy)
    policy.load_state_dict(merged, strict=True)
    logger.info(
        "initialized PPO policy from imitation checkpoint %s "
        "(encoder+actor loaded, value head reset; %d parameter tensors)",
        path, len(encoder_actor),
    )
    return info
