"""Robust PPO checkpoint persistence: complete, atomic, versioned training state.

A checkpoint written here contains everything needed to *continue* PPO training
rather than merely reload weights: policy weights, optimizer state, update /
episode / step counters, PPO hyperparameters, model metadata and the Python,
NumPy, PyTorch (and, when used, CUDA) RNG states.  Temporary rollout data is not
saved because the existing trainer always finishes an update before a safe
training boundary, so the rollout buffer is empty at save time.

The module is deliberately decoupled from the trainer: it works with any object
exposing ``policy``, ``optimizer``, ``config``, ``num_actions``, ``iteration``,
``episode_count`` and ``total_steps``.  :class:`CheckpointManager` adds the
``checkpoint_latest.pt`` / ``checkpoint_best.pt`` lifecycle, keeping the best
checkpoint according to a caller-supplied metric.

Writes are atomic: the payload is written to a temporary file in the same
directory and then ``os.replace``-d onto the destination, so a crash while
saving never destroys the previous valid checkpoint.  Version and model metadata
allow incompatible checkpoints to be rejected instead of loaded silently.
"""

from __future__ import annotations

import logging
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..agents.action_space import (
    DEFAULT_ACTION_SPACE,
    OPEN_ONLY,
)

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT = "minesweeper_ppo"
#: v5 = flag-aware OPEN+FLAG+CHORD 11-channel / 1440-action experiment.
#: v4 = the earlier OPEN-only 10-channel / 480-action experiment, still loadable
#: only by an OPEN_ONLY model so the completed baseline is never invalidated.
#: Any other version is rejected rather than silently loaded.
CHECKPOINT_VERSION = 5
LEGACY_OPEN_ONLY_VERSION = 4
LATEST_FILENAME = "checkpoint_latest.pt"
BEST_FILENAME = "checkpoint_best.pt"


class CheckpointError(ValueError):
    """Raised when a checkpoint is missing, unreadable or incompatible."""


def _uses_cuda(trainer: Any) -> bool:
    """Return whether the trainer's policy is (or should be) running on CUDA."""

    device = getattr(trainer, "device", None)
    if device is not None and getattr(device, "type", None) == "cuda":
        return True
    try:
        return any(parameter.is_cuda for parameter in trainer.policy.parameters())
    except AttributeError:
        return False


def _encode_numpy_state(state: tuple[Any, ...]) -> dict[str, Any]:
    """Encode ``np.random.get_state()`` as plain, ``weights_only``-safe data."""

    name, keys, position, has_gauss, cached_gaussian = state
    return {
        "name": str(name),
        "keys": np.asarray(keys).astype(np.int64).tolist(),
        "pos": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _decode_numpy_state(data: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(data["name"]),
        np.asarray(data["keys"], dtype=np.uint32),
        int(data["pos"]),
        int(data["has_gauss"]),
        float(data["cached_gaussian"]),
    )


def capture_rng_state(*, include_cuda: bool = False) -> dict[str, Any]:
    """Capture the global Python, NumPy and PyTorch RNG states.

    PyTorch generator state is stored as a plain list of ints so the checkpoint
    remains loadable with ``weights_only=True``.  CUDA state is only captured
    when ``include_cuda`` is true *and* CUDA is available.
    """

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": _encode_numpy_state(np.random.get_state()),
        "torch": torch.get_rng_state().tolist(),
    }
    if include_cuda and torch.cuda.is_available():
        state["cuda"] = [
            tensor.detach().cpu().to(torch.uint8).tolist()
            for tensor in torch.cuda.get_rng_state_all()
        ]
    return state


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    """Restore RNG states captured by :func:`capture_rng_state`."""

    if not state:
        return
    if state.get("python") is not None:
        random.setstate(tuple(state["python"]))
    if state.get("numpy") is not None:
        np.random.set_state(_decode_numpy_state(state["numpy"]))
    if state.get("torch") is not None:
        torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    cuda = state.get("cuda")
    if cuda and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.tensor(device_state, dtype=torch.uint8) for device_state in cuda]
        )


def _policy_channels(policy: Any) -> int | None:
    """Best-effort inference of the CNN hidden channel width."""

    channels = getattr(policy, "channels", None)
    if channels is not None:
        return int(channels)
    encoder = getattr(policy, "encoder", None)
    try:
        return int(encoder[0].out_channels)
    except (TypeError, IndexError, AttributeError):
        return None


def _observation_channels(policy: Any) -> int | None:
    """Best-effort inference of the CNN input (observation) channel count."""

    channels = getattr(policy, "observation_channels", None)
    if channels is not None:
        return int(channels)
    encoder = getattr(policy, "encoder", None)
    try:
        return int(encoder[0].in_channels)
    except (TypeError, IndexError, AttributeError):
        return None


def action_space_of(trainer: Any) -> str:
    """Return the action-space identity of a trainer (policy then env)."""

    for owner in (getattr(trainer, "policy", None), getattr(trainer, "env", None)):
        action_space = getattr(owner, "action_space", None)
        if action_space:
            return str(action_space)
    return DEFAULT_ACTION_SPACE


def _observation_mismatch_message(saved: int, expected: int) -> str:
    return (
        f"checkpoint was trained with {saved} observation channel(s) but the "
        f"current model expects {expected}; the observation encoding is "
        f"incompatible. The checkpoint was left untouched - start a new run "
        f"(--fresh archives the old checkpoint instead of deleting it)."
    )


def model_info(trainer: Any) -> dict[str, Any]:
    """Return architecture metadata used to detect incompatible checkpoints."""

    policy = trainer.policy
    info: dict[str, Any] = {
        "class": type(policy).__name__,
        "width": int(getattr(policy, "width", -1)),
        "height": int(getattr(policy, "height", -1)),
        "num_actions": int(
            getattr(policy, "num_actions", getattr(trainer, "num_actions", -1))
        ),
        "action_space": action_space_of(trainer),
    }
    channels = _policy_channels(policy)
    if channels is not None:
        info["channels"] = channels
    observation_channels = _observation_channels(policy)
    if observation_channels is not None:
        info["observation_channels"] = observation_channels
    return info


def _config_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    return dict(config)


def build_checkpoint(
    trainer: Any,
    *,
    metrics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the complete, serialisable training-state payload.

    ``metadata`` records experiment provenance (algorithm, seed, URL, ...) so a
    run can be identified and reproduced later; it is never used by training.
    """

    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "action_space": action_space_of(trainer),
        "policy": trainer.policy.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "config": _config_dict(trainer.config),
        "num_actions": int(trainer.num_actions),
        "iteration": int(getattr(trainer, "iteration", 0)),
        "episode_count": int(getattr(trainer, "episode_count", 0)),
        "total_steps": int(getattr(trainer, "total_steps", 0)),
        "model": model_info(trainer),
        "rng": capture_rng_state(include_cuda=_uses_cuda(trainer)),
        "metrics": dict(metrics or {}),
        "metadata": dict(metadata or {}),
    }


def atomic_save(payload: Any, path: str | Path) -> Path:
    """Serialise ``payload`` to ``path`` atomically (temp file then replace)."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=destination.name + ".", suffix=".tmp"
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_checkpoint(
    trainer: Any,
    path: str | Path,
    *,
    metrics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Build and atomically write a checkpoint for ``trainer``."""

    return atomic_save(
        build_checkpoint(trainer, metrics=metrics, metadata=metadata), path
    )


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read a checkpoint payload, raising :class:`CheckpointError` on failure."""

    source = Path(path)
    if not source.exists():
        raise CheckpointError(f"checkpoint not found: {source}")
    try:
        data = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - surface a single clear error type
        raise CheckpointError(f"could not read checkpoint {source}: {error}") from error
    if not isinstance(data, Mapping) or "policy" not in data:
        raise CheckpointError(f"{source} is not a PPO checkpoint")
    return dict(data)


def validate_checkpoint(data: Mapping[str, Any], trainer: Any) -> None:
    """Reject a checkpoint that is incompatible with ``trainer``.

    Checks the format, version, action space and the board size, action-space
    size and model architecture metadata.  A v4 checkpoint is the completed
    OPEN-only baseline and is accepted only by an OPEN_ONLY model; every other
    action-space combination is rejected with a clear error.
    """

    checkpoint_format = data.get("format")
    if checkpoint_format != CHECKPOINT_FORMAT:
        raise CheckpointError(
            f"incompatible checkpoint format: {checkpoint_format!r} "
            f"(expected {CHECKPOINT_FORMAT!r})"
        )

    current_space = action_space_of(trainer)
    saved_model = data.get("model")
    checkpoint_space = data.get("action_space")
    if checkpoint_space is None and isinstance(saved_model, Mapping):
        checkpoint_space = saved_model.get("action_space")

    version = data.get("version")
    if version is None:
        raise CheckpointError(
            "checkpoint has no version; it was written by an incompatible "
            f"older revision (expected version {CHECKPOINT_VERSION}). The "
            "checkpoint was left untouched"
        )
    if not isinstance(version, int):
        raise CheckpointError(f"checkpoint version {version!r} is invalid")
    if version == LEGACY_OPEN_ONLY_VERSION:
        if current_space != OPEN_ONLY:
            raise CheckpointError(
                f"checkpoint version {LEGACY_OPEN_ONLY_VERSION} is the OPEN-only "
                f"baseline but the current model uses action space "
                f"{current_space!r}; loading it would silently reinterpret the "
                f"OPEN actions. Use an OPEN_ONLY model or start a new run "
                f"(--fresh archives the old checkpoint instead of deleting it)."
            )
    elif version != CHECKPOINT_VERSION:
        raise CheckpointError(
            f"checkpoint version {version!r} is not supported "
            f"(expected {CHECKPOINT_VERSION} for {current_space!r}, or "
            f"{LEGACY_OPEN_ONLY_VERSION} for the OPEN-only baseline)"
        )

    if checkpoint_space is not None and str(checkpoint_space) != current_space:
        raise CheckpointError(
            f"checkpoint action space {checkpoint_space!r} does not match the "
            f"current model action space {current_space!r}"
        )

    expected_actions = int(getattr(trainer, "num_actions"))
    saved_actions = data.get("num_actions")
    if saved_actions is not None and int(saved_actions) != expected_actions:
        raise CheckpointError(
            f"checkpoint num_actions {saved_actions} does not match trainer "
            f"num_actions {expected_actions}"
        )

    current = model_info(trainer)
    current_observation = current.get("observation_channels")
    if saved_model:
        for key in ("class", "width", "height", "channels", "action_space"):
            if key in saved_model and key in current and saved_model[key] != current[key]:
                raise CheckpointError(
                    f"checkpoint model {key} {saved_model[key]!r} does not match "
                    f"trainer {key} {current[key]!r}"
                )
        saved_observation = saved_model.get("observation_channels")
        if (
            saved_observation is not None
            and current_observation is not None
            and int(saved_observation) != int(current_observation)
        ):
            raise CheckpointError(
                _observation_mismatch_message(
                    int(saved_observation), int(current_observation)
                )
            )

    # Older checkpoints (pre-v3) did not record the observation-channel count.
    # Inspect the serialized first convolution so an incompatible encoding is
    # rejected here with a clear error instead of a raw load_state_dict size
    # mismatch later.
    policy_state = data.get("policy")
    weight = (
        policy_state.get("encoder.0.weight")
        if isinstance(policy_state, Mapping)
        else None
    )
    if (
        weight is not None
        and current_observation is not None
        and getattr(weight, "ndim", 0) == 4
    ):
        saved_observation = int(weight.shape[1])
        if saved_observation != int(current_observation):
            raise CheckpointError(
                _observation_mismatch_message(saved_observation, int(current_observation))
            )


def _restore_config(base: Any, values: Mapping[str, Any]) -> Any:
    """Rebuild a config object from its serialised form, preserving its type."""

    if is_dataclass(base):
        return type(base)(**values)
    return dict(values)


def load_checkpoint(trainer: Any, data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate ``data`` and restore the full training state into ``trainer``."""

    validate_checkpoint(data, trainer)
    trainer.policy.load_state_dict(data["policy"])
    if "optimizer" in data:
        trainer.optimizer.load_state_dict(data["optimizer"])
    if "config" in data:
        current = _config_dict(trainer.config)
        saved = dict(data["config"])
        if current != saved:
            logger.warning(
                "checkpoint hyperparameters differ from the current run; using "
                "the checkpoint values to keep the experiment consistent"
            )
        trainer.config = _restore_config(trainer.config, data["config"])
    trainer.iteration = int(data.get("iteration", 0))
    # Legacy (v1) checkpoints predate ``episode_count``; in the episode-batch
    # training loop one update corresponds to one episode, so fall back to it.
    trainer.episode_count = int(
        data.get("episode_count", data.get("iteration", getattr(trainer, "episode_count", 0)))
    )
    trainer.total_steps = int(data.get("total_steps", 0))
    restore_rng_state(data.get("rng"))
    return dict(data)


class CheckpointManager:
    """Own the ``latest`` and ``best`` checkpoint files for one run directory.

    ``best`` is only overwritten when the supplied metric improves on the best
    seen so far (higher is better unless ``maximize=False``).  The best metric is
    recovered from the best checkpoint on disk so it survives process restarts.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        latest_name: str = LATEST_FILENAME,
        best_name: str = BEST_FILENAME,
        maximize: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.run_dir / latest_name
        self.best_path = self.run_dir / best_name
        self.maximize = bool(maximize)
        self.metadata = dict(metadata or {})
        self.best_metric: float | None = None
        self._recover_best_metric()

    def _recover_best_metric(self) -> None:
        if not self.best_path.exists():
            return
        try:
            data = read_checkpoint(self.best_path)
        except CheckpointError as error:
            logger.warning("ignoring unreadable best checkpoint: %s", error)
            return
        metric = data.get("metrics", {}).get("selection_metric")
        if metric is not None:
            self.best_metric = float(metric)

    def _is_improvement(self, metric: float | None) -> bool:
        if metric is None:
            return False
        if self.best_metric is None:
            return True
        return metric > self.best_metric if self.maximize else metric < self.best_metric

    def save_latest(
        self, trainer: Any, *, metrics: Mapping[str, Any] | None = None
    ) -> Path:
        """Always write the most recent recoverable state."""

        return write_checkpoint(
            trainer, self.latest_path, metrics=metrics, metadata=self.metadata
        )

    def save_best(
        self,
        trainer: Any,
        *,
        metric: float | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> Path | None:
        """Write the best checkpoint; return ``None`` if the metric is not better."""

        merged = dict(metrics or {})
        if metric is None and "selection_metric" in merged:
            metric = float(merged["selection_metric"])
        if metric is not None:
            merged["selection_metric"] = float(metric)
        if not self._is_improvement(metric):
            logger.info(
                "best checkpoint kept (metric=%s, best=%s)", metric, self.best_metric
            )
            return None
        path = write_checkpoint(
            trainer, self.best_path, metrics=merged, metadata=self.metadata
        )
        self.best_metric = float(metric)
        return path

    def save(
        self,
        trainer: Any,
        *,
        metric: float | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> tuple[Path, Path | None]:
        """Write both checkpoints, returning ``(latest, best_or_none)``."""

        latest = self.save_latest(trainer, metrics=metrics)
        best = self.save_best(trainer, metric=metric, metrics=metrics)
        return latest, best

    def load_latest(self, trainer: Any) -> dict[str, Any]:
        """Restore the latest checkpoint into ``trainer``."""

        return load_checkpoint(trainer, read_checkpoint(self.latest_path))

    def load_best(self, trainer: Any) -> dict[str, Any]:
        """Restore the best checkpoint into ``trainer``."""

        return load_checkpoint(trainer, read_checkpoint(self.best_path))
