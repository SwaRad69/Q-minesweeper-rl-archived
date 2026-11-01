"""Device selection for CNN+PPO training and evaluation.

The CNN/PPO computation should use an available CUDA GPU; everything else
(Playwright, the browser, DOM parsing, JSONL/checkpoint I/O) stays on CPU.
Only tensors cross the CPU/GPU boundary, and the boundary is owned by
:class:`~minesweeper_rl.agents.ppo.PPOTrainer` (policy, optimizer, rollout and
minibatch tensors).

``resolve_device`` turns a user request into a concrete device:

* ``auto`` -- CUDA when available, otherwise CPU (the safe default).
* ``cuda`` -- CUDA only; raises a clear error when CUDA is unavailable so a
  mistaken request is never silently downgraded.
* ``cpu``  -- always CPU.

The selected device is always reported through :meth:`DeviceInfo.log` so a run
never *claims* GPU acceleration while actually running on CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

#: Valid values for the ``--device`` CLI option.
DEVICE_CHOICES = ("auto", "cuda", "cpu")


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Resolved device plus the information needed to log it honestly."""

    device: torch.device
    requested: str
    cuda_available: bool
    cuda_device_name: str | None = None

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def describe(self) -> str:
        if self.is_cuda:
            return f"device=cuda ({self.cuda_device_name})"
        return "device=cpu"

    def log(self, logger: logging.Logger) -> None:
        """Log the requested/selected device and CUDA availability."""

        logger.info("device selection | requested=%s | %s", self.requested, self.describe())
        logger.info("cuda available: %s", self.cuda_available)
        if self.cuda_available and self.cuda_device_name:
            logger.info("cuda device: %s", self.cuda_device_name)


def resolve_device(request: str = "auto") -> DeviceInfo:
    """Resolve a device request into a concrete, available :class:`DeviceInfo`."""

    normalized = (request or "auto").strip().lower()
    if normalized not in DEVICE_CHOICES:
        raise ValueError(
            f"device must be one of {DEVICE_CHOICES}, got {request!r}"
        )

    cuda_available = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if cuda_available else None

    if normalized == "cpu":
        return DeviceInfo(torch.device("cpu"), normalized, cuda_available, name)
    if normalized == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "device 'cuda' was requested but CUDA is not available; use "
                "--device cpu or --device auto"
            )
        return DeviceInfo(torch.device("cuda"), normalized, cuda_available, name)

    chosen = "cuda" if cuda_available else "cpu"
    return DeviceInfo(torch.device(chosen), normalized, cuda_available, name)
