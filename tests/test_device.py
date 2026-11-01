"""Device-selection and GPU-correctness tests (no browser, no live site).

These tests never *require* CUDA: on a CPU-only machine they verify the CPU
fallback and that a mistaken ``cuda`` request is rejected.  When CUDA is
available they additionally run a tiny synthetic forward/backward smoke test.
"""

from __future__ import annotations

import pytest
import torch

from minesweeper_rl.agents import (
    OPEN_ONLY,
    PPOConfig,
    PPOTrainer,
    PolicyValueNetwork,
    TargetCellEnv,
)
from minesweeper_rl.experiment.device import DEVICE_CHOICES, DeviceInfo, resolve_device


def test_device_choices_are_declared() -> None:
    assert DEVICE_CHOICES == ("auto", "cuda", "cpu")


def test_auto_selects_cuda_when_available_else_cpu() -> None:
    info = resolve_device("auto")

    assert isinstance(info, DeviceInfo)
    assert info.cuda_available == torch.cuda.is_available()
    if torch.cuda.is_available():
        assert info.device.type == "cuda"
        assert info.is_cuda is True
        assert info.cuda_device_name
    else:
        assert info.device.type == "cpu"
        assert info.is_cuda is False


def test_cpu_is_always_available() -> None:
    info = resolve_device("cpu")
    assert info.device.type == "cpu"
    assert info.describe() == "device=cpu"


def test_explicit_cuda_request_requires_cuda() -> None:
    if torch.cuda.is_available():
        info = resolve_device("cuda")
        assert info.device.type == "cuda"
    else:
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda")


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_device("quantum-tpu")


def test_trainer_moves_model_parameters_to_selected_device() -> None:
    info = resolve_device("auto")
    env = TargetCellEnv(width=4)
    policy = PolicyValueNetwork(
        width=4, height=1, channels=8, action_space=OPEN_ONLY
    )
    trainer = PPOTrainer(
        policy,
        env,
        num_actions=env.num_actions,
        config=PPOConfig(rollout_steps=4, num_epochs=1, minibatch_size=4),
        device=info.device,
    )

    assert trainer.device.type == info.device.type
    devices = {parameter.device.type for parameter in trainer.policy.parameters()}
    assert devices == {info.device.type}


def test_rollout_and_update_run_on_selected_device() -> None:
    info = resolve_device("auto")
    env = TargetCellEnv(width=4)
    policy = PolicyValueNetwork(
        width=4, height=1, channels=8, action_space=OPEN_ONLY
    )
    trainer = PPOTrainer(
        policy,
        env,
        num_actions=env.num_actions,
        config=PPOConfig(rollout_steps=8, num_epochs=2, minibatch_size=4),
        device=info.device,
    )

    trainer.collect_rollout()
    metrics = trainer.update()

    assert metrics["updates"] > 0
    for parameter in trainer.policy.parameters():
        assert parameter.device.type == info.device.type


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_synthetic_forward_backward_smoke() -> None:
    """A tiny end-to-end PPO update must work on CUDA when a GPU is present."""

    device = torch.device("cuda")
    env = TargetCellEnv(width=4)
    policy = PolicyValueNetwork(
        width=4, height=1, channels=8, action_space=OPEN_ONLY
    )
    trainer = PPOTrainer(
        policy,
        env,
        num_actions=env.num_actions,
        config=PPOConfig(rollout_steps=8, num_epochs=2, minibatch_size=4),
        device=device,
    )

    trainer.collect_rollout()
    metrics = trainer.update()

    assert trainer.device.type == "cuda"
    assert all(p.is_cuda for p in trainer.policy.parameters())
    assert metrics["updates"] > 0
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))
