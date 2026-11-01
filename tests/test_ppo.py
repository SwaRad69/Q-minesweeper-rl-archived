"""Tests for the reusable PPO trainer.

Covers GAE, the clipped surrogate, rollout collection with masking, parameter
updates, checkpoint round-trips, and end-to-end learning on the deterministic
mock environment.  The trainer is exercised through the plain
``reset()``/``step(action)`` interface, with no browser involved.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from minesweeper_rl.agents import (
    OPEN_ONLY,
    PPOConfig,
    PPOTrainer,
    PolicyValueNetwork,
    TargetCellEnv,
    clipped_surrogate_loss,
    compute_gae,
)


def make_trainer(**config_overrides: object) -> PPOTrainer:
    env = TargetCellEnv(width=4)
    policy = PolicyValueNetwork(
        width=4, height=1, channels=16, action_space=OPEN_ONLY
    )
    config = PPOConfig(
        rollout_steps=16,
        num_epochs=2,
        minibatch_size=8,
        learning_rate=1e-2,
        **config_overrides,
    )
    return PPOTrainer(policy, env, num_actions=4, config=config)


def test_compute_gae_matches_manual_discounted_advantages() -> None:
    advantages, returns = compute_gae(
        rewards=np.array([0.0, 0.0, 1.0]),
        values=np.array([0.0, 0.0, 0.0]),
        terminated=np.array([False, False, False]),
        last_value=0.0,
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert advantages == pytest.approx([1.0, 1.0, 1.0])
    assert returns == pytest.approx([1.0, 1.0, 1.0])


def test_compute_gae_does_not_bootstrap_terminated_steps() -> None:
    advantages, _ = compute_gae(
        rewards=np.array([0.0, 1.0]),
        values=np.array([0.0, 0.0]),
        terminated=np.array([False, True]),
        last_value=100.0,
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert advantages == pytest.approx([1.0, 1.0])


def test_compute_gae_applies_gamma_and_lambda() -> None:
    advantages, _ = compute_gae(
        rewards=np.array([1.0, 0.0]),
        values=np.array([0.0, 0.0]),
        terminated=np.array([False, False]),
        last_value=0.0,
        gamma=0.5,
        gae_lambda=0.5,
    )
    # step 1: delta = 0; step 0: delta = 1, gae = 1 + 0.5*0.5*0 = 1
    assert advantages == pytest.approx([1.0, 0.0])


def test_clipped_surrogate_limits_large_ratios() -> None:
    advantages = torch.tensor([1.0, 1.0])
    ratios = torch.tensor([10.0, 1.1])
    loss = clipped_surrogate_loss(ratios, advantages, clip_coef=0.2)
    # First ratio is clipped to 1.2, second is not clipped.
    assert loss.item() == pytest.approx((1.2 + 1.1) / 2)


def test_rollout_collects_masked_finite_transitions() -> None:
    trainer = make_trainer()
    stats = trainer.collect_rollout()
    buffer = trainer._buffer

    assert stats.steps == 16
    assert len(buffer) == 16
    assert all(mask.shape == (4,) for mask in buffer.masks)
    assert all(np.isfinite(log_prob) for log_prob in buffer.log_probs)
    assert all(np.isfinite(value) for value in buffer.values)

    # One-step episodes: every transition terminates and the clue cell (action
    # 0) is always masked out.
    assert all(buffer.terminated)
    for action, mask in zip(buffer.actions, buffer.masks):
        assert mask[action]
        assert not mask[0]
        assert action in (1, 2, 3)


def test_predict_never_returns_masked_action() -> None:
    trainer = make_trainer()
    for _ in range(20):
        observation = trainer.env.reset()
        action, value = trainer.predict(observation)
        assert trainer.action_mask(observation)[action]
        assert 0 <= action < 4
        assert np.isfinite(value)


def test_update_changes_parameters_and_reports_metrics() -> None:
    trainer = make_trainer()
    trainer.collect_rollout()
    before = trainer.policy.actor.weight.detach().clone()

    metrics = trainer.update()

    assert not torch.equal(before, trainer.policy.actor.weight.detach())
    for key in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "total_loss",
    ):
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_update_requires_a_rollout() -> None:
    trainer = make_trainer()
    with pytest.raises(RuntimeError, match="collect a rollout"):
        trainer.update()


def test_checkpoint_round_trip_restores_policy_and_counters(tmp_path) -> None:
    trainer = make_trainer()
    trainer.train(iterations=2)
    path = trainer.save(tmp_path / "checkpoint.pt")

    assert path.exists()
    restored = make_trainer()
    restored.load(path)

    assert restored.iteration == trainer.iteration
    assert restored.total_steps == trainer.total_steps
    for original, loaded in zip(
        trainer.policy.state_dict().values(), restored.policy.state_dict().values()
    ):
        assert torch.equal(original, loaded)

    restored.collect_rollout()
    restored.update()


def test_checkpoint_rejects_action_count_mismatch(tmp_path) -> None:
    trainer = make_trainer()
    trainer.train(iterations=1)
    path = trainer.save(tmp_path / "checkpoint.pt")

    other = PPOTrainer(
        PolicyValueNetwork(
            width=3, height=1, channels=8, action_space=OPEN_ONLY
        ),
        TargetCellEnv(width=3),
        num_actions=3,
        config=PPOConfig(rollout_steps=8, num_epochs=1, minibatch_size=8),
    )
    with pytest.raises(ValueError, match="num_actions"):
        other.load(path)


def test_ppo_learns_the_target_cell_mock_environment() -> None:
    torch.manual_seed(0)
    env = TargetCellEnv(width=4)
    policy = PolicyValueNetwork(
        width=4, height=1, channels=16, action_space=OPEN_ONLY
    )
    config = PPOConfig(
        rollout_steps=64,
        num_epochs=8,
        minibatch_size=64,
        learning_rate=1e-2,
        entropy_coef=0.0,
        gamma=0.99,
        gae_lambda=0.95,
    )
    trainer = PPOTrainer(policy, env, num_actions=4, config=config)

    def greedy_accuracy(episodes: int = 60) -> float:
        correct = 0
        for _ in range(episodes):
            observation = env.reset()
            action, _ = trainer.predict(observation, greedy=True)
            _, _, _, _, info = env.step(action)
            correct += int(info["correct"])
        return correct / episodes

    accuracy_before = greedy_accuracy()
    history = trainer.train(iterations=120)
    assert history
    assert np.isfinite(history[-1]["total_loss"])

    accuracy_after = greedy_accuracy()
    assert accuracy_before < 0.5
    assert accuracy_after == 1.0
    assert accuracy_after > accuracy_before
