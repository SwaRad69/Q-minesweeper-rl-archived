"""Reusable PPO (Proximal Policy Optimization) for an actor-critic policy.

The trainer is environment agnostic.  It only requires an environment object
with::

    reset() -> observation
    step(action) -> (observation, reward, terminated, truncated, info)

and a policy module mapping a batch of observations to ``(action_logits,
value)``.  Action masking is optional and supplied by the caller (or read from
an ``action_mask(observation)`` method on the environment).  No Minesweeper or
browser logic lives here, so the same trainer works for other tasks.

Implemented features: rollout collection, action masking, log probabilities,
value estimates, rewards, discounted returns, Generalized Advantage Estimation,
the clipped surrogate objective, value loss, entropy bonus, minibatch
optimization, multiple epochs per rollout, optional KL early stopping, and
checkpoint saving/loading.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_


class Environment(Protocol):
    """Minimal environment interface consumed by :class:`PPOTrainer`."""

    def reset(self) -> Any:
        """Start a new episode and return the initial observation."""

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Apply an action and return ``(obs, reward, terminated, truncated, info)``."""


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Hyperparameters for :class:`PPOTrainer`."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    rollout_steps: int = 128
    num_epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    target_kl: float | None = None

    def __post_init__(self) -> None:
        if self.rollout_steps < 1:
            raise ValueError("rollout_steps must be at least 1")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be at least 1")
        if self.minibatch_size < 1:
            raise ValueError("minibatch_size must be at least 1")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")


@dataclass(slots=True)
class RolloutStats:
    """Summary of one rollout collection."""

    steps: int
    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)

    @property
    def episodes(self) -> int:
        return len(self.episode_returns)

    @property
    def mean_return(self) -> float:
        if not self.episode_returns:
            return 0.0
        return float(np.mean(self.episode_returns))


@dataclass(slots=True)
class RolloutBuffer:
    """Per-step data gathered during rollout collection."""

    observations: list[Any] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)

    def clear(self) -> None:
        self.observations.clear()
        self.masks.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.terminated.clear()


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    last_value: float,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(advantages, returns)`` from a rollout using GAE(lambda)."""

    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=bool)
    if not (rewards.shape == values.shape == terminated.shape):
        raise ValueError("rewards, values and terminated must share a shape")

    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    for step in reversed(range(len(rewards))):
        next_value = last_value if step == len(rewards) - 1 else values[step + 1]
        non_terminal = 1.0 - float(terminated[step])
        delta = rewards[step] + gamma * next_value * non_terminal - values[step]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[step] = last_gae
    return advantages, advantages + values


def clipped_surrogate_loss(
    ratios: Tensor, advantages: Tensor, clip_coef: float
) -> Tensor:
    """Return the mean PPO clipped surrogate objective (to be maximized)."""

    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
    return torch.min(unclipped, clipped).mean()


class PPOTrainer:
    """Actor-critic PPO trainer over the :class:`Environment` interface."""

    def __init__(
        self,
        policy: nn.Module,
        env: Environment,
        *,
        num_actions: int,
        mask_fn: Callable[[Any], np.ndarray] | None = None,
        config: PPOConfig = PPOConfig(),
        device: str | torch.device = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        if num_actions < 1:
            raise ValueError("num_actions must be at least 1")
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.env = env
        self.num_actions = int(num_actions)
        self.config = config
        if mask_fn is None and hasattr(env, "action_mask"):
            mask_fn = env.action_mask  # type: ignore[assignment]
        self._mask_fn = mask_fn
        if optimizer is None:
            optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=config.learning_rate
            )
        self.optimizer = optimizer
        self.iteration = 0
        self.episode_count = 0
        self.total_steps = 0
        self._buffer = RolloutBuffer()
        self._last_obs: Any = None
        self.last_rollout: RolloutStats | None = None

    def action_mask(self, observation: Any) -> np.ndarray:
        """Return the boolean validity mask for one observation."""

        if self._mask_fn is None:
            return np.ones(self.num_actions, dtype=bool)
        mask = np.asarray(self._mask_fn(observation), dtype=bool).reshape(-1)
        if mask.shape != (self.num_actions,):
            raise ValueError(
                f"mask must have {self.num_actions} entries, got {mask.shape}"
            )
        return mask

    def _obs_tensor(self, observations: Any) -> Tensor:
        array = np.stack([np.asarray(obs) for obs in observations])
        return torch.as_tensor(array, device=self.device)

    def _mask_tensor(self, masks: Any) -> Tensor:
        return torch.as_tensor(
            np.stack([np.asarray(mask) for mask in masks]),
            dtype=torch.bool,
            device=self.device,
        )

    def _forward(self, observations: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
        logits, values = self.policy(observations)
        logits = logits.reshape(logits.shape[0], -1)
        if logits.shape[1] != self.num_actions:
            raise ValueError(
                f"policy produced {logits.shape[1]} actions, expected {self.num_actions}"
            )
        masks = masks.reshape(logits.shape)
        logits = logits.masked_fill(~masks, float("-inf"))
        values = values.reshape(-1)
        return logits, values

    def collect_rollout(
        self, *, steps: int | None = None, stop_at_episode_end: bool = False
    ) -> RolloutStats:
        """Collect transitions into the buffer.

        By default collects ``config.rollout_steps`` transitions.  Pass
        ``steps`` to override, and ``stop_at_episode_end=True`` to stop as soon
        as one episode terminates (useful for episode-batch training).
        """

        limit = self.config.rollout_steps if steps is None else steps
        if limit < 1:
            raise ValueError("steps must be at least 1")

        buffer = RolloutBuffer()
        observation = self.env.reset()
        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        running_return = 0.0
        running_length = 0

        while len(buffer) < limit:
            mask = self.action_mask(observation)
            if not mask.any():
                observation = self.env.reset()
                continue

            obs_tensor = self._obs_tensor([observation])
            mask_tensor = self._mask_tensor([mask])
            with torch.no_grad():
                logits, values = self._forward(obs_tensor, mask_tensor)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                log_prob = distribution.log_prob(action)

            action_index = int(action.item())
            next_observation, reward, terminated, truncated, _ = self.env.step(
                action_index
            )
            done = bool(terminated) or bool(truncated)

            buffer.observations.append(observation)
            buffer.masks.append(mask)
            buffer.actions.append(action_index)
            buffer.log_probs.append(float(log_prob.item()))
            buffer.values.append(float(values.item()))
            buffer.rewards.append(float(reward))
            buffer.terminated.append(done)

            running_return += float(reward)
            running_length += 1
            if done:
                episode_returns.append(running_return)
                episode_lengths.append(running_length)
                running_return = 0.0
                running_length = 0
                if stop_at_episode_end:
                    observation = next_observation
                    break
                observation = self.env.reset()
            else:
                observation = next_observation

        # An episode-batch collection whose step budget ran out before the game
        # reached a terminal state (for example a live game that never ends) is
        # closed as a truncated episode.  This guarantees the completed-episode
        # counter advances, so the training loop cannot spin forever on a stuck
        # game; the truncated transition is marked terminal so GAE does not
        # bootstrap past it.
        if stop_at_episode_end and running_length > 0 and buffer.terminated:
            buffer.terminated[-1] = True
            episode_returns.append(running_return)
            episode_lengths.append(running_length)

        self.total_steps += len(buffer)
        self.episode_count += len(episode_returns)
        self._buffer = buffer
        self._last_obs = observation
        self.last_rollout = RolloutStats(
            steps=len(buffer),
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
        )
        return self.last_rollout

    def collect_episode(self, *, max_steps: int = 2_000) -> RolloutStats:
        """Collect one episode, truncating and counting it at ``max_steps``.

        A single Expert game cannot need anywhere near ``max_steps`` legitimate
        actions (381 safe cells), so hitting the budget means the game is stuck.
        The partial episode is recorded as a truncated episode so the caller
        always makes progress.
        """

        return self.collect_rollout(steps=max_steps, stop_at_episode_end=True)

    def _bootstrap_value(self) -> float:
        mask = self.action_mask(self._last_obs)
        if not mask.any():
            return 0.0
        with torch.no_grad():
            _, values = self._forward(
                self._obs_tensor([self._last_obs]), self._mask_tensor([mask])
            )
        return float(values.item())

    def predict(self, observation: Any, *, greedy: bool = True) -> tuple[int, float]:
        """Return ``(action, value)`` for one observation under the policy.

        Masked (invalid) actions are never selected.
        """

        mask = self.action_mask(observation)
        with torch.no_grad():
            logits, values = self._forward(
                self._obs_tensor([observation]), self._mask_tensor([mask])
            )
            if greedy:
                action = int(logits.argmax(dim=1).item())
            else:
                action = int(Categorical(logits=logits).sample().item())
        return action, float(values.item())

    def action_probabilities(
        self, observation: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(probabilities, mask)`` for one observation.

        Masked actions have probability exactly ``0`` and the probabilities sum
        to ``1`` over the valid actions.
        """

        mask = self.action_mask(observation)
        with torch.no_grad():
            logits, _ = self._forward(
                self._obs_tensor([observation]), self._mask_tensor([mask])
            )
            probabilities = torch.softmax(logits, dim=1)[0]
        return probabilities.cpu().numpy(), mask

    def update(self) -> dict[str, float]:
        """Run PPO epochs/minibatches over the most recent rollout."""

        buffer = self._buffer
        if len(buffer) == 0:
            raise RuntimeError("collect a rollout before calling update")

        config = self.config
        observations = self._obs_tensor(buffer.observations)
        masks = self._mask_tensor(buffer.masks)
        actions = torch.as_tensor(buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(
            buffer.log_probs, dtype=torch.float32, device=self.device
        )

        advantages, returns = compute_gae(
            np.asarray(buffer.rewards, dtype=np.float64),
            np.asarray(buffer.values, dtype=np.float64),
            np.asarray(buffer.terminated, dtype=bool),
            self._bootstrap_value(),
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        advantages_t = torch.as_tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        if config.normalize_advantages:
            advantages_t = (advantages_t - advantages_t.mean()) / (
                advantages_t.std(unbiased=False) + 1e-8
            )

        total = len(buffer)
        batch_size = min(config.minibatch_size, total)
        parameters = [p for p in self.policy.parameters() if p.requires_grad]
        metrics: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "total_loss": 0.0,
        }
        update_count = 0
        kl_for_epoch = 0.0
        early_stopped = False

        for _ in range(config.num_epochs):
            permutation = torch.randperm(total, device=self.device)
            kl_for_epoch = 0.0
            epoch_batches = 0
            for start in range(0, total, batch_size):
                indices = permutation[start : start + batch_size]
                logits, values = self._forward(
                    observations[indices], masks[indices]
                )
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(actions[indices])
                entropy = distribution.entropy().mean()
                ratios = torch.exp(new_log_probs - old_log_probs[indices])

                policy_loss = -clipped_surrogate_loss(
                    ratios, advantages_t[indices], config.clip_coef
                )
                value_loss = (values - returns_t[indices]).pow(2).mean()
                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                if parameters:
                    clip_grad_norm_(parameters, config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs[indices] - new_log_probs).mean()
                    clip_fraction = (
                        (ratios - 1.0).abs() > config.clip_coef
                    ).float().mean()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy.item())
                metrics["approx_kl"] += float(approx_kl.item())
                metrics["clip_fraction"] += float(clip_fraction.item())
                metrics["total_loss"] += float(loss.item())
                kl_for_epoch += float(approx_kl.item())
                update_count += 1
                epoch_batches += 1

            if (
                config.target_kl is not None
                and epoch_batches > 0
                and kl_for_epoch / epoch_batches > config.target_kl
            ):
                early_stopped = True
                break

        if update_count:
            for key in metrics:
                metrics[key] /= update_count
        metrics["updates"] = float(update_count)
        metrics["early_stopped"] = float(early_stopped)
        return metrics

    def train(
        self,
        *,
        iterations: int | None = None,
        total_steps: int | None = None,
    ) -> list[dict[str, float]]:
        """Alternate rollout collection and PPO updates.

        Provide either ``iterations`` or a ``total_steps`` budget; when only the
        latter is given the iteration count is derived from ``rollout_steps``.
        Returns one metrics dictionary per iteration.
        """

        if iterations is None:
            if total_steps is None:
                raise ValueError("provide iterations or total_steps")
            iterations = max(1, total_steps // self.config.rollout_steps)
        if iterations < 1:
            raise ValueError("iterations must be at least 1")

        history: list[dict[str, float]] = []
        for _ in range(iterations):
            stats = self.collect_rollout()
            metrics = self.update()
            metrics["iteration"] = float(self.iteration)
            metrics["steps"] = float(stats.steps)
            metrics["episodes"] = float(stats.episodes)
            metrics["mean_episode_return"] = stats.mean_return
            self.iteration += 1
            history.append(metrics)
        return history

    def save(self, path: str | Path) -> Path:
        """Write a complete, atomic checkpoint of the training state."""

        from ..experiment.checkpoint import write_checkpoint

        return write_checkpoint(self, path)

    def load(self, path: str | Path) -> None:
        """Restore a checkpoint written by :meth:`save`."""

        from ..experiment.checkpoint import load_checkpoint, read_checkpoint

        load_checkpoint(self, read_checkpoint(path))
