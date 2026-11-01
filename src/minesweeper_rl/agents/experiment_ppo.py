"""Small, heavily instrumented live PPO experiment on the Minesweeper website.

This connects the verified PPO trainer to the real browser environment and
plays a short run (default 15 games) while checking, every episode, that:

* every selected action is valid under the reveal mask
* masked actions receive zero probability and the distribution sums to one
* gradients are non-zero after the PPO update
* a PPO update actually runs and moves the parameters
* checkpoints are written
* rewards are recorded and vary across the run
* the browser environment stays stable (no step/reset errors)

It is intentionally tiny and does not alter the model architecture::

    $env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
    python -m minesweeper_rl.agents.experiment_ppo --games 15 --checkpoint ppo_exp.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..browser import BrowserController
from ..config import EnvironmentConfig
from ..dom import GameStatus
from ..environment import MinesweeperEnvironment
from .action_space import FLAGS
from .minesweeper_env import MinesweeperPPOEnv
from .policy import PolicyValueNetwork
from .ppo import PPOConfig, PPOTrainer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EpisodeReport:
    """Verification results for one live game."""

    game: int
    steps: int
    total_reward: float
    outcome: str
    invalid_actions: int
    probability_errors: int
    max_probability_deviation: float
    grad_norm: float
    parameter_delta: float
    updates: int
    saved_checkpoint: bool
    duration_s: float


@dataclass(slots=True)
class ExperimentSummary:
    """Aggregate verification results for the run."""

    reports: list[EpisodeReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    checkpoint_written: bool = False

    @property
    def games(self) -> int:
        return len(self.reports)

    def mean_reward(self, reports: list[EpisodeReport]) -> float:
        if not reports:
            return 0.0
        return float(np.mean([r.total_reward for r in reports]))

    def format(self) -> str:
        if not self.reports:
            return "SUMMARY: no completed games"
        rewards = [r.total_reward for r in self.reports]
        half = len(self.reports) // 2
        first = self.mean_reward(self.reports[:half])
        second = self.mean_reward(self.reports[half:])
        return (
            "SUMMARY "
            f"| games={self.games} errors={len(self.errors)} "
            f"| invalid_actions={sum(r.invalid_actions for r in self.reports)} "
            f"probability_errors={sum(r.probability_errors for r in self.reports)} "
            f"| max_prob_dev={max(r.max_probability_deviation for r in self.reports):.2e} "
            f"| min_grad_norm={min(r.grad_norm for r in self.reports):.3e} "
            f"| updates={sum(r.updates for r in self.reports)} "
            f"| reward[min={min(rewards):.1f} max={max(rewards):.1f} "
            f"unique={len(set(round(r, 3) for r in rewards))}] "
            f"| mean_reward first_half={first:.2f} second_half={second:.2f} "
            f"| checkpoint={self.checkpoint_written}"
        )


def _grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().item())
    return float(np.sqrt(total))


def _parameter_checksum(module: nn.Module) -> float:
    return float(sum(p.detach().abs().sum().item() for p in module.parameters()))


def classify_outcome(terminated: bool, game_status: object) -> str:
    """Return ``"won"``/``"lost"``/``"ended"`` from explicit status, never reward.

    Terminal rewards now include revealed-cell progress, so the old
    ``last_reward == WIN_REWARD`` check is no longer valid; the game status is
    authoritative.
    """

    if not terminated:
        return "ended"
    value = getattr(game_status, "value", game_status)
    if value in (GameStatus.WON.value, GameStatus.LOST.value):
        return str(value)
    return "ended"


def _episode_outcome(trainer: PPOTrainer, buffer: object) -> str:
    """Classify a finished episode using the buffer flag and the env status."""

    terminated_list = getattr(buffer, "terminated", None) or []
    last_terminated = bool(terminated_list[-1]) if terminated_list else False
    env = getattr(trainer, "env", None)
    last_info = getattr(env, "last_info", None) or {}
    return classify_outcome(last_terminated, last_info.get("game_status"))


def _verify_buffer(trainer: PPOTrainer) -> tuple[int, int, float]:
    """Return ``(invalid_actions, probability_errors, max_probability_deviation)``."""

    buffer = trainer._buffer
    invalid_actions = 0
    probability_errors = 0
    max_deviation = 0.0
    for observation, mask, action in zip(
        buffer.observations, buffer.masks, buffer.actions
    ):
        if not bool(mask[action]):
            invalid_actions += 1
        probabilities, current_mask = trainer.action_probabilities(observation)
        if np.any(probabilities[~current_mask] > 1e-12):
            probability_errors += 1
        max_deviation = max(
            max_deviation, abs(float(probabilities.sum()) - 1.0)
        )
    return invalid_actions, probability_errors, max_deviation


def run_experiment(
    trainer: PPOTrainer,
    *,
    games: int,
    checkpoint: Path | None = None,
    checkpoint_every: int = 5,
) -> ExperimentSummary:
    """Play ``games`` live games with per-episode verification and PPO updates."""

    summary = ExperimentSummary()
    for game in range(1, games + 1):
        started = time.perf_counter()
        try:
            stats = trainer.collect_episode()
            buffer = trainer._buffer
            if len(buffer) == 0:
                raise RuntimeError("episode collected no transitions")

            invalid, prob_errors, max_dev = _verify_buffer(trainer)

            before = _parameter_checksum(trainer.policy)
            metrics = trainer.update()
            after = _parameter_checksum(trainer.policy)
            grad = _grad_norm(trainer.policy)
            trainer.iteration += 1

            total_reward = float(sum(buffer.rewards))
            outcome = _episode_outcome(trainer, buffer)

            saved = False
            if checkpoint is not None and game % checkpoint_every == 0:
                trainer.save(checkpoint)
                saved = True

            report = EpisodeReport(
                game=game,
                steps=stats.steps,
                total_reward=total_reward,
                outcome=outcome,
                invalid_actions=invalid,
                probability_errors=prob_errors,
                max_probability_deviation=max_dev,
                grad_norm=grad,
                parameter_delta=abs(after - before),
                updates=int(metrics["updates"]),
                saved_checkpoint=saved,
                duration_s=time.perf_counter() - started,
            )
            summary.reports.append(report)
            logger.info(
                "game %02d | steps=%3d reward=%+7.2f %-7s | invalid=%d prob_err=%d "
                "max_prob_dev=%.1e | grad=%.2e dparam=%.2e updates=%d %s | %.2fs",
                report.game,
                report.steps,
                report.total_reward,
                report.outcome.upper(),
                report.invalid_actions,
                report.probability_errors,
                report.max_probability_deviation,
                report.grad_norm,
                report.parameter_delta,
                report.updates,
                "saved" if saved else "",
                report.duration_s,
            )
        except Exception as error:  # noqa: BLE001 - record and keep the run diagnosable
            message = f"game {game}: {type(error).__name__}: {error}"
            summary.errors.append(message)
            logger.exception("game %02d failed: %s", game, message)
            break

    if checkpoint is not None:
        trainer.save(checkpoint)
        summary.checkpoint_written = checkpoint.exists()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=15,
                        help="number of live games to play (keep this small)")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("ppo_experiment.pt"))
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--url", default=os.environ.get("MINESWEEPER_URL"))
    parser.add_argument("--profile", default=os.environ.get("MINESWEEPER_PROFILE",
                        "browser_profile"))
    parser.add_argument("--diagnostics", action="store_true",
                        default=os.environ.get("MINESWEEPER_DIAGNOSTICS", "").lower()
                        in {"1", "true", "yes"})
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("provide --url or set MINESWEEPER_URL")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        rollout_steps=1,
    )

    controller = BrowserController(args.url, args.profile, wait_for_board=False)
    summary = ExperimentSummary()
    try:
        page = controller.start()
        environment = MinesweeperEnvironment(
            page, EnvironmentConfig(diagnostics=args.diagnostics)
        )
        env = MinesweeperPPOEnv(environment, action_space=FLAGS)
        logger.info(
            "connected | url=%s board=%dx%d actions=%d",
            args.url, env.width, env.height, env.num_actions,
        )
        policy = PolicyValueNetwork(
            width=env.width, height=env.height, action_space=FLAGS
        )
        trainer = PPOTrainer(policy, env, num_actions=env.num_actions, config=config)

        summary = run_experiment(
            trainer,
            games=args.games,
            checkpoint=args.checkpoint,
            checkpoint_every=args.checkpoint_every,
        )
        logger.info("%s", summary.format())
        for message in summary.errors:
            logger.error("error: %s", message)
    finally:
        controller.shutdown()
    return 0 if not summary.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
