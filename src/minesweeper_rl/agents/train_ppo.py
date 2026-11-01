"""Train PPO on the live Minesweeper website.

This is the only place that wires the reusable :class:`PPOTrainer` to the
browser.  It builds the Playwright controller, wraps the browser environment
in :class:`MinesweeperPPOEnv` (flat actions + reveal mask), and runs alternating
rollout collection and PPO updates, checkpointing periodically.

Requires a game URL.  Every environment step is a real browser click, so keep
``--rollout-steps`` modest and expect a training run to be slow::

    $env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
    python -m minesweeper_rl.agents.train_ppo --iterations 50 --checkpoint ppo.pt
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

from ..browser import BrowserController
from ..config import EnvironmentConfig
from ..environment import MinesweeperEnvironment
from .action_space import FLAGS
from .minesweeper_env import MinesweeperPPOEnv
from .policy import PolicyValueNetwork
from .ppo import PPOConfig, PPOTrainer

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50,
                        help="number of rollout/update iterations")
    parser.add_argument("--rollout-steps", type=int, default=32,
                        help="environment steps collected per iteration")
    parser.add_argument("--epochs", type=int, default=4,
                        help="PPO epochs per rollout")
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="checkpoint path to write and optionally resume from")
    parser.add_argument("--resume", action="store_true",
                        help="load --checkpoint before training when it exists")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="save the checkpoint every N iterations")
    parser.add_argument("--url", default=os.environ.get("MINESWEEPER_URL"),
                        help="Minesweeper game URL (or MINESWEEPER_URL)")
    parser.add_argument("--profile", default=os.environ.get("MINESWEEPER_PROFILE",
                        "browser_profile"))
    parser.add_argument("--diagnostics", action="store_true",
                        default=os.environ.get("MINESWEEPER_DIAGNOSTICS", "").lower()
                        in {"1", "true", "yes"})
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.url:
        build_parser().error("provide --url or set MINESWEEPER_URL")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        learning_rate=args.learning_rate,
        rollout_steps=args.rollout_steps,
        num_epochs=args.epochs,
        minibatch_size=args.minibatch_size,
    )

    controller = BrowserController(args.url, args.profile, wait_for_board=False)
    try:
        page = controller.start()
        environment = MinesweeperEnvironment(
            page, EnvironmentConfig(diagnostics=args.diagnostics)
        )
        env = MinesweeperPPOEnv(environment, action_space=FLAGS)
        policy = PolicyValueNetwork(
            width=env.width, height=env.height, action_space=FLAGS
        )
        trainer = PPOTrainer(policy, env, num_actions=env.num_actions, config=config)

        if args.resume and args.checkpoint and args.checkpoint.exists():
            trainer.load(args.checkpoint)
            logger.info("resumed from %s at iteration %d", args.checkpoint, trainer.iteration)

        for _ in range(args.iterations):
            stats = trainer.collect_rollout()
            metrics = trainer.update()
            logger.info(
                "iter %04d | steps=%d episodes=%d mean_return=%.2f | "
                "policy=%.4f value=%.4f entropy=%.4f kl=%.5f early_stop=%.0f",
                trainer.iteration,
                stats.steps,
                stats.episodes,
                stats.mean_return,
                metrics["policy_loss"],
                metrics["value_loss"],
                metrics["entropy"],
                metrics["approx_kl"],
                metrics["early_stopped"],
            )
            trainer.iteration += 1
            if (
                args.checkpoint is not None
                and args.checkpoint_every > 0
                and trainer.iteration % args.checkpoint_every == 0
            ):
                trainer.save(args.checkpoint)
                logger.info("saved checkpoint to %s", args.checkpoint)

        if args.checkpoint is not None:
            trainer.save(args.checkpoint)
            logger.info("saved final checkpoint to %s", args.checkpoint)
    finally:
        controller.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
