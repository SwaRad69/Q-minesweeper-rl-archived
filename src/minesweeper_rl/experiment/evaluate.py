"""Evaluation entry point: run a frozen checkpoint without learning.

Kept separate from training: it loads a checkpoint, selects actions greedily
(argmax of the masked policy) under ``torch.no_grad()``, never touches an
optimizer, and asserts that the weights are byte-for-byte unchanged after the
run.  It reports win rate and the other episode metrics.

FLAGS-only, matching the canonical training path: 11-channel observation and the
1440-action space.  There is no ``--action-space`` option.

::

    $env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
    python -m minesweeper_rl.experiment.evaluate \
        --checkpoint experiments/cnn_ppo_flags_v1/checkpoint_latest.pt --games 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from ..agents.minesweeper_env import MinesweeperPPOEnv
from ..agents.action_space import FLAGS
from ..agents.policy import PolicyValueNetwork
from ..config import EnvironmentConfig
from .browser import open_environment
from .checkpoint import read_checkpoint, validate_checkpoint
from .device import DEVICE_CHOICES, resolve_device
from .metrics import MetricsLogger, MetricsSummary, summarize
from .plots import plot_evaluation
from .tracking import TrackedEnvironment

logger = logging.getLogger(__name__)


EVALUATION_RESULTS_FILE = "evaluation_results.json"


def write_evaluation_results(run_dir: str | Path, payload: dict) -> Path:
    """Write a machine-readable evaluation summary atomically.

    The raw per-episode records stay in ``episodes.jsonl``; this is the small
    comparable summary later cross-experiment tables read without touching the
    original data.
    """

    destination = Path(run_dir) / EVALUATION_RESULTS_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=destination.name + ".", suffix=".tmp"
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_frozen_policy(
    checkpoint: str | Path,
    policy: nn.Module,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load policy weights after the same compatibility checks as training.

    The checkpoint is validated (format, version, board, action count, channels
    and observation-channel count) before any weights are touched.  An
    incompatible checkpoint -- including one written with the old 10-channel
    observation encoding -- raises :class:`~.checkpoint.CheckpointError` instead
    of a raw ``load_state_dict`` size mismatch, and is never overwritten.
    """

    data = read_checkpoint(checkpoint)
    validate_checkpoint(
        data,
        SimpleNamespace(policy=policy, num_actions=int(policy.num_actions)),
    )
    policy.load_state_dict(data["policy"])
    policy.to(device)
    policy.eval()
    return data


def parameter_checksum(module: nn.Module) -> float:
    """Sum of absolute parameter values, used to prove weights do not change."""

    return float(sum(p.detach().abs().sum().item() for p in module.parameters()))


def greedy_action(
    policy: nn.Module,
    observation: Any,
    mask: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> tuple[int, float]:
    """Deterministic masked argmax action and its value estimate."""

    observation_tensor = torch.as_tensor(np.asarray(observation), device=device)
    mask_tensor = torch.as_tensor(
        np.asarray(mask), dtype=torch.bool, device=device
    )
    with torch.no_grad():
        logits, value = policy(observation_tensor)
        logits = logits.reshape(-1).masked_fill(~mask_tensor, float("-inf"))
        action = int(logits.argmax().item())
        estimate = float(value.reshape(-1).item())
    return action, estimate


@dataclass(slots=True)
class EvaluationReport:
    """Outcome of one evaluation session."""

    games: int
    summary: MetricsSummary
    weights_unchanged: bool
    stopped_early: bool = False
    errors: list[str] = field(default_factory=list)


def run_evaluation(
    policy: nn.Module,
    tracker: TrackedEnvironment,
    metrics: MetricsLogger,
    *,
    games: int,
    device: str | torch.device = "cpu",
    max_consecutive_failures: int = 3,
    stop_event: threading.Event | None = None,
    emit: Callable[[str], None] | None = None,
) -> EvaluationReport:
    """Play ``games`` games greedily and record metrics; never update weights."""

    say = emit or logger.info
    policy.eval()
    before = parameter_checksum(policy)
    errors: list[str] = []
    stopped_early = False
    consecutive_failures = 0
    completed = 0

    for game in range(1, games + 1):
        if stop_event is not None and stop_event.is_set():
            stopped_early = True
            break
        try:
            observation = tracker.reset()
            done = False
            while not done:
                mask = tracker.action_mask(observation)
                action, _ = greedy_action(policy, observation, mask, device=device)
                observation, _, terminated, truncated, _ = tracker.step(action)
                done = bool(terminated) or bool(truncated)
            consecutive_failures = 0
            completed += 1
            episode = tracker.last_episode
            if episode is not None:
                metrics.log_episode(episode)
                say(
                    f"eval game {game:03d} | steps={episode.steps:3d} "
                    f"reward={episode.reward:+7.2f} outcome={episode.outcome:<7} "
                    f"mine={int(episode.mine_hit)} invalid={episode.invalid_actions}"
                )
        except KeyboardInterrupt:
            stopped_early = True
            break
        except Exception as error:  # noqa: BLE001
            consecutive_failures += 1
            message = f"eval game {game}: {type(error).__name__}: {error}"
            errors.append(message)
            logger.exception("evaluation failure: %s", message)
            if consecutive_failures >= max_consecutive_failures:
                stopped_early = True
                break

    after = parameter_checksum(policy)
    weights_unchanged = bool(np.isclose(before, after, rtol=1e-12, atol=0.0))
    if not weights_unchanged:
        raise RuntimeError(
            "evaluation modified policy weights; it must stay frozen"
        )

    summary = summarize(metrics.episodes, environment_failures=tracker.failures)
    return EvaluationReport(
        games=completed,
        summary=summary,
        weights_unchanged=weights_unchanged,
        stopped_early=stopped_early,
        errors=errors,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("experiments/cnn_ppo_flags_v1/checkpoint_latest.pt"),
                        help="frozen checkpoint to evaluate")
    parser.add_argument("--games", type=int, default=10,
                        help="number of evaluation games")
    parser.add_argument("--run-dir", type=Path,
                        default=Path("experiments/cnn_ppo_flags_v1_eval"),
                        help="directory for evaluation metrics and plots")
    parser.add_argument("--device", default="auto", choices=list(DEVICE_CHOICES),
                        help="compute device for greedy action selection: "
                             "auto (CUDA if available else CPU), cuda, or cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--no-plots", action="store_true")
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
    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        device_info = resolve_device(args.device)
    except (ValueError, RuntimeError) as error:
        logger.error("invalid --device: %s", error)
        return 1
    device_info.log(logger)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = MetricsLogger(run_dir)
    stop_event = threading.Event()

    with open_environment(
        args.url, args.profile, config=EnvironmentConfig(diagnostics=args.diagnostics)
    ) as environment:
        env = MinesweeperPPOEnv(environment, action_space=FLAGS)
        policy = PolicyValueNetwork(
            width=env.width, height=env.height, action_space=FLAGS
        )
        load_frozen_policy(args.checkpoint, policy, device=device_info.device)
        tracker = TrackedEnvironment(env, phase="eval")
        logger.info(
            "evaluating %s | board=%dx%d games=%d run_dir=%s",
            args.checkpoint, env.width, env.height, args.games, run_dir,
        )
        report = run_evaluation(
            policy,
            tracker,
            metrics,
            games=args.games,
            device=device_info.device,
            max_consecutive_failures=args.max_consecutive_failures,
            stop_event=stop_event,
        )

    logger.info("evaluation summary | %s", report.summary.format())
    logger.info(
        "games=%d stopped_early=%s weights_unchanged=%s errors=%d",
        report.games, report.stopped_early, report.weights_unchanged, len(report.errors),
    )
    results_path = write_evaluation_results(
        run_dir,
        {
            "checkpoint": str(args.checkpoint),
            "run_dir": str(run_dir),
            "seed": args.seed,
            "device": str(device_info.device),
            "device_requested": device_info.requested,
            "gpu": device_info.cuda_device_name,
            "games": report.games,
            "stopped_early": report.stopped_early,
            "weights_unchanged": report.weights_unchanged,
            "errors": report.errors,
            "summary": report.summary.as_dict(),
        },
    )
    logger.info("wrote evaluation results -> %s", results_path)
    if not args.no_plots:
        plot_evaluation(run_dir)
    return 0 if not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
