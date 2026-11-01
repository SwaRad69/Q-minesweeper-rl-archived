"""Training entry point: resumable, interruptible PPO training on the live site.

**FLAGS-only.**  This is the canonical training path.  It always uses the
flag-aware 11-channel observation and the 1440-action space
(OPEN ``0..479``, FLAG ``480..959``, CHORD ``960..1439``); there is deliberately
no ``--action-space`` option, so the main experiment cannot be run OPEN-only.

Separate from evaluation.  A single ``--games`` value is the *total* number of
completed episodes for the experiment, so the same command can be re-run to
continue where a previous process stopped:

::

    $env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
    python -m minesweeper_rl.experiment.train --games 1000 --run-dir experiments/cnn_ppo_flags_v1

On start the runner looks for ``<run-dir>/checkpoint_latest.pt``.  If present it
is loaded automatically (model, optimizer, episode/update/step counters, RNG and
hyperparameters) and training continues from the next episode.  ``--fresh``
intentionally starts a new run, archiving any existing artifacts first so
nothing useful is destroyed silently.

``Ctrl+C`` (SIGINT) or SIGTERM sets a stop flag: the current episode and its PPO
update finish, metrics are recorded, ``checkpoint_latest.pt`` is written, the
browser is closed and the process exits.  A second interrupt forces termination.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from ..agents.minesweeper_env import MinesweeperPPOEnv
from ..agents.action_space import FLAGS, OPEN_ONLY
from ..agents.policy import PolicyValueNetwork
from ..agents.ppo import PPOConfig, PPOTrainer
from ..config import EnvironmentConfig
from .browser import open_environment
from .checkpoint import (
    BEST_FILENAME,
    LATEST_FILENAME,
    CheckpointError,
    CheckpointManager,
    load_checkpoint,
    read_checkpoint,
)
from .device import DEVICE_CHOICES, DeviceInfo, resolve_device
from .initialization import (
    PolicyCompatibilityError,
    initialize_policy_from_imitation,
)
from .metrics import EPISODES_FILE, UPDATES_FILE, EpisodeMetrics, MetricsLogger
from .metrics import MetricsSummary, UpdateMetrics, summarize
from .plots import plot_training
from .tracking import TrackedEnvironment

logger = logging.getLogger(__name__)

ARCHIVE_DIRNAME = "archive"
RUN_CONFIG_FILE = "run_config.json"

#: PPO initialization modes.
TRAINING_MODES = ("scratch", "human_pretrained")
#: Value recorded in run_config.json for each mode.
INITIALIZATION_SCRATCH = "random"
INITIALIZATION_HUMAN = "human_pretrained"
#: Observation/action representation and reward identities recorded per run.
OBSERVATION_VERSION = "flag_aware_v1"
REWARD_DEFINITION = "newly_revealed_cells + terminal_reward (FLAG neutral)"

#: Rolling window used for the live win-rate / mean-reward console display.
CONSOLE_WINDOW = 50

#: Configuration keys that must not change silently while resuming a run.
STRICT_CONFIG_KEYS = (
    "action_space",
    "observation_channels",
    "num_actions",
    "action_type",
    "width",
    "height",
    "mines",
    "gamma",
    "gae_lambda",
    "clip_coef",
    "value_coef",
    "entropy_coef",
    "learning_rate",
    "minibatch_size",
    "num_epochs",
    "max_grad_norm",
    "normalize_advantages",
    "target_kl",
)


@dataclass(slots=True)
class TrainingReport:
    """Outcome of one training session."""

    started_at_episode: int
    games_completed: int
    stopped_early: bool
    checkpoint_path: Path | None
    summary: MetricsSummary
    errors: list[str] = field(default_factory=list)
    final_episode: int = 0
    total_updates: int = 0
    total_steps: int = 0
    best_checkpoint_path: Path | None = None
    best_metric: float | None = None


def completed_episodes(trainer: object) -> int:
    """Return the number of episodes completed so far for ``trainer``."""

    return int(getattr(trainer, "episode_count", getattr(trainer, "iteration", 0)))


def seed_everything(seed: int) -> None:
    """Seed every RNG the training stack relies on.

    PPO also restores these RNG states from checkpoints, but a run must start
    from a fully specified seed to be reproducible in the first place.
    """

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def install_stop_handlers(stop_event: threading.Event) -> None:
    """Request a graceful stop on SIGINT/SIGTERM instead of crashing.

    The first signal sets ``stop_event`` so the loop can finish the current
    episode and checkpoint.  A second signal force-terminates, which is
    acceptable during shutdown if the first request appears stuck.
    """

    state = {"signals": 0}

    def handler(signum: int, _frame: object) -> None:
        state["signals"] += 1
        if state["signals"] == 1:
            logger.warning(
                "stop requested (signal %s); finishing the current episode", signum
            )
            stop_event.set()
            return
        logger.warning("second stop request (signal %s); terminating immediately", signum)
        os._exit(130)

    for name in ("SIGINT", "SIGTERM"):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            signal.signal(number, handler)
        except (ValueError, OSError):
            pass


def archive_existing_run(
    run_dir: Path, *, extra_paths: Sequence[Path] = ()
) -> Path | None:
    """Move an existing run's artifacts aside and return the archive directory.

    Used by ``--fresh`` so an intentional restart never deletes checkpoints or
    metrics.  ``extra_paths`` covers an explicitly overridden ``--checkpoint``
    that may live outside ``run_dir``.  Returns ``None`` when nothing exists.
    """

    run_dir = Path(run_dir)
    candidates = [run_dir / name for name in (LATEST_FILENAME, BEST_FILENAME,
                                              EPISODES_FILE, UPDATES_FILE,
                                              RUN_CONFIG_FILE)]
    candidates.extend(Path(path) for path in extra_paths)

    artifacts: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path.exists() and path not in seen:
            artifacts.append(path)
            seen.add(path)
    if not artifacts:
        return None

    stamp = time.strftime("%Y%m%d_%H%M%S")
    destination = run_dir / ARCHIVE_DIRNAME / stamp
    suffix = 1
    while destination.exists():
        destination = run_dir / ARCHIVE_DIRNAME / f"{stamp}_{suffix}"
        suffix += 1
    destination.mkdir(parents=True)
    for path in artifacts:
        shutil.move(str(path), str(destination / path.name))
    logger.info("archived previous run artifacts to %s", destination)
    return destination


def _rolling_win_rate(episodes: Sequence[EpisodeMetrics], window: int) -> float:
    recent = list(episodes[-window:]) if window > 0 else list(episodes)
    if not recent:
        return 0.0
    return sum(1 for episode in recent if episode.won) / len(recent)


def _rolling_mean_reward(episodes: Sequence[EpisodeMetrics], window: int) -> float:
    recent = list(episodes[-window:]) if window > 0 else list(episodes)
    if not recent:
        return 0.0
    return sum(episode.reward for episode in recent) / len(recent)


def _package_version(name: str) -> str | None:
    """Return an installed distribution version, or ``None`` if unknown."""

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    """Return the current Git commit hash when the source tree is a repo."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode == 0:
        return result.stdout.strip() or None
    return None


def _environment_info(device: DeviceInfo, url: str, profile: str) -> dict:
    """Collect software/hardware provenance for the machine-readable run config.

    This only records facts; it never influences training.  It lets a later
    reader identify the exact stack a run was produced with.
    """

    cuda_version = getattr(torch.version, "cuda", None)
    return {
        "python": str(sys.version.split()[0]),
        "platform": str(platform.platform()),
        # ``torch.__version__`` is a ``TorchVersion`` (str subclass) that
        # ``weights_only=True`` checkpoints cannot unpickle; store plain str.
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "playwright": _package_version("playwright"),
        "cuda_available": bool(device.cuda_available),
        "cuda_version": None if cuda_version is None else str(cuda_version),
        "gpu": None if device.cuda_device_name is None else str(device.cuda_device_name),
        "git_commit": _git_commit(),
        "url": str(url),
        "profile": str(profile),
    }


def build_run_config(
    policy: PolicyValueNetwork,
    env: MinesweeperPPOEnv,
    ppo_config: PPOConfig,
    *,
    games: int,
    checkpoint_interval: int,
    seed: int | None,
    device: DeviceInfo,
    url: str,
    profile: str,
    experiment: str | None = None,
    action_space: str | None = None,
    initialization: str = INITIALIZATION_SCRATCH,
    pretrained_policy: str | None = None,
    pretrained_info: Mapping | None = None,
    observation_version: str = OBSERVATION_VERSION,
) -> dict:
    """Assemble the complete, reproducible configuration for one run."""

    if action_space is None:
        action_space = (
            getattr(env, "action_space", None)
            or getattr(policy, "action_space", None)
            or FLAGS
        )
    flag_aware = action_space == FLAGS
    return {
        "experiment": experiment,
        "algorithm": "ppo",
        "architecture": "cnn",
        "action_space": str(action_space),
        "observation_version": observation_version,
        "observation_channels": int(policy.observation_channels),
        "num_actions": int(env.num_actions),
        "action_type": "OPEN_FLAG_CHORD" if flag_aware else "OPEN_ONLY",
        "flags": bool(flag_aware),
        "chords": bool(flag_aware),
        "board": {
            "width": int(env.width),
            "height": int(env.height),
            "mines": int(getattr(env, "mines", 99)),
            "difficulty": str(getattr(env, "difficulty", "expert")),
        },
        "ppo": asdict(ppo_config),
        "games": int(games),
        "checkpoint_interval": int(checkpoint_interval),
        "seed": None if seed is None else int(seed),
        "device": str(device.device),
        "device_requested": device.requested,
        "human_data": initialization == INITIALIZATION_HUMAN,
        "initialization": initialization,
        "pretrained": initialization == INITIALIZATION_HUMAN,
        "pretrained_policy": pretrained_policy,
        "pretrained_info": dict(pretrained_info) if pretrained_info else None,
        "reward": REWARD_DEFINITION,
        "url": url,
        "profile": str(profile),
        "environment": _environment_info(device, url, profile),
    }


def _flatten_run_config(config: Mapping) -> dict:
    """Flatten a run-config snapshot into the strictly compared keys."""

    board = config.get("board", {}) or {}
    ppo = config.get("ppo", {}) or {}
    flat: dict = {
        # Run configurations written before the flag-aware revision had no
        # action-space field; the only space then was OPEN-only.
        "action_space": config.get("action_space", OPEN_ONLY),
        "observation_channels": config.get("observation_channels"),
        "num_actions": config.get("num_actions"),
        "action_type": config.get("action_type"),
        "width": board.get("width"),
        "height": board.get("height"),
        "mines": board.get("mines"),
    }
    for key in (
        "gamma",
        "gae_lambda",
        "clip_coef",
        "value_coef",
        "entropy_coef",
        "learning_rate",
        "minibatch_size",
        "num_epochs",
        "max_grad_norm",
        "normalize_advantages",
        "target_kl",
    ):
        flat[key] = ppo.get(key)
    return flat


def config_mismatches(saved: Mapping, current: Mapping) -> list[str]:
    """Return human-readable descriptions of incompatible configuration keys.

    Only the keys in :data:`STRICT_CONFIG_KEYS` are compared: target games,
    checkpoint interval and device may legitimately change between sessions.
    """

    saved_flat = _flatten_run_config(saved)
    current_flat = _flatten_run_config(current)
    mismatches: list[str] = []
    for key in STRICT_CONFIG_KEYS:
        left = saved_flat.get(key)
        right = current_flat.get(key)
        if left != right:
            mismatches.append(f"{key}: checkpoint={left!r} requested={right!r}")
    return mismatches


def write_run_config(run_dir: str | Path, config: Mapping) -> Path:
    """Write the configuration snapshot atomically (temp file then replace)."""

    destination = Path(run_dir) / RUN_CONFIG_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=destination.name + ".", suffix=".tmp"
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def read_run_config(run_dir: str | Path) -> dict | None:
    """Read the configuration snapshot for a run directory, if present."""

    path = Path(run_dir) / RUN_CONFIG_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_training(
    trainer: PPOTrainer,
    tracker: TrackedEnvironment,
    metrics: MetricsLogger,
    *,
    games: int,
    checkpoint: Path | None = None,
    checkpoint_interval: int = 10,
    max_consecutive_failures: int = 3,
    stop_event: threading.Event | None = None,
    emit: Callable[[str], None] | None = None,
    manager: CheckpointManager | None = None,
    best_window: int = 50,
    best_min_episodes: int = 10,
) -> TrainingReport:
    """Play until ``games`` total episodes are completed, updating PPO per game.

    ``games`` is an absolute target on the experiment's completed-episode
    counter, so a resumed run finishes the remaining episodes rather than
    adding ``games`` more.
    """

    say = emit or logger.info
    if manager is None and checkpoint is not None:
        checkpoint = Path(checkpoint)
        manager = CheckpointManager(checkpoint.parent, latest_name=checkpoint.name)

    started_at = completed_episodes(trainer)
    # Detect a disagreement between the checkpoint and the JSONL history rather
    # than silently picking one.  If the metric log has more episodes the
    # process was killed between logging and saving, so trust the log to avoid
    # duplicate episode ids.  If the log has fewer, the history was truncated
    # or lost -- report it and continue from the authoritative checkpoint.
    if metrics.episodes:
        recorded = max(episode.episode for episode in metrics.episodes)
        if recorded > started_at:
            say(
                f"metrics recorded {recorded} episodes but the checkpoint is at "
                f"{started_at}; continuing from episode {recorded} to avoid "
                "duplicate ids"
            )
            started_at = recorded
            trainer.episode_count = recorded
        elif recorded < started_at:
            say(
                f"checkpoint is at episode {started_at} but metrics only recorded "
                f"{recorded}; continuing from the checkpoint (episode "
                f"{started_at + 1}). Existing metrics are kept, not overwritten."
            )
    tracker.episode_offset = started_at
    errors: list[str] = []
    stopped_early = False
    consecutive_failures = 0
    device = str(getattr(trainer, "device", "cpu"))
    learning_rate = float(trainer.config.learning_rate)
    session_started = time.perf_counter()

    while completed_episodes(trainer) < games:
        if stop_event is not None and stop_event.is_set():
            stopped_early = True
            say(f"stop requested at episode {completed_episodes(trainer)}; stopping")
            break
        try:
            tracker_episode_before = tracker.last_episode
            episodes_before = completed_episodes(trainer)
            trainer.collect_episode()
            # A stuck game is closed by the trainer internally and advances the
            # episode counter without the environment reporting a terminal
            # state, so the wrapper never finalizes an EpisodeMetrics.  Close it
            # here; otherwise the next iteration would log the previous episode
            # id again.
            if (
                completed_episodes(trainer) > episodes_before
                and tracker.last_episode is tracker_episode_before
            ):
                tracker.finish_truncated_episode()
            update = trainer.update()
            trainer.iteration += 1
            consecutive_failures = 0

            metrics.log_update(
                UpdateMetrics(
                    iteration=trainer.iteration,
                    phase="train",
                    policy_loss=update["policy_loss"],
                    value_loss=update["value_loss"],
                    entropy=update["entropy"],
                    approx_kl=update["approx_kl"],
                    clip_fraction=update["clip_fraction"],
                    total_loss=update["total_loss"],
                    updates=int(update["updates"]),
                    learning_rate=learning_rate,
                    device=device,
                )
            )
            episode = tracker.last_episode
            if episode is not None:
                metrics.log_episode(episode)
                win_rate = _rolling_win_rate(metrics.episodes, CONSOLE_WINDOW)
                mean_reward = _rolling_mean_reward(metrics.episodes, CONSOLE_WINDOW)
                elapsed = time.perf_counter() - session_started
                done_this_session = completed_episodes(trainer) - started_at
                per_hour = (
                    done_this_session / elapsed * 3600.0 if elapsed > 0 else 0.0
                )
                best = (
                    f"{manager.best_metric:.1%}"
                    if manager is not None and manager.best_metric is not None
                    else "n/a"
                )
                say(
                    f"episode {episode.episode:04d}/{games} | update {trainer.iteration:04d} "
                    f"| steps={episode.steps:3d} reward={episode.reward:+7.2f} "
                    f"outcome={episode.outcome:<7} revealed={episode.cells_revealed:3d} "
                    f"open={episode.open_actions:3d} "
                    f"| winrate={win_rate:.1%} meanR={mean_reward:+.2f} best={best} "
                    f"| policy={update['policy_loss']:+.4f} value={update['value_loss']:.4f} "
                    f"entropy={update['entropy']:.4f} kl={update['approx_kl']:.5f} "
                    f"| {per_hour:5.1f} games/h elapsed={elapsed / 60.0:.1f}m device={device}"
                )
                if manager is not None and episode.episode >= best_min_episodes:
                    win_rate = _rolling_win_rate(metrics.episodes, best_window)
                    saved_best = manager.save_best(
                        trainer,
                        metric=win_rate,
                        metrics={"win_rate": win_rate, "window": best_window},
                    )
                    if saved_best is not None:
                        say(f"new best win rate {win_rate:.1%} -> {saved_best}")

            if (
                manager is not None
                and checkpoint_interval > 0
                and completed_episodes(trainer) % checkpoint_interval == 0
            ):
                manager.save_latest(
                    trainer, metrics={"episode": completed_episodes(trainer)}
                )
                say(
                    f"saved checkpoint at episode {completed_episodes(trainer)} "
                    f"-> {manager.latest_path}"
                )
        except KeyboardInterrupt:
            stopped_early = True
            say("keyboard interrupt; stopping after a final checkpoint")
            break
        except Exception as error:  # noqa: BLE001 - keep the run diagnosable
            consecutive_failures += 1
            message = (
                f"episode {completed_episodes(trainer) + 1}: "
                f"{type(error).__name__}: {error}"
            )
            errors.append(message)
            logger.exception("training failure: %s", message)
            if consecutive_failures >= max_consecutive_failures:
                stopped_early = True
                say(f"too many consecutive failures ({consecutive_failures}); stopping")
                break

    if manager is not None:
        manager.save_latest(trainer, metrics={"episode": completed_episodes(trainer)})
        say(f"saved checkpoint -> {manager.latest_path}")

    summary = summarize(metrics.episodes, environment_failures=tracker.failures)
    return TrainingReport(
        started_at_episode=started_at,
        games_completed=completed_episodes(trainer) - started_at,
        stopped_early=stopped_early,
        checkpoint_path=manager.latest_path if manager is not None else checkpoint,
        summary=summary,
        errors=errors,
        final_episode=completed_episodes(trainer),
        total_updates=trainer.iteration,
        total_steps=trainer.total_steps,
        best_checkpoint_path=manager.best_path if manager is not None else None,
        best_metric=manager.best_metric if manager is not None else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20,
                        help="total episodes to train for (experiment target)")
    parser.add_argument("--run-dir", type=Path,
                        default=Path("experiments/cnn_ppo_flags_v1"),
                        help="isolated directory for this experiment's metrics, "
                             "plots, config and checkpoints (one directory per "
                             "experiment under experiments/)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="override the latest checkpoint path "
                             f"(default: <run-dir>/{LATEST_FILENAME})")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore existing checkpoints and start a new run "
                             "(existing artifacts are archived, not deleted)")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                        help="save every N episodes (0 disables interim saves)")
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--training-mode", default="scratch",
                        choices=list(TRAINING_MODES),
                        help="'scratch' = random CNN initialization; "
                             "'human_pretrained' = initialize the CNN/actor "
                             "from a behavioral-cloning checkpoint")
    parser.add_argument("--pretrained-policy", type=Path, default=None,
                        help="human imitation checkpoint to initialize the PPO "
                             "policy from (required for --training-mode "
                             "human_pretrained)")
    parser.add_argument("--device", default="auto", choices=list(DEVICE_CHOICES),
                        help="compute device for the CNN/PPO: auto (CUDA if "
                             "available else CPU), cuda, or cpu")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--no-advantage-normalization", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--url", default=os.environ.get("MINESWEEPER_URL"))
    parser.add_argument("--profile", default=os.environ.get("MINESWEEPER_PROFILE",
                        "browser_profile"))
    parser.add_argument("--diagnostics", action="store_true",
                        default=os.environ.get("MINESWEEPER_DIAGNOSTICS", "").lower()
                        in {"1", "true", "yes"})
    return parser


def _log_header(run_dir: Path, target: int, *, resuming: bool, data: dict | None,
                best_metric: float | None) -> None:
    logger.info("=" * 40)
    logger.info("Minesweeper PPO Training")
    logger.info("=" * 40)
    logger.info("")
    logger.info("Run directory:")
    logger.info("%s", run_dir)
    logger.info("")
    logger.info("Status:")
    logger.info("%s", "RESUMING" if resuming else "NEW RUN")
    if resuming and data is not None:
        previous = int(data.get("episode_count", data.get("iteration", 0)))
        logger.info("")
        logger.info("Checkpoint found.")
        logger.info("Resuming training.")
        logger.info("")
        logger.info("Previous episode: %d / %d", previous, target)
        logger.info("Previous update: %d", int(data.get("iteration", 0)))
        logger.info("Previous total steps: %d", int(data.get("total_steps", 0)))
        logger.info(
            "Previous best win rate: %s",
            f"{best_metric:.1%}" if best_metric is not None else "n/a",
        )
        logger.info("")
        logger.info("Continuing from episode %d...", previous + 1)
    else:
        logger.info("")
        logger.info("Episode:")
        logger.info("0 / %d", target)
    logger.info("-" * 40)


def _log_summary(run_dir: Path, report: TrainingReport, latest: Path | None) -> None:
    logger.info("=" * 40)
    logger.info("Training stopped safely" if report.stopped_early else "Training complete")
    logger.info("=" * 40)
    logger.info("")
    logger.info("Episodes completed: %d", report.final_episode)
    logger.info("Total updates: %d", report.total_updates)
    logger.info("Total steps: %d", report.total_steps)
    if report.best_metric is not None:
        logger.info("Best win rate: %.1f%%", report.best_metric * 100.0)
    logger.info("")
    logger.info("Checkpoint:")
    logger.info("%s", latest)
    if report.errors:
        logger.info("")
        logger.info("Errors: %d", len(report.errors))
    logger.info("")
    logger.info("Browser closed successfully.")


def main(argv: list[str] | None = None, *, stop_event: threading.Event | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("provide --url or set MINESWEEPER_URL")
    if args.games < 1:
        parser.error("--games must be at least 1")
    if args.seed is not None:
        seed_everything(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        device_info = resolve_device(args.device)
    except (ValueError, RuntimeError) as error:
        logger.error("invalid --device: %s", error)
        return 1

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        extra = [args.checkpoint] if args.checkpoint is not None else []
        archive_existing_run(run_dir, extra_paths=extra)

    manager = CheckpointManager(
        run_dir,
        metadata={
            "algorithm": "ppo",
            "seed": args.seed,
            "action_space": FLAGS,
            "url": args.url,
            "profile": str(args.profile),
            "device": str(device_info.device),
            "human_data": False,
            "initialization": "random",
        },
    )
    if args.checkpoint is not None:
        manager.latest_path = Path(args.checkpoint)
        manager.best_path = manager.latest_path.with_name(BEST_FILENAME)

    resume_data: dict | None = None
    if not args.fresh and manager.latest_path.exists():
        try:
            resume_data = read_checkpoint(manager.latest_path)
        except CheckpointError as error:
            logger.error(
                "refusing to start: %s", error
            )
            logger.error(
                "the existing checkpoint could not be read; passing --fresh "
                "will ARCHIVE it (not delete it) and start a new run"
            )
            return 1

    # The initialization mode is recorded permanently in run_config.json; on
    # resume it is adopted from the saved configuration so the fact is never lost.
    saved_config: dict | None = None
    if resume_data is not None:
        saved_config = (resume_data.get("metadata") or {}).get("run_config")
    if saved_config is None:
        saved_config = read_run_config(run_dir)

    if resume_data is not None and saved_config:
        initialization = str(
            saved_config.get("initialization", INITIALIZATION_SCRATCH)
        )
        pretrained_policy = saved_config.get("pretrained_policy")
        pretrained_info = saved_config.get("pretrained_info")
        training_mode = (
            "human_pretrained"
            if initialization == INITIALIZATION_HUMAN
            else "scratch"
        )
    else:
        training_mode = args.training_mode
        initialization = (
            INITIALIZATION_HUMAN
            if training_mode == "human_pretrained"
            else INITIALIZATION_SCRATCH
        )
        pretrained_policy = (
            str(args.pretrained_policy) if args.pretrained_policy else None
        )
        pretrained_info = None

    if resume_data is None:
        if training_mode == "human_pretrained" and not pretrained_policy:
            logger.error(
                "--training-mode human_pretrained requires "
                "--pretrained-policy <imitation checkpoint>"
            )
            return 1
        if training_mode == "scratch" and args.pretrained_policy is not None:
            logger.error(
                "--pretrained-policy is only valid together with "
                "--training-mode human_pretrained"
            )
            return 1

    manager.metadata["initialization"] = initialization
    manager.metadata["pretrained_policy"] = pretrained_policy
    manager.metadata["human_data"] = initialization == INITIALIZATION_HUMAN

    metrics = MetricsLogger(run_dir)
    _log_header(
        run_dir,
        args.games,
        resuming=resume_data is not None,
        data=resume_data,
        best_metric=manager.best_metric,
    )
    device_info.log(logger)

    config = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        normalize_advantages=not args.no_advantage_normalization,
        rollout_steps=1,
    )

    if stop_event is None:
        stop_event = threading.Event()
    install_stop_handlers(stop_event)

    report: TrainingReport
    with open_environment(
        args.url, args.profile, config=EnvironmentConfig(diagnostics=args.diagnostics)
    ) as environment:
        env = MinesweeperPPOEnv(environment, action_space=FLAGS)
        policy = PolicyValueNetwork(
            width=env.width, height=env.height, action_space=FLAGS
        )
        if resume_data is None and initialization == INITIALIZATION_HUMAN:
            try:
                info = initialize_policy_from_imitation(policy, pretrained_policy)
            except PolicyCompatibilityError as error:
                logger.error("pretrained policy is incompatible: %s", error)
                return 1
            pretrained_info = info.as_dict()
            logger.info(
                "pretrained initialization | checkpoint=%s epoch=%s "
                "human_games=%s",
                pretrained_policy,
                pretrained_info.get("epoch"),
                (pretrained_info.get("human_dataset") or {}).get("games"),
            )
        tracker = TrackedEnvironment(env, phase="train")
        # The policy, optimizer and every rollout/minibatch tensor live on the
        # selected device; the browser environment above stays on CPU.
        trainer = PPOTrainer(
            policy,
            tracker,
            num_actions=env.num_actions,
            config=config,
            device=device_info.device,
        )
        parameter = next(policy.parameters(), None)
        logger.info(
            "model | observation_channels=%d actions=%d parameter_device=%s",
            policy.observation_channels,
            env.num_actions,
            parameter.device if parameter is not None else "n/a",
        )

        run_config = build_run_config(
            policy,
            env,
            config,
            games=args.games,
            checkpoint_interval=args.checkpoint_interval,
            seed=args.seed,
            device=device_info,
            url=args.url,
            profile=str(args.profile),
            experiment=run_dir.name,
            action_space=FLAGS,
            initialization=initialization,
            pretrained_policy=pretrained_policy,
            pretrained_info=pretrained_info,
        )

        if resume_data is not None:
            if saved_config is not None:
                mismatches = config_mismatches(saved_config, run_config)
                if mismatches:
                    logger.error(
                        "refusing to resume: requested configuration is "
                        "incompatible with the checkpoint"
                    )
                    for mismatch in mismatches:
                        logger.error("  %s", mismatch)
                    logger.error(
                        "pass --fresh to start a new experiment (existing "
                        "artifacts are archived, not deleted)"
                    )
                    return 1
            else:
                logger.warning(
                    "the checkpoint has no run-config snapshot; compatibility "
                    "is limited to the model/action/observation checks"
                )
            try:
                load_checkpoint(trainer, resume_data)
            except CheckpointError as error:
                logger.error("refusing to resume: %s", error)
                logger.error(
                    "the checkpoint is incompatible with this run; it was left "
                    "untouched for inspection"
                )
                return 1
            metrics.load_existing()
            logger.info(
                "restored | episode=%d update=%d total_steps=%d",
                trainer.episode_count, trainer.iteration, trainer.total_steps,
            )
        else:
            # Experiment immutability: a directory that already carries a run
            # configuration but no checkpoint cannot be resumed (the model state
            # is gone), and silently overwriting it would mix two experiments.
            # Require an explicit --fresh, which archives rather than deletes.
            existing = read_run_config(run_dir)
            if existing is not None:
                logger.error(
                    "refusing to start: %s already contains a run configuration "
                    "but no checkpoint. This directory belongs to an existing "
                    "experiment; pass --fresh to archive it and start a new run, "
                    "or use a different --run-dir.",
                    run_dir,
                )
                return 1
            path = write_run_config(run_dir, run_config)
            logger.info("wrote configuration snapshot -> %s", path)
            if initialization == INITIALIZATION_HUMAN:
                logger.info(
                    "fresh run from HUMAN-PRETRAINED initialization (%s)",
                    pretrained_policy,
                )
            else:
                logger.info(
                    "fresh run from RANDOM initialization (no human demonstrations)"
                )

        # Embed the configuration snapshot in every checkpoint written now on.
        manager.metadata["run_config"] = run_config

        report = run_training(
            trainer,
            tracker,
            metrics,
            games=args.games,
            checkpoint_interval=args.checkpoint_interval,
            max_consecutive_failures=args.max_consecutive_failures,
            stop_event=stop_event,
            manager=manager,
        )

    _log_summary(run_dir, report, manager.latest_path)
    if not args.no_plots:
        plot_training(run_dir)
    return 0 if not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
