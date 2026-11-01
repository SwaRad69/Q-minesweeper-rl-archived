"""Plots for training and evaluation runs.

Matplotlib is an optional dependency: if it is not installed the functions log
a warning and return ``None`` so experiments still run headless.  All figures
are written into the run directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .metrics import EpisodeMetrics, UpdateMetrics, load_run

logger = logging.getLogger(__name__)

TRAINING_PLOT = "training_progress.png"
EVALUATION_PLOT = "evaluation.png"


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    window = max(1, min(window, values.size))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _rolling_window(size: int, cap: int = 20) -> int:
    return max(1, min(cap, size))


def _import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # noqa: BLE001 - plotting is optional
        logger.warning("matplotlib unavailable, skipping plots: %s", error)
        return None
    return plt


def plot_training(
    run_dir: str | Path, *, rolling: int = 20
) -> Path | None:
    """Write a multi-panel training-progress figure for a run directory."""

    plt = _import_pyplot()
    if plt is None:
        return None

    episodes, updates = load_run(run_dir)
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / TRAINING_PLOT

    figure, axes = plt.subplots(4, 2, figsize=(13, 15))
    rewards = np.array([e.reward for e in episodes], dtype=float)
    steps = np.array([e.steps for e in episodes], dtype=float)
    wins = np.array([1.0 if e.won else 0.0 for e in episodes], dtype=float)
    mines = np.array([1.0 if e.mine_hit else 0.0 for e in episodes], dtype=float)
    revealed = np.array([e.cells_revealed for e in episodes], dtype=float)
    opens = np.array([e.open_actions for e in episodes], dtype=float)
    window = _rolling_window(len(episodes), rolling)

    if rewards.size:
        axes[0, 0].plot(rewards, alpha=0.35, label="reward")
        axes[0, 0].plot(
            np.arange(window - 1, rewards.size), _rolling(rewards, window),
            label=f"mean({window})",
        )
        axes[0, 0].set_ylabel("episode reward")
    axes[0, 0].set_title("Reward")
    axes[0, 0].legend(loc="best")

    if wins.size:
        axes[0, 1].plot(
            np.arange(window - 1, wins.size), _rolling(wins, window), color="green"
        )
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_ylabel("win rate")
    axes[0, 1].set_title(f"Win rate (rolling {window})")

    if steps.size:
        axes[1, 0].plot(
            np.arange(window - 1, steps.size), _rolling(steps, window), color="orange"
        )
    axes[1, 0].set_ylabel("steps")
    axes[1, 0].set_title(f"Episode length (rolling {window})")

    if mines.size:
        axes[1, 1].plot(
            np.arange(window - 1, mines.size), _rolling(mines, window), color="red"
        )
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_ylabel("mine-hit rate")
    axes[1, 1].set_title(f"Mine-hit rate (rolling {window})")

    if revealed.size:
        axes[2, 0].plot(revealed, alpha=0.35, color="teal", label="cells revealed")
        axes[2, 0].plot(
            np.arange(window - 1, revealed.size), _rolling(revealed, window),
            color="teal", label=f"mean({window})",
        )
        axes[2, 0].legend(loc="best")
    axes[2, 0].set_ylabel("cells")
    axes[2, 0].set_title("Cells revealed per episode")

    if opens.size:
        axes[2, 1].plot(
            np.arange(window - 1, opens.size), _rolling(opens, window),
            color="steelblue", label="OPEN",
        )
        axes[2, 1].legend(loc="best")
    axes[2, 1].set_ylabel("actions")
    axes[2, 1].set_title(f"OPEN actions (rolling {window})")

    if updates:
        iterations = [u.iteration for u in updates]
        axes[3, 0].plot(iterations, [u.policy_loss for u in updates], label="policy")
        axes[3, 0].plot(iterations, [u.value_loss for u in updates], label="value")
        axes[3, 0].legend(loc="best")
        axes[3, 0].set_xlabel("iteration")
        axes[3, 0].set_ylabel("loss")
        axes[3, 0].set_title("PPO losses")
        axes[3, 1].plot(iterations, [u.entropy for u in updates], label="entropy")
        axes[3, 1].plot(
            iterations, [u.approx_kl for u in updates], label="approx KL"
        )
        axes[3, 1].legend(loc="best")
        axes[3, 1].set_xlabel("iteration")
        axes[3, 1].set_title("Entropy / KL")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("Training progress")
    figure.tight_layout()
    figure.savefig(output, dpi=120)
    plt.close(figure)
    logger.info("wrote training plot to %s", output)
    return output


def plot_evaluation(run_dir: str | Path) -> Path | None:
    """Write a summary bar chart of evaluation metrics."""

    plt = _import_pyplot()
    if plt is None:
        return None

    episodes, _ = load_run(run_dir)
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / EVALUATION_PLOT
    if not episodes:
        logger.warning("no evaluation episodes to plot in %s", destination)
        return None

    games = len(episodes)
    win_rate = sum(1 for e in episodes if e.won) / games
    mine_rate = sum(1 for e in episodes if e.mine_hit) / games
    mean_reward = float(np.mean([e.reward for e in episodes]))
    mean_steps = float(np.mean([e.steps for e in episodes]))
    mean_cells = float(np.mean([e.cells_revealed for e in episodes]))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(
        ["win rate", "mine-hit rate"],
        [win_rate, mine_rate],
        color=["green", "red"],
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Outcome rates")
    axes[1].bar(
        ["mean reward", "mean steps", "mean cells"],
        [mean_reward, mean_steps, mean_cells],
        color=["steelblue", "orange", "gray"],
    )
    axes[1].set_title("Per-episode averages")
    for axis in axes:
        axis.grid(alpha=0.2, axis="y")
    figure.suptitle(f"Evaluation over {games} games")
    figure.tight_layout()
    figure.savefig(output, dpi=120)
    plt.close(figure)
    logger.info("wrote evaluation plot to %s", output)
    return output
