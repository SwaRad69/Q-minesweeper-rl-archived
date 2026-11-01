"""Behavioral cloning (imitation) pretraining on human demonstrations.

This is a standalone supervised entry point, deliberately independent from the
PPO trainer.  It loads the persistent human dataset with
:class:`~minesweeper_rl.data.human_demonstrations.HumanDemonstrationDataset`,
splits it **by game**, and trains the existing flag-aware
:class:`~minesweeper_rl.agents.policy.PolicyValueNetwork` with cross-entropy on
the 1440-class action id::

    loss = cross_entropy(policy_logits, human_action_id)

Only observable state/action pairs are used; hidden mine layouts are never part
of a sample.  This is imitation pretraining -- accuracy here does not mean the
policy can solve Minesweeper.

::

    python -m minesweeper_rl.experiment.imitation_train \
        --dataset data/human \
        --output experiments/cnn_imitation_expert_v1 \
        --epochs 20 --batch-size 64 --learning-rate 1e-3 --val-split 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..agents.action_space import FLAGS, action_count
from ..agents.policy import PolicyValueNetwork
from ..data.human_demonstrations import (
    DatasetStatistics,
    HumanDemonstrationDataset,
    HumanSample,
    split_games,
)
from .checkpoint import atomic_save, capture_rng_state, restore_rng_state
from .device import DEVICE_CHOICES, DeviceInfo, resolve_device

logger = logging.getLogger(__name__)

IMITATION_CHECKPOINT_FORMAT = "minesweeper_imitation"
IMITATION_CHECKPOINT_VERSION = 1
LATEST_FILENAME = "checkpoint_latest.pt"
BEST_FILENAME = "checkpoint_best.pt"
RUN_CONFIG_FILE = "run_config.json"
HISTORY_FILE = "history.json"

#: Action-type labels by code (0 = OPEN, 1 = FLAG, 2 = CHORD).
ACTION_LABELS = ("open", "flag", "chord")


class ImitationError(ValueError):
    """Raised for invalid imitation configuration or incompatible checkpoints."""


@dataclass(slots=True)
class BCConfig:
    """Behavioral-cloning hyperparameters."""

    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    val_fraction: float = 0.2
    seed: int = 42
    weight_decay: float = 0.0
    channels: int = 16

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ImitationReport:
    """Summary of one imitation training run."""

    epochs: int
    train_games: int
    val_games: int
    train_transitions: int
    val_transitions: int
    best_val_accuracy: float | None
    latest_checkpoint: Path | None
    best_checkpoint: Path | None
    history: list[dict[str, Any]] = field(default_factory=list)


def _atomic_write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
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


def encode_samples(
    samples: list[HumanSample], *, width: int, height: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(boards[N,H,W] int8, actions[N] int64)`` for supervision."""

    if samples:
        boards = np.stack([np.asarray(sample.board) for sample in samples])
        boards = np.ascontiguousarray(boards, dtype=np.int8)
        actions = np.asarray([sample.action_id for sample in samples], dtype=np.int64)
    else:
        boards = np.zeros((0, height, width), dtype=np.int8)
        actions = np.zeros((0,), dtype=np.int64)
    return torch.from_numpy(boards), torch.from_numpy(actions)


@torch.no_grad()
def evaluate(
    policy: nn.Module,
    boards: torch.Tensor,
    actions: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    cells: int,
) -> dict[str, Any]:
    """Return loss, action accuracy, top-k and per-action-type accuracy."""

    policy.eval()
    total = int(actions.numel())
    if total == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "top3": 0.0,
            "top5": 0.0,
            "per_type_accuracy": {name: None for name in ACTION_LABELS},
            "per_type_support": {name: 0 for name in ACTION_LABELS},
            "per_type_predicted": {name: 0 for name in ACTION_LABELS},
        }

    loss_sum = 0.0
    correct = 0
    top3 = 0
    top5 = 0
    per_type_correct = {name: 0 for name in ACTION_LABELS}
    per_type_support = {name: 0 for name in ACTION_LABELS}
    per_type_predicted = {name: 0 for name in ACTION_LABELS}

    for start in range(0, total, batch_size):
        batch_boards = boards[start : start + batch_size].to(device)
        batch_actions = actions[start : start + batch_size].to(device)
        logits, _ = policy(batch_boards)
        loss_sum += float(F.cross_entropy(logits, batch_actions, reduction="sum").item())
        predictions = logits.argmax(dim=1)
        correct += int((predictions == batch_actions).sum().item())
        k = min(5, logits.shape[1])
        top = logits.topk(k, dim=1).indices
        hits = top == batch_actions.unsqueeze(1)
        top3 += int(hits[:, : min(3, k)].any(dim=1).sum().item())
        top5 += int(hits.any(dim=1).sum().item())
        target_types = (batch_actions // cells).tolist()
        predicted_types = (predictions // cells).tolist()
        is_correct = (predictions == batch_actions).tolist()
        for index, type_code in enumerate(target_types):
            name = ACTION_LABELS[int(type_code)]
            per_type_support[name] += 1
            per_type_correct[name] += int(is_correct[index])
        for type_code in predicted_types:
            name = ACTION_LABELS[int(type_code)]
            per_type_predicted[name] += 1

    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "top3": top3 / total,
        "top5": top5 / total,
        "per_type_accuracy": {
            name: (per_type_correct[name] / per_type_support[name])
            if per_type_support[name]
            else None
            for name in ACTION_LABELS
        },
        "per_type_support": per_type_support,
        "per_type_predicted": per_type_predicted,
    }


def train_imitation(
    policy: nn.Module,
    train_samples: list[HumanSample],
    val_samples: list[HumanSample],
    config: BCConfig,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    start_epoch: int = 1,
    cells: int,
    width: int,
    height: int,
    history: list[dict[str, Any]] | None = None,
    on_epoch=None,
) -> tuple[list[dict[str, Any]], torch.optim.Optimizer]:
    """Run behavioral cloning and return ``(history, optimizer)``."""

    train_boards, train_actions = encode_samples(
        train_samples, width=width, height=height
    )
    val_boards, val_actions = encode_samples(val_samples, width=width, height=height)
    total = int(train_actions.numel())
    if total == 0:
        raise ImitationError("no training transitions were loaded")

    policy.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    history = list(history) if history else []
    for epoch in range(start_epoch, start_epoch + config.epochs):
        policy.train()
        permutation = torch.randperm(total)
        running_loss = 0.0
        for start in range(0, total, config.batch_size):
            indices = permutation[start : start + config.batch_size]
            batch_boards = train_boards[indices].to(device)
            batch_actions = train_actions[indices].to(device)
            logits, _ = policy(batch_boards)
            loss = F.cross_entropy(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(batch_actions.numel())

        train_metrics = evaluate(
            policy, train_boards, train_actions,
            device=device, batch_size=config.batch_size, cells=cells,
        )
        val_metrics = evaluate(
            policy, val_boards, val_actions,
            device=device, batch_size=config.batch_size, cells=cells,
        )
        record = {
            "epoch": epoch,
            "optimizer_loss": running_loss / total,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        if on_epoch is not None:
            on_epoch(record, history)

    return history, optimizer


def model_info(policy: nn.Module) -> dict[str, Any]:
    """Architecture metadata for checkpoint compatibility checks."""

    first = policy.encoder[0]
    return {
        "class": type(policy).__name__,
        "width": int(getattr(policy, "width", -1)),
        "height": int(getattr(policy, "height", -1)),
        "channels": int(first.out_channels),
        "num_actions": int(getattr(policy, "num_actions", -1)),
        "observation_channels": int(getattr(policy, "observation_channels", -1)),
        "action_space": str(getattr(policy, "action_space", FLAGS)),
    }


def build_imitation_checkpoint(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    config: BCConfig,
    dataset_stats: DatasetStatistics,
    train_stats: DatasetStatistics,
    val_stats: DatasetStatistics,
    history: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latest = history[-1] if history else {}
    return {
        "format": IMITATION_CHECKPOINT_FORMAT,
        "version": IMITATION_CHECKPOINT_VERSION,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "config": config.as_dict(),
        "model": model_info(policy),
        "dataset": dataset_stats.as_dict(),
        "train_dataset": train_stats.as_dict(),
        "val_dataset": val_stats.as_dict(),
        "history": history,
        "metrics": {
            "train_loss": latest.get("train", {}).get("loss"),
            "val_loss": latest.get("val", {}).get("loss"),
            "train_accuracy": latest.get("train", {}).get("accuracy"),
            "val_accuracy": latest.get("val", {}).get("accuracy"),
            "val_top3": latest.get("val", {}).get("top3"),
            "val_top5": latest.get("val", {}).get("top5"),
            "val_per_type_accuracy": latest.get("val", {}).get("per_type_accuracy"),
        },
        "rng": capture_rng_state(include_cuda=torch.cuda.is_available()),
        "metadata": dict(extra or {}),
    }


def read_imitation_checkpoint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise ImitationError(f"checkpoint not found: {source}")
    try:
        data = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001
        raise ImitationError(f"could not read checkpoint {source}: {error}") from error
    if not isinstance(data, dict) or data.get("format") != IMITATION_CHECKPOINT_FORMAT:
        raise ImitationError(f"{source} is not an imitation checkpoint")
    return data


def validate_imitation_checkpoint(data: dict[str, Any], policy: nn.Module) -> None:
    if data.get("format") != IMITATION_CHECKPOINT_FORMAT:
        raise ImitationError(f"unexpected checkpoint format: {data.get('format')!r}")
    if int(data.get("version", -1)) != IMITATION_CHECKPOINT_VERSION:
        raise ImitationError(
            f"unsupported checkpoint version {data.get('version')!r}; expected "
            f"{IMITATION_CHECKPOINT_VERSION}"
        )
    saved = data.get("model") or {}
    current = model_info(policy)
    for key in ("class", "width", "height", "channels", "num_actions",
                "observation_channels", "action_space"):
        if key in saved and saved[key] != current[key]:
            raise ImitationError(
                f"checkpoint model {key} {saved[key]!r} does not match the "
                f"current policy {current[key]!r}"
            )


def load_imitation_checkpoint(
    path: str | Path,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    data = read_imitation_checkpoint(path)
    validate_imitation_checkpoint(data, policy)
    policy.load_state_dict(data["policy"])
    if optimizer is not None and "optimizer" in data:
        optimizer.load_state_dict(data["optimizer"])
    restore_rng_state(data.get("rng"))
    return data


def _format_metrics(metrics: dict[str, Any]) -> str:
    per_type = metrics.get("per_type_accuracy") or {}
    parts = []
    for name in ACTION_LABELS:
        value = per_type.get(name)
        parts.append(f"{name}={'n/a' if value is None else f'{value:.3f}'}")
    return (
        f"loss={metrics['loss']:.4f} acc={metrics['accuracy']:.3f} "
        f"top3={metrics.get('top3', 0):.3f} top5={metrics.get('top5', 0):.3f} "
        f"| {' '.join(parts)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/human"),
                        help="persistent human dataset directory")
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/cnn_imitation_expert_v1"),
                        help="output experiment directory for checkpoints/metrics")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="fraction of GAMES used for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channels", type=int, default=16,
                        help="CNN hidden channel width")
    parser.add_argument("--device", default="auto", choices=list(DEVICE_CHOICES))
    parser.add_argument("--game", action="append", default=None,
                        help="restrict to a game id (repeatable), e.g. game_000001")
    parser.add_argument("--winning-only", action="store_true",
                        help="train only on games the human won")
    parser.add_argument("--exclude-incomplete", action="store_true",
                        help="drop interrupted games")
    parser.add_argument("--resume", action="store_true",
                        help="resume from checkpoint_latest.pt in the output dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    try:
        device_info = resolve_device(args.device)
    except (ValueError, RuntimeError) as error:
        logger.error("invalid --device: %s", error)
        return 1
    device_info.log(logger)

    try:
        dataset = HumanDemonstrationDataset(
            args.dataset,
            games=args.game,
            winning_only=args.winning_only,
            exclude_incomplete=args.exclude_incomplete,
        )
    except Exception as error:  # noqa: BLE001
        logger.error("could not load dataset: %s", error)
        return 1
    if not dataset.games:
        logger.error("the selected dataset contains no games; nothing to train on")
        return 1

    stats = dataset.statistics()
    logger.info("=" * 60)
    logger.info("Human dataset: %s", dataset.root)
    logger.info("=" * 60)
    logger.info("%s", stats.format())

    train_ids, val_ids = split_games(
        dataset.game_ids(), val_fraction=args.val_split, seed=args.seed
    )
    if not train_ids:
        logger.error("no training games after the %.2f validation split", args.val_split)
        return 1
    train_dataset = dataset.subset(train_ids)
    val_dataset = dataset.subset(val_ids)
    train_stats = DatasetStatistics.from_games(train_dataset.games)
    val_stats = DatasetStatistics.from_games(val_dataset.games)

    logger.info("-" * 60)
    logger.info("split (by game, seed=%d, val_split=%.2f)", args.seed, args.val_split)
    logger.info("training games:   %d", train_stats.games)
    logger.info("validation games: %d", val_stats.games)
    logger.info("training transitions:   %d", train_stats.transitions)
    logger.info("validation transitions: %d", val_stats.transitions)
    logger.info("train action types: OPEN=%d FLAG=%d CHORD=%d",
                train_stats.open_count, train_stats.flag_count, train_stats.chord_count)
    logger.info("val action types:   OPEN=%d FLAG=%d CHORD=%d",
                val_stats.open_count, val_stats.flag_count, val_stats.chord_count)

    config = BCConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_fraction=args.val_split,
        seed=args.seed,
        weight_decay=args.weight_decay,
        channels=args.channels,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = PolicyValueNetwork(
        width=dataset.width,
        height=dataset.height,
        channels=args.channels,
        action_space=FLAGS,
    )
    cells = dataset.cells
    logger.info(
        "model | class=%s observation_channels=%d actions=%d device=%s",
        type(policy).__name__, policy.observation_channels, policy.num_actions,
        device_info.device,
    )
    if policy.num_actions != action_count(FLAGS, cells=cells):
        logger.error("model action count does not match the dataset")
        return 1

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    latest_path = output / LATEST_FILENAME
    best_path = output / BEST_FILENAME

    optimizer = torch.optim.Adam(
        policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if args.resume and latest_path.exists():
        data = load_imitation_checkpoint(latest_path, policy, optimizer)
        history = list(data.get("history", []))
        start_epoch = int(data.get("epoch", 0)) + 1
        logger.info("resumed from %s at epoch %d", latest_path, start_epoch - 1)
    elif latest_path.exists():
        try:
            existing = read_imitation_checkpoint(latest_path)
        except ImitationError as error:
            logger.error("refusing to overwrite %s: %s", latest_path, error)
            return 1
        logger.warning("overwriting previous imitation run in %s", output)

    _atomic_write_json(
        output / RUN_CONFIG_FILE,
        {
            "algorithm": "behavioral_cloning",
            "format": IMITATION_CHECKPOINT_FORMAT,
            "version": IMITATION_CHECKPOINT_VERSION,
            "dataset": str(dataset.root),
            "filters": {
                "games": args.game,
                "winning_only": args.winning_only,
                "exclude_incomplete": args.exclude_incomplete,
            },
            "config": config.as_dict(),
            "dataset_statistics": stats.as_dict(),
            "train_statistics": train_stats.as_dict(),
            "val_statistics": val_stats.as_dict(),
            "device": str(device_info.device),
            "device_requested": device_info.requested,
            "cuda_available": device_info.cuda_available,
        },
    )

    best_val_accuracy: float | None = None

    def on_epoch(record: dict[str, Any], running_history: list[dict[str, Any]]) -> None:
        nonlocal best_val_accuracy
        epoch = record["epoch"]
        logger.info(
            "epoch %03d | train %s",
            epoch, _format_metrics(record["train"]),
        )
        logger.info(
            "           | val   %s",
            _format_metrics(record["val"]),
        )
        payload = build_imitation_checkpoint(
            policy, optimizer,
            epoch=epoch,
            config=config,
            dataset_stats=stats,
            train_stats=train_stats,
            val_stats=val_stats,
            history=running_history,
            extra={"dataset": str(dataset.root)},
        )
        atomic_save(payload, latest_path)
        val_accuracy = float(record["val"]["accuracy"])
        if best_val_accuracy is None or val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            atomic_save(payload, best_path)
            logger.info("new best val accuracy %.3f -> %s", val_accuracy, best_path)

    history, optimizer = train_imitation(
        policy, train_dataset.samples(), val_dataset.samples(), config,
        device_info.device, optimizer=optimizer, start_epoch=start_epoch,
        cells=cells, width=dataset.width, height=dataset.height,
        history=history, on_epoch=on_epoch,
    )

    _atomic_write_json(output / HISTORY_FILE, {"history": history})

    final = history[-1] if history else {}
    logger.info("=" * 60)
    logger.info("imitation training complete")
    logger.info("epochs=%d train_games=%d val_games=%d",
                len(history), train_stats.games, val_stats.games)
    if final:
        logger.info("final train %s", _format_metrics(final["train"]))
        logger.info("final val   %s", _format_metrics(final["val"]))
        logger.info(
            "val per-action-type accuracy: OPEN=%s FLAG=%s CHORD=%s",
            *(final["val"]["per_type_accuracy"][name] for name in ACTION_LABELS),
        )
    logger.info("latest checkpoint: %s", latest_path)
    logger.info("best checkpoint:   %s", best_path if best_path.exists() else "n/a")
    logger.info(
        "NOTE: action accuracy measures imitation of humans, not Minesweeper skill."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
