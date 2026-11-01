"""Experiment infrastructure: metrics, tracking, training, evaluation, plots."""

from __future__ import annotations

import importlib

from .browser import open_environment
from .checkpoint import (
    BEST_FILENAME,
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    LATEST_FILENAME,
    CheckpointError,
    CheckpointManager,
)
from .device import DEVICE_CHOICES, DeviceInfo, resolve_device
from .metrics import (
    EpisodeMetrics,
    MetricsLogger,
    MetricsSummary,
    UpdateMetrics,
    load_run,
    summarize,
)
from .plots import plot_evaluation, plot_training
from .tracking import TrackedEnvironment

# The CLI modules are imported lazily so ``python -m
# minesweeper_rl.experiment.train`` does not trigger a runpy double-import
# warning while still allowing ``from minesweeper_rl.experiment import
# run_training``.
_LAZY = {
    "EvaluationReport": ("evaluate", "EvaluationReport"),
    "greedy_action": ("evaluate", "greedy_action"),
    "load_frozen_policy": ("evaluate", "load_frozen_policy"),
    "run_evaluation": ("evaluate", "run_evaluation"),
    "PolicyCompatibilityError": ("initialization", "PolicyCompatibilityError"),
    "PretrainedPolicyInfo": ("initialization", "PretrainedPolicyInfo"),
    "check_policy_compatibility": ("initialization", "check_policy_compatibility"),
    "initialize_policy_from_imitation": (
        "initialization", "initialize_policy_from_imitation"),
    "inspect_imitation_checkpoint": ("initialization", "inspect_imitation_checkpoint"),
    "format_comparison": ("compare_evaluations", "format_comparison"),
    "load_evaluation_results": ("compare_evaluations", "load_evaluation_results"),
    "RUN_CONFIG_FILE": ("train", "RUN_CONFIG_FILE"),
    "TrainingReport": ("train", "TrainingReport"),
    "build_run_config": ("train", "build_run_config"),
    "config_mismatches": ("train", "config_mismatches"),
    "read_run_config": ("train", "read_run_config"),
    "run_training": ("train", "run_training"),
    "write_run_config": ("train", "write_run_config"),
}


def __getattr__(name: str):
    if name in _LAZY:
        module_name, attribute = _LAZY[name]
        module = importlib.import_module(f".{module_name}", __name__)
        value = getattr(module, attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BEST_FILENAME",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "CheckpointError",
    "CheckpointManager",
    "DEVICE_CHOICES",
    "DeviceInfo",
    "EpisodeMetrics",
    "EvaluationReport",
    "LATEST_FILENAME",
    "MetricsLogger",
    "MetricsSummary",
    "PolicyCompatibilityError",
    "PretrainedPolicyInfo",
    "RUN_CONFIG_FILE",
    "TrackedEnvironment",
    "TrainingReport",
    "UpdateMetrics",
    "build_run_config",
    "check_policy_compatibility",
    "config_mismatches",
    "format_comparison",
    "greedy_action",
    "initialize_policy_from_imitation",
    "inspect_imitation_checkpoint",
    "load_evaluation_results",
    "load_frozen_policy",
    "load_run",
    "open_environment",
    "plot_evaluation",
    "plot_training",
    "read_run_config",
    "resolve_device",
    "run_evaluation",
    "run_training",
    "summarize",
    "write_run_config",
]
