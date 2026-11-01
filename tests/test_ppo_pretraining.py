"""Tests for human-pretrained PPO initialization and training modes.

No browser and no live data.  A tiny flag-aware imitation checkpoint is forged
with the real behavioral-cloning checkpoint builder, then used to initialize the
PPO policy; the training entry point is driven against a deterministic fake
environment with ``open_environment`` monkeypatched.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from minesweeper_rl.agents import FLAGS, OPEN_ONLY, PPOConfig, PolicyValueNetwork
from minesweeper_rl.agents.minesweeper_env import MinesweeperPPOEnv
from minesweeper_rl.data import DatasetStatistics
from minesweeper_rl.dom import FLAGGED, HIDDEN, GameStatus
from minesweeper_rl.experiment import compare_evaluations, imitation_train, train as train_module
from minesweeper_rl.experiment.checkpoint import LATEST_FILENAME, read_checkpoint
from minesweeper_rl.experiment.imitation_train import BCConfig, build_imitation_checkpoint
from minesweeper_rl.experiment.initialization import (
    PolicyCompatibilityError,
    check_policy_compatibility,
    initialize_policy_from_imitation,
    inspect_imitation_checkpoint,
)
from minesweeper_rl.experiment.train import (
    INITIALIZATION_HUMAN,
    INITIALIZATION_SCRATCH,
    build_run_config,
    main,
    read_run_config,
)

WIDTH, HEIGHT = 4, 2
CELLS = WIDTH * HEIGHT


class FakeFlagBrowserEnv:
    """Deterministic, mine-free flag-aware board that ends in a win."""

    def __init__(self, *, stop_event=None, stop_after=None,
                 width: int = WIDTH, height: int = HEIGHT) -> None:
        self.config = SimpleNamespace(width=width, height=height)
        self.width = width
        self.height = height
        self._stop_event = stop_event
        self._stop_after = stop_after
        self.episodes = 0
        self._board = np.full((height, width), HIDDEN, dtype=np.int8)

    def reset(self) -> np.ndarray:
        self._board = np.full((self.height, self.width), HIDDEN, dtype=np.int8)
        self._board[0, 0] = 0
        return self._board.copy()

    def apply_action(self, action_type: int, x: int, y: int):
        from minesweeper_rl.action_types import ACTION_CHORD, ACTION_FLAG, ACTION_OPEN

        self._board = self._board.copy()
        newly = 0
        if action_type == ACTION_OPEN and self._board[y, x] == HIDDEN:
            self._board[y, x] = 0
            newly = 1
        elif action_type == ACTION_FLAG and self._board[y, x] in (HIDDEN, FLAGGED):
            self._board[y, x] = HIDDEN if self._board[y, x] == FLAGGED else FLAGGED
        elif action_type == ACTION_CHORD:
            for ny in range(self.height):
                for nx in range(self.width):
                    if self._board[ny, nx] == HIDDEN:
                        self._board[ny, nx] = 0
                        newly += 1
        done = bool((self._board != HIDDEN).all())
        status = GameStatus.WON if done else GameStatus.RUNNING
        info = {
            "action": (x, y),
            "action_type": action_type,
            "operation": "chord" if action_type == 2 else (
                "flag" if action_type == 1 else "open"),
            "game_status": status,
            "newly_revealed": newly,
            "hit_mine": False,
        }
        if done:
            self.episodes += 1
            if (
                self._stop_event is not None
                and self._stop_after is not None
                and self.episodes >= self._stop_after
            ):
                self._stop_event.set()
        return self._board.copy(), 0.0, done, False, info


def make_open(env, closed):
    @contextmanager
    def _open(url, profile, *, config=None):
        try:
            yield env
        finally:
            closed.append(True)

    return _open


def make_imitation_checkpoint(
    path, *, width: int = WIDTH, height: int = HEIGHT,
    channels: int = 16, action_space: str = FLAGS,
) -> PolicyValueNetwork:
    policy = PolicyValueNetwork(
        width=width, height=height, channels=channels, action_space=action_space
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    stats = DatasetStatistics(
        games=3, complete_games=2, winning_games=1, lost_games=1,
        incomplete_games=1, transitions=9, open_count=3, flag_count=3, chord_count=3,
    )
    payload = build_imitation_checkpoint(
        policy, optimizer, epoch=2, config=BCConfig(), dataset_stats=stats,
        train_stats=stats, val_stats=stats, history=[],
    )
    imitation_train.atomic_save(payload, path)
    return policy


def cli(run_dir, *extra: str, games: int = 2) -> list[str]:
    return [
        "--url", "http://example.test", "--games", str(games),
        "--run-dir", str(run_dir), "--checkpoint-interval", "1", "--no-plots",
        "--device", "cpu", "--seed", "7", *extra,
    ]


# ---------------------------------------------------------------------------
# Checkpoint -> PPO policy initialization
# ---------------------------------------------------------------------------


def test_initialize_loads_encoder_and_actor_and_resets_value_head(tmp_path) -> None:
    path = tmp_path / "imitation.pt"
    source = make_imitation_checkpoint(path)
    target = PolicyValueNetwork(width=WIDTH, height=HEIGHT)

    info = initialize_policy_from_imitation(target, path)

    assert info.num_actions == 3 * CELLS
    assert info.action_space == FLAGS
    assert info.human_dataset["games"] == 3
    assert torch.equal(target.encoder[0].weight, source.encoder[0].weight)
    assert torch.equal(target.actor.weight, source.actor.weight)
    assert torch.equal(target.actor.bias, source.actor.bias)
    # The value head is reset, not carried over from the untrusted BC critic.
    assert torch.equal(target.critic[2].weight, torch.zeros_like(target.critic[2].weight))
    assert torch.equal(target.critic[2].bias, torch.zeros_like(target.critic[2].bias))


def test_initialize_rejects_incompatible_architectures(tmp_path) -> None:
    path = tmp_path / "imitation.pt"
    make_imitation_checkpoint(path)

    wrong_width = PolicyValueNetwork(width=3, height=HEIGHT)
    with pytest.raises(PolicyCompatibilityError, match="width"):
        initialize_policy_from_imitation(wrong_width, path)

    wrong_channels = PolicyValueNetwork(width=WIDTH, height=HEIGHT, channels=8)
    with pytest.raises(PolicyCompatibilityError, match="channels"):
        initialize_policy_from_imitation(wrong_channels, path)

    wrong_space = PolicyValueNetwork(width=WIDTH, height=HEIGHT, action_space=OPEN_ONLY)
    with pytest.raises(PolicyCompatibilityError):
        initialize_policy_from_imitation(wrong_space, path)


def test_failed_initialization_leaves_policy_untouched(tmp_path) -> None:
    path = tmp_path / "imitation.pt"
    make_imitation_checkpoint(path)
    policy = PolicyValueNetwork(width=3, height=HEIGHT)
    before = policy.encoder[0].weight.detach().clone()

    with pytest.raises(PolicyCompatibilityError):
        initialize_policy_from_imitation(policy, path)

    assert torch.equal(policy.encoder[0].weight, before)


def test_inspect_reports_provenance(tmp_path) -> None:
    path = tmp_path / "imitation.pt"
    make_imitation_checkpoint(path)
    info = inspect_imitation_checkpoint(path)
    assert info.format == "minesweeper_imitation"
    assert info.epoch == 2
    assert info.observation_channels == 11
    check_policy_compatibility(info, PolicyValueNetwork(width=WIDTH, height=HEIGHT))


# ---------------------------------------------------------------------------
# Training modes through the PPO entry point
# ---------------------------------------------------------------------------


def test_scratch_training_records_random_initialization(tmp_path, monkeypatch) -> None:
    closed: list[bool] = []
    monkeypatch.setattr(
        train_module, "open_environment", make_open(FakeFlagBrowserEnv(), closed)
    )
    code = main(cli(tmp_path / "scratch"))

    assert code == 0
    assert closed == [True]
    config = read_run_config(tmp_path / "scratch")
    assert config["initialization"] == INITIALIZATION_SCRATCH
    assert config["human_data"] is False
    assert config["pretrained"] is False
    assert config["pretrained_policy"] is None
    assert config["pretrained_info"] is None
    assert config["architecture"] == "cnn"
    assert config["action_space"] == FLAGS
    assert config["observation_version"] == "flag_aware_v1"
    assert "reward" in config and "ppo" in config and "seed" in config


def test_pretrained_training_records_initialization(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "imitation.pt"
    make_imitation_checkpoint(checkpoint)
    closed: list[bool] = []
    monkeypatch.setattr(
        train_module, "open_environment", make_open(FakeFlagBrowserEnv(), closed)
    )

    code = main(cli(
        tmp_path / "human", "--training-mode", "human_pretrained",
        "--pretrained-policy", str(checkpoint),
    ))

    assert code == 0
    config = read_run_config(tmp_path / "human")
    assert config["initialization"] == INITIALIZATION_HUMAN
    assert config["human_data"] is True
    assert config["pretrained"] is True
    assert config["pretrained_policy"] == str(checkpoint)
    assert config["pretrained_info"]["num_actions"] == 3 * CELLS
    assert config["pretrained_info"]["human_dataset"]["games"] == 3

    data = read_checkpoint(tmp_path / "human" / LATEST_FILENAME)
    assert data["metadata"]["initialization"] == INITIALIZATION_HUMAN
    assert data["metadata"]["pretrained_policy"] == str(checkpoint)
    assert data["metadata"]["run_config"]["initialization"] == INITIALIZATION_HUMAN


def test_pretrained_mode_requires_checkpoint(tmp_path, monkeypatch) -> None:
    def boom(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("browser should not be opened")

    monkeypatch.setattr(train_module, "open_environment", boom)
    code = main(cli(
        tmp_path / "bad", "--training-mode", "human_pretrained",
    ))
    assert code == 1


def test_scratch_mode_rejects_pretrained_policy(tmp_path, monkeypatch) -> None:
    def boom(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("browser should not be opened")

    monkeypatch.setattr(train_module, "open_environment", boom)
    checkpoint = tmp_path / "imitation.pt"
    make_imitation_checkpoint(checkpoint)
    code = main(cli(
        tmp_path / "bad2", "--training-mode", "scratch",
        "--pretrained-policy", str(checkpoint),
    ))
    assert code == 1


def test_pretrained_training_resumes(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "imitation.pt"
    make_imitation_checkpoint(checkpoint)
    run_dir = tmp_path / "human"
    closed: list[bool] = []
    stop_event = threading.Event()

    env = FakeFlagBrowserEnv(stop_event=stop_event, stop_after=1)
    monkeypatch.setattr(train_module, "open_environment", make_open(env, closed))
    first = main(
        cli(run_dir, "--training-mode", "human_pretrained",
            "--pretrained-policy", str(checkpoint), games=3),
        stop_event=stop_event,
    )
    assert first == 0
    assert int(read_checkpoint(run_dir / LATEST_FILENAME)["episode_count"]) == 1

    # Resume without re-specifying the initialization: it is adopted from the
    # permanently recorded run_config.json.
    closed2: list[bool] = []
    monkeypatch.setattr(
        train_module, "open_environment", make_open(FakeFlagBrowserEnv(), closed2)
    )
    second = main(cli(run_dir, games=3))
    assert second == 0
    assert int(read_checkpoint(run_dir / LATEST_FILENAME)["episode_count"]) == 3
    assert read_run_config(run_dir)["initialization"] == INITIALIZATION_HUMAN
    data = read_checkpoint(run_dir / LATEST_FILENAME)
    assert data["metadata"]["run_config"]["initialization"] == INITIALIZATION_HUMAN


def test_build_run_config_records_required_keys() -> None:
    policy = PolicyValueNetwork(width=WIDTH, height=HEIGHT)
    env = MinesweeperPPOEnv(FakeFlagBrowserEnv(), action_space=FLAGS)
    from minesweeper_rl.experiment.device import DeviceInfo

    config = build_run_config(
        policy, env, PPOConfig(),
        games=5, checkpoint_interval=1, seed=1,
        device=DeviceInfo(torch.device("cpu"), "cpu", False),
        url="http://example.test", profile="p",
        initialization=INITIALIZATION_HUMAN,
        pretrained_policy="/tmp/ckpt.pt",
        pretrained_info={"num_actions": 3 * CELLS, "human_dataset": {"games": 3}},
    )
    for key in (
        "experiment", "architecture", "action_space", "observation_version",
        "pretrained", "pretrained_policy", "pretrained_info", "human_data",
        "initialization", "ppo", "seed", "device", "reward",
    ):
        assert key in config
    assert config["human_data"] is True
    assert config["pretrained"] is True


# ---------------------------------------------------------------------------
# Evaluation comparison
# ---------------------------------------------------------------------------


def test_evaluation_summary_tracks_action_counts() -> None:
    from minesweeper_rl.experiment.metrics import EpisodeMetrics, summarize

    episodes = [
        EpisodeMetrics(1, "eval", "lost", False, -5.0, 10, True, 20, 1.0, 0,
                       open_actions=7, flag_actions=2, chord_actions=1),
        EpisodeMetrics(2, "eval", "won", True, 120.0, 40, False, 300, 2.0, 0,
                       open_actions=30, flag_actions=8, chord_actions=2),
    ]
    summary = summarize(episodes)
    assert summary.mean_open_actions == pytest.approx(18.5)
    assert summary.mean_flag_actions == pytest.approx(5.0)
    assert summary.mean_chord_actions == pytest.approx(1.5)
    assert "open=" in summary.format() and "chord=" in summary.format()


def test_format_comparison_table() -> None:
    scratch = {"summary": {"games": 10, "win_rate": 0.1, "mean_reward": -3.0,
                           "mean_steps": 12.0, "mean_cells_revealed": 40.0,
                           "mine_hit_rate": 0.9, "mean_open_actions": 11.0,
                           "mean_flag_actions": 1.0, "mean_chord_actions": 0.0}}
    human = {"summary": {"games": 10, "win_rate": 0.2, "mean_reward": 5.0,
                         "mean_steps": 16.0, "mean_cells_revealed": 60.0,
                         "mine_hit_rate": 0.8, "mean_open_actions": 13.0,
                         "mean_flag_actions": 3.0, "mean_chord_actions": 0.0}}
    table = compare_evaluations.format_comparison({"scratch": scratch, "human": human})
    assert "scratch" in table and "human" in table
    assert "win_rate" in table and "open" in table and "chord" in table
