"""Training-readiness tests: config snapshots, metrics continuity, resume.

No live website is used.  The real ``MinesweeperPPOEnv`` / ``TrackedEnvironment``
/ ``PPOTrainer`` stack drives a deterministic fake browser environment while
``open_environment`` is monkeypatched, exactly as in ``test_train_resume``.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch

from minesweeper_rl.action_types import ACTION_CHORD, ACTION_FLAG, ACTION_OPEN
from minesweeper_rl.agents import PPOConfig, PolicyValueNetwork
from minesweeper_rl.agents.action_space import FLAGS, OPEN_ONLY
from minesweeper_rl.agents.minesweeper_env import MinesweeperPPOEnv
from minesweeper_rl.dom import FLAGGED, HIDDEN, GameStatus
from minesweeper_rl.experiment import load_run
from minesweeper_rl.experiment import train as train_module
from minesweeper_rl.experiment.checkpoint import LATEST_FILENAME, read_checkpoint
from minesweeper_rl.experiment.device import DeviceInfo
from minesweeper_rl.experiment.tracking import TrackedEnvironment
from minesweeper_rl.experiment.train import (
    RUN_CONFIG_FILE,
    build_run_config,
    config_mismatches,
    main,
    read_run_config,
    write_run_config,
)

WIDTH, HEIGHT = 4, 2


class FakeBrowserEnv:
    """Mine-free board where OPEN reveals and FLAG toggles, win on completion."""

    def __init__(self, *, stop_event=None, stop_after=None,
                 width: int = WIDTH, height: int = HEIGHT) -> None:
        self.config = SimpleNamespace(width=width, height=height)
        self.width = width
        self.height = height
        self._stop_event = stop_event
        self._stop_after = stop_after
        self.episodes = 0
        self._step_count = 0
        self._board = np.full((height, width), -1, dtype=np.int8)

    def _budget(self) -> int:
        # A real game always ends; cap absurd non-terminating flag loops so the
        # FLAGS action space cannot keep a mine-free fake alive forever.
        return 4 * self.width * self.height

    def reset(self) -> np.ndarray:
        self._board = np.full((self.height, self.width), -1, dtype=np.int8)
        self._board[0, 0] = 0
        self._step_count = 0
        return self._board.copy()

    def step(self, action):
        x, y = action
        if self._board[y, x] != HIDDEN:
            raise ValueError(f"cell ({x}, {y}) already revealed")
        self._board = self._board.copy()
        self._board[y, x] = 0
        self._step_count += 1
        win = (self._board < 0).sum() == 0
        timed_out = not win and self._step_count >= self._budget()
        if win:
            status, reward, terminated = GameStatus.WON, 100.0, True
        elif timed_out:
            status, reward, terminated = GameStatus.LOST, -10.0, True
        else:
            status, reward, terminated = GameStatus.RUNNING, 1.0, False
        info = {
            "newly_revealed": 1,
            "hit_mine": timed_out,
            "game_status": status,
            "operation": "open",
        }
        if terminated:
            self.episodes += 1
            if (
                self._stop_event is not None
                and self._stop_after is not None
                and self.episodes >= self._stop_after
            ):
                self._stop_event.set()
        return self._board.copy(), reward, terminated, False, info

    def apply_action(self, action_type: int, x: int, y: int):
        """FLAGS dispatch used by ``MinesweeperPPOEnv`` on the canonical path."""

        if action_type == ACTION_OPEN:
            return self.step((x, y))
        self._board = self._board.copy()
        newly = 0
        if action_type == ACTION_FLAG:
            if self._board[y, x] not in (HIDDEN, FLAGGED):
                raise ValueError(f"cell ({x}, {y}) cannot be flagged")
            self._board[y, x] = HIDDEN if self._board[y, x] == FLAGGED else FLAGGED
        elif action_type == ACTION_CHORD:
            for ny in range(self.height):
                for nx in range(self.width):
                    if self._board[ny, nx] == HIDDEN:
                        self._board[ny, nx] = 0
                        newly += 1
        self._step_count += 1
        win = bool((self._board < 0).sum() == 0)
        timed_out = not win and self._step_count >= self._budget()
        terminated = win or timed_out
        status = GameStatus.WON if win else (GameStatus.LOST if timed_out else GameStatus.RUNNING)
        reward = 0.0 if action_type == ACTION_FLAG and not timed_out else (
            -10.0 if timed_out else (100.0 if win else float(newly))
        )
        info = {
            "newly_revealed": newly,
            "hit_mine": timed_out,
            "game_status": status,
            "operation": "flag" if action_type == ACTION_FLAG else "open",
        }
        if terminated:
            self.episodes += 1
            if (
                self._stop_event is not None
                and self._stop_after is not None
                and self.episodes >= self._stop_after
            ):
                self._stop_event.set()
        return self._board.copy(), reward, terminated, False, info


def make_fake_open(env, closed):
    @contextmanager
    def _open(url, profile, *, config=None):
        try:
            yield env
        finally:
            closed.append(True)

    return _open


def _device_info() -> DeviceInfo:
    return DeviceInfo(torch.device("cpu"), "cpu", torch.cuda.is_available())


def _fake_env_for_config() -> SimpleNamespace:
    return SimpleNamespace(width=WIDTH, height=HEIGHT, num_actions=WIDTH * HEIGHT)


# ---------------------------------------------------------------------------
# Random initialization / research boundary.
# ---------------------------------------------------------------------------


def test_fresh_models_are_randomly_initialized() -> None:
    first = PolicyValueNetwork(width=WIDTH, height=HEIGHT, channels=8)
    second = PolicyValueNetwork(width=WIDTH, height=HEIGHT, channels=8)

    first_state = first.state_dict()
    second_state = second.state_dict()
    assert not all(
        torch.equal(first_state[key], second_state[key]) for key in first_state
    )


def test_fresh_model_matches_open_only_architecture() -> None:
    policy = PolicyValueNetwork(width=WIDTH, height=HEIGHT, action_space=OPEN_ONLY)
    assert policy.observation_channels == 10
    assert policy.num_actions == WIDTH * HEIGHT
    assert policy.encoder[0].in_channels == 10
    assert policy.actor.out_channels == 1


def test_fresh_model_matches_flag_aware_architecture() -> None:
    policy = PolicyValueNetwork(width=WIDTH, height=HEIGHT)
    assert policy.action_space == FLAGS
    assert policy.observation_channels == 11
    assert policy.num_actions == 3 * WIDTH * HEIGHT
    assert policy.encoder[0].in_channels == 11
    assert policy.actor.out_channels == 3


def test_run_config_declares_random_initialization_and_no_human_data() -> None:
    policy = PolicyValueNetwork(
        width=WIDTH, height=HEIGHT, channels=8, action_space=OPEN_ONLY
    )
    config = build_run_config(
        policy,
        _fake_env_for_config(),
        PPOConfig(),
        games=10,
        checkpoint_interval=5,
        seed=42,
        device=_device_info(),
        url="http://example.test",
        profile="browser_profile",
        experiment="cnn_ppo_flags_v1",
    )
    assert config["experiment"] == "cnn_ppo_flags_v1"
    assert config["algorithm"] == "ppo"
    assert config["architecture"] == "cnn"
    assert config["human_data"] is False
    assert config["initialization"] == "random"
    assert config["action_space"] == OPEN_ONLY
    assert config["observation_channels"] == 10
    assert config["num_actions"] == WIDTH * HEIGHT
    assert config["action_type"] == "OPEN_ONLY"
    assert config["flags"] is False
    assert config["chords"] is False
    assert config["reward"] == "newly_revealed_cells + terminal_reward (FLAG neutral)"
    assert config["games"] == 10
    assert config["seed"] == 42


def test_run_config_declares_flag_aware_experiment() -> None:
    env = SimpleNamespace(
        width=WIDTH, height=HEIGHT, num_actions=3 * WIDTH * HEIGHT,
        action_space=FLAGS, mines=99, difficulty="expert",
    )
    policy = PolicyValueNetwork(width=WIDTH, height=HEIGHT, channels=8)
    config = build_run_config(
        policy,
        env,
        PPOConfig(),
        games=5,
        checkpoint_interval=1,
        seed=1,
        device=_device_info(),
        url="http://example.test",
        profile="browser_profile",
        experiment="cnn_ppo_flags_v1",
    )
    assert config["action_space"] == FLAGS
    assert config["action_type"] == "OPEN_FLAG_CHORD"
    assert config["flags"] is True
    assert config["chords"] is True
    assert config["observation_channels"] == 11
    assert config["num_actions"] == 3 * WIDTH * HEIGHT


def test_default_run_dir_is_primary_experiment_directory() -> None:
    from minesweeper_rl.experiment.train import build_parser

    args = build_parser().parse_args(["--url", "http://example.test"])
    assert args.run_dir.as_posix() == "experiments/cnn_ppo_flags_v1"


# ---------------------------------------------------------------------------
# Configuration snapshot: round-trip and compatibility.
# ---------------------------------------------------------------------------


def test_run_config_round_trip(tmp_path) -> None:
    policy = PolicyValueNetwork(width=WIDTH, height=HEIGHT, channels=8)
    config = build_run_config(
        policy,
        _fake_env_for_config(),
        PPOConfig(),
        games=5,
        checkpoint_interval=2,
        seed=None,
        device=_device_info(),
        url="http://example.test",
        profile="p",
    )
    path = write_run_config(tmp_path, config)
    assert path.exists()
    assert read_run_config(tmp_path) == config


def test_config_mismatch_detects_hyperparameter_change() -> None:
    policy = PolicyValueNetwork(width=WIDTH, height=HEIGHT, channels=8)
    env = _fake_env_for_config()
    saved = build_run_config(
        policy, env, PPOConfig(learning_rate=3e-4),
        games=10, checkpoint_interval=5, seed=1,
        device=_device_info(), url="u", profile="p",
    )
    requested = build_run_config(
        policy, env, PPOConfig(learning_rate=9e-3),
        games=10, checkpoint_interval=5, seed=1,
        device=_device_info(), url="u", profile="p",
    )

    mismatches = config_mismatches(saved, requested)
    assert any("learning_rate" in mismatch for mismatch in mismatches)


def test_config_mismatch_allows_target_and_device_changes() -> None:
    policy = PolicyValueNetwork(width=WIDTH, height=HEIGHT, channels=8)
    env = _fake_env_for_config()
    saved = build_run_config(
        policy, env, PPOConfig(),
        games=10, checkpoint_interval=5, seed=1,
        device=DeviceInfo(torch.device("cpu"), "cpu", False), url="u", profile="p",
    )
    requested = build_run_config(
        policy, env, PPOConfig(),
        games=25, checkpoint_interval=50, seed=None,
        device=DeviceInfo(torch.device("cuda"), "cuda", False), url="u", profile="p",
    )

    assert config_mismatches(saved, requested) == []


def test_config_mismatch_treats_missing_action_space_as_open_only() -> None:
    # Run configurations written before the flag-aware revision had no
    # ``action_space`` field; they describe the OPEN-only baseline.
    policy = PolicyValueNetwork(
        width=WIDTH, height=HEIGHT, channels=8, action_space=OPEN_ONLY
    )
    requested = build_run_config(
        policy, _fake_env_for_config(), PPOConfig(),
        games=10, checkpoint_interval=5, seed=1,
        device=_device_info(), url="u", profile="p",
    )
    saved = dict(requested)
    saved.pop("action_space")

    assert config_mismatches(saved, requested) == []


def test_config_mismatch_detects_action_space_change() -> None:
    open_policy = PolicyValueNetwork(
        width=WIDTH, height=HEIGHT, channels=8, action_space=OPEN_ONLY
    )
    flag_policy = PolicyValueNetwork(width=WIDTH, height=HEIGHT, channels=8)
    env = _fake_env_for_config()
    saved = build_run_config(
        open_policy, env, PPOConfig(),
        games=10, checkpoint_interval=5, seed=1,
        device=_device_info(), url="u", profile="p",
    )
    requested = build_run_config(
        flag_policy, env, PPOConfig(),
        games=10, checkpoint_interval=5, seed=1,
        device=_device_info(), url="u", profile="p",
    )

    assert any("action_space" in mismatch for mismatch in config_mismatches(saved, requested))


# ---------------------------------------------------------------------------
# OPEN/FLAG accounting and metrics continuity.
# ---------------------------------------------------------------------------


def test_tracked_environment_counts_open_actions() -> None:
    tracker = TrackedEnvironment(
        MinesweeperPPOEnv(FakeBrowserEnv(), action_space=OPEN_ONLY)
    )
    tracker.reset()                    # cell 0 is already revealed by reset
    for cell in range(1, WIDTH * HEIGHT):
        tracker.step(cell)             # OPEN the remaining cells -> win

    episode = tracker.last_episode
    assert episode is not None
    assert episode.outcome == "won"
    assert episode.open_actions == WIDTH * HEIGHT - 1
    assert episode.open_actions == episode.steps
    assert episode.flag_actions == 0
    assert episode.chord_actions == 0
    assert episode.source == "agent"


# ---------------------------------------------------------------------------
# Total-games semantics and pause/resume.
# ---------------------------------------------------------------------------


def test_resume_continues_to_total_target_without_duplicates(tmp_path, monkeypatch) -> None:
    closed: list[bool] = []
    stop_event = threading.Event()
    env = FakeBrowserEnv(stop_event=stop_event, stop_after=3)
    monkeypatch.setattr(train_module, "open_environment", make_fake_open(env, closed))

    first = main(
        ["--url", "http://example.test", "--games", "10", "--device", "cpu",
         "--run-dir", str(tmp_path),
         "--checkpoint-interval", "1", "--no-plots"],
        stop_event=stop_event,
    )

    assert first == 0
    assert closed == [True]
    episodes, _ = load_run(tmp_path)
    assert [episode.episode for episode in episodes] == [1, 2, 3]
    assert int(read_checkpoint(tmp_path / LATEST_FILENAME)["episode_count"]) == 3

    closed2: list[bool] = []
    env2 = FakeBrowserEnv()
    monkeypatch.setattr(train_module, "open_environment", make_fake_open(env2, closed2))

    second = main(
        ["--url", "http://example.test", "--games", "10", "--device", "cpu",
         "--run-dir", str(tmp_path),
         "--checkpoint-interval", "1", "--no-plots"]
    )

    assert second == 0
    assert closed2 == [True]
    episodes, _ = load_run(tmp_path)
    assert [episode.episode for episode in episodes] == list(range(1, 11))
    assert int(read_checkpoint(tmp_path / LATEST_FILENAME)["episode_count"]) == 10


def test_resume_rejects_incompatible_hyperparameters(tmp_path, monkeypatch) -> None:
    closed: list[bool] = []
    monkeypatch.setattr(
        train_module, "open_environment", make_fake_open(FakeBrowserEnv(), closed)
    )
    first = main(
        ["--url", "http://example.test", "--games", "2", "--device", "cpu",
         "--run-dir", str(tmp_path), "--no-plots"]
    )
    assert first == 0
    assert read_run_config(tmp_path) is not None

    closed2: list[bool] = []
    monkeypatch.setattr(
        train_module, "open_environment", make_fake_open(FakeBrowserEnv(), closed2)
    )
    code = main(
        ["--url", "http://example.test", "--games", "5", "--device", "cpu",
         "--learning-rate", "9e-3",
         "--run-dir", str(tmp_path), "--no-plots"]
    )

    assert code == 1
    assert closed2 == [True]
    # The existing checkpoint is untouched (still at 2 episodes).
    assert int(read_checkpoint(tmp_path / LATEST_FILENAME)["episode_count"]) == 2
    assert (tmp_path / RUN_CONFIG_FILE).exists()
