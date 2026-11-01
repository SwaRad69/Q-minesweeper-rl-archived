"""Focused reliability tests for the training persistence system.

Covers the simulated workflow: new run, graceful stop, resume, continuous vs
resumed determinism, --fresh, corrupted/incompatible checkpoints and atomic
checkpoint writes.  No live website: a deterministic coordinate-based mock
environment is driven through the real PPO stack, and ``open_environment`` is
monkeypatched so Playwright never launches.
"""

from __future__ import annotations

import random
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from minesweeper_rl.action_types import ACTION_CHORD, ACTION_FLAG, ACTION_OPEN
from minesweeper_rl.agents import FLAGS, PPOConfig, PPOTrainer, PolicyValueNetwork
from minesweeper_rl.agents.action_space import OPEN_ONLY
from minesweeper_rl.agents.minesweeper_env import MinesweeperPPOEnv
from minesweeper_rl.dom import FLAGGED, HIDDEN, GameStatus
from minesweeper_rl.experiment import MetricsLogger, load_run
from minesweeper_rl.experiment import train as train_module
from minesweeper_rl.experiment.checkpoint import (
    LATEST_FILENAME,
    CheckpointError,
    load_checkpoint,
    read_checkpoint,
    write_checkpoint,
)
from minesweeper_rl.experiment.metrics import EpisodeMetrics
from minesweeper_rl.experiment.tracking import TrackedEnvironment
from minesweeper_rl.experiment.train import main, run_training, seed_everything

WIDTH, HEIGHT = 4, 2
BASE_CONFIG = PPOConfig(
    rollout_steps=1, num_epochs=1, minibatch_size=8, learning_rate=1e-2
)


class StatelessBoardEnv:
    """Deterministic, memoryless board: reset always yields the same board."""

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
        done = win or timed_out
        status = GameStatus.WON if win else (GameStatus.LOST if timed_out else GameStatus.RUNNING)
        reward = 100.0 if win else (-10.0 if timed_out else 1.0)
        info = {"newly_revealed": 1, "hit_mine": timed_out, "game_status": status}
        if done:
            self.episodes += 1
            if (
                self._stop_event is not None
                and self._stop_after is not None
                and self.episodes >= self._stop_after
            ):
                self._stop_event.set()
        return self._board.copy(), reward, done, False, info

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
        done = win or timed_out
        status = GameStatus.WON if win else (GameStatus.LOST if timed_out else GameStatus.RUNNING)
        reward = 0.0 if action_type == ACTION_FLAG and not timed_out else (
            -10.0 if timed_out else (100.0 if win else float(newly))
        )
        info = {"newly_revealed": newly, "hit_mine": timed_out, "game_status": status}
        if done:
            self.episodes += 1
            if (
                self._stop_event is not None
                and self._stop_after is not None
                and self.episodes >= self._stop_after
            ):
                self._stop_event.set()
        return self._board.copy(), reward, done, False, info


def make_trainer(env: StatelessBoardEnv | None = None):
    env = env or StatelessBoardEnv()
    tracker = TrackedEnvironment(
        MinesweeperPPOEnv(env, action_space=FLAGS), phase="train"
    )
    policy = PolicyValueNetwork(
        width=env.width, height=env.height, channels=16, action_space=FLAGS
    )
    trainer = PPOTrainer(
        policy,
        tracker,
        num_actions=3 * env.width * env.height,
        config=BASE_CONFIG,
    )
    return trainer, tracker


def make_open(env, opened, closed):
    @contextmanager
    def _open(url, profile, *, config=None):
        opened.append(True)
        try:
            yield env
        finally:
            closed.append(True)

    return _open


def seed_run(run_dir, episodes: int = 2):
    trainer, tracker = make_trainer()
    return run_training(
        trainer, tracker, MetricsLogger(run_dir),
        games=episodes,
        checkpoint=Path(run_dir) / LATEST_FILENAME,
        emit=lambda _: None,
    )


def cli(run_dir, games: int, *extra: str) -> list[str]:
    return ["--url", "http://example.test", "--games", str(games),
            "--run-dir", str(run_dir), "--no-plots", *extra]


# ---------------------------------------------------------------------------
# TEST 1 - new run
# ---------------------------------------------------------------------------


def test_new_run_checkpoint_contents(tmp_path) -> None:
    trainer, tracker = make_trainer()
    metrics = MetricsLogger(tmp_path)
    report = run_training(
        trainer, tracker, metrics, games=4,
        checkpoint=tmp_path / LATEST_FILENAME, emit=lambda _: None,
    )

    path = tmp_path / LATEST_FILENAME
    assert path.exists()
    data = read_checkpoint(path)

    assert data["policy"], "model state must be present"
    assert data["optimizer"]["state"], "optimizer state must be present"
    assert int(data["episode_count"]) == 4
    assert int(data["iteration"]) == 4
    assert int(data["total_steps"]) == report.total_steps
    assert int(data["total_steps"]) == sum(e.steps for e in metrics.episodes)

    episodes, updates = load_run(tmp_path)
    assert len(episodes) == 4
    assert len(updates) == 4


# ---------------------------------------------------------------------------
# TEST 2 - stop (Ctrl+C equivalent) and cleanup
# ---------------------------------------------------------------------------


def test_stop_is_graceful_and_checkpoint_loadable(tmp_path) -> None:
    stop_event = threading.Event()
    env = StatelessBoardEnv(stop_event=stop_event, stop_after=2)
    trainer, tracker = make_trainer(env)
    metrics = MetricsLogger(tmp_path)

    report = run_training(
        trainer, tracker, metrics, games=10,
        checkpoint=tmp_path / LATEST_FILENAME, stop_event=stop_event,
        emit=lambda _: None,
    )

    assert report.stopped_early is True
    assert report.final_episode == 2
    assert len(metrics.episodes) == 2

    data = read_checkpoint(tmp_path / LATEST_FILENAME)
    restored, _ = make_trainer()
    load_checkpoint(restored, data)  # must be loadable
    assert restored.episode_count == 2


def test_stop_closes_browser(tmp_path, monkeypatch) -> None:
    opened: list[bool] = []
    closed: list[bool] = []
    stop_event = threading.Event()
    env = StatelessBoardEnv(stop_event=stop_event, stop_after=2)
    monkeypatch.setattr(
        train_module, "open_environment", make_open(env, opened, closed)
    )

    code = main(cli(tmp_path, 10), stop_event=stop_event)

    assert code == 0
    assert opened == [True]
    assert closed == [True]
    assert int(read_checkpoint(tmp_path / LATEST_FILENAME)["episode_count"]) == 2
    assert len(load_run(tmp_path)[0]) == 2


# ---------------------------------------------------------------------------
# TEST 3 - resume
# ---------------------------------------------------------------------------


def test_resume_restores_model_and_optimizer_before_training(tmp_path) -> None:
    seed_run(tmp_path, episodes=2)
    data = read_checkpoint(tmp_path / LATEST_FILENAME)

    restored, _ = make_trainer()
    load_checkpoint(restored, data)

    for key, value in data["policy"].items():
        assert torch.equal(value, restored.policy.state_dict()[key]), key
    assert restored.optimizer.state_dict()["state"]
    assert restored.episode_count == 2
    assert restored.iteration == 2
    assert restored.total_steps == int(data["total_steps"])


def test_resume_continues_episode_numbering(tmp_path) -> None:
    seed_run(tmp_path, episodes=2)

    trainer, tracker = make_trainer()
    trainer.load(tmp_path / LATEST_FILENAME)
    metrics = MetricsLogger(tmp_path)
    metrics.load_existing()

    report = run_training(
        trainer, tracker, metrics, games=5,
        checkpoint=tmp_path / LATEST_FILENAME, emit=lambda _: None,
    )

    assert report.started_at_episode == 2
    assert report.games_completed == 3
    ids = [e.episode for e in load_run(tmp_path)[0]]
    assert ids == [1, 2, 3, 4, 5]


def test_main_auto_detects_and_continues(tmp_path, monkeypatch) -> None:
    seed_run(tmp_path, episodes=2)
    opened: list[bool] = []
    closed: list[bool] = []
    stop_event = threading.Event()
    env = StatelessBoardEnv(stop_event=stop_event, stop_after=1)
    monkeypatch.setattr(
        train_module, "open_environment", make_open(env, opened, closed)
    )

    code = main(cli(tmp_path, 4), stop_event=stop_event)

    assert code == 0
    assert opened == [True] and closed == [True]
    ids = [e.episode for e in load_run(tmp_path)[0]]
    assert ids == [1, 2, 3], "must continue, not reset, the episode counter"


# ---------------------------------------------------------------------------
# TEST 4 - continuous vs resumed determinism
# ---------------------------------------------------------------------------


def _seeded_trainer(seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = StatelessBoardEnv()
    tracker = TrackedEnvironment(
        MinesweeperPPOEnv(env, action_space=OPEN_ONLY), phase="train"
    )
    policy = PolicyValueNetwork(
        width=env.width, height=env.height, channels=8, action_space=OPEN_ONLY
    )
    config = PPOConfig(
        rollout_steps=1, num_epochs=2, minibatch_size=4, learning_rate=1e-2
    )
    trainer = PPOTrainer(
        policy, tracker, num_actions=env.width * env.height, config=config
    )
    return trainer, tracker


def test_continuous_and_resumed_training_are_identical(tmp_path) -> None:
    total, split = 6, 3

    # Run A: one uninterrupted session.
    trainer_a, tracker_a = _seeded_trainer(0)
    run_training(
        trainer_a, tracker_a, MetricsLogger(tmp_path / "A"), games=total,
        checkpoint=tmp_path / "A" / LATEST_FILENAME, emit=lambda _: None,
    )

    # Run B: same seed, train `split`, save, then resume in a "new process".
    trainer_b, tracker_b = _seeded_trainer(0)
    run_training(
        trainer_b, tracker_b, MetricsLogger(tmp_path / "B"), games=split,
        checkpoint=tmp_path / "B" / LATEST_FILENAME, emit=lambda _: None,
    )
    data = read_checkpoint(tmp_path / "B" / LATEST_FILENAME)
    assert int(data["episode_count"]) == split

    trainer_b2, tracker_b2 = _seeded_trainer(0)
    load_checkpoint(trainer_b2, data)  # restores weights, optimizer, RNG
    metrics_b2 = MetricsLogger(tmp_path / "B")
    metrics_b2.load_existing()
    run_training(
        trainer_b2, tracker_b2, metrics_b2, games=total,
        checkpoint=tmp_path / "B" / LATEST_FILENAME, emit=lambda _: None,
    )

    # Final weights must be bit-for-bit identical.
    for key, value in trainer_a.policy.state_dict().items():
        assert torch.equal(value, trainer_b2.policy.state_dict()[key]), key

    # Update records from the resumed portion must match the continuous run.
    updates_a = load_run(tmp_path / "A")[1]
    updates_b = load_run(tmp_path / "B")[1]
    assert len(updates_a) == len(updates_b) == total
    for a, b in zip(updates_a[split:], updates_b[split:]):
        assert a.iteration == b.iteration
        assert a.policy_loss == b.policy_loss
        assert a.value_loss == b.value_loss
        assert a.entropy == b.entropy
        assert a.approx_kl == b.approx_kl
        assert a.total_loss == b.total_loss


# ---------------------------------------------------------------------------
# TEST 5 - fresh
# ---------------------------------------------------------------------------


def test_fresh_starts_over_and_archives(tmp_path, monkeypatch) -> None:
    seed_run(tmp_path, episodes=3)
    opened: list[bool] = []
    closed: list[bool] = []
    env = StatelessBoardEnv()
    monkeypatch.setattr(
        train_module, "open_environment", make_open(env, opened, closed)
    )

    code = main(cli(tmp_path, 2, "--fresh"))

    assert code == 0
    ids = [e.episode for e in load_run(tmp_path)[0]]
    assert ids == [1, 2], "a fresh run must restart episode numbering"
    assert int(read_checkpoint(tmp_path / LATEST_FILENAME)["episode_count"]) == 2

    archives = list((tmp_path / "archive").iterdir())
    assert len(archives) == 1
    assert (archives[0] / LATEST_FILENAME).exists()


# ---------------------------------------------------------------------------
# TEST 6 - corrupted checkpoint
# ---------------------------------------------------------------------------


def test_corrupted_checkpoint_fails_clearly(tmp_path, monkeypatch) -> None:
    seed_run(tmp_path, episodes=2)
    latest = tmp_path / LATEST_FILENAME
    latest.write_bytes(b"\x00\x01 not a torch checkpoint \xff")
    corrupt_bytes = latest.read_bytes()

    opened: list[bool] = []
    closed: list[bool] = []
    env = StatelessBoardEnv()
    monkeypatch.setattr(
        train_module, "open_environment", make_open(env, opened, closed)
    )

    code = main(cli(tmp_path, 5))

    assert code != 0, "a corrupt checkpoint must not silently start a new run"
    assert opened == [], "no browser should be opened"
    assert latest.read_bytes() == corrupt_bytes, "corrupt file must be untouched"
    assert len(load_run(tmp_path)[0]) == 2, "metrics must be untouched"


def test_fresh_recovers_from_corrupted_checkpoint(tmp_path, monkeypatch) -> None:
    seed_run(tmp_path, episodes=2)
    (tmp_path / LATEST_FILENAME).write_bytes(b"garbage")

    opened: list[bool] = []
    closed: list[bool] = []
    env = StatelessBoardEnv()
    monkeypatch.setattr(
        train_module, "open_environment", make_open(env, opened, closed)
    )

    code = main(cli(tmp_path, 2, "--fresh"))

    assert code == 0
    assert opened == [True] and closed == [True]
    assert len(load_run(tmp_path)[0]) == 2
    assert len(list((tmp_path / "archive").iterdir())) == 1


# ---------------------------------------------------------------------------
# TEST 7 - incompatible checkpoint
# ---------------------------------------------------------------------------


def test_incompatible_checkpoint_rejected(tmp_path, monkeypatch) -> None:
    # A checkpoint trained for a 4x1 board (num_actions == 4).
    source = PPOTrainer(
        PolicyValueNetwork(width=4, height=1, channels=8, action_space=OPEN_ONLY),
        SimpleNamespace(),
        num_actions=4,
        config=PPOConfig(),
    )
    write_checkpoint(source, tmp_path / LATEST_FILENAME)

    opened: list[bool] = []
    closed: list[bool] = []
    env = StatelessBoardEnv(width=4, height=2)  # num_actions == 8
    monkeypatch.setattr(
        train_module, "open_environment", make_open(env, opened, closed)
    )

    code = main(cli(tmp_path, 1))

    assert code != 0
    assert opened == [True] and closed == [True], "browser must still be closed"
    # Original (incompatible) checkpoint left intact for inspection.
    assert int(read_checkpoint(tmp_path / LATEST_FILENAME)["num_actions"]) == 4


def test_incompatible_checkpoint_raises_at_layer(tmp_path) -> None:
    source = PPOTrainer(
        PolicyValueNetwork(width=4, height=1, channels=8, action_space=OPEN_ONLY),
        SimpleNamespace(),
        num_actions=4,
        config=PPOConfig(),
    )
    data = write_checkpoint(source, tmp_path / "bad.pt")
    payload = read_checkpoint(data)

    recipient, _ = make_trainer()  # 4x2 FLAGS, num_actions == 24
    with pytest.raises(CheckpointError, match="action space"):
        load_checkpoint(recipient, payload)


# ---------------------------------------------------------------------------
# TEST 8 - interrupted checkpoint write
# ---------------------------------------------------------------------------


def test_interrupted_checkpoint_preserves_previous(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / LATEST_FILENAME
    trainer, tracker = make_trainer()
    run_training(
        trainer, tracker, MetricsLogger(tmp_path), games=1,
        checkpoint=checkpoint, emit=lambda _: None,
    )
    assert int(read_checkpoint(checkpoint)["episode_count"]) == 1
    original = checkpoint.read_bytes()

    def exploding_save(*_args, **_kwargs):
        raise RuntimeError("simulated interruption during checkpoint write")

    monkeypatch.setattr(torch, "save", exploding_save)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_training(
            trainer, tracker, MetricsLogger(tmp_path), games=2,
            checkpoint=checkpoint, emit=lambda _: None,
        )
    monkeypatch.undo()

    assert checkpoint.read_bytes() == original
    data = read_checkpoint(checkpoint)
    assert int(data["episode_count"]) == 1
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Extra - metrics ahead of checkpoint must not duplicate episode ids
# ---------------------------------------------------------------------------


def test_metrics_ahead_of_checkpoint_does_not_duplicate_ids(tmp_path) -> None:
    seed_run(tmp_path, episodes=2)
    metrics = MetricsLogger(tmp_path)
    metrics.load_existing()
    for episode_id in (3, 4):
        metrics.log_episode(
            EpisodeMetrics(episode_id, "train", "won", True, 100.0, 7, False, 7, 1.0, 0)
        )

    trainer, tracker = make_trainer()
    trainer.load(tmp_path / LATEST_FILENAME)
    resumed = MetricsLogger(tmp_path)
    resumed.load_existing()

    report = run_training(
        trainer, tracker, resumed, games=5,
        checkpoint=tmp_path / LATEST_FILENAME, emit=lambda _: None,
    )

    ids = [e.episode for e in resumed.episodes]
    assert ids == [1, 2, 3, 4, 5]
    assert len(ids) == len(set(ids))
    assert report.started_at_episode == 4
    assert report.games_completed == 1


def test_multiple_resume_cycles_keep_episode_ids_unique(tmp_path) -> None:
    seed_run(tmp_path, episodes=2)

    for target in (4, 6):
        trainer, tracker = make_trainer()
        trainer.load(tmp_path / LATEST_FILENAME)
        metrics = MetricsLogger(tmp_path)
        metrics.load_existing()
        run_training(
            trainer, tracker, metrics, games=target,
            checkpoint=tmp_path / LATEST_FILENAME, emit=lambda _: None,
        )

    ids = [e.episode for e in load_run(tmp_path)[0]]
    assert ids == [1, 2, 3, 4, 5, 6]
    assert len(ids) == len(set(ids))


def test_seed_everything_is_reproducible() -> None:
    seed_everything(5)
    first = (torch.rand(3).tolist(), random.random(), float(np.random.rand()))
    seed_everything(5)
    second = (torch.rand(3).tolist(), random.random(), float(np.random.rand()))
    assert first == second


def test_main_records_seed_and_algorithm(tmp_path, monkeypatch) -> None:
    opened: list[bool] = []
    closed: list[bool] = []
    env = StatelessBoardEnv()
    monkeypatch.setattr(
        train_module, "open_environment", make_open(env, opened, closed)
    )

    code = main(cli(tmp_path, 1, "--seed", "7"))

    assert code == 0
    data = read_checkpoint(tmp_path / LATEST_FILENAME)
    assert data["metadata"]["seed"] == 7
    assert data["metadata"]["algorithm"] == "ppo"
    assert data["model"]["width"] == WIDTH
    assert data["model"]["height"] == HEIGHT


# ---------------------------------------------------------------------------
# Stuck episodes: the trainer truncates internally, the env never terminates.
# ---------------------------------------------------------------------------


class SometimesStuckEnv:
    """Wins immediately on the first episode, then never terminates again.

    This reproduces a stuck live game: ``PPOTrainer.collect_episode`` closes the
    second episode internally (step budget) without the environment reporting a
    terminal state, so the tracking wrapper must be finalized by the training
    loop or the previous episode id is logged twice.
    """

    def __init__(self, width: int = WIDTH, height: int = HEIGHT) -> None:
        self.config = SimpleNamespace(width=width, height=height)
        self.width = width
        self.height = height
        self.episode = 0
        self._board = np.full((height, width), HIDDEN, dtype=np.int8)

    def reset(self) -> np.ndarray:
        self.episode += 1
        # A valid board with at least one hidden cell so the mask is non-empty.
        self._board = np.zeros((self.height, self.width), dtype=np.int8)
        self._board[0, 0] = HIDDEN
        return self._board.copy()

    def apply_action(self, action_type: int, x: int, y: int):
        if self.episode == 1:
            self._board = np.zeros((self.height, self.width), dtype=np.int8)
            info = {
                "newly_revealed": 1,
                "hit_mine": False,
                "game_status": GameStatus.WON,
                "action_type": action_type,
            }
            return self._board.copy(), 100.0, True, False, info
        # Stuck from here on: the board never changes and the game never ends.
        info = {
            "newly_revealed": 0,
            "hit_mine": False,
            "game_status": GameStatus.RUNNING,
            "action_type": action_type,
        }
        return self._board.copy(), 0.0, False, False, info


def test_internally_truncated_episode_is_logged_once_with_unique_id(tmp_path) -> None:
    torch.manual_seed(0)
    tracker = TrackedEnvironment(
        MinesweeperPPOEnv(SometimesStuckEnv(), action_space=FLAGS), phase="train"
    )
    policy = PolicyValueNetwork(
        width=WIDTH, height=HEIGHT, channels=8, action_space=FLAGS
    )
    config = PPOConfig(
        rollout_steps=1, num_epochs=1, minibatch_size=2048, learning_rate=1e-2
    )
    trainer = PPOTrainer(
        policy, tracker, num_actions=3 * WIDTH * HEIGHT, config=config
    )

    report = run_training(
        trainer,
        tracker,
        MetricsLogger(tmp_path),
        games=2,
        checkpoint=tmp_path / LATEST_FILENAME,
        emit=lambda _: None,
    )

    metrics = MetricsLogger.load(tmp_path)
    ids = [episode.episode for episode in metrics.episodes]
    assert ids == [1, 2], "a truncated episode must not reuse the previous id"
    assert len(ids) == len(set(ids))
    assert metrics.episodes[0].outcome == "won"
    assert metrics.episodes[1].outcome == "truncated"
    assert report.final_episode == 2
    assert not report.errors
