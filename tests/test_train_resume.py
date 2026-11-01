"""Tests for resumable training: stop, resume, --fresh and browser cleanup.

No live website is used.  A deterministic coordinate-based fake environment is
wrapped by the real ``MinesweeperPPOEnv`` / ``TrackedEnvironment`` / ``PPOTrainer``
stack, and ``open_environment`` is monkeypatched so no Playwright browser is
launched.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from minesweeper_rl.action_types import ACTION_CHORD, ACTION_FLAG, ACTION_OPEN
from minesweeper_rl.agents import FLAGS, PPOConfig, PPOTrainer, PolicyValueNetwork
from minesweeper_rl.agents.minesweeper_env import MinesweeperPPOEnv
from minesweeper_rl.dom import FLAGGED, HIDDEN, GameStatus
from minesweeper_rl.experiment import MetricsLogger, load_run
from minesweeper_rl.experiment import train as train_module
from minesweeper_rl.experiment.checkpoint import LATEST_FILENAME, read_checkpoint
from minesweeper_rl.experiment.tracking import TrackedEnvironment
from minesweeper_rl.experiment.train import (
    archive_existing_run,
    main,
    run_training,
)

WIDTH, HEIGHT = 4, 2


class FakeBrowserEnv:
    """Coordinate-based, mine-free board that always terminates in a win."""

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
        info = {"newly_revealed": 1, "hit_mine": timed_out, "game_status": status}
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
        info = {"newly_revealed": newly, "hit_mine": timed_out, "game_status": status}
        if terminated:
            self.episodes += 1
            if (
                self._stop_event is not None
                and self._stop_after is not None
                and self.episodes >= self._stop_after
            ):
                self._stop_event.set()
        return self._board.copy(), reward, terminated, False, info


def make_trainer():
    env = MinesweeperPPOEnv(FakeBrowserEnv(), action_space=FLAGS)
    tracker = TrackedEnvironment(env, phase="train")
    policy = PolicyValueNetwork(
        width=WIDTH, height=HEIGHT, channels=16, action_space=FLAGS
    )
    config = PPOConfig(
        rollout_steps=1, num_epochs=1, minibatch_size=8, learning_rate=1e-2
    )
    trainer = PPOTrainer(
        policy, tracker, num_actions=3 * WIDTH * HEIGHT, config=config
    )
    return trainer, tracker


def make_trainer_for(env: FakeBrowserEnv):
    tracker = TrackedEnvironment(
        MinesweeperPPOEnv(env, action_space=FLAGS), phase="train"
    )
    policy = PolicyValueNetwork(
        width=env.width, height=env.height, channels=16, action_space=FLAGS
    )
    config = PPOConfig(
        rollout_steps=1, num_epochs=1, minibatch_size=8, learning_rate=1e-2
    )
    trainer = PPOTrainer(
        policy, tracker, num_actions=3 * env.width * env.height, config=config
    )
    return trainer, tracker


def make_fake_open(env, closed):
    @contextmanager
    def _open(url, profile, *, config=None):
        try:
            yield env
        finally:
            closed.append(True)

    return _open


def seed_run(run_dir, episodes: int = 2):
    trainer, tracker = make_trainer()
    metrics = MetricsLogger(run_dir)
    return run_training(
        trainer,
        tracker,
        metrics,
        games=episodes,
        checkpoint=run_dir / LATEST_FILENAME,
        emit=lambda _: None,
    )


# ---------------------------------------------------------------------------
# Core loop: new run, normal completion, graceful stop, checkpointing.
# ---------------------------------------------------------------------------


def test_new_training_run_creates_checkpoint_and_metrics(tmp_path) -> None:
    report = seed_run(tmp_path, episodes=3)

    assert report.stopped_early is False
    assert report.final_episode == 3
    assert (tmp_path / LATEST_FILENAME).exists()
    episodes, updates = load_run(tmp_path)
    assert [episode.episode for episode in episodes] == [1, 2, 3]
    assert len(updates) == 3


def test_run_training_normal_completion(tmp_path) -> None:
    trainer, tracker = make_trainer()
    report = run_training(
        trainer, tracker, MetricsLogger(tmp_path),
        games=2, checkpoint=tmp_path / LATEST_FILENAME, emit=lambda _: None,
    )

    assert report.stopped_early is False
    assert report.games_completed == 2
    assert report.final_episode == 2


def test_ctrl_c_graceful_stop_writes_checkpoint(tmp_path) -> None:
    stop_event = threading.Event()
    env = FakeBrowserEnv(stop_event=stop_event, stop_after=2)
    trainer, tracker = make_trainer_for(env)
    metrics = MetricsLogger(tmp_path)

    report = run_training(
        trainer, tracker, metrics,
        games=10,
        checkpoint=tmp_path / LATEST_FILENAME,
        stop_event=stop_event,
        emit=lambda _: None,
    )

    assert report.stopped_early is True
    assert report.final_episode == 2
    assert len(metrics.episodes) == 2
    data = read_checkpoint(tmp_path / LATEST_FILENAME)
    assert int(data["episode_count"]) == 2
    assert int(data["total_steps"]) == report.total_steps


def test_run_training_writes_best_checkpoint(tmp_path) -> None:
    trainer, tracker = make_trainer()
    report = run_training(
        trainer, tracker, MetricsLogger(tmp_path),
        games=3, checkpoint=tmp_path / LATEST_FILENAME,
        best_min_episodes=1, emit=lambda _: None,
    )

    assert report.best_checkpoint_path is not None
    assert report.best_checkpoint_path.exists()
    assert report.best_metric == pytest.approx(1.0)


def test_run_training_survives_repeated_failures_and_saves(tmp_path) -> None:
    class FailingEnv(FakeBrowserEnv):
        def reset(self):
            raise RuntimeError("environment unavailable")

    trainer, tracker = make_trainer_for(FailingEnv())
    report = run_training(
        trainer, tracker, MetricsLogger(tmp_path),
        games=5, checkpoint=tmp_path / LATEST_FILENAME,
        max_consecutive_failures=2, emit=lambda _: None,
    )

    assert report.stopped_early is True
    assert len(report.errors) == 2
    assert (tmp_path / LATEST_FILENAME).exists()


# ---------------------------------------------------------------------------
# Resume: counters, metrics continuity and automatic detection.
# ---------------------------------------------------------------------------


def test_resume_restores_counters(tmp_path) -> None:
    seed_run(tmp_path, episodes=2)
    data = read_checkpoint(tmp_path / LATEST_FILENAME)

    trainer, _ = make_trainer()
    trainer.load(tmp_path / LATEST_FILENAME)

    assert trainer.episode_count == int(data["episode_count"]) == 2
    assert trainer.iteration == int(data["iteration"]) == 2
    assert trainer.total_steps == int(data["total_steps"])


def test_resume_continues_episode_ids_without_duplicates(tmp_path) -> None:
    seed_run(tmp_path, episodes=2)

    trainer, tracker = make_trainer()
    trainer.load(tmp_path / LATEST_FILENAME)
    metrics = MetricsLogger(tmp_path)
    metrics.load_existing()

    report = run_training(
        trainer, tracker, metrics,
        games=5, checkpoint=tmp_path / LATEST_FILENAME, emit=lambda _: None,
    )

    assert report.started_at_episode == 2
    assert report.games_completed == 3
    episodes, _ = load_run(tmp_path)
    assert [episode.episode for episode in episodes] == [1, 2, 3, 4, 5]


def test_main_auto_resumes_existing_checkpoint(tmp_path, monkeypatch) -> None:
    seed_run(tmp_path, episodes=2)
    closed: list[bool] = []
    stop_event = threading.Event()
    env = FakeBrowserEnv(stop_event=stop_event, stop_after=1)
    monkeypatch.setattr(
        train_module, "open_environment", make_fake_open(env, closed)
    )

    code = main(
        ["--url", "http://example.test", "--games", "4",
         "--run-dir", str(tmp_path), "--no-plots"],
        stop_event=stop_event,
    )

    assert code == 0
    assert closed == [True]
    episodes, _ = load_run(tmp_path)
    assert [episode.episode for episode in episodes] == [1, 2, 3]
    assert int(read_checkpoint(tmp_path / LATEST_FILENAME)["episode_count"]) == 3


def test_main_new_run_completes_and_closes_browser(tmp_path, monkeypatch) -> None:
    closed: list[bool] = []
    env = FakeBrowserEnv()
    monkeypatch.setattr(
        train_module, "open_environment", make_fake_open(env, closed)
    )

    code = main(
        ["--url", "http://example.test", "--games", "3",
         "--run-dir", str(tmp_path), "--no-plots"]
    )

    assert code == 0
    assert closed == [True]
    episodes, _ = load_run(tmp_path)
    assert [episode.episode for episode in episodes] == [1, 2, 3]
    assert (tmp_path / LATEST_FILENAME).exists()


def test_main_fresh_archives_and_restarts(tmp_path, monkeypatch) -> None:
    seed_run(tmp_path, episodes=2)
    assert (tmp_path / LATEST_FILENAME).exists()

    closed: list[bool] = []
    env = FakeBrowserEnv()
    monkeypatch.setattr(
        train_module, "open_environment", make_fake_open(env, closed)
    )

    code = main(
        ["--url", "http://example.test", "--games", "2",
         "--run-dir", str(tmp_path), "--fresh", "--no-plots"]
    )

    assert code == 0
    assert closed == [True]
    archives = list((tmp_path / "archive").iterdir())
    assert len(archives) == 1
    assert (archives[0] / LATEST_FILENAME).exists()
    episodes, _ = load_run(tmp_path)
    assert [episode.episode for episode in episodes] == [1, 2]
    assert int(read_checkpoint(tmp_path / LATEST_FILENAME)["episode_count"]) == 2


def test_main_closes_browser_on_exception(tmp_path, monkeypatch) -> None:
    closed: list[bool] = []
    env = FakeBrowserEnv()
    monkeypatch.setattr(
        train_module, "open_environment", make_fake_open(env, closed)
    )

    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(train_module, "run_training", boom)

    with pytest.raises(RuntimeError, match="kaboom"):
        main(["--url", "http://example.test", "--games", "1",
              "--run-dir", str(tmp_path), "--no-plots"])
    assert closed == [True]


def test_archive_existing_run_moves_artifacts(tmp_path) -> None:
    (tmp_path / "episodes.jsonl").write_text('{"episode": 1}\n', encoding="utf-8")

    destination = archive_existing_run(tmp_path)

    assert destination is not None
    assert (destination / "episodes.jsonl").exists()
    assert not (tmp_path / "episodes.jsonl").exists()
    assert archive_existing_run(tmp_path) is None
