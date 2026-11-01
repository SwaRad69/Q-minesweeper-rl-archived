"""Tests for the experiment infrastructure (no browser involved)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from minesweeper_rl.agents import (
    OPEN_ONLY,
    PolicyValueNetwork,
    PPOConfig,
    PPOTrainer,
)
from minesweeper_rl.dom import GameStatus
from minesweeper_rl.experiment import (
    EpisodeMetrics,
    MetricsLogger,
    TrackedEnvironment,
    UpdateMetrics,
    greedy_action,
    plot_evaluation,
    plot_training,
    run_evaluation,
    run_training,
    summarize,
)


class FakeBoardEnv:
    """Deterministic, mine-free board env with flat actions and a reveal mask."""

    def __init__(self, width: int = 4, height: int = 2) -> None:
        self.config = SimpleNamespace(width=width, height=height)
        self.width = width
        self.height = height
        self.num_actions = width * height
        self._board = np.full((height, width), -1, dtype=np.int8)

    def reset(self) -> np.ndarray:
        self._board = np.full((self.height, self.width), -1, dtype=np.int8)
        self._board[0, 0] = 0
        return self._board.copy()

    def step(self, action: int):
        x, y = action % self.width, action // self.width
        if self._board[y, x] != -1:
            raise ValueError(f"cell ({x}, {y}) already revealed")
        self._board = self._board.copy()
        self._board[y, x] = np.int8(0)
        if (self._board == -1).sum() == 0:
            status, reward, terminated = GameStatus.WON, 100.0, True
        else:
            status, reward, terminated = GameStatus.RUNNING, 1.0, False
        info = {"newly_revealed": 1, "hit_mine": False, "game_status": status}
        return self._board.copy(), reward, terminated, False, info

    def action_mask(self, observation=None) -> np.ndarray:
        board = self._board if observation is None else np.asarray(observation)
        return (board.reshape(-1) == -1)


def make_trainer(config: PPOConfig | None = None):
    tracker = TrackedEnvironment(FakeBoardEnv(), phase="train")
    policy = PolicyValueNetwork(
        width=4, height=2, channels=8, action_space=OPEN_ONLY
    )
    trainer = PPOTrainer(
        policy,
        tracker,
        num_actions=8,
        config=config
        or PPOConfig(rollout_steps=1, num_epochs=1, minibatch_size=8, learning_rate=1e-2),
    )
    return trainer, tracker


def test_summarize_computes_expected_metrics() -> None:
    episodes = [
        EpisodeMetrics(1, "train", "won", True, 100.0, 10, False, 25, 2.0, 0),
        EpisodeMetrics(2, "train", "lost", False, -10.0, 3, True, 2, 1.0, 1),
    ]
    summary = summarize(episodes, environment_failures=2)
    assert summary.games == 2
    assert summary.wins == 1
    assert summary.losses == 1
    assert summary.win_rate == pytest.approx(0.5)
    assert summary.mean_reward == pytest.approx(45.0)
    assert summary.mean_steps == pytest.approx(6.5)
    assert summary.mine_hit_rate == pytest.approx(0.5)
    assert summary.mean_cells_revealed == pytest.approx(13.5)
    assert summary.invalid_actions == 1
    assert summary.environment_failures == 2


def test_metrics_logger_round_trip(tmp_path) -> None:
    logger = MetricsLogger(tmp_path)
    logger.log_episode(
        EpisodeMetrics(1, "train", "won", True, 100.0, 10, False, 25, 2.0, 0)
    )
    logger.log_update(
        UpdateMetrics(1, "train", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 4)
    )

    restored = MetricsLogger.load(tmp_path)
    assert restored.episodes == logger.episodes
    assert restored.updates == logger.updates
    assert restored.episodes[0].won is True


def test_tracked_environment_records_completed_episode() -> None:
    tracker = TrackedEnvironment(FakeBoardEnv())
    tracker.reset()
    terminated = False
    while not terminated:
        mask = tracker.action_mask()
        action = int(np.argmax(mask))
        _, _, terminated, _, _ = tracker.step(action)

    episode = tracker.last_episode
    assert episode is not None
    assert episode.won is True
    assert episode.outcome == "won"
    assert episode.steps == 7
    assert episode.cells_revealed == 7
    assert episode.invalid_actions == 0
    assert tracker.failures == 0
    assert tracker.episodes == [episode]


def test_tracked_environment_counts_invalid_actions_and_failures() -> None:
    tracker = TrackedEnvironment(FakeBoardEnv())
    tracker.reset()
    tracker.step(1)  # reveal (1, 0)
    with pytest.raises(ValueError):
        tracker.step(1)  # already revealed -> invalid + environment failure
    assert tracker.invalid_actions == 1
    assert tracker.failures == 1


def test_run_training_logs_games_and_saves_checkpoint(tmp_path) -> None:
    trainer, tracker = make_trainer()
    metrics = MetricsLogger(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"

    report = run_training(
        trainer,
        tracker,
        metrics,
        games=3,
        checkpoint=checkpoint,
        checkpoint_interval=2,
        emit=lambda _: None,
    )

    assert report.games_completed == 3
    assert not report.stopped_early
    assert checkpoint.exists()
    assert len(metrics.episodes) == 3
    assert report.summary.games == 3
    assert report.summary.win_rate == pytest.approx(1.0)


def test_run_training_resumes_from_checkpoint(tmp_path) -> None:
    trainer, tracker = make_trainer()
    metrics = MetricsLogger(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"

    run_training(trainer, tracker, metrics, games=2, checkpoint=checkpoint,
                 emit=lambda _: None)

    resumed, resumed_tracker = make_trainer()
    resumed.load(checkpoint)
    resumed_metrics = MetricsLogger(tmp_path)
    resumed_metrics.load_existing()

    report = run_training(
        resumed, resumed_tracker, resumed_metrics, games=5,
        checkpoint=checkpoint, emit=lambda _: None,
    )

    assert report.started_at_episode == 2
    assert report.games_completed == 3
    assert len(resumed_metrics.episodes) == 5
    assert [e.episode for e in resumed_metrics.episodes] == [1, 2, 3, 4, 5]


def test_run_training_honours_stop_event(tmp_path) -> None:
    trainer, tracker = make_trainer()
    stop_event = threading.Event()
    stop_event.set()

    report = run_training(
        trainer,
        tracker,
        MetricsLogger(tmp_path),
        games=5,
        checkpoint=tmp_path / "checkpoint.pt",
        stop_event=stop_event,
        emit=lambda _: None,
    )
    assert report.stopped_early is True
    assert report.games_completed == 0
    assert (tmp_path / "checkpoint.pt").exists()


def test_run_evaluation_is_greedy_and_frozen(tmp_path) -> None:
    policy = PolicyValueNetwork(
        width=4, height=2, channels=8, action_space=OPEN_ONLY
    )
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    tracker = TrackedEnvironment(FakeBoardEnv(), phase="eval")
    metrics = MetricsLogger(tmp_path)

    report = run_evaluation(
        policy, tracker, metrics, games=3, emit=lambda _: None
    )

    assert report.games == 3
    assert report.weights_unchanged is True
    assert report.summary.win_rate == pytest.approx(1.0)
    assert report.summary.invalid_actions == 0
    for key, value in policy.state_dict().items():
        assert torch.equal(value, before[key])


def test_greedy_action_is_deterministic_and_masked() -> None:
    policy = PolicyValueNetwork(
        width=4, height=2, channels=8, action_space=OPEN_ONLY
    )
    observation = FakeBoardEnv().reset()
    mask = np.array([False, True, True, True, True, True, True, True])

    first = greedy_action(policy, observation, mask)
    second = greedy_action(policy, observation, mask)
    assert first[0] == second[0]
    assert first[0] != 0  # masked-out action


def test_plots_are_written(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    logger = MetricsLogger(tmp_path)
    for index in range(1, 6):
        logger.log_episode(
            EpisodeMetrics(index, "train", "lost", False, -float(index),
                           index, True, index, 1.0, 0)
        )
        logger.log_update(
            UpdateMetrics(index, "train", 0.1 * index, 0.2, 0.3, 0.01, 0.0, 0.4, 4)
        )

    training_plot = plot_training(tmp_path)
    evaluation_plot = plot_evaluation(tmp_path)
    assert training_plot is not None and training_plot.exists()
    assert evaluation_plot is not None and evaluation_plot.exists()
