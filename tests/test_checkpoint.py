"""Tests for robust PPO checkpoint persistence (no browser involved).

Exercises versioning, atomic writes, latest/best management, full state
round-trips (weights, optimizer, counters, RNG) and incompatibility detection
using the deterministic :class:`TargetCellEnv` mock trainer.
"""

from __future__ import annotations

import logging
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from minesweeper_rl.agents import (
    FLAGS,
    OBSERVATION_CHANNELS,
    OBSERVATION_CHANNELS_FLAGS,
    OPEN_ONLY,
    PPOConfig,
    PPOTrainer,
    PolicyValueNetwork,
    TargetCellEnv,
)
from minesweeper_rl.dom import FLAGGED, HIDDEN
from minesweeper_rl.experiment.checkpoint import (
    BEST_FILENAME,
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    LATEST_FILENAME,
    CheckpointError,
    CheckpointManager,
    build_checkpoint,
    load_checkpoint,
    model_info,
    read_checkpoint,
    validate_checkpoint,
    write_checkpoint,
)
from minesweeper_rl.experiment.evaluate import load_frozen_policy


def make_trainer(width: int = 4, *, channels: int = 16) -> PPOTrainer:
    env = TargetCellEnv(width=width)
    policy = PolicyValueNetwork(
        width=width, height=1, channels=channels, action_space=OPEN_ONLY
    )
    config = PPOConfig(
        rollout_steps=16, num_epochs=2, minibatch_size=8, learning_rate=1e-2
    )
    return PPOTrainer(policy, env, num_actions=width, config=config)


def make_flag_trainer(width: int = 4, *, channels: int = 16) -> PPOTrainer:
    """Flag-aware trainer over a tiny board: OPEN + FLAG + CHORD per cell."""

    env = TargetCellEnv(width=width)
    policy = PolicyValueNetwork(
        width=width, height=1, channels=channels, action_space=FLAGS
    )
    config = PPOConfig(
        rollout_steps=16, num_epochs=2, minibatch_size=8, learning_rate=1e-2
    )
    return PPOTrainer(policy, env, num_actions=3 * width, config=config)


def assert_state_dicts_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key, value in left.items():
        assert torch.equal(value, right[key]), key


def test_save_checkpoint_creates_versioned_file(tmp_path) -> None:
    trainer = make_trainer()
    path = write_checkpoint(trainer, tmp_path / LATEST_FILENAME)

    assert path.exists()
    data = read_checkpoint(path)
    assert data["format"] == CHECKPOINT_FORMAT
    assert data["version"] == CHECKPOINT_VERSION
    assert "model" in data and "rng" in data and "metrics" in data


def test_save_via_trainer_uses_checkpoint_layer(tmp_path) -> None:
    trainer = make_trainer()
    path = trainer.save(tmp_path / "checkpoint.pt")
    data = read_checkpoint(path)
    assert data["format"] == CHECKPOINT_FORMAT


def test_load_restores_model_weights(tmp_path) -> None:
    trainer = make_trainer()
    trainer.train(iterations=2)
    path = write_checkpoint(trainer, tmp_path / "checkpoint.pt")

    restored = make_trainer()
    load_checkpoint(restored, read_checkpoint(path))
    assert_state_dicts_equal(
        trainer.policy.state_dict(), restored.policy.state_dict()
    )


def test_load_via_trainer_restores_weights(tmp_path) -> None:
    trainer = make_trainer()
    trainer.train(iterations=1)
    path = trainer.save(tmp_path / "checkpoint.pt")

    restored = make_trainer()
    restored.load(path)
    assert_state_dicts_equal(
        trainer.policy.state_dict(), restored.policy.state_dict()
    )


def test_load_restores_optimizer_state(tmp_path) -> None:
    trainer = make_trainer()
    trainer.train(iterations=2)
    path = write_checkpoint(trainer, tmp_path / "checkpoint.pt")

    restored = make_trainer()
    load_checkpoint(restored, read_checkpoint(path))

    original = trainer.optimizer.state_dict()
    loaded = restored.optimizer.state_dict()
    assert original["state"], "training should have populated Adam state"
    assert original["param_groups"][0]["lr"] == loaded["param_groups"][0]["lr"]
    for parameter_id, state in original["state"].items():
        for name, value in state.items():
            current = loaded["state"][parameter_id][name]
            if torch.is_tensor(value):
                assert torch.equal(value, current)
            else:
                assert value == current


def test_load_restores_update_counter(tmp_path) -> None:
    trainer = make_trainer()
    trainer.iteration = 7
    path = write_checkpoint(trainer, tmp_path / "checkpoint.pt")

    restored = make_trainer()
    load_checkpoint(restored, read_checkpoint(path))
    assert restored.iteration == 7


def test_load_restores_episode_counter(tmp_path) -> None:
    trainer = make_trainer()
    trainer.train(iterations=3)
    assert trainer.episode_count > 0
    trainer.episode_count = 42
    path = write_checkpoint(trainer, tmp_path / "checkpoint.pt")

    restored = make_trainer()
    load_checkpoint(restored, read_checkpoint(path))
    assert restored.episode_count == 42


def test_load_restores_total_steps(tmp_path) -> None:
    trainer = make_trainer()
    trainer.train(iterations=2)
    assert trainer.total_steps > 0
    trainer.total_steps = 1234
    path = write_checkpoint(trainer, tmp_path / "checkpoint.pt")

    restored = make_trainer()
    load_checkpoint(restored, read_checkpoint(path))
    assert restored.total_steps == 1234


def test_load_restores_rng_state(tmp_path) -> None:
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    trainer = make_trainer()

    path = write_checkpoint(trainer, tmp_path / "checkpoint.pt")
    expected = (
        torch.rand(4).tolist(),
        random.random(),
        float(np.random.rand()),
    )

    torch.manual_seed(999)
    random.seed(999)
    np.random.seed(999)
    restored = make_trainer()
    load_checkpoint(restored, read_checkpoint(path))

    actual = (
        torch.rand(4).tolist(),
        random.random(),
        float(np.random.rand()),
    )
    assert actual == expected


def test_rejects_newer_checkpoint_version(tmp_path) -> None:
    trainer = make_trainer()
    data = build_checkpoint(trainer)
    data["version"] = CHECKPOINT_VERSION + 1

    with pytest.raises(CheckpointError, match="version"):
        load_checkpoint(make_trainer(), data)


def test_rejects_unknown_checkpoint_format(tmp_path) -> None:
    trainer = make_trainer()
    data = build_checkpoint(trainer)
    data["format"] = "some_other_project"

    with pytest.raises(CheckpointError, match="format"):
        load_checkpoint(make_trainer(), data)


def test_read_rejects_non_checkpoint_file(tmp_path) -> None:
    path = tmp_path / "not_a_checkpoint.pt"
    torch.save({"hello": "world"}, path)

    with pytest.raises(CheckpointError, match="not a PPO checkpoint"):
        read_checkpoint(path)


def test_rejects_action_space_mismatch(tmp_path) -> None:
    trainer = make_trainer(width=4)
    data = build_checkpoint(trainer)
    other = make_trainer(width=3)

    with pytest.raises(CheckpointError, match="num_actions"):
        load_checkpoint(other, data)


def test_rejects_model_shape_mismatch(tmp_path) -> None:
    trainer = make_trainer(width=4)
    data = build_checkpoint(trainer)

    policy = PolicyValueNetwork(
        width=2, height=2, channels=16, action_space=OPEN_ONLY
    )
    other = PPOTrainer(
        policy,
        TargetCellEnv(width=4),
        num_actions=4,
        config=PPOConfig(rollout_steps=8, num_epochs=1, minibatch_size=8),
    )

    with pytest.raises(CheckpointError, match="width"):
        load_checkpoint(other, data)


def test_rejects_channel_mismatch(tmp_path) -> None:
    trainer = make_trainer(width=4, channels=16)
    data = build_checkpoint(trainer)
    other = make_trainer(width=4, channels=8)

    with pytest.raises(CheckpointError, match="channels"):
        load_checkpoint(other, data)


def test_checkpoint_records_observation_channels(tmp_path) -> None:
    data = build_checkpoint(make_trainer())
    assert data["model"]["observation_channels"] == OBSERVATION_CHANNELS == 10
    assert data["action_space"] == OPEN_ONLY
    assert CHECKPOINT_VERSION == 5


def test_rejects_observation_channel_metadata_mismatch(tmp_path) -> None:
    trainer = make_trainer(width=4)
    data = build_checkpoint(trainer)
    # A flag-aware (11-channel) model against the OPEN-only (10-channel) model.
    data["model"]["observation_channels"] = 11

    with pytest.raises(CheckpointError, match="observation channel"):
        load_checkpoint(make_trainer(width=4), data)
    # The in-memory checkpoint payload is not the file on disk, but the original
    # trainer's state must be untouched by the failed load.
    assert trainer.policy.encoder[0].in_channels == OBSERVATION_CHANNELS


def test_rejects_legacy_flag_aware_weights_without_metadata(tmp_path) -> None:
    trainer = make_trainer(width=4)
    data = build_checkpoint(trainer)
    # Simulate an old flag-aware checkpoint: no observation_channels metadata
    # and an 11-channel first convolution weight.
    data["model"].pop("observation_channels", None)
    weight = data["policy"]["encoder.0.weight"]
    expanded = weight.new_zeros((weight.shape[0], 11, *weight.shape[2:]))
    expanded[:, : weight.shape[1]] = weight
    data["policy"]["encoder.0.weight"] = expanded

    with pytest.raises(CheckpointError, match="observation channel"):
        load_checkpoint(make_trainer(width=4), data)


def test_checkpoint_round_trip_with_open_board(tmp_path) -> None:
    trainer = make_trainer(width=4)
    board = np.full((1, 4), HIDDEN, dtype=np.int8)
    board[0, 0] = 0
    board[0, 1] = 3
    logits, _ = trainer.policy(board)
    trainer.iteration = 3

    path = write_checkpoint(trainer, tmp_path / "c.pt")
    restored = make_trainer(width=4)
    load_checkpoint(restored, read_checkpoint(path))

    assert restored.iteration == 3
    restored_logits, _ = restored.policy(board)
    assert torch.equal(logits, restored_logits)


def test_frozen_policy_loads_compatible_checkpoint(tmp_path) -> None:
    trainer = make_trainer(width=4)
    trainer.train(iterations=1)
    path = write_checkpoint(trainer, tmp_path / "ok.pt")

    policy = PolicyValueNetwork(
        width=4, height=1, channels=16, action_space=OPEN_ONLY
    )
    load_frozen_policy(path, policy)

    assert_state_dicts_equal(trainer.policy.state_dict(), policy.state_dict())


def test_frozen_policy_rejects_incompatible_checkpoint(tmp_path) -> None:
    trainer = make_trainer(width=4)
    data = build_checkpoint(trainer)
    data["model"].pop("observation_channels", None)
    weight = data["policy"]["encoder.0.weight"]
    expanded = weight.new_zeros((weight.shape[0], 11, *weight.shape[2:]))
    expanded[:, : weight.shape[1]] = weight
    data["policy"]["encoder.0.weight"] = expanded
    path = tmp_path / "old.pt"
    torch.save(data, path)

    policy = PolicyValueNetwork(
        width=4, height=1, channels=16, action_space=OPEN_ONLY
    )
    with pytest.raises(CheckpointError, match="observation channel"):
        load_frozen_policy(path, policy)


def test_default_model_action_space_is_1440() -> None:
    policy = PolicyValueNetwork()  # default 30x16 flag-aware Expert
    info = model_info(SimpleNamespace(policy=policy, num_actions=policy.num_actions))
    assert info["observation_channels"] == OBSERVATION_CHANNELS_FLAGS == 11
    assert info["num_actions"] == 1440
    assert info["action_space"] == FLAGS


def test_open_only_model_action_space_is_480() -> None:
    policy = PolicyValueNetwork(action_space=OPEN_ONLY)
    info = model_info(SimpleNamespace(policy=policy, num_actions=policy.num_actions))
    assert info["observation_channels"] == OBSERVATION_CHANNELS == 10
    assert info["num_actions"] == 480
    assert info["action_space"] == OPEN_ONLY


def test_flag_aware_checkpoint_round_trip(tmp_path) -> None:
    trainer = make_flag_trainer(width=4)
    board = np.full((1, 4), FLAGGED, dtype=np.int8)
    board[0, 0] = HIDDEN
    board[0, 1] = 3
    logits, _ = trainer.policy(board)
    trainer.iteration = 2

    path = write_checkpoint(trainer, tmp_path / "flags.pt")
    data = read_checkpoint(path)
    assert data["action_space"] == FLAGS
    assert data["num_actions"] == 12
    assert data["model"]["observation_channels"] == OBSERVATION_CHANNELS_FLAGS

    restored = make_flag_trainer(width=4)
    load_checkpoint(restored, data)
    assert restored.iteration == 2
    restored_logits, _ = restored.policy(board)
    assert torch.equal(logits, restored_logits)


def test_open_only_checkpoint_rejected_by_flag_aware_model(tmp_path) -> None:
    data = build_checkpoint(make_trainer(width=4))  # OPEN-only, 10 channels
    data["version"] = 4  # the completed baseline version

    with pytest.raises(CheckpointError, match="OPEN-only"):
        load_checkpoint(make_flag_trainer(width=4), data)


def test_flag_aware_checkpoint_rejected_by_open_only_model(tmp_path) -> None:
    data = build_checkpoint(make_flag_trainer(width=4))  # FLAGS, 11 channels

    with pytest.raises(CheckpointError):
        load_checkpoint(make_trainer(width=4), data)


def test_v4_open_only_checkpoint_still_loads_into_open_only_model(tmp_path) -> None:
    trainer = make_trainer(width=4)
    data = build_checkpoint(trainer)
    data["version"] = 4  # simulate the completed OPEN-only baseline checkpoint

    restored = make_trainer(width=4)
    load_checkpoint(restored, data)
    assert_state_dicts_equal(trainer.policy.state_dict(), restored.policy.state_dict())


def test_rejects_flag_aware_960_checkpoint(tmp_path) -> None:
    trainer = make_trainer(width=4)  # OPEN-only, num_actions == 4
    data = build_checkpoint(trainer)
    # Forge a flag-aware checkpoint payload: 960-style action count and
    # 11 observation channels.
    data["num_actions"] = 960
    data["model"]["num_actions"] = 960
    data["model"]["observation_channels"] = 11

    with pytest.raises(CheckpointError):
        validate_checkpoint(
            data,
            SimpleNamespace(
                policy=trainer.policy, num_actions=trainer.policy.num_actions
            ),
        )


def test_frozen_policy_rejects_flag_aware_checkpoint(tmp_path) -> None:
    trainer = make_trainer(width=4)  # OPEN-only, 4 actions
    data = build_checkpoint(trainer)
    data["num_actions"] = 960
    data["model"]["num_actions"] = 960
    data["model"]["observation_channels"] = 11
    path = tmp_path / "flag_aware.pt"
    torch.save(data, path)

    policy = PolicyValueNetwork(
        width=4, height=1, channels=16, action_space=OPEN_ONLY
    )
    with pytest.raises(CheckpointError):
        load_frozen_policy(path, policy)


def test_legacy_checkpoint_is_rejected(tmp_path) -> None:
    # The pre-v3 ``minesweeper_ppo_v1`` files (e.g. ppo_experiment.pt) have no
    # version and predate the OPEN-only format; they must not be loaded silently.
    trainer = make_trainer()
    trainer.iteration = 5
    data = {
        "format": "minesweeper_ppo_v1",
        "policy": trainer.policy.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "config": {"gamma": 0.99},
        "num_actions": 4,
        "iteration": 5,
        "total_steps": 80,
    }
    path = tmp_path / "legacy.pt"
    torch.save(data, path)

    with pytest.raises(CheckpointError, match="format"):
        load_checkpoint(make_trainer(), read_checkpoint(path))


def test_best_checkpoint_only_updates_on_improvement(tmp_path) -> None:
    trainer = make_trainer()
    manager = CheckpointManager(tmp_path)

    assert manager.save_best(trainer, metric=0.2) is not None
    assert manager.best_metric == pytest.approx(0.2)

    assert manager.save_best(trainer, metric=0.1) is None
    assert manager.best_metric == pytest.approx(0.2)

    assert manager.save_best(trainer, metric=0.3) is not None
    assert manager.best_metric == pytest.approx(0.3)

    recovered = CheckpointManager(tmp_path)
    assert recovered.best_metric == pytest.approx(0.3)


def test_best_checkpoint_can_be_loaded(tmp_path) -> None:
    trainer = make_trainer()
    trainer.iteration = 8
    manager = CheckpointManager(tmp_path)
    manager.save_best(trainer, metric=0.5)

    restored = make_trainer()
    manager.load_best(restored)
    assert restored.iteration == 8
    assert_state_dicts_equal(
        trainer.policy.state_dict(), restored.policy.state_dict()
    )


def test_manager_save_writes_latest_always_and_best_once(tmp_path) -> None:
    trainer = make_trainer()
    manager = CheckpointManager(tmp_path)

    latest, best = manager.save(trainer, metric=0.4)
    assert latest == tmp_path / LATEST_FILENAME
    assert best == tmp_path / BEST_FILENAME

    latest, best = manager.save(trainer, metric=0.4)
    assert latest.exists()
    assert best is None
    assert (tmp_path / BEST_FILENAME).exists()


def test_manager_loads_latest_checkpoint(tmp_path) -> None:
    trainer = make_trainer()
    trainer.train(iterations=1)
    manager = CheckpointManager(tmp_path)
    manager.save_latest(trainer, metrics={"win_rate": 0.5})

    restored = make_trainer()
    manager.load_latest(restored)
    assert restored.iteration == trainer.iteration
    assert_state_dicts_equal(
        trainer.policy.state_dict(), restored.policy.state_dict()
    )


def test_atomic_write_keeps_previous_checkpoint_on_failure(tmp_path, monkeypatch) -> None:
    trainer = make_trainer()
    path = tmp_path / LATEST_FILENAME
    write_checkpoint(trainer, path)
    original = path.read_bytes()

    def exploding_save(*_args, **_kwargs):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(torch, "save", exploding_save)
    with pytest.raises(RuntimeError, match="simulated disk failure"):
        write_checkpoint(trainer, path)

    monkeypatch.undo()
    assert path.read_bytes() == original
    assert read_checkpoint(path)["format"] == CHECKPOINT_FORMAT
    assert list(tmp_path.glob("*.tmp")) == []


def test_checkpoint_records_metadata(tmp_path) -> None:
    trainer = make_trainer()
    path = write_checkpoint(
        trainer, tmp_path / "c.pt", metadata={"algorithm": "ppo", "seed": 7}
    )

    data = read_checkpoint(path)
    assert data["metadata"] == {"algorithm": "ppo", "seed": 7}


def test_manager_persists_metadata_into_checkpoints(tmp_path) -> None:
    trainer = make_trainer()
    manager = CheckpointManager(tmp_path, metadata={"algorithm": "ppo", "seed": 3})

    manager.save_latest(trainer, metrics={"episode": 1})

    assert read_checkpoint(tmp_path / LATEST_FILENAME)["metadata"]["seed"] == 3


def test_load_warns_when_hyperparameters_differ(tmp_path, caplog) -> None:
    trainer = make_trainer()
    data = build_checkpoint(trainer)

    other = make_trainer()
    other.config = PPOConfig(
        rollout_steps=16, num_epochs=2, minibatch_size=8, learning_rate=9e-3
    )

    with caplog.at_level(logging.WARNING, logger="minesweeper_rl.experiment.checkpoint"):
        load_checkpoint(other, data)

    assert any("hyperparameters differ" in record.message for record in caplog.records)
    # The checkpoint values win, keeping the experiment consistent.
    assert other.config.learning_rate == pytest.approx(data["config"]["learning_rate"])
