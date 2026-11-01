"""Unit tests for the action space, observation encoding and CNN policy.

Covers both action spaces: the flag-aware default (OPEN + FLAG + CHORD, 1440
actions, 11 observation channels) and the OPEN-only baseline (480 actions, 10
channels) that remains selectable for the completed experiment.  No PPO
optimization or browser connection is exercised.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from minesweeper_rl.agents import (
    ACTION_CHORD,
    ACTION_FLAG,
    ACTION_OPEN,
    ACTION_TYPES,
    CELLS,
    DEFAULT_ACTION_SPACE,
    FLAGS,
    HEIGHT,
    NUM_ACTIONS_FLAGS,
    NUM_ACTIONS_OPEN_ONLY,
    NUM_VALUES,
    OBSERVATION_CHANNELS,
    OBSERVATION_CHANNELS_FLAGS,
    OBSERVATION_VALUES,
    OBSERVATION_VALUES_FLAGS,
    OPEN_ONLY,
    VALUE_TO_CHANNEL,
    VALUE_TO_CHANNEL_FLAGS,
    WIDTH,
    PolicyValueNetwork,
    action_count,
    action_mask,
    action_to_coordinate,
    action_type_mask,
    apply_action_mask,
    board_to_tensor,
    cell_of,
    chord_mask,
    coordinate_to_action,
    decode_action,
    encode_action,
    flag_mask,
    join_action,
    observation_channels,
    observation_values,
    open_mask,
    split_action,
)
from minesweeper_rl.dom import FLAGGED, HIDDEN, MAX_NUMBER


def board_fixture(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(HIDDEN, MAX_NUMBER + 1, size=(HEIGHT, WIDTH)).astype(np.int8)


def flag_board_fixture(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(FLAGGED, MAX_NUMBER + 1, size=(HEIGHT, WIDTH)).astype(np.int8)


# ---------------------------------------------------------------------------
# Counts and constants
# ---------------------------------------------------------------------------


def test_default_board_and_action_counts() -> None:
    assert (WIDTH, HEIGHT) == (30, 16)
    assert CELLS == 480
    assert DEFAULT_ACTION_SPACE == FLAGS
    assert NUM_ACTIONS_OPEN_ONLY == 480
    assert NUM_ACTIONS_FLAGS == 1440
    assert ACTION_TYPES == (ACTION_OPEN, ACTION_FLAG, ACTION_CHORD)
    assert action_count(FLAGS) == 1440
    assert action_count(OPEN_ONLY) == 480


def test_observation_channel_constants() -> None:
    assert OBSERVATION_CHANNELS == 10
    assert NUM_VALUES == OBSERVATION_CHANNELS
    assert OBSERVATION_VALUES == (HIDDEN, 0, 1, 2, 3, 4, 5, 6, 7, 8)
    assert VALUE_TO_CHANNEL[HIDDEN] == 0 and VALUE_TO_CHANNEL[8] == 9
    assert OBSERVATION_CHANNELS_FLAGS == 11
    assert OBSERVATION_VALUES_FLAGS == (FLAGGED, HIDDEN, 0, 1, 2, 3, 4, 5, 6, 7, 8)
    assert VALUE_TO_CHANNEL_FLAGS[FLAGGED] == 0
    assert VALUE_TO_CHANNEL_FLAGS[HIDDEN] == 1
    assert VALUE_TO_CHANNEL_FLAGS[8] == 10
    assert observation_channels(FLAGS) == 11
    assert observation_values(FLAGS) == OBSERVATION_VALUES_FLAGS


# ---------------------------------------------------------------------------
# Cell indexing (OPEN-only compatibility)
# ---------------------------------------------------------------------------


def test_coordinate_action_round_trip_is_deterministic() -> None:
    assert coordinate_to_action(0, 0) == 0
    assert coordinate_to_action(29, 0) == 29
    assert coordinate_to_action(0, 1) == 30
    assert coordinate_to_action(29, 15) == 479

    seen: set[int] = set()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            action = coordinate_to_action(x, y)
            assert action not in seen
            seen.add(action)
            assert action_to_coordinate(action) == (x, y)
    assert seen == set(range(CELLS))


# ---------------------------------------------------------------------------
# Compound encoding: action_id = action_type * cells + cell
# ---------------------------------------------------------------------------


def test_compound_encoding_is_type_major() -> None:
    assert encode_action(0, action_type=ACTION_OPEN) == 0
    assert encode_action(479, action_type=ACTION_OPEN) == 479
    assert encode_action(0, action_type=ACTION_FLAG) == 480
    assert encode_action(479, action_type=ACTION_FLAG) == 959
    assert encode_action(0, action_type=ACTION_CHORD) == 960
    assert encode_action(479, action_type=ACTION_CHORD) == 1439


def test_split_action_round_trip_for_every_cell_and_type() -> None:
    for action_type in ACTION_TYPES:
        for cell in range(CELLS):
            action = encode_action(cell, action_type=action_type)
            assert split_action(action) == (action_type, cell)
            assert cell_of(action) == cell
            assert join_action(action_type, cell) == action


def test_split_action_rejects_out_of_range() -> None:
    with pytest.raises(IndexError):
        split_action(NUM_ACTIONS_FLAGS)
    with pytest.raises(IndexError):
        split_action(-1)
    with pytest.raises(TypeError):
        split_action(1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        split_action(True)


def test_encode_action_rejects_bad_action_type() -> None:
    with pytest.raises(ValueError):
        encode_action(0, action_type=3)
    for bad in (-1, NUM_ACTIONS_OPEN_ONLY, 1000):
        with pytest.raises(IndexError):
            decode_action(bad)
    with pytest.raises(TypeError):
        decode_action(1.0)  # type: ignore[arg-type]


def test_decode_action_selects_type_block() -> None:
    assert decode_action(480, action_type=ACTION_FLAG) == 0
    assert decode_action(1439, action_type=ACTION_CHORD) == 479
    with pytest.raises(IndexError):
        decode_action(0, action_type=ACTION_FLAG)


# ---------------------------------------------------------------------------
# Legality masks
# ---------------------------------------------------------------------------


def test_open_mask_semantics() -> None:
    board = np.full((HEIGHT, WIDTH), HIDDEN, dtype=np.int8)
    board[0, 0] = 1
    board[0, 2] = FLAGGED

    mask = open_mask(board)
    assert mask.shape == (CELLS,)
    assert mask[1]
    assert not mask[0]
    assert not mask[2]


def test_flag_mask_allows_hidden_and_flagged_only() -> None:
    board = np.full((2, 3), HIDDEN, dtype=np.int8)
    board[0, 0] = 1
    board[0, 1] = FLAGGED

    mask = flag_mask(board, width=3, height=2)
    assert not mask[0], "an opened cell cannot be flagged"
    assert mask[1], "an already flagged cell can be unflagged"
    assert mask[2], "a hidden cell can be flagged"


def test_chord_mask_requires_matching_flags_and_hidden_neighbour() -> None:
    board = np.array(
        [[1, FLAGGED, HIDDEN], [HIDDEN, HIDDEN, HIDDEN]], dtype=np.int8
    )
    mask = chord_mask(board, width=3, height=2)
    assert mask[0], "1 with one flagged neighbour and a hidden neighbour chords"
    assert not mask[1], "a flagged cell is not a chord target"
    assert not mask[2], "a hidden cell is not a chord target"


def test_flag_action_mask_concatenates_blocks() -> None:
    board = np.array(
        [[1, FLAGGED, HIDDEN], [HIDDEN, HIDDEN, HIDDEN]], dtype=np.int8
    )
    cells = 6
    mask = action_mask(board, action_space=FLAGS, width=3, height=2)

    assert mask.shape == (3 * cells,)
    assert list(map(bool, mask)) == list(map(bool, np.concatenate([
        action_type_mask(board, ACTION_OPEN, width=3, height=2),
        action_type_mask(board, ACTION_FLAG, width=3, height=2),
        action_type_mask(board, ACTION_CHORD, width=3, height=2),
    ])))


def test_open_only_action_mask_shape() -> None:
    board = np.full((HEIGHT, WIDTH), HIDDEN, dtype=np.int8)
    mask = action_mask(board, action_space=OPEN_ONLY)
    assert mask.shape == (CELLS,)

    batched = action_mask(board, action_space=OPEN_ONLY)
    assert batched.shape == (CELLS,)


def test_action_mask_batched_shapes_and_validation() -> None:
    board = np.full((HEIGHT, WIDTH), HIDDEN, dtype=np.int8)
    batched = action_mask(np.stack([board, board]))
    assert batched.shape == (2, NUM_ACTIONS_FLAGS)

    open_batched = action_mask(np.stack([board, board]), action_space=OPEN_ONLY)
    assert open_batched.shape == (2, NUM_ACTIONS_OPEN_ONLY)

    with pytest.raises(ValueError, match="trailing shape"):
        action_mask(np.zeros((2, 3), dtype=np.int8))


# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------


def test_board_to_tensor_flag_channel_mapping() -> None:
    values = [FLAGGED, HIDDEN, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    board = np.array([values], dtype=np.int8)

    tensor = board_to_tensor(board, action_space=FLAGS, width=len(values), height=1)

    assert tensor.shape == (OBSERVATION_CHANNELS_FLAGS, 1, len(values))
    for channel, value in enumerate(values):
        assert VALUE_TO_CHANNEL_FLAGS[value] == channel
        assert tensor[channel, 0, channel].item() == 1.0
        assert tensor[:, 0, channel].sum().item() == 1.0


def test_board_to_tensor_open_only_rejects_flag_state() -> None:
    board = np.full((1, 2), HIDDEN, dtype=np.int8)
    board[0, 0] = FLAGGED
    with pytest.raises(ValueError, match="values"):
        board_to_tensor(board, action_space=OPEN_ONLY, width=2, height=1)


def test_board_to_tensor_one_hot_invariant_for_both_modes() -> None:
    for action_space, fixture in ((FLAGS, flag_board_fixture), (OPEN_ONLY, board_fixture)):
        board = fixture(seed=7)
        tensor = board_to_tensor(board, action_space=action_space)
        assert tensor.shape == (observation_channels(action_space), HEIGHT, WIDTH)
        assert torch.all(tensor.sum(dim=0) == 1.0)
        assert torch.all((tensor == 0.0) | (tensor == 1.0))


def test_board_to_tensor_rejects_out_of_range_values() -> None:
    for bad in (-3, 9):
        with pytest.raises(ValueError, match="values"):
            board_to_tensor(np.full((HEIGHT, WIDTH), bad, dtype=np.int8))
    with pytest.raises(ValueError, match="trailing shape"):
        board_to_tensor(np.zeros((2, 3), dtype=np.int8))
    with pytest.raises(ValueError, match="2D or 3D"):
        board_to_tensor(np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.int8))


def test_board_to_tensor_accepts_torch() -> None:
    board = flag_board_fixture()
    assert torch.equal(
        board_to_tensor(torch.as_tensor(board)), board_to_tensor(board)
    )


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------


def test_default_network_is_flag_aware_with_1440_logits() -> None:
    network = PolicyValueNetwork()
    assert network.action_space == FLAGS
    assert network.num_actions == NUM_ACTIONS_FLAGS == 1440
    assert network.observation_channels == OBSERVATION_CHANNELS_FLAGS == 11
    assert network.encoder[0].in_channels == 11
    assert network.actor.out_channels == 3
    logits, value = network(flag_board_fixture())
    assert logits.shape == (1440,)
    assert value.shape == (1,)


def test_open_only_network_is_single_headed() -> None:
    network = PolicyValueNetwork(
        width=4, height=2, action_space=OPEN_ONLY
    )
    assert network.num_actions == 8
    assert network.actor.out_channels == 1
    logits, _ = network(np.zeros((2, 4), dtype=np.int8))
    assert logits.shape == (8,)


def test_actor_and_critic_output_shapes_batched() -> None:
    network = PolicyValueNetwork(action_space=OPEN_ONLY)
    logits, value = network(board_fixture())
    assert logits.shape == (NUM_ACTIONS_OPEN_ONLY,)
    assert value.shape == (1,)

    batched_logits, batched_value = network(np.stack([board_fixture()] * 4))
    assert batched_logits.shape == (4, NUM_ACTIONS_OPEN_ONLY)
    assert batched_value.shape == (4, 1)


def test_flag_network_actor_flatten_matches_action_encoding() -> None:
    # The actor is a 3-plane 1x1 convolution over the same spatial order as the
    # cell index, so flattening yields action_type * cells + cell directly.
    network = PolicyValueNetwork(width=3, height=2)
    assert network.num_actions == 18
    assert network.actor.out_channels == 3

    with torch.no_grad():
        network.actor.weight.zero_()
        network.actor.bias.copy_(torch.tensor([10.0, 20.0, 30.0]))
    logits, _ = network(np.zeros((2, 3), dtype=np.int8))

    for action_type, expected in (
        (ACTION_OPEN, 10.0),
        (ACTION_FLAG, 20.0),
        (ACTION_CHORD, 30.0),
    ):
        for cell in range(6):
            action = encode_action(
                cell, action_type=action_type, width=3, height=2
            )
            assert logits[action].item() == expected, (
                action_type,
                cell,
                action,
            )


def test_forward_applies_action_mask_flags() -> None:
    board = flag_board_fixture(seed=2)
    mask = action_mask(board)
    logits, _ = PolicyValueNetwork()(board, action_mask=mask)

    invalid = ~torch.as_tensor(mask)
    assert torch.isneginf(logits[invalid]).all()
    assert torch.isfinite(logits[~invalid]).all()


def test_apply_action_mask_removes_invalid_actions() -> None:
    logits = torch.zeros(5)
    mask = torch.tensor([True, False, True, False, True])

    masked = apply_action_mask(logits, mask)
    assert masked[0].item() == 0.0
    assert torch.isneginf(masked[1])
    assert torch.isneginf(masked[3])
    probabilities = torch.softmax(masked, dim=0)
    assert probabilities.sum().item() == pytest.approx(1.0)


def test_forward_backward_produces_gradients() -> None:
    network = PolicyValueNetwork()
    logits, value = network(flag_board_fixture(seed=3))
    (logits.sum() + value.sum()).backward()

    assert network.encoder[0].weight.grad is not None
    assert network.encoder[2].weight.grad is not None
    assert network.actor.weight.grad is not None
    assert network.actor.bias.grad is not None
    assert network.critic[2].weight.grad is not None


def test_masked_batch_forward_backward() -> None:
    network = PolicyValueNetwork()
    boards = np.stack([flag_board_fixture(seed=5)] * 3)
    mask = action_mask(boards)

    logits, value = network(boards, action_mask=mask)
    assert logits.shape == (3, NUM_ACTIONS_FLAGS)
    (logits.sum() + value.sum()).backward()
    assert network.actor.weight.grad is not None


def test_network_stays_small() -> None:
    parameters = sum(p.numel() for p in PolicyValueNetwork().parameters())
    assert parameters < 100_000
