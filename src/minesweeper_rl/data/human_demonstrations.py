"""Load and preprocess persistent human demonstrations for imitation learning.

This module is a **read-only** view over the raw recorder output in
``data/human/`` (see :mod:`minesweeper_rl.data.human_dataset`).  The raw game
files are the source of truth and are never modified.  The derived training
representation is a deterministic function of them: each training sample is an
observable board ``observation_t`` paired with the human ``action_id`` in the
project's 1440-action space.

A sample is built only from fields the player could observe at decision time:

* ``observation_before`` -- the flag-aware board (``FLAGGED=-2``, ``HIDDEN=-1``,
  ``OPEN=0..8``) decoded to ``int8``; exactly the representation the flag-aware
  CNN policy consumes;
* ``action.action_id`` -- ``action_type * (width * height) + (y * width + x)``.

The post-action board and any ground-truth mine information are never part of a
training sample.  Splitting is by **game**, so no transition from one game can
appear in both the training and validation split.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..action_types import ACTION_NAMES, compound_action_id
from .human_dataset import (
    ACTION_SPACE,
    GAMES_DIRNAME,
    MANIFEST_FILENAME,
    legacy_dataset_reason,
    semantic_game_result,
)
from .schema import DataError, decode_board

#: Fraction of games used for validation by default.
DEFAULT_VAL_FRACTION = 0.2
#: Deterministic default split seed.
DEFAULT_SPLIT_SEED = 0


@dataclass(frozen=True, slots=True)
class HumanSample:
    """One observable ``observation_t -> human_action_t`` training pair."""

    game_id: str
    step: int
    board: npt.NDArray[np.int8]
    action_id: int
    action_type: int
    x: int
    y: int


@dataclass(slots=True)
class HumanGame:
    """One loaded human game and its ordered training samples."""

    game_id: str
    game_index: int
    outcome: str
    complete: bool
    width: int
    height: int
    samples: list[HumanSample] = field(default_factory=list)
    source: str = "human"
    mines: int | None = None
    difficulty: str | None = None
    valid: bool = True
    invalid_reasons: list[str] = field(default_factory=list)
    first_board: npt.NDArray[np.int8] | None = None
    final_board: npt.NDArray[np.int8] | None = None

    @property
    def won(self) -> bool:
        return self.outcome == "won"


@dataclass(slots=True)
class DatasetStatistics:
    """Summary of the human data selected for training."""

    games: int = 0
    complete_games: int = 0
    winning_games: int = 0
    lost_games: int = 0
    incomplete_games: int = 0
    transitions: int = 0
    open_count: int = 0
    flag_count: int = 0
    chord_count: int = 0
    action_counts: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_games(cls, games: Iterable[HumanGame]) -> "DatasetStatistics":
        stats = cls()
        for game in games:
            stats.games += 1
            if game.complete:
                stats.complete_games += 1
            if game.outcome == "won":
                stats.winning_games += 1
            elif game.outcome == "lost":
                stats.lost_games += 1
            else:
                stats.incomplete_games += 1
            for sample in game.samples:
                stats.transitions += 1
                stats.action_counts[sample.action_id] = (
                    stats.action_counts.get(sample.action_id, 0) + 1
                )
                if sample.action_type == 0:
                    stats.open_count += 1
                elif sample.action_type == 1:
                    stats.flag_count += 1
                else:
                    stats.chord_count += 1
        return stats

    @property
    def average_transitions_per_game(self) -> float:
        return self.transitions / self.games if self.games else 0.0

    def action_type_distribution(self) -> dict[str, int]:
        return {
            ACTION_NAMES[0]: self.open_count,
            ACTION_NAMES[1]: self.flag_count,
            ACTION_NAMES[2]: self.chord_count,
        }

    def normalized_action_distribution(self) -> dict[int, float]:
        if not self.transitions:
            return {}
        return {
            action_id: count / self.transitions
            for action_id, count in sorted(self.action_counts.items())
        }

    def top_actions(self, limit: int = 15) -> list[tuple[int, int]]:
        ordered = sorted(
            self.action_counts.items(), key=lambda item: (-item[1], item[0])
        )
        return ordered[:limit]

    def as_dict(self) -> dict[str, Any]:
        return {
            "games": self.games,
            "complete_games": self.complete_games,
            "winning_games": self.winning_games,
            "lost_games": self.lost_games,
            "incomplete_games": self.incomplete_games,
            "transitions": self.transitions,
            "open_count": self.open_count,
            "flag_count": self.flag_count,
            "chord_count": self.chord_count,
            "average_transitions_per_game": self.average_transitions_per_game,
            "distinct_actions": len(self.action_counts),
            "action_counts": {str(k): v for k, v in sorted(self.action_counts.items())},
        }

    def format(self, *, top: int = 15) -> str:
        lines = [
            f"games={self.games} (won={self.winning_games} lost={self.lost_games} "
            f"incomplete={self.incomplete_games}) complete={self.complete_games}",
            f"transitions={self.transitions} "
            f"avg/game={self.average_transitions_per_game:.1f}",
            "action types: "
            f"OPEN={self.open_count} FLAG={self.flag_count} CHORD={self.chord_count}",
            f"distinct action ids used: {len(self.action_counts)} / 1440",
        ]
        if self.transitions:
            shares = self.action_type_distribution()
            total = self.transitions
            lines.append(
                "distribution: "
                + " ".join(
                    f"{name}={count} ({count / total:.1%})"
                    for name, count in shares.items()
                )
            )
            lines.append("top action ids:")
            for action_id, count in self.top_actions(top):
                lines.append(
                    f"  {action_id:4d} x{count:<6d} ({count / total:5.1%})"
                )
        return "\n".join(lines)


def _game_files(root: str | Path) -> list[Path]:
    root = Path(root)
    reason = legacy_dataset_reason(root)
    if reason is not None:
        raise DataError(
            f"{root} is not a canonical human dataset: {reason}. The canonical "
            f"dataset is data/human/ with dataset_manifest.json and "
            f"games/game_XXXXXX.json."
        )
    games_dir = root / GAMES_DIRNAME
    if not games_dir.exists():
        raise DataError(f"human dataset has no {GAMES_DIRNAME}/ directory: {root}")
    return sorted(games_dir.glob("game_*.json"))


def _load_game(path: Path) -> HumanGame:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataError(f"could not read game {path}: {error}") from error
    if not isinstance(data, dict):
        raise DataError(f"game {path} is not a JSON object")

    game_id = data.get("game_id")
    if not isinstance(game_id, str) or not game_id:
        raise DataError(f"game {path} has no game_id")
    action_space = data.get("action_space")
    if action_space is not None and action_space != ACTION_SPACE:
        raise DataError(
            f"game {game_id} declares action_space {action_space!r}, not "
            f"{ACTION_SPACE!r}; legacy chord-as-open data must not be used"
        )
    width = int(data.get("width", 0))
    height = int(data.get("height", 0))
    if width < 1 or height < 1:
        raise DataError(f"game {game_id} has invalid dimensions {width}x{height}")
    cells = width * height

    transitions = data.get("transitions")
    if not isinstance(transitions, list):
        raise DataError(f"game {game_id} has no transitions list")

    samples: list[HumanSample] = []
    for expected_step, transition in enumerate(transitions):
        if not isinstance(transition, dict):
            raise DataError(f"game {game_id} step {expected_step}: not an object")
        step = int(transition.get("step", -1))
        if step != expected_step:
            raise DataError(
                f"game {game_id}: step index {step} out of order (expected "
                f"{expected_step})"
            )
        action = transition.get("action")
        if not isinstance(action, dict):
            raise DataError(f"game {game_id} step {step}: missing action")
        action_type = int(action.get("action_type", -1))
        if action_type not in ACTION_NAMES:
            raise DataError(
                f"game {game_id} step {step}: invalid action_type {action_type!r}"
            )
        x = int(action.get("x", -1))
        y = int(action.get("y", -1))
        if not (0 <= x < width and 0 <= y < height):
            raise DataError(
                f"game {game_id} step {step}: coordinate outside board ({x}, {y})"
            )
        action_id = int(action.get("action_id", -1))
        expected_id = compound_action_id(action_type, y * width + x, cells=cells)
        if action_id != expected_id:
            raise DataError(
                f"game {game_id} step {step}: action_id {action_id} does not "
                f"match type/cell -> {expected_id}"
            )
        board = decode_board(
            transition.get("observation_before"), width=width, height=height
        )
        samples.append(
            HumanSample(
                game_id=game_id,
                step=step,
                board=np.ascontiguousarray(board, dtype=np.int8),
                action_id=action_id,
                action_type=action_type,
                x=x,
                y=y,
            )
        )

    outcome = str(data.get("outcome", "incomplete"))
    semantic = semantic_game_result(data)
    first_board: npt.NDArray[np.int8] | None = None
    final_board: npt.NDArray[np.int8] | None = None
    if transitions:
        try:
            first_board = decode_board(
                transitions[0].get("observation_before"), width=width, height=height
            )
            final_board = decode_board(
                transitions[-1].get("observation_after"), width=width, height=height
            )
        except (DataError, AttributeError):
            first_board = final_board = None
    return HumanGame(
        game_id=game_id,
        game_index=int(data.get("game_index", 0)),
        outcome=outcome,
        complete=bool(data.get("complete", False)),
        width=width,
        height=height,
        samples=samples,
        source=str(data.get("source", "human")),
        mines=data.get("mines"),
        difficulty=data.get("difficulty"),
        valid=semantic.valid,
        invalid_reasons=list(semantic.reasons),
        first_board=first_board,
        final_board=final_board,
    )


def _matches(game: HumanGame, selectors: Sequence[str | int]) -> bool:
    for selector in selectors:
        if isinstance(selector, int):
            if game.game_index == selector:
                return True
        elif selector == game.game_id:
            return True
    return False


class HumanDemonstrationDataset:
    """Read-only, filterable view over a persistent human dataset."""

    def __init__(
        self,
        root: str | Path,
        *,
        games: Sequence[str | int] | None = None,
        winning_only: bool = False,
        exclude_incomplete: bool = False,
        include_invalid: bool = False,
    ) -> None:
        self.root = Path(root)
        self._manifest = self._read_manifest()
        loaded = [_load_game(path) for path in _game_files(self.root)]
        loaded.sort(key=lambda game: (game.game_index, game.game_id))
        self._mark_phantom_continuations(loaded)

        invalid = [game for game in loaded if not game.valid]
        self.invalid_games = invalid
        if not include_invalid:
            loaded = [game for game in loaded if game.valid]
            if not loaded and invalid:
                raise DataError(
                    f"all {len(invalid)} games in {self.root} are semantically "
                    f"invalid; refusing to build a training dataset. Run "
                    f"validate_dataset() for details, or pass "
                    f"include_invalid=True to inspect them."
                )

        if games:
            loaded = [game for game in loaded if _matches(game, games)]
        if winning_only:
            loaded = [game for game in loaded if game.won]
        if exclude_incomplete:
            loaded = [game for game in loaded if game.complete]

        self.games = loaded
        self._check_dimensions()

    @staticmethod
    def _mark_phantom_continuations(loaded: Sequence[HumanGame]) -> None:
        """Flag games whose first board continues the previous game's final board."""

        previous: HumanGame | None = None
        for game in loaded:
            if (
                previous is not None
                and previous.final_board is not None
                and game.first_board is not None
                and previous.final_board.shape == game.first_board.shape
                and np.array_equal(previous.final_board, game.first_board)
            ):
                game.valid = False
                game.invalid_reasons.append(
                    "first state equals the previous game's final state "
                    "(phantom continuation)"
                )
            previous = game

    def _read_manifest(self) -> dict[str, Any]:
        path = self.root / MANIFEST_FILENAME
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _check_dimensions(self) -> None:
        dimensions = {(game.width, game.height) for game in self.games}
        if len(dimensions) > 1:
            raise DataError(
                f"human dataset mixes board dimensions: {sorted(dimensions)}"
            )
        board = self._manifest.get("board")
        if dimensions and isinstance(board, dict):
            width, height = dimensions.pop()
            if int(board.get("width", width)) != width or int(
                board.get("height", height)
            ) != height:
                raise DataError(
                    "game board dimensions disagree with dataset_manifest.json"
                )

    # -- accessors --------------------------------------------------------

    @property
    def width(self) -> int:
        if not self.games:
            raise DataError("the selected dataset contains no games")
        return self.games[0].width

    @property
    def height(self) -> int:
        if not self.games:
            raise DataError("the selected dataset contains no games")
        return self.games[0].height

    @property
    def cells(self) -> int:
        return self.width * self.height

    @property
    def num_actions(self) -> int:
        return len(ACTION_NAMES) * self.cells

    def game_ids(self) -> list[str]:
        return [game.game_id for game in self.games]

    def boards(self) -> list[npt.NDArray[np.int8]]:
        return [sample.board for game in self.games for sample in game.samples]

    def action_ids(self) -> list[int]:
        return [sample.action_id for game in self.games for sample in game.samples]

    def samples(self) -> list[HumanSample]:
        return [sample for game in self.games for sample in game.samples]

    def statistics(self) -> DatasetStatistics:
        return DatasetStatistics.from_games(self.games)

    def subset(self, game_ids: Sequence[str]) -> "HumanDemonstrationDataset":
        """Return a new view restricted to ``game_ids`` without re-reading files."""

        selected = set(game_ids)
        view = HumanDemonstrationDataset.__new__(HumanDemonstrationDataset)
        view.root = self.root
        view._manifest = self._manifest
        view.games = [game for game in self.games if game.game_id in selected]
        return view


def split_games(
    game_ids: Sequence[str],
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[str], list[str]]:
    """Split game ids into ``(train, validation)`` deterministically by game.

    The split is a pure function of ``game_ids``, ``val_fraction`` and ``seed``:
    shuffling with a seeded RNG and taking the first ``n_val`` games as
    validation.  No transition from one game can appear in both splits.
    """

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    ids = list(game_ids)
    if not ids:
        return [], []
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_val = int(round(n * val_fraction))
    if val_fraction > 0.0 and n >= 2:
        n_val = max(1, min(n - 1, n_val))
    else:
        n_val = max(0, min(n, n_val))
    return ids[n_val:], ids[:n_val]

