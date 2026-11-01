"""Experiment metrics: per-episode/update records, JSONL logging and summaries.

Training and evaluation both emit the same :class:`EpisodeMetrics` records; only
the ``phase`` field differs.  Updates additionally emit :class:`UpdateMetrics`.
Records are appended as JSON lines so a run can be resumed, plotted and
summarised without holding everything in memory.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

EPISODES_FILE = "episodes.jsonl"
UPDATES_FILE = "updates.jsonl"


@dataclass(slots=True)
class EpisodeMetrics:
    """Outcome and cost of one complete game.

    ``open_actions``/``flag_actions``/``chord_actions`` count the OPEN, FLAG and
    CHORD operations in the episode.  ``source`` is always ``"agent"`` for
    autonomous training (human demonstration data is not part of this pipeline).
    """

    episode: int
    phase: str
    outcome: str
    won: bool
    reward: float
    steps: int
    mine_hit: bool
    cells_revealed: int
    duration_s: float
    invalid_actions: int
    timestamp: float = field(default_factory=time.time)
    open_actions: int = 0
    #: FLAG/UNFLAG operations (always neutral reward, never terminate a game).
    flag_actions: int = 0
    #: CHORD operations on an opened number.
    chord_actions: int = 0
    source: str = "agent"
    #: Total genuinely revealed cells on the final board (``board >= 0``),
    #: including the controlled first move.  ``cells_revealed`` counts only the
    #: agent's own progress; this is the coverage numerator.
    revealed_total: int = 0


@dataclass(slots=True)
class UpdateMetrics:
    """One PPO update cycle (rollout + optimization)."""

    iteration: int
    phase: str
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    total_loss: float
    updates: int
    timestamp: float = field(default_factory=time.time)
    learning_rate: float = 0.0
    device: str = ""


@dataclass(slots=True)
class MetricsSummary:
    """Aggregate statistics over a set of episodes."""

    games: int
    wins: int
    losses: int
    win_rate: float
    mean_reward: float
    mean_steps: float
    mine_hit_rate: float
    mean_cells_revealed: float
    mean_duration_s: float
    invalid_actions: int
    environment_failures: int
    #: Mean per-episode operation counts (OPEN / FLAG / CHORD).
    mean_open_actions: float = 0.0
    mean_flag_actions: float = 0.0
    mean_chord_actions: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def format(self) -> str:
        return (
            f"games={self.games} wins={self.wins} losses={self.losses} "
            f"win_rate={self.win_rate:.1%} | mean_reward={self.mean_reward:.3f} "
            f"mean_steps={self.mean_steps:.2f} mine_hit_rate={self.mine_hit_rate:.1%} "
            f"cells_revealed={self.mean_cells_revealed:.1f} "
            f"duration={self.mean_duration_s:.2f}s | "
            f"open={self.mean_open_actions:.1f} flag={self.mean_flag_actions:.1f} "
            f"chord={self.mean_chord_actions:.1f} | invalid={self.invalid_actions} "
            f"env_failures={self.environment_failures}"
        )


def summarize(
    episodes: list[EpisodeMetrics], *, environment_failures: int = 0
) -> MetricsSummary:
    """Aggregate episode records into a :class:`MetricsSummary`."""

    games = len(episodes)
    if games == 0:
        return MetricsSummary(
            games=0, wins=0, losses=0, win_rate=0.0, mean_reward=0.0,
            mean_steps=0.0, mine_hit_rate=0.0, mean_cells_revealed=0.0,
            mean_duration_s=0.0, invalid_actions=0,
            environment_failures=environment_failures,
        )
    wins = sum(1 for episode in episodes if episode.won)
    losses = sum(1 for episode in episodes if episode.outcome == "lost")
    return MetricsSummary(
        games=games,
        wins=wins,
        losses=losses,
        win_rate=wins / games,
        mean_reward=sum(e.reward for e in episodes) / games,
        mean_steps=sum(e.steps for e in episodes) / games,
        mine_hit_rate=sum(1 for e in episodes if e.mine_hit) / games,
        mean_cells_revealed=sum(e.cells_revealed for e in episodes) / games,
        mean_duration_s=sum(e.duration_s for e in episodes) / games,
        invalid_actions=sum(e.invalid_actions for e in episodes),
        environment_failures=environment_failures,
        mean_open_actions=sum(e.open_actions for e in episodes) / games,
        mean_flag_actions=sum(e.flag_actions for e in episodes) / games,
        mean_chord_actions=sum(e.chord_actions for e in episodes) / games,
    )


class MetricsLogger:
    """Append-only JSONL metrics writer and reader for one run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.run_dir / EPISODES_FILE
        self.updates_path = self.run_dir / UPDATES_FILE
        self.episodes: list[EpisodeMetrics] = []
        self.updates: list[UpdateMetrics] = []

    def _append(self, path: Path, record: object) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def log_episode(self, episode: EpisodeMetrics) -> None:
        self.episodes.append(episode)
        self._append(self.episodes_path, episode)

    def log_update(self, update: UpdateMetrics) -> None:
        self.updates.append(update)
        self._append(self.updates_path, update)

    def load_existing(self) -> None:
        """Load previously written records into memory (for resume/plots)."""

        self.episodes = _load(self.episodes_path, EpisodeMetrics)
        self.updates = _load(self.updates_path, UpdateMetrics)

    @classmethod
    def load(cls, run_dir: str | Path) -> "MetricsLogger":
        logger = cls(run_dir)
        logger.load_existing()
        return logger


def _load(path: Path, record_type: type) -> list:
    records: list = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(record_type(**json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            # A process killed mid-write can leave a truncated final JSON line.
            # Skip it (and any other malformed line) instead of aborting a
            # resume: the checkpoint remains the authoritative training state.
            logger.warning("skipping malformed metrics line in %s: %s", path, error)
    return records


def load_run(run_dir: str | Path) -> tuple[list[EpisodeMetrics], list[UpdateMetrics]]:
    """Return ``(episodes, updates)`` stored in a run directory."""

    run = Path(run_dir)
    return (
        _load(run / EPISODES_FILE, EpisodeMetrics),
        _load(run / UPDATES_FILE, UpdateMetrics),
    )
