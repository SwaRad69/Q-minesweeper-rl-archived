"""Random Minesweeper agent.

A minimal baseline that plays complete games through the browser environment
and reports per-episode statistics.  It exists to prove the environment can be
driven autonomously end to end; it is not a learning algorithm.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..browser import BrowserController
from ..config import EnvironmentConfig
from ..dom import HIDDEN, GameStatus
from ..environment import MinesweeperEnvironment

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Outcome and statistics for one complete game."""

    episode: int
    status: GameStatus
    steps: int
    total_reward: float
    cells_revealed: int
    hit_mine: bool
    duration_s: float

    @property
    def won(self) -> bool:
        return self.status is GameStatus.WON


def hidden_cells(board: npt.NDArray[np.int8]) -> list[tuple[int, int]]:
    """Return every hidden ``(x, y)`` cell from an observed board.

    The random agent never flags cells, so a hidden cell is always a valid
    reveal target.
    """

    ys, xs = np.where(board == HIDDEN)
    return [(int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist())]


def play_episode(
    env: MinesweeperEnvironment, episode: int, rng: random.Random
) -> EpisodeResult:
    """Play one game to termination with uniformly random valid reveals."""

    started = time.perf_counter()
    board = env.reset()
    total_reward = 0.0
    steps = 0
    hit_mine = False
    status = GameStatus.RUNNING

    while True:
        actions = hidden_cells(board)
        if not actions:
            break
        action = rng.choice(actions)
        board, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        status = info["game_status"]
        hit_mine = bool(info.get("hit_mine", False))
        if terminated or truncated:
            break

    return EpisodeResult(
        episode=episode,
        status=status,
        steps=steps,
        total_reward=total_reward,
        cells_revealed=int(np.count_nonzero(board >= 0)),
        hit_mine=hit_mine,
        duration_s=time.perf_counter() - started,
    )


def format_episode(result: EpisodeResult) -> str:
    """Render one episode as a single log line."""

    outcome = result.status.value.upper()
    mine = "yes" if result.hit_mine else "no"
    return (
        f"episode {result.episode:03d} | {outcome:<7} | steps={result.steps:>3} | "
        f"reward={result.total_reward:>7.1f} | revealed={result.cells_revealed:>3} | "
        f"mine={mine:<3} | {result.duration_s:6.2f}s"
    )


def summarize(results: list[EpisodeResult]) -> str:
    """Return a concise multi-line summary over all episodes."""

    games = len(results)
    if games == 0:
        return "random agent summary: no episodes played"
    wins = sum(1 for result in results if result.won)
    losses = sum(1 for result in results if result.status is GameStatus.LOST)
    total_time = sum(result.duration_s for result in results)
    return (
        "random agent summary "
        f"| games={games} wins={wins} losses={losses} win_rate={wins / games:.1%} "
        f"| avg_steps={sum(r.steps for r in results) / games:.1f} "
        f"avg_reward={sum(r.total_reward for r in results) / games:.1f} "
        f"avg_revealed={sum(r.cells_revealed for r in results) / games:.1f} "
        f"| total={total_time:.1f}s avg_episode={total_time / games:.1f}s"
    )


def run_random_agent(
    env: MinesweeperEnvironment,
    games: int,
    *,
    rng: random.Random | None = None,
    on_episode: Callable[[EpisodeResult], None] | None = None,
) -> list[EpisodeResult]:
    """Play ``games`` complete games and return their results."""

    if games < 1:
        raise ValueError("games must be at least 1")
    generator = rng if rng is not None else random.Random()
    results: list[EpisodeResult] = []
    for episode in range(1, games + 1):
        result = play_episode(env, episode, generator)
        results.append(result)
        if on_episode is not None:
            on_episode(result)
        logger.info("%s", format_episode(result))
    logger.info("%s", summarize(results))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20, help="number of games to play")
    parser.add_argument("--seed", type=int, default=None, help="optional RNG seed")
    parser.add_argument("--url", default=os.environ.get("MINESWEEPER_URL"),
                        help="Minesweeper game URL (or MINESWEEPER_URL)")
    parser.add_argument("--profile", default=os.environ.get("MINESWEEPER_PROFILE", "browser_profile"),
                        help="persistent profile directory")
    parser.add_argument("--diagnostics", action="store_true",
                        default=os.environ.get("MINESWEEPER_DIAGNOSTICS", "").lower() in {"1", "true", "yes"},
                        help="enable environment DEBUG logging")
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("provide --url or set MINESWEEPER_URL")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    controller = BrowserController(args.url, args.profile, wait_for_board=False)
    try:
        page = controller.start()
        env = MinesweeperEnvironment(page, EnvironmentConfig(diagnostics=args.diagnostics))
        run_random_agent(env, args.games, rng=random.Random(args.seed))
    finally:
        controller.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
