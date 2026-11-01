"""Stress-test the live Minesweeper environment and report reliability.

The harness drives full games with random valid reveals, deliberately exercises
invalid actions, DOM synchronisation, parsing, page reloads and unexpected page
states, and measures where the wall-clock time actually goes.  It reports what
it finds rather than fixing anything, so failures are visible.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import numpy as np

from .actions import InvalidActionError
from .agents.random_agent import hidden_cells
from .browser import BrowserController
from .config import EnvironmentConfig
from .dom import FLAGGED, MAX_NUMBER, GameStatus
from .environment import MinesweeperEnvironment
from .metrics import EnvironmentMetrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EpisodeReport:
    """Outcome of one stress episode."""

    episode: int
    phase: str
    status: GameStatus | None
    steps: int
    total_reward: float
    duration_s: float
    error: str | None = None
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class CheckReport:
    """Result of one targeted reliability check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class StressReport:
    """Everything collected during a stress run."""

    episodes: list[EpisodeReport] = field(default_factory=list)
    checks: list[CheckReport] = field(default_factory=list)
    metrics: EnvironmentMetrics = field(default_factory=EnvironmentMetrics)
    recovery_attempts: int = 0
    recovery_succeeded: int = 0

    @property
    def attempted(self) -> int:
        return len(self.episodes)

    @property
    def failed(self) -> list[EpisodeReport]:
        return [episode for episode in self.episodes if episode.error is not None]

    @property
    def successful(self) -> int:
        return self.attempted - len(self.failed)

    @property
    def reliability_percent(self) -> float:
        return self.successful / self.attempted * 100.0 if self.attempted else 0.0


def play_stress_episode(
    env: MinesweeperEnvironment, episode: int, phase: str, rng: random.Random
) -> EpisodeReport:
    """Play one random episode, capturing any environment failure."""

    started = time.perf_counter()
    steps = 0
    total_reward = 0.0
    status: GameStatus | None = None
    try:
        board = env.reset()
        status = env.get_game_status()
        while status not in (GameStatus.WON, GameStatus.LOST):
            actions = hidden_cells(board)
            if not actions:
                break
            board, reward, terminated, truncated, info = env.step(rng.choice(actions))
            total_reward += reward
            steps += 1
            status = info["game_status"]
            if terminated or truncated:
                break
    except Exception as error:  # noqa: BLE001 - the harness records every failure
        return EpisodeReport(
            episode=episode,
            phase=phase,
            status=None,
            steps=steps,
            total_reward=total_reward,
            duration_s=time.perf_counter() - started,
            error=str(error),
            error_type=type(error).__name__,
        )
    return EpisodeReport(
        episode=episode,
        phase=phase,
        status=status,
        steps=steps,
        total_reward=total_reward,
        duration_s=time.perf_counter() - started,
    )


def _expect_exception(func, action: object, exc_type: type[BaseException], name: str) -> CheckReport:
    try:
        func(action)
    except exc_type:
        return CheckReport(name, True)
    except Exception as error:  # noqa: BLE001 - wrong exception is a failure
        return CheckReport(name, False, f"raised {type(error).__name__}: {error}")
    return CheckReport(name, False, "no exception raised")


def check_invalid_actions(env: MinesweeperEnvironment) -> list[CheckReport]:
    """Invalid actions must raise the documented errors and not corrupt state."""

    board = env.observe()
    width, height = env.config.width, env.config.height
    checks = [
        _expect_exception(env.step, (width, 0), IndexError, "invalid:out_of_bounds"),
        _expect_exception(env.step, (0.5, 0), TypeError, "invalid:non_integer"),
        _expect_exception(env.step, (0,), TypeError, "invalid:malformed_action"),
    ]
    revealed = [
        (x, y) for y in range(height) for x in range(width) if board[y, x] >= 0
    ]
    if revealed:
        checks.append(
            _expect_exception(
                env.step, revealed[0], InvalidActionError, "invalid:already_revealed"
            )
        )
    checks.append(CheckReport("invalid:no_state_corruption", np.array_equal(board, env.observe())))
    return checks


def check_board_parsing(env: MinesweeperEnvironment) -> CheckReport:
    board = env.observe()
    shape_ok = board.shape == (env.config.height, env.config.width)
    dtype_ok = board.dtype == np.int8
    values_ok = bool(
        np.isin(board, np.arange(FLAGGED, MAX_NUMBER + 1)).all()
    )
    passed = shape_ok and dtype_ok and values_ok
    return CheckReport(
        "board_parsing:shape_dtype_values",
        passed,
        f"shape={board.shape} dtype={board.dtype}",
    )


def check_dom_sync(env: MinesweeperEnvironment, samples: int = 5) -> CheckReport:
    boards = [env.observe() for _ in range(samples)]
    stable = all(np.array_equal(boards[0], board) for board in boards[1:])
    return CheckReport("dom_sync:stable_idle_observations", stable, f"{samples} reads")


def check_reload_recovery(env: MinesweeperEnvironment, page) -> CheckReport:
    try:
        page.reload(wait_until="domcontentloaded")
        env.reset()
    except Exception as error:  # noqa: BLE001 - recovery failure is the result
        return CheckReport("recovery:page_reload", False, f"{type(error).__name__}: {error}")
    return CheckReport("recovery:page_reload", True)


def check_unexpected_page_state(
    env: MinesweeperEnvironment, page, base_url: str
) -> CheckReport:
    """A boardless page must fail fast, then recover after navigating back."""

    page.goto("about:blank", wait_until="domcontentloaded")
    had_board = env.board_exists()
    short = MinesweeperEnvironment(
        page,
        EnvironmentConfig(
            width=env.config.width,
            height=env.config.height,
            ready_timeout_ms=2_000,
            new_game_timeout_ms=2_000,
        ),
    )
    try:
        short.reset()
    except RuntimeError:
        fast_failure = True
    except Exception as error:  # noqa: BLE001 - unexpected error type is a failure
        return CheckReport(
            "unexpected_state:boardless_page",
            False,
            f"raised {type(error).__name__}: {error}",
        )
    else:
        fast_failure = False

    recovered = False
    try:
        page.goto(base_url, wait_until="domcontentloaded")
        env.reset()
        recovered = True
    except Exception as error:  # noqa: BLE001 - recovery failure is the result
        detail = f"{type(error).__name__}: {error}"
        return CheckReport("unexpected_state:boardless_page", False, detail)

    return CheckReport(
        "unexpected_state:boardless_page",
        (not had_board) and fast_failure and recovered,
        f"board_exists={had_board} fast_failure={fast_failure} recovered={recovered}",
    )


def run_win_phase(page, base_url: str, rng: random.Random, max_games: int = 8) -> CheckReport:
    """Exercise the win path on a small naturally solvable 9x9 board.

    The first reveal can already solve the board, so the status is checked
    right after ``reset`` as well as after every step.  Mines stay hidden after
    a win, so ``hidden_cells`` alone is not a terminal signal.
    """

    parts = urlsplit(base_url)
    win_url = f"{parts.scheme}://{parts.netloc}/start/9x9/0"
    try:
        page.goto(win_url, wait_until="domcontentloaded")
        env = MinesweeperEnvironment(page, EnvironmentConfig(width=9, height=9))
        for game in range(1, max_games + 1):
            board = env.reset()
            status = env.get_game_status()
            while status not in (GameStatus.WON, GameStatus.LOST):
                actions = hidden_cells(board)
                if not actions:
                    break
                board, _reward, terminated, truncated, info = env.step(rng.choice(actions))
                status = info["game_status"]
                if terminated or truncated:
                    break
            if status is GameStatus.WON:
                return CheckReport("win_path:9x9", True, f"won game {game}")
    except Exception as error:  # noqa: BLE001 - failure is the result
        return CheckReport("win_path:9x9", False, f"{type(error).__name__}: {error}")
    return CheckReport("win_path:9x9", False, "no win within max games")


def _recover(env: MinesweeperEnvironment, page, base_url: str) -> bool:
    try:
        env.reset()
        return True
    except Exception:  # noqa: BLE001 - fall through to a full navigation
        pass
    try:
        page.goto(base_url, wait_until="domcontentloaded")
        env.reset()
        return True
    except Exception:  # noqa: BLE001 - recovery itself failed
        return False


def run_stress(
    controller: BrowserController,
    base_url: str,
    games: int,
    *,
    seed: int | None = None,
    diagnostics: bool = False,
    win_phase: bool = True,
) -> StressReport:
    rng = random.Random(seed)
    report = StressReport()
    page = controller.start()
    env = MinesweeperEnvironment(page, EnvironmentConfig(diagnostics=diagnostics))
    env.metrics = report.metrics

    for episode in range(1, games + 1):
        result = play_stress_episode(env, episode, "main", rng)
        report.episodes.append(result)
        logger.info(
            "episode %03d | %-7s | steps=%3d | reward=%7.1f | %5.1fs%s",
            result.episode,
            result.status.value.upper() if result.status else "FAILED",
            result.steps,
            result.total_reward,
            result.duration_s,
            f" | {result.error_type}: {result.error}" if result.error else "",
        )
        if result.error is not None:
            report.recovery_attempts += 1
            if _recover(env, page, base_url):
                report.recovery_succeeded += 1
                logger.info("recovered after episode %03d", result.episode)
            else:
                logger.warning("recovery failed after episode %03d", result.episode)

    if games < 1:
        env.reset()

    report.checks.extend(check_invalid_actions(env))
    report.checks.append(check_board_parsing(env))
    report.checks.append(check_dom_sync(env))
    report.checks.append(check_reload_recovery(env, page))
    report.checks.append(check_unexpected_page_state(env, page, base_url))

    if win_phase:
        report.checks.append(run_win_phase(page, base_url, rng))

    return report


def _latency_row(label: str, latency) -> str:
    return (
        f"  {label:<10} {latency.mean_ms:>9.1f} {latency.median_ms:>9.1f} "
        f"{latency.p95_ms:>9.1f} {latency.max_ms:>9.1f} {latency.count:>7d}"
    )


def summarize_stress(report: StressReport) -> str:
    """Render the final reliability and latency report."""

    metrics = report.metrics
    losses = sum(1 for e in report.episodes if e.status is GameStatus.LOST)
    wins = sum(1 for e in report.episodes if e.status is GameStatus.WON)
    lengths = [e.steps for e in report.episodes]
    avg_length = sum(lengths) / len(lengths) if lengths else 0.0

    lines = [
        "==== Environment stress summary ====",
        f"episodes attempted : {report.attempted}",
        f"episodes successful: {report.successful}",
        f"episodes failed    : {len(report.failed)}",
        f"reliability        : {report.reliability_percent:.1f}%",
        f"wins / losses      : {wins} / {losses}",
        f"avg episode length : {avg_length:.1f} steps",
        f"recovery           : {report.recovery_succeeded}/{report.recovery_attempts} "
        "failed episodes recovered",
        "",
        "latency (ms)     mean    median       p95       max   samples",
        _latency_row("step", metrics.step),
        _latency_row("reset", metrics.reset),
        _latency_row("observe", metrics.observe),
        _latency_row("dom_read", metrics.dom_read),
        _latency_row("parse", metrics.parse),
        _latency_row("click", metrics.click),
        "",
        f"clicks={metrics.clicks} event_fallbacks={metrics.event_fallbacks}",
        "",
        "checks:",
    ]
    for check in report.checks:
        marker = "PASS" if check.passed else "FAIL"
        detail = f" ({check.detail})" if check.detail else ""
        lines.append(f"  [{marker}] {check.name}{detail}")

    failures = Counter(
        f"{episode.error_type}: {episode.error}" for episode in report.failed
    )
    lines.append("")
    lines.append("failure modes:")
    if failures:
        for failure, count in failures.most_common():
            lines.append(f"  x{count} {failure}")
    else:
        lines.append("  none")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=30, help="number of stress games")
    parser.add_argument("--seed", type=int, default=None, help="optional RNG seed")
    parser.add_argument("--url", default=os.environ.get("MINESWEEPER_URL"),
                        help="Minesweeper game URL (or MINESWEEPER_URL)")
    parser.add_argument("--profile", default=os.environ.get("MINESWEEPER_PROFILE", "browser_profile"),
                        help="persistent profile directory")
    parser.add_argument("--diagnostics", action="store_true",
                        default=os.environ.get("MINESWEEPER_DIAGNOSTICS", "").lower() in {"1", "true", "yes"},
                        help="enable environment DEBUG logging")
    parser.add_argument("--no-win-phase", action="store_true",
                        help="skip the mine-free win-path check")
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("provide --url or set MINESWEEPER_URL")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    controller = BrowserController(args.url, args.profile, wait_for_board=False)
    try:
        report = run_stress(
            controller,
            args.url,
            args.games,
            seed=args.seed,
            diagnostics=args.diagnostics,
            win_phase=not args.no_win_phase,
        )
    finally:
        controller.shutdown()
    print(summarize_stress(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
