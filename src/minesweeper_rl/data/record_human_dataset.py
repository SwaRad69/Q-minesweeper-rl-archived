"""Record human Minesweeper demonstrations into a persistent dataset.

Unlike :mod:`minesweeper_rl.data.record_human` (which writes a single append-only
JSONL store), this entry point writes one immutable JSON file per completed game
under a persistent ``data/human/`` dataset, so play sessions accumulate without
ever rewriting or losing previous games and without a fixed game target.

A game is recorded as a sequence of observable ``state -> action -> next_state``
transitions using the project's flag-aware action encoding::

    action_id = action_type * (width * height) + (y * width + x)

with ``action_type`` 0 = OPEN, 1 = FLAG, 2 = CHORD.  Hidden mine positions are
never recorded.  Interrupted games are written marked ``complete: false`` unless
``--drop-incomplete`` is given.

::

    $env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
    python -m minesweeper_rl.data.record_human_dataset --dataset data/human
"""

from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
import threading

from ..config import EnvironmentConfig
from .human_dataset import HumanDatasetStore, PersistentHumanRecorder

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "data/human"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="persistent human dataset directory")
    parser.add_argument("--games", type=int, default=0,
                        help="stop after N completed games (0 = keep recording "
                             "until interrupted)")
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--mines", type=int, default=99)
    parser.add_argument("--difficulty", default="expert")
    parser.add_argument("--poll-interval", type=float, default=0.05,
                        help="seconds between event/board polls")
    parser.add_argument("--action-timeout", type=float, default=5.0,
                        help="initial seconds to wait for a stable board after a "
                             "click (the press remains pending for recovery)")
    parser.add_argument("--quiescence-time", type=float, default=0.3,
                        help="seconds the board must stay unchanged to count as settled")
    parser.add_argument("--drop-incomplete", action="store_true",
                        help="do not persist interrupted games (default: persist "
                             "them marked complete=false)")
    parser.add_argument("--url", default=os.environ.get("MINESWEEPER_URL"))
    parser.add_argument("--profile", default=os.environ.get("MINESWEEPER_PROFILE",
                        "browser_profile"))
    parser.add_argument("--diagnostics", action="store_true",
                        default=os.environ.get("MINESWEEPER_DIAGNOSTICS", "").lower()
                        in {"1", "true", "yes"})
    parser.add_argument("--log-file", help="timestamped recorder audit log (default: DATASET/recorder.log)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("provide --url or set MINESWEEPER_URL")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log_path = Path(args.log_file) if args.log_file else Path(args.dataset) / "recorder.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    audit = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    audit.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(audit)

    # Imported lazily so the module stays light for unit tests.
    from ..experiment.browser import open_environment
    from ..experiment.train import install_stop_handlers

    config = EnvironmentConfig(
        width=args.width,
        height=args.height,
        mines=args.mines,
        difficulty=args.difficulty,
        diagnostics=args.diagnostics,
    )
    store = HumanDatasetStore(
        args.dataset,
        width=config.width,
        height=config.height,
        mines=config.mines,
        difficulty=config.difficulty,
    )
    sink = PersistentHumanRecorder(
        store,
        width=config.width,
        height=config.height,
        mines=config.mines,
        difficulty=config.difficulty,
        store_incomplete=not args.drop_incomplete,
    )

    stop_event = threading.Event()
    install_stop_handlers(stop_event)

    logger.info("=" * 40)
    logger.info("Human Minesweeper demonstration recorder")
    logger.info("=" * 40)
    logger.info("Dataset: %s", store.root)
    logger.info("Board: %dx%d mines=%s difficulty=%s",
                config.width, config.height, config.mines, config.difficulty)
    logger.info("Existing games: %d", store.game_count)
    logger.info("Next game index: %d", store.next_game_index)
    logger.info("Play Minesweeper in the browser window; press Ctrl+C to stop.")

    with open_environment(args.url, args.profile, config=config) as environment:
        from .record_human import HumanTrajectoryRecorder

        human = HumanTrajectoryRecorder(
            environment,
            sink,
            poll_interval=args.poll_interval,
            action_timeout=args.action_timeout,
            quiescence_time=args.quiescence_time,
            diagnostics=args.diagnostics,
        )
        target = args.games if args.games and args.games > 0 else sys.maxsize
        report = human.run(max_games=target, stop_event=stop_event)

    logger.info("=" * 40)
    logger.info("Recording stopped")
    logger.info("=" * 40)
    logger.info("Dataset: %s", store.root)
    logger.info("%s", report.format())
    for error in report.errors:
        logger.info("Recorder rejection: %s", error)
    return 0 if not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
