"""Manual live test for the long-lived Minesweeper browser controller."""

from __future__ import annotations

import argparse
import os
import time

from minesweeper_rl.browser import BrowserController


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("MINESWEEPER_URL"),
                        help="Minesweeper game URL (or MINESWEEPER_URL)")
    parser.add_argument("--profile", default=os.environ.get("MINESWEEPER_PROFILE", "browser_profile"),
                        help="Persistent profile directory (or MINESWEEPER_PROFILE)")
    parser.add_argument("--delay", type=float, default=float(os.environ.get("BROWSER_INSPECTION_DELAY", "5")),
                        help="Seconds to leave the headed browser open")
    parser.add_argument("--board-timeout", type=float,
                        default=float(os.environ.get("MINESWEEPER_BOARD_TIMEOUT_MS", "60000")),
                        help="Milliseconds to wait for .cell elements")
    args = parser.parse_args()
    if not args.url:
        parser.error("provide --url or set MINESWEEPER_URL")

    controller = BrowserController(args.url, args.profile, board_timeout_ms=args.board_timeout)
    try:
        controller.start()
        print(f"Minesweeper board found: {controller.board_exists()}")
        time.sleep(max(0, args.delay))
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
