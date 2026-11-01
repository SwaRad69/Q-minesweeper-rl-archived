"""Validate and summarise a persistent human demonstration dataset.

Checks corrupted files, duplicate game ids, invalid action ids, invalid
coordinates, invalid state shapes, missing/inconsistent transitions, impossible
action types and incomplete games, then reports dataset statistics.

::

    python -m minesweeper_rl.data.validate_human_dataset --dataset data/human
"""

from __future__ import annotations

import argparse
import logging

from .human_dataset import validate_dataset

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/human",
                        help="persistent human dataset directory to validate")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = validate_dataset(args.dataset)
    print(report.format())
    if report.invalid_reasons:
        print("invalidation reasons:")
        for category, count in sorted(
            report.invalid_reasons.items(), key=lambda item: (-item[1], item[0])
        ):
            print(f"  {category}: {count}")
    for error in report.errors:
        print(f"  - {error}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
