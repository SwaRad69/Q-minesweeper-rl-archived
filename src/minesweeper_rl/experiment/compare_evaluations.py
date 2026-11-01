"""Compare frozen-policy evaluation results from two or more runs.

Every ``experiment.evaluate`` run writes ``evaluation_results.json`` with the
same schema, so scratch PPO and human-pretrained PPO can be compared directly::

    python -m minesweeper_rl.experiment.compare_evaluations \
        scratch=experiments/cnn_ppo_expert_v1_eval \
        human=experiments/cnn_ppo_expert_human_v1_eval
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

EVALUATION_RESULTS_FILE = "evaluation_results.json"

#: (field in the evaluation summary, column label, format)
_METRICS = (
    ("games", "games", "{:.0f}"),
    ("win_rate", "win_rate", "{:.1%}"),
    ("mean_reward", "mean_reward", "{:.2f}"),
    ("mean_steps", "steps", "{:.1f}"),
    ("mean_cells_revealed", "cells", "{:.1f}"),
    ("mine_hit_rate", "mine_hit", "{:.1%}"),
    ("mean_open_actions", "open", "{:.1f}"),
    ("mean_flag_actions", "flag", "{:.1f}"),
    ("mean_chord_actions", "chord", "{:.1f}"),
)


def load_evaluation_results(path: str | Path) -> dict:
    """Load an evaluation summary from a run directory or JSON file."""

    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / EVALUATION_RESULTS_FILE
    if not candidate.exists():
        raise FileNotFoundError(f"evaluation results not found: {candidate}")
    data = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "summary" not in data:
        raise ValueError(f"{candidate} is not an evaluation results file")
    return data


def format_comparison(runs: Mapping[str, Mapping]) -> str:
    """Return a fixed-width table comparing named evaluation summaries."""

    names = list(runs)
    if not names:
        return "no runs to compare"
    width = max(len(name) for name in names) + 2
    header = "metric".ljust(14) + "".join(name.rjust(max(width, 12)) for name in names)
    lines = [header, "-" * len(header)]
    for field, label, spec in _METRICS:
        row = label.ljust(14)
        for name in names:
            summary = runs[name].get("summary", {})
            value = summary.get(field)
            cell = "n/a" if value is None else spec.format(value)
            row += cell.rjust(max(width, 12))
        lines.append(row)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", nargs="+", metavar="NAME=PATH",
        help="named evaluation run directory or evaluation_results.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    runs: dict[str, dict] = {}
    for item in args.runs:
        if "=" not in item:
            parser.error(f"expected NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        try:
            runs[name] = load_evaluation_results(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
    print(format_comparison(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
