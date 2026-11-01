# LEGACY / NON-CANONICAL — do not use for new human data

This directory was produced by the older append-only JSONL recorder
(`minesweeper_rl.data.record_human` -> `TrajectoryStore`), not by the canonical
human demonstration recorder.

- `games.jsonl` / `transitions.jsonl` / `meta.json` use the legacy schema.
- The legacy schema has no `action_id` and **folds a chord into an OPEN action**
  (`action_type` values are strings such as `"reveal"` and `"open"`).

The canonical human demonstration dataset is `data/human/` (per-game
`games/game_XXXXXX.json` with OPEN/FLAG/CHORD = 0/1/2 and
`action_id = action_type * cells + cell_index`).

The canonical loaders detect and reject this legacy format explicitly, so a
CHORD demonstration can never be silently reinterpreted as an OPEN. See
[`../../HUMAN_DATA.md`](../../HUMAN_DATA.md).
