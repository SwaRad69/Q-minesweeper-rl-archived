# Live human smoke test (FLAGS-only)

Run this **before** any large-scale human collection. It verifies that the
recorder actually produces semantically valid FLAGS demonstrations from live
play. A merely recorded JSON file is **not** proof — the semantic validator must
confirm the games.

Do not automate this as a browser test; a human must play. The deterministic
recorder semantics are covered by the non-live test suite
(`tests/test_human_recorder_lifecycle.py`,
`tests/test_flags_only_architecture.py`).

## Preconditions

- `data/human/` is the canonical dataset
  (`recorder_version: 3`, `action_space: "flags"`, `num_actions: 1440`). New games append
  monotonically to existing records without overwriting.
- The repository's non-live tests pass:
  `python -m pytest -m "not live"`.

## Command

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
python -m minesweeper_rl.data.record_human_dataset --dataset data/human --diagnostics
```

`--diagnostics` logs every cell press with its accept/reject reason. Play several
Expert games (mix OPEN, FLAG and at least one CHORD), then `Ctrl+C`.

## Checks (all must hold)

1. **Browser launches** and the board is observed at all.
2. **Genuinely fresh board detected** — the first captured transition's
   `observation_before` is all `HIDDEN` (`-1`), no flags, no open cells.
3. **First human action is OPEN** — `action.action_type == 0` and it targets a
   hidden cell.
4. **OPEN captured** — opens its clicked cell (`observation_after` cell `>= 0`).
5. **FLAG captured** — `action.action_type == 1`, toggles the clicked cell between
   `HIDDEN` and `FLAGGED`.
6. **CHORD captured** — `action.action_type == 2` on an opened numbered cell, with
   `action_id == 2 * 480 + cell_index`.
7. **Observations before/after are correct** — each transition's
   `observation_after` equals the next transition's `observation_before`; the
   clicked cell's change matches the action type.
8. **Termination captured** — the final transition of a finished game has
   `terminated == true` and `game_status` `won`/`lost`; no clicks after it.
9. **No phantom games** — no game starts from a non-fresh board, and no game
   continues the previous game's final board.
10. **Dataset validation passes.**

## Validation

```powershell
python -m minesweeper_rl.data.validate_human_dataset --dataset data/human
```

Expected: `status=OK`, with non-zero `open`/`flag`/`chord` counts and the number
of games you played. Any error (e.g. `pre-fix recorder version`, `non-fresh
initial board`, `phantom continuation`, `CHORD no-op`, `terminal no-op`,
`action_id ... does not match`) means **stop** and inspect before collecting
more.

## Loader check

```python
from minesweeper_rl.data.human_demonstrations import HumanDemonstrationDataset

dataset = HumanDemonstrationDataset("data/human")
print(dataset.statistics().format())
assert dataset.num_actions == 1440
assert all(0 <= a < 1440 for a in dataset.action_ids())
```

## Then stop

Do **not** start large-scale collection yet. Inspect the first few games
(`data/human/games/*.json`, `metadata.jsonl`) and confirm the OPEN/FLAG/CHORD
counts and transitions look right. Only then collect a large expert dataset.
