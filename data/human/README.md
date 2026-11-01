# Human Minesweeper demonstrations (FLAGS-only, canonical)

`data/human/` is the **single canonical human demonstration dataset**. It is
**FLAGS-only**: OPEN + FLAG + CHORD, 1440 actions, model-independent observable
boards. Recording only observes play; it never drives the board and never touches
the RL environment.

This dataset was **reset from scratch** for the FLAGS-only architecture. Every
game recorded before the reset (the invalid `recorder_version: 1` set and the
`human_smoke_v3` smoke sets) was moved to [`../archive/`](../archive/); none of it
is used, merged, or converted.

> The older legacy-format JSONL store (`games.jsonl` / `transitions.jsonl` /
> `meta.json`, originally `datasets/human_expert/`, now
> `archive/human_expert/`) folds a chord into an OPEN action and is
> **non-canonical**. The canonical loaders detect and reject it explicitly. See
> [`../../HUMAN_DATA.md`](../../HUMAN_DATA.md).

## Frozen identity (from `dataset_manifest.json`)

```text
observation_representation : flag_aware_v1   (FLAGGED=-2, HIDDEN=-1, OPEN 0..8)
observation_version        : 1
observation_channels       : 11
action_space               : flags
action_encoding            : action_type*cells+cell
action_types               : {"0": "open", "1": "flag", "2": "chord"}
cells                      : 480
num_actions                : 1440
recorder_version           : 3
format_version             : 2
board                      : 30 x 16, 99 mines (expert)
```

## 1. Record a session

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
python -m minesweeper_rl.data.record_human_dataset --dataset data/human
```

The browser opens on the live site. There is no fixed game target: the recorder
keeps going and the dataset grows. Press `Ctrl+C` in the terminal to stop (a game
in progress is saved marked incomplete unless `--drop-incomplete` is given).
Useful options: `--games N`, `--width/--height/--mines/--difficulty`, `--url`,
`--profile`, `--diagnostics`.

`record_human_dataset` writes recorder-v3 games only. The legacy
`python -m minesweeper_rl.data.record_human` entry point (append-only JSONL,
chord-as-open) was removed; do not recreate it.

## 2. How to play

Play Minesweeper normally in the browser window:

- **left click** on a hidden cell -> OPEN
- **right click** on a covered cell -> FLAG (toggles the flag)
- **left click** on an opened number -> CHORD

Each action is stored as an observable `state -> action -> next_state` transition.
Invalid / unattributable clicks (stale-board, phantom, terminal stray clicks,
invalid first actions) are rejected and never create a transition or a game.
Hidden mine positions are never stored.

## 3. Layout

```text
data/human/
├── games/
│   ├── game_000001.json
│   └── ...
├── metadata.jsonl          # one summary line per game (append-only)
├── dataset_manifest.json   # format identity, board, counters, next id
└── README.md
```

Each transition stores `step`, `timestamp`, `observation_before`,
`observation_after` (base64 `int8`), the action (`type`, `action_type`, `x`, `y`,
`cell_index`, `action_id`), `game_status`, `terminated`, `truncated`,
`newly_revealed`, `opened_cells` and `reward`. `reward`, `newly_revealed` and
`opened_cells` are **derived**; the raw facts are the two board snapshots and the
action.

### Action encoding

```text
action_type : 0 = OPEN, 1 = FLAG, 2 = CHORD
cell_index  = y * width + x
action_id   = action_type * (width * height) + cell_index
```

For the Expert board (480 cells): `0..479` OPEN, `480..959` FLAG,
`960..1439` CHORD.

## 4. Validate

```powershell
python -m minesweeper_rl.data.validate_human_dataset --dataset data/human
```

It rejects corrupted/duplicate games, wrong `action_space` (`flags` only),
wrong `num_actions` (1440 only), unknown action ids/types, out-of-range
coordinates, non-11-channel/non-flag-aware observations, board discontinuities,
phantom continuations, pre-v3 recorder versions and terminal-semantics
violations, and exits non-zero on any problem.

## 5. Live smoke test (manual, before large-scale collection)

Before collecting a large expert set, record a few games and confirm:

1. the browser launches and a genuinely fresh all-hidden board is detected;
2. the first human action is captured as OPEN;
3. OPEN is captured; 4. FLAG is captured; 5. CHORD is captured (use a number whose
   flags match);
6. `observation_before` / `observation_after` reflect the board on either side;
7. win/loss is captured as a terminal transition;
8. no phantom games appear between play sessions;
9. `validate_human_dataset` reports `status=OK` with the expected
   open/flag/chord counts;
10. `HumanDemonstrationDataset("data/human")` loads the games with 1440-action
    ids.

A recorded JSON file alone is **not** proof: the semantic validator must confirm
the games. See [`../../HUMAN_DATA.md`](../../HUMAN_DATA.md) and
[`../../HUMAN_SMOKE_TEST.md`](../../HUMAN_SMOKE_TEST.md).

## 6. Imitation training

```powershell
python -m minesweeper_rl.experiment.imitation_train `
  --dataset data/human `
  --output experiments/cnn_bc_flags_v1 `
  --epochs 20 --batch-size 64 --learning-rate 1e-3 --val-split 0.2 --seed 42
```
