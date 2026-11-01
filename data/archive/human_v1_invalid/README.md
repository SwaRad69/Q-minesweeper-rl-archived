# Human Minesweeper demonstrations

`data/human/` is the **single canonical human demonstration dataset** for this
project (OPEN + FLAG + CHORD, 1440 actions, model-independent observable boards).
Persistent dataset of **human** Minesweeper games for later imitation learning
and other RL agents. It is independent from PPO training: recording only
observes play, it never drives the board, and it never touches the RL
environment.

> The older legacy-format JSONL store (originally `datasets/human_expert/`, now
> preserved at `archive/human_expert/`) is **legacy / non-canonical**.
> Its schema folds a chord into an OPEN action, so it must never be read as this
> dataset's 1440-action semantics. The canonical loaders detect and reject it
> explicitly. See [`../../HUMAN_DATA.md`](../../HUMAN_DATA.md).

## 1. Start the recorder

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
python -m minesweeper_rl.data.record_human_dataset --dataset data/human
```

The browser opens on the live site. There is no fixed game target: the recorder
keeps going and the dataset simply grows. Press `Ctrl+C` in the terminal to stop
(any game in progress is saved marked incomplete by default; add
`--drop-incomplete` to discard it).

Useful options: `--games N` (stop after N completed games), `--width/--height/
--mines/--difficulty` (defaults 30/16/99/expert), `--url`, `--profile`,
`--diagnostics`.

## 2. How to play

Play Minesweeper normally in the browser window:

- **left click** on a hidden cell → OPEN
- **right click** on a covered cell → FLAG (toggles the flag)
- **left click** on an opened number → CHORD

Each action is captured and stored as an observable
`state -> action -> next_state` transition. Invalid or unattributable clicks are
ignored (they do not create transitions). Hidden mine positions are never stored.

## 3. Where data is stored

```text
data/human/
├── games/
│   ├── game_000001.json
│   ├── game_000002.json
│   └── ...
├── metadata.jsonl          # one summary line per game (append-only)
├── dataset_manifest.json   # format identity, board, counters, next id
└── README.md
```

Each `games/game_XXXXXX.json` is an immutable record of one game: game-level
metadata plus the list of transitions. `metadata.jsonl` is append-only and the
small manifest is replaced atomically, so recording a new game never rewrites
earlier data.

Each transition stores: `step`, `timestamp`, `observation_before`,
`observation_after` (base64 `int8` boards), the action (`type`,
`action_type`, `x`, `y`, `cell_index`, `action_id`), `game_status`,
`terminated`, `truncated`, `newly_revealed`, `opened_cells` and `reward`.

`reward`, `newly_revealed` and `opened_cells` are **derived** fields (listed in
the game's `derived_fields`); they are retained for validation and offline
experiments but can be recomputed from the observable boards. The **raw**,
model-independent facts are the two board snapshots and the action. Each game
and `dataset_manifest.json` carries the frozen identity (`action_space`,
`action_types`, `num_actions`, `cells`, `observation_version`, `reward_spec`) so
a future CNN/GNN/quantum loader can interpret the file without importing code.
See [`../../HUMAN_DATA.md`](../../HUMAN_DATA.md).

### Action encoding

```text
action_type: 0 = OPEN, 1 = FLAG, 2 = CHORD
cell_index  = y * width + x
action_id   = action_type * (width * height) + cell_index
```

For the Expert board (480 cells): `0..479` OPEN, `480..959` FLAG,
`960..1439` CHORD — the same encoding as the flag-aware RL environment.

### Observation representation

`FLAGGED = -2`, `HIDDEN = -1`, `OPEN = 0..8`, row-major `[y, x]`, base64
`int8`. This is the flag-aware representation (`flag_aware_v1`) used by the
current environment.

## 4. How games are numbered

Game ids are `game_000001`, `game_000002`, ... (`game_{index:06d}`). The next
index lives in `dataset_manifest.json` and is reconciled against the files that
actually exist, so ids are unique and monotonic across restarts. Stopping and
restarting the recorder never reuses an id or rewrites an existing game.

## 5. Resume collection

Just run the same command again. The recorder reads the existing dataset,
reports the current game count, and continues from the next index.

```powershell
python -m minesweeper_rl.data.record_human_dataset --dataset data/human
```

Example: 5 games today, then 12 more next week → 17 games total, numbered
`game_000001` .. `game_000017`.

## 6. Validate the dataset

```powershell
python -m minesweeper_rl.data.validate_human_dataset --dataset data/human
```

It checks corrupted files, duplicate game ids, invalid action ids, invalid
coordinates, invalid state shapes, missing/inconsistent transitions, impossible
action types and incomplete games, and exits non-zero if any problem is found.

## 7. Inspect dataset statistics

The validation command prints a summary:

```text
status=OK games=17 (won=6 lost=9 incomplete=2) transitions=842 open=600 flag=180 chord=62 avg_length=49.5
```

- `games` / `won` / `lost` / `incomplete`
- `transitions` total, and the OPEN / FLAG / CHORD counts
- `avg_length` average transitions per game

## Reproducibility metadata

`dataset_manifest.json` records `format_version`, `recorder_version`,
`environment_version`, `observation_representation`, `action_encoding`, and the
board (`width`, `height`, `mines`, `difficulty`). Each game file repeats the
recorder/observation/action versions it was written with.

## Loading games in Python

```python
from minesweeper_rl.data import HumanDatasetStore, validate_dataset

store = HumanDatasetStore("data/human", width=30, height=16, mines=99)
for summary in store.iter_games():
    print(summary.game_id, summary.outcome, summary.total_steps)

game = store.load_game("game_000001")
for step in game["transitions"]:
    print(step["observation_before"], step["action"], step["observation_after"])

print(validate_dataset("data/human").format())
```

## Imitation training (behavioral cloning)

A read-only loader turns the raw games into `observation_t -> human_action_t`
samples (flag-aware board + 1440-class action id) and a standalone trainer
performs supervised imitation. Raw game files are never modified.

```powershell
# Train on every game currently recorded under data/human
python -m minesweeper_rl.experiment.imitation_train `
  --dataset data/human `
  --output experiments/cnn_imitation_expert_v1 `
  --epochs 20 --batch-size 64 --learning-rate 1e-3 --val-split 0.2 --seed 42 `
  --device auto
```

Useful options:

- `--game game_000001` (repeatable) select specific games
- `--winning-only` train only on won games
- `--exclude-incomplete` drop interrupted games
- `--resume` continue from `checkpoint_latest.pt` in the output directory

The trainer splits data **by game** (configurable ratio and deterministic seed),
prints the class distribution before training, and writes
`checkpoint_latest.pt`, `checkpoint_best.pt`, `run_config.json` and
`history.json` into the output directory. This is imitation pretraining only;
high action accuracy does not mean the policy can solve Minesweeper.

```python
from minesweeper_rl.data import HumanDemonstrationDataset, split_games

dataset = HumanDemonstrationDataset("data/human", exclude_incomplete=True)
train_ids, val_ids = split_games(dataset.game_ids(), val_fraction=0.2, seed=42)
print(dataset.statistics().format())
for sample in dataset.samples():
    print(sample.board.shape, sample.action_id, sample.action_type)
```
