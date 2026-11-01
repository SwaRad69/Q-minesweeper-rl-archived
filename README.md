# Non AI text - "

My intention for this project was to make benchmarks for minesweeper using classical and quantum reinforcement learning which can play minesweeper in minesweeper.online. 
I was supper hooked while I started this project when i was hooked playing minesweeper and i was desparate for a proeject. I lost interest afterwards. GGZ.

player profile : https://minesweeper.online/player/55188183







"



# 🚫 Minesweeper RL (Archived)

Reinforcement-learning platform that uses the real Minesweeper website as its
environment. Quantum and local-simulator components are intentionally not
included.

The project is **FLAGS-only**. The canonical action space is the human-like
**OPEN + FLAG + CHORD** space (1440 actions, 11 observation channels). It is the
only action space the main training path can run; there is no `--action-space`
option. The completed OPEN-only 480-action / 10-channel baseline is historical
and was moved to `archive/experiments/` (it can no longer be trained or resumed,
by design).

## CNN + PPO: OPEN + FLAG + CHORD

- Environment: `minesweeper.online` Expert, 30 x 16 board, 99 mines.
- Action space: 1440 discrete actions. `action_id = action_type * 480 + cell`,
  with `cell = y * width + x`; `action_type` is 0 = OPEN, 1 = FLAG, 2 = CHORD,
  so `0..479` = OPEN, `480..959` = FLAG, `960..1439` = CHORD.
- Observation: canonical board (`FLAGGED = -2`, `HIDDEN = -1`, `OPEN = 0..8`)
  one-hot encoded into 11 channels (channel 0 = flagged, channel 1 = hidden,
  channels 2..10 = open 0..8).
- Action mask: OPEN only on hidden cells; FLAG on covered cells (hidden or
  already flagged); CHORD only on an opened number whose flagged neighbours
  equal the number and that still has a hidden neighbour.
- Reward: `revealed(after) - revealed(before) + terminal` for OPEN and CHORD;
  FLAG is neutral (`0.0`). WIN = `+100 + progress`, LOSS = `-10 + progress`. No
  reward shaping and no hidden-mine information.
- Random initialization, no human demonstrations.

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
python -m minesweeper_rl.experiment.train --games 2000 `
  --run-dir experiments/cnn_ppo_flags_v1 --checkpoint-interval 5 --seed 42
```

Evaluation is FLAGS-only too (no `--action-space`):

```powershell
python -m minesweeper_rl.experiment.evaluate `
  --checkpoint experiments/cnn_ppo_flags_v1/checkpoint_latest.pt `
  --games 50 --run-dir experiments/cnn_ppo_flags_v1_eval
```

Run output is isolated under `experiments/<experiment_name>/` (config, metrics,
checkpoints, plots, logs); existing directories are never overwritten without
`--fresh` (which archives). See `experiments/README.md` for the naming
convention.

## Structure

```text
src/minesweeper_rl/
  browser/       Playwright lifecycle and page setup
  dom/           Minesweeper.com DOM parsing and board validation
  actions/       Mouse actions sent to the browser page
  environment.py Small environment facade (`observe`, `reveal`, `get_game_status`)
  config.py      Board and selector configuration
  logging_config.py Package logger access
  agents/        Reserved for future RL agent implementations
notebooks/       Reserved for exploratory notebooks
tests/           Unit and browser integration tests
```

The environment does only what has been verified against the site: it waits
for cells, observes a `numpy.int8` board with hidden cells as `-1` and revealed
numbers as `0..8`, opens one hidden cell with `reveal(x, y)` (or
`reveal((x, y))`), starts a fresh game with `reset()`, and takes one RL
transition with `step((x, y))`. Opening rejects cells that are not currently
hidden, performs a Playwright mouse click, and falls back to the verified
`mousedown`/`mouseup`/`click` event sequence if the click does not change the
board. It then waits for a meaningful board change and returns the new state.

### Step

`step(action)` takes an `(x, y)` pair, validates it against the current board,
performs the reveal, waits for the board to settle, and returns
`(next_state, reward, terminated, truncated, info)`. Reward precedence is
mine before win before safe reveal:

| Outcome | Reward | Terminated |
| --- | --- | --- |
| Mine revealed (`GameStatus.LOST`) | `-10.0` | yes |
| Board solved (`GameStatus.WON`) | `+100.0` | yes |
| Any other safe reveal (`RUNNING`) | `+1.0` | no |

`truncated` is always `False` for now because there is no step or time limit.
`info` contains `action`, `previous_value`, `new_value`, `game_status`,
`hit_mine`, `newly_revealed`, `opened_cells`, and `elapsed_ms` for debugging.
Stepping after the episode is over raises `InvalidActionError` until `reset()`
is called.

### Reset

`reset()` clicks the verified restart control `#top_area_face`. On the live
site that resets an untouched board in place but navigates to a new
`/game/<id>` once play has started, so reset waits for a stable, all-hidden,
`READY` board rather than using any fixed delay. It then makes one
deterministic controlled first reveal at the board centre (override with
`EnvironmentConfig.first_move`) and returns the validated post-move board.

The site guarantees the first reveal is never a mine, so reset never starts the
agent on an all-hidden board and never loses to the opening gamble. If that
guarantee ever fails, reset restarts and retries the same cell up to
`first_move_attempts` times. Configure `new_game_selector`,
`new_game_timeout_ms`, `first_move`, and `first_move_attempts` through
`EnvironmentConfig`.

`MinesweeperEnvironment.observe()` captures the relevant cell DOM data and
requires two identical, complete snapshots before returning a board. This
prevents a board being returned while the site is still inserting or updating
cells. Configure `observation_timeout_ms`, `observation_poll_ms`,
`action_timeout_ms`, `action_poll_ms`, and `diagnostics` through
`EnvironmentConfig`; diagnostics use the `minesweeper_rl.environment` logger
at DEBUG level.

### Game status

`MinesweeperEnvironment.get_game_status()` returns a `GameStatus` derived only
from observable DOM state, never from reward. The indicators were discovered on
the live site:

| Status | Observable DOM signal |
| --- | --- |
| `WON` | `#top_area_face` has class `hd_top-area-face-win` (or `nnhd_...`) |
| `LOST` | `#top_area_face` has class `hd_top-area-face-lose` (or `nnhd_...`) |
| `READY` | face is `*-top-area-face-unpressed`/`-pressed` and no cell is opened |
| `RUNNING` | face is unpressed/pressed and at least one cell is opened |

The face classes are read only after a stable cell snapshot, because a win or
loss always changes the revealed cells. A no-guessing game uses the `nnhd_`
prefix; the classifiers match either prefix.

## Install and test

```powershell
python -m pip install -e ".[test]"
python -m playwright install chromium
python -m pytest -m "not live"
```

Playwright uses its installed Chromium executable. If the browser binary has
not been installed yet, run `python -m playwright install chromium` once.

The first test can be run directly with:

```powershell
python -m pytest tests/test_environment.py
```

## Browser controller manual test

The controller launches one headed Chromium instance using a persistent
profile, navigates to the configured URL, and checks for at least one `.cell`
element. It does not log in, click, parse, or otherwise interact with the
game. The browser remains visible for five seconds by default.

Set the URL and run the standalone manual test:

```powershell
$env:MINESWEEPER_URL = "https://your-game-url"
python test_browser.py
```

The profile defaults to `browser_profile/` in the project directory. Override
it with `MINESWEEPER_PROFILE` or `--profile`; override the inspection delay
with `BROWSER_INSPECTION_DELAY` or `--delay`. If manual login is needed, use a
larger `--board-timeout` (milliseconds) so there is time to complete it in the
headed browser, for example `python test_browser.py --board-timeout 120000`.

If login is required, log in manually in the headed browser. The persistent
profile may retain that session for later runs where the site permits it.

## Live tests

Set the game URL inspected during manual testing. `https://minesweeper.online/start/3`
starts a fresh expert game. The observation tests only read the DOM; the reveal
test clicks one hidden cell. Enable `MINESWEEPER_DIAGNOSTICS=1` for DEBUG logs.

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
$env:MINESWEEPER_DIAGNOSTICS = "1"
python -m pytest -m live -s
```

The live parser test `tests/test_live_board_matrix.py` prints the parsed
16-by-30 matrix and can be run directly with:

```powershell
python -m pytest tests/test_live_board_matrix.py -s
```

## Random agent

The random baseline resets the environment, reveals uniformly random hidden
cells until the game ends, and repeats for a requested number of games. It
proves the environment can play complete games autonomously; it is not a
learning algorithm.

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
python -m minesweeper_rl.agents.random_agent --games 20 --seed 42
```

Each episode is logged with its number, win/loss, steps, total reward, cells
revealed, whether a mine was hit, and duration, followed by a summary with win
rate and averages. Use `--url`/`--profile` to override the environment
variables, `--seed` for reproducible action choices, and `--diagnostics` for
environment DEBUG logging.

## Stress test

`minesweeper_rl.stress` plays many random games while recording reset, step,
DOM-read, parse and click latencies, then runs targeted checks for invalid
actions, board parsing, DOM synchronisation, page-reload recovery, unexpected
boardless pages, and the win path.

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
python -m minesweeper_rl.stress --games 40 --seed 2026
```

Latency telemetry is opt-in: the environment only records timings when
`MinesweeperEnvironment.metrics` is set (the stress harness does this), so
normal use is unaffected. Use `--no-win-phase` to skip the small-board win
check.

## Browser controller use

```python
from minesweeper_rl.browser import BrowserController

browser = BrowserController("YOUR_GAME_URL", "browser_profile")
try:
    browser.start()
    from minesweeper_rl.environment import MinesweeperEnvironment
    state = MinesweeperEnvironment(browser).observe()
    print(state.shape)
finally:
    browser.shutdown()
```

## Human demonstration dataset

Record human Expert games into the persistent, append-only FLAGS-only dataset
(recorder v3, 1440 actions) for imitation learning. See
`data/human/README.md` and `HUMAN_DATA.md` for details. The old datasets were
archived under `data/archive/`; `data/human/` is a fresh empty dataset.

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
python -m minesweeper_rl.data.record_human_dataset --dataset data/human   # play, Ctrl+C to stop
python -m minesweeper_rl.data.validate_human_dataset --dataset data/human  # validate + stats
```

Behavioral-cloning pretraining on the recorded games (read-only, split by game):

```powershell
python -m minesweeper_rl.experiment.imitation_train `
  --dataset data/human --output experiments/cnn_bc_flags_v1 `
  --epochs 20 --batch-size 64 --learning-rate 1e-3 --val-split 0.2 --seed 42
```

Human-pretrained PPO initializes the flag-aware CNN/actor from the imitation
checkpoint (the value head is reset to zero) and records the choice permanently
in `run_config.json`. See `experiments/cnn_ppo_flags_human_v1/README.md`.

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
python -m minesweeper_rl.experiment.train --games 2000 `
  --training-mode human_pretrained `
  --pretrained-policy experiments/cnn_bc_flags_v1/checkpoint_latest.pt `
  --run-dir experiments/cnn_ppo_flags_human_v1 `
  --checkpoint-interval 10 --seed 42 --device auto
```
