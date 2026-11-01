# Human Data Collection

A comprehensive, operational guide to recording, validating, and managing human demonstration trajectories on `minesweeper.online` Expert for the Minesweeper RL project.

---

## 1. Purpose

This human demonstration dataset is the single canonical, model-independent repository of expert human Minesweeper gameplay. Its objectives are:

1. **Behavioral Cloning (Imitation Learning)**: Supervised pretraining of neural network policy heads via `minesweeper_rl.experiment.imitation_train`, teaching the agent spatial deduction, safe cell openings, flagging, and chording before reinforcement learning begins.
2. **PPO Warm-Starting (`human_pretrained`)**: Initializing the convolutional actor backbone of the Proximal Policy Optimization agent (`minesweeper_rl.experiment.train --training-mode human_pretrained`) from human imitation checkpoints, bypassing hundreds of thousands of blind random exploration steps.
3. **Pure Markov Decision Process (MDP) Storage**: Capturing raw, observable-only state-action-next_state transitions without embedding neural-network-specific assumptions (such as one-hot CNN tensors, GNN graphs, or quantum state encodings). Downstream adapters derive their required tensor representations on demand without altering the raw files.
4. **Authentic Human Gameplay**: Preserving the natural, full action vocabulary of real human players (revealing hidden cells, toggling flags, and chording opened numbers) while enforcing strict game lifecycle boundaries.

---

## 2. Current Collection Architecture

The collection pipeline operates as a passive observer over a headed Playwright browser session:

```text
Browser (Playwright Chromium persistent session on minesweeper.online/start/3)
    │
    ▼
Injected Capture Listener (window capture-phase pointerdown listener snapshotting cell & face)
    │
    ▼
Game Boundary Detection (_handle_idle_board checks for verified 16x30 all-hidden board)
    │
    ▼
Continuous Board Observation (MinesweeperEnvironment.observe() with 2-snapshot fingerprint matching)
    │
    ▼
Action Event Drain & Pre-Validation (drain_events() checks coordinates, primary/secondary buttons)
    │
    ▼
Quiescence & Change Polling (_wait_for_stable_change verifies cell change & 0.3s cascade settling)
    │
    ▼
Action Effect Confirmation (_confirm_action verifies reveal opened cell, flag toggled, or chord reacted)
    │
    ▼
Transition Capture (record_step builds canonical transition, encodes board, computes derived reward)
    │
    ▼
Game Termination Check (detects GameStatus.WON, GameStatus.LOST, or mid-game ABANDONED)
    │
    ▼
Dataset Storage (HumanDatasetStore commits atomic game JSON, metadata.jsonl, and manifest)
```

### Component Roles
- **Playwright Controller** (`minesweeper_rl.browser.controller`): Manages the headed Chromium process with a persistent user profile (`browser_profile/`) preserving session preferences and cookies.
- **Capture Listener** (`_CLICK_CAPTURE_SCRIPT` in `record_human.py`): Injected at `window` creation via `page.add_init_script`. Intercepts `pointerdown` events in the capture phase before the website's scripts execute. Captures cell dataset coordinates `(x, y)`, mouse button, cell class list, and `#top_area_face` class snapshot.
- **Human Observer** (`HumanTrajectoryRecorder` in `record_human.py`): Runs the recording state machine (`WAITING → READY → RECORDING → RESYNCING`). Reconciles the browser events against the authoritative `_stable_board`.
- **Dataset Sink** (`PersistentHumanRecorder` in `human_dataset.py`): Drops in as a trajectory sink, buffers transitions for the active game, validates action semantics and terminal status, and dispatches writes to storage.
- **Storage Layer** (`HumanDatasetStore` and `storage.py`): Implements atomic file writing with exponential-backoff retry logic on Windows file locks, monotonic ID generation, and metadata indexing.

---

## 3. Supported Human Actions

The system faithfully records all three authentic human Minesweeper operations:

| Action | Mouse Input | Code (`action_type`) | Action ID Range (Expert) | Pre-condition on Target `(x, y)` | Post-condition Effect |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **OPEN** | Left-click (Button 0) | `0` | `0 .. 479` | `board_before[y, x] == -1` (Hidden) | Target cell becomes revealed (`0 .. 8`), or game ends in `LOST` (detonated mine). |
| **FLAG** | Right-click (Button 2) | `1` | `480 .. 959` | `board_before[y, x] == -1` (Hidden) | Target cell toggles from Hidden (`-1`) to Flagged (`-2`). |
| **UNFLAG** | Right-click (Button 2) | `1` | `480 .. 959` | `board_before[y, x] == -2` (Flagged) | Target cell toggles from Flagged (`-2`) to Hidden (`-1`). |
| **CHORD** | Left-click (Button 0) | `2` | `960 .. 1439` | `1 <= board_before[y, x] <= 8` (Opened Number) | At least one adjacent covered neighbor opens, or game ends in `LOST`. |

### Actions Handled Specially
- **First Move of a Game**: MUST be an `OPEN` action targeting a hidden cell (`board_before[y, x] == -1`). If a player attempts to place a `FLAG` or perform a `CHORD` as the very first move of a fresh game, the event is rejected and logged.
- **Left-Click on Flagged Cell**: Ignored and rejected. Left-clicking a flag does nothing on the website and is never recorded as an action.
- **Middle-Click (Button 1) & Other Buttons**: Logged and counted under `report.other_clicks` and `game.other_clicks`, but ignored by the MDP trajectory recorder.
- **Left-Click on Numbered Cell with Insufficient Flags**: If the player clicks a number whose adjacent flags do not match its count, the board does not change; the recorder flags this as a "chord changed nothing" and rejects it without recording an invalid transition.

---

## 4. Observation Format

Observations represent the exact board state visible to the player at decision time.

### Board Dimensions & Values
- **Grid Size**: $16 \text{ rows} \times 30 \text{ columns} = 480 \text{ cells}$ (Standard Expert difficulty).
- **Internal Array Representation**: 2D `numpy.int8` array of shape `(16, 30)` in row-major `[y, x]` order.
- **Canonical Cell Encoding**:
  - `-2` (`FLAGGED`): Covered cell marked with a player flag (`hd_flag` / `hd_closed_flag`).
  - `-1` (`HIDDEN`): Covered, unflagged cell (`hd_closed`).
  - `0 .. 8` (`OPEN`): Revealed cell displaying its exact adjacent mine count (`hd_opened` + `hd_type0` through `hd_type8`).
- **Post-Mortem Mine Normalization**: On game termination (`LOST`), the website displays exploded mines (`hd_type10`), unexploded mines (`hd_type11`), and misflagged mines (`hd_type12`). The parser normalizes all terminal mine artifacts to `HIDDEN` (`-1`). This strictly guarantees that hidden mine positions are **never** leaked into observations.

### Serialized Storage Encoding
In saved game JSON files, each board observation is serialized as a base64-encoded string representing the raw, C-contiguous `int8` byte stream (exactly 480 bytes):

```python
# Encoding
encoded = base64.b64encode(board.astype(np.int8).tobytes(order="C")).decode("ascii")

# Decoding
board = np.frombuffer(base64.b64decode(encoded), dtype=np.int8).reshape(16, 30)
```

- **Identity Metadata**:
  - `observation_representation`: `"flag_aware_v1"`
  - `observation_version`: `1`

---

## 5. Action Format

Actions utilize a unified, discrete compound action encoding mapping every coordinate and action type into a single integer:

### Encoding Formula
$$\text{cell\_index} = y \times \text{width} + x$$
$$\text{action\_id} = \text{action\_type} \times (\text{width} \times \text{height}) + \text{cell\_index}$$

Where for Expert:
- $\text{width} = 30$, $\text{height} = 16$, $\text{cells} = 480$.
- $\text{action\_type} \in \{0: \text{OPEN}, 1: \text{FLAG}, 2: \text{CHORD}\}$.
- $\text{num\_actions} = 3 \times 480 = 1440$.

### Action Sub-ranges
- **OPEN** (`0 .. 479`): `action_id = 0 * 480 + cell_index = cell_index`
- **FLAG** (`480 .. 959`): `action_id = 1 * 480 + cell_index = 480 + cell_index`
- **CHORD** (`960 .. 1439`): `action_id = 2 * 480 + cell_index = 960 + cell_index`

### Coordinate Conversion
The DOM elements provide `data-x` and `data-y` attributes (`x` along width $0 .. 29$, `y` along height $0 .. 15$).
- Browser $(x, y) \leftrightarrow$ Dataset $(x, y)$ are identical.
- Example 1: OPEN cell at $(x=15, y=8) \rightarrow \text{cell\_index} = 8 \times 30 + 15 = 255 \rightarrow \text{action\_id} = 0 \times 480 + 255 = \mathbf{255}$.
- Example 2: FLAG cell at $(x=15, y=8) \rightarrow \text{cell\_index} = 255 \rightarrow \text{action\_id} = 1 \times 480 + 255 = \mathbf{735}$.
- Example 3: CHORD cell at $(x=15, y=8) \rightarrow \text{cell\_index} = 255 \rightarrow \text{action\_id} = 2 \times 480 + 255 = \mathbf{1215}$.

---

## 6. Transition Format

Each entry in a game's `transitions` array records a complete state-action-next_state step:

```json
{
  "step": 0,
  "timestamp": 1789644627.0682545,
  "observation_before": "////////////////////////////////////////////////////////////////... (base64 int8)",
  "observation_after": "AAH/////////////////////////////////////AAH/////////////////... (base64 int8)",
  "action": {
    "type": "open",
    "action_type": 0,
    "x": 0,
    "y": 0,
    "cell_index": 0,
    "action_id": 0
  },
  "game_status": "running",
  "terminated": false,
  "truncated": false,
  "reward": 8.0,
  "newly_revealed": 8,
  "opened_cells": 8
}
```

### Transition Schema Specification
- `step` (int): 0-indexed step counter within the game.
- `timestamp` (float): High-resolution Unix epoch timestamp (seconds) when step occurred.
- `observation_before` (string): Base64-encoded board state before the action.
- `observation_after` (string): Base64-encoded board state after the action and cascade settled.
- `action` (object):
  - `type` (string): `"open"`, `"flag"`, or `"chord"`.
  - `action_type` (int): `0` (open), `1` (flag), `2` (chord).
  - `x` (int): Grid column (`0 .. 29`).
  - `y` (int): Grid row (`0 .. 15`).
  - `cell_index` (int): Flat cell index (`0 .. 479`).
  - `action_id` (int): Compound discrete action ID (`0 .. 1439`).
- `game_status` (string): Observable status post-action: `"running"`, `"won"`, or `"lost"`.
- `terminated` (bool): `true` if this move caused a win or loss; otherwise `false`.
- `truncated` (bool): Always `false` during human recording.
- `reward` (float): Derived transition reward under `reward_spec: "reveal_progress_terminal_v1"`.
  - Flag or Unflag: `0.0`.
  - Open / Chord (non-terminal): $+1.0 \times \text{newly\_revealed}$.
  - Win: $\text{progress} + 100.0$.
  - Loss: $\text{progress} - 10.0$.
- `newly_revealed` (int): Number of newly revealed numbers ($0 .. 8$) produced by this move.
- `opened_cells` (int): Cumulative count of opened numbers on `observation_after`.

---

## 7. Game Lifecycle

The recording system executes an explicit state machine:

```text
┌──────────────────────────────────────────────┐
│            WAITING_FOR_FRESH_BOARD           │
└──────────────────────┬───────────────────────┘
                       │ Verified all-hidden board (480 cells == -1, face != terminal)
                       ▼
┌──────────────────────────────────────────────┐
│                     READY                    │
└──────────────────────┬───────────────────────┘
                       │ First action (MUST be OPEN on a hidden cell)
                       ▼
┌──────────────────────────────────────────────┐
│                   RECORDING                  │
└──────┬───────────────────────┬───────────────┘
       │                       │
       │ Won / Lost            │ In-progress reset / Ctrl+C / desync
       ▼                       ▼
┌──────────────┐       ┌────────────────────────┐
│   TERMINAL   │       │ ABANDONED (incomplete) │
└──────┬───────┘       └───────────┬────────────┘
       │                           │
       └───────────────┬───────────┘
                       ▼
┌──────────────────────────────────────────────┐
│                   RESYNCING                  │
└──────────────────────┬───────────────────────┘
                       │ Player clicks smiley face -> New fresh board appears
                       ▼
               (Back to READY)
```

### Stage Details
1. **WAITING_FOR_FRESH_BOARD**: Initial state upon launch or navigation. The recorder observes the DOM until a completely unplayed board is confirmed (`_is_fresh_board()`).
2. **READY**: A fresh board is verified. The recorder waits for the human's opening move. Any attempt to flag or chord is rejected.
3. **RECORDING**: The game ID is allocated (`game_XXXXXX`). The recorder tracks continuous stable board states, processes human clicks, validates quiescence, and appends transitions.
4. **TERMINAL**: When a move hits a mine or solves the board, `terminated` becomes `true`. The final step is saved, `end_game()` writes the file, and the recorder transitions to `RESYNCING`.
5. **RESYNCING**: The finished board remains on screen. Clicks on the dead board are tagged `ignored_terminal` and ignored. The recorder waits until the user clicks the restart face `#top_area_face`, transitioning back to `READY`.

---

## 8. Abandoned / Invalid Games

### What "ABANDONED" Means
An "abandoned" game is a recording session that ended for a verified lifecycle
reason before a valid terminal transition was captured. Time elapsed is never
such a reason. Files use `"result": "abandoned"`, `"outcome": "incomplete"`,
`"complete": false`, plus a machine-readable `"termination_reason"`.

### When a Game Receives ABANDONED Status
1. **User Restarts Mid-Game**: The player clicks the smiley face `#top_area_face` (or presses F2/space) while a game is underway with $\ge 1$ steps recorded.
2. **Keyboard Interrupt (Ctrl+C)**: The operator stops the recording session in PowerShell while a game is in progress.
3. **Page Closure**: The active Playwright page is positively reported closed.
4. **Missed Terminal Transition**: If `#top_area_face` turns to win/loss but no corresponding terminal click transition can be recovered, the partial recording is invalid and is committed as abandoned rather than fabricating an action.
5. **Verified Capture Gap**: A stable board discontinuity cannot be attributed to the captured pending press, or transition validation proves the recorded chain invalid.

Temporary board/status/event-queue failures remain `UNKNOWN` and are retried.
After the action wait expires, the captured press remains pending and is
recovered from a later stable board or the next press's atomic pre-click
snapshot. Long games, long think-time between actions, slow rendering, and a
temporary file/DOM/Playwright failure do not produce `ABANDONED`.

### Did Genuine Losses Previously Become ABANDONED?
- **Root Cause of Past Issue**: In earlier versions (recorder v1/v2), fast browser renders could update `#top_area_face` to `-lose` before Python drained the pointer event queue. When Python inspected `status_before`, it saw `LOST` and concluded the click was a stray click on an already-dead board, prematurely abandoning the game.
- **How It Is Fixed in Current Code (Recorder v3)**:
  1. The browser capture script (`_CLICK_CAPTURE_SCRIPT`) snapshots `faceClasses` inside the browser event listener at the exact microsecond of `pointerdown`.
  2. `_pre_action_status(event)` checks this snapshot: if the face was still `RUNNING` when the user pressed the mouse, the action is recognized as the move that *caused* the loss, even if the DOM changed before Python drained the queue.
  3. `mine_grace` (1.0s window) allows the DOM to settle post-mine detonation.
  4. Losing chords (where the canonical board does not change because detonated mines normalize to `-1`) are explicitly recognized as terminal losses.
  5. The active dataset contains **42 verified losses and 0 invalid games**, confirming genuine losses are cleanly captured.

### Handling and Inspection of Incomplete Games
- **Zero-Step Discard**: A game with 0 steps recorded is discarded in memory (`discard_game()`) and never written to disk.
- **Persistence**: By default (`store_incomplete=True`), partial games ($\ge 1$ steps) are written to `games/game_XXXXXX.json` and logged in `metadata.jsonl`.
- **Filtering**: If you run with `--drop-incomplete`, abandoned games are discarded without persisting.
- **Training Safety**: `HumanDemonstrationDataset` validates all games upon loading. Valid incomplete games can provide safe step demonstrations, but downstream training loaders can filter them by checking `complete == True` or `outcome == "won"`.

---

## 9. Data Integrity Protections

The repository enforces multi-layered defense mechanisms against corrupted or invalid data:

1. **Capture-Phase Pointerdown Interception**: Injected script hooks `window.addEventListener('pointerdown', handler, true)`. Capturing pointerdown intercepts human intent prior to website script execution.
2. **Trusted Event Filtering**: `event.isTrusted === true` is strictly verified. Synthetic browser events generated by scripts or extensions are rejected.
3. **Two-Snapshot Stability Matching**: `MinesweeperEnvironment.observe()` requires two consecutive atomic DOM snapshots to match their coordinate-and-class fingerprint before returning a board.
4. **Cascade Quiescence Window**: After an action reveals cells, `_wait_for_stable_change()` polls until the board remains unchanged for `quiescence_time` (0.3s). Large multi-stage cascades are never sampled mid-animation.
5. **Fresh Board Gating**: A new game can only be instantiated when `_is_fresh_board()` confirms all 480 cells are `HIDDEN` (`-1`) and the face is non-terminal. Mid-game fragments and phantom games are impossible.
6. **Phantom Continuation Prevention**: The validator ensures the first state of game $N+1$ does not equal the final state of game $N$.
7. **Transition Continuity Check**: `board_before` of step $t$ must byte-for-byte match `board_after` of step $t-1$.
8. **Impossible Transition Rejection**: Rules strictly reject impossible transitions:
   - Safe OPEN ending in LOST.
   - OPEN revealing an already-open cell.
   - Flag toggling an open cell.
   - Non-terminal transition with no board change.
9. **Atomic Storage & File Lock Retries**:
   - Every game JSON is written to a unique temporary file (`.tmp`) and replaced atomically using `os.replace`.
   - On Windows, file-sync engines (OneDrive, antivirus scanners) can hold transient locks. `_replace_with_retry` and `_append_text` implement up to 6 retry attempts with exponential backoff (`winerror` 5, 32, 33).
10. **Reconciled Monotonic Numbering**: Monotonic game IDs (`game_000001`, `game_000002`) are reconciled on startup against disk files, guaranteeing restarts never overwrite existing games.
11. **Hidden Mine Isolation**: `FORBIDDEN_LAYOUT_KEYS` (`mine_positions`, `hidden_mines`, `solution`, etc.) are blocked from appearing in any game or transition record.

---

## 10. Directory Structure

The canonical dataset directory structure is organized as follows:

```text
rl minesweeper/
├── data/
│   ├── human/
│   │   ├── dataset_manifest.json   # Identity, board config, total counters, next id
│   │   ├── metadata.jsonl          # Fast append-only summary index (1 line per game)
│   │   ├── README.md               # Dataset documentation
│   │   ├── COLLECTION_STATUS.md    # Current collection status note
│   │   └── games/                  # Immutable per-game records
│   │       ├── game_000001.json
│   │       ├── game_000002.json
│   │       └── ...
│   └── archive/                    # Archived invalid/smoke datasets (read-only)
│       ├── human_v1_invalid/
│       ├── human_smoke_v3/
│       └── human_smoke_v3_interrupted/
```

---

## 11. Installation & Prerequisites

Ensure the package and browser automation binaries are installed:

```powershell
# 1. Install package in editable mode with test dependencies
python -m pip install -e ".[test]"

# 2. Install Chromium browser binary for Playwright
python -m playwright install chromium

# 3. Verify non-live test suite passes
python -m pytest -m "not live"
```

---

## 12. Starting the Recorder

### Standard Command
To start recording human demonstrations into `data/human/`:

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
python -m minesweeper_rl.data.record_human_dataset --dataset data/human
```

### With Diagnostics Logging (Recommended for First Session)
Diagnostics mode prints every pointerdown event, coordinate, button, and acceptance/rejection reason:

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
python -m minesweeper_rl.data.record_human_dataset --dataset data/human --diagnostics
```

### Useful CLI Flags
- `--dataset <dir>`: Path to persistent dataset (default: `data/human`).
- `--games <N>`: Stop automatically after $N$ completed games (default: `0` = record indefinitely until `Ctrl+C`).
- `--drop-incomplete`: Discard interrupted/abandoned games instead of saving them marked `complete: false`.
- `--quiescence-time <sec>`: Settling duration for board changes (default: `0.3`).
- `--action-timeout <sec>`: Initial wait for a stable board reaction after a click (default: `5.0`); expiry keeps the press pending for recovery and does not abandon the game.
- `--profile <dir>`: Browser profile directory (default: `browser_profile`).

---

## 13. How I Should Play During Collection

To ensure your collected demonstrations are clean, valid, and rich in learning signals:

### Before Starting
1. When the browser launches, check the board. If the board is already partially played or terminal from a previous run, **click the smiley face (`#top_area_face`)** once to reset it to an all-hidden, fresh board.
2. The terminal will log: `fresh board detected; ready to record a new game`.

### Playing the Game
1. **First Move**: Left-click on any hidden cell to start the game. The first action **must** be an OPEN.
2. **Revealing**: Left-click hidden cells to open them.
3. **Flagging**: Right-click covered cells to place or remove flags.
4. **Chording**: Left-click an opened numbered cell when you have flagged the exact number of neighboring mines.
5. **Cascade Reveals**: During large cascade openings, allow a brief pause (approx. 0.3s) for the animation and settling to finish before your next click.

### Game Conclusion
1. **When You Win**: The win face appears. The terminal logs `Game X ... Result: WON`.
2. **When You Lose**: The lose face appears. The terminal logs `Game X ... Result: LOST`.
3. **Starting the Next Game**: Click the smiley face `#top_area_face` to generate a fresh board. The recorder detects the reset, logs ready status, and waits for your next opening click.
4. **Stopping the Recorder**: Press `Ctrl+C` in your PowerShell terminal between games or after finishing a game.

### What NOT to Do
- **Do NOT** right-click (flag) as your first move. Move 1 must be OPEN.
- **Do NOT** click the board after game over before clicking the smiley face. Clicks on a terminal board are ignored.
- **Do NOT** click frantically while a cascade is actively rendering.
- **Do NOT** use synthetic mouse emulation or third-party auto-solvers.

---

## 14. Recommended Collection Protocol

To compile a high-quality demonstration dataset for imitation and RL pretraining:

1. **Aim for Complete Games**: Play each game with genuine intent to solve it. Avoid deliberately abandoning games unless stuck.
2. **Genuine Losses are Valuable**: Do not restart prematurely when a guess is required. Losing on an unavoidable 50/50 guess provides critical negative training signal (distinguishing safe deductions from mine hits).
3. **Balanced Action Mix**: Use flags and chording naturally as you would in real play. Chords provide dense multi-cell reveal examples.
4. **Consistent Difficulty**: Maintain standard Expert settings (30x16, 99 mines) on `https://minesweeper.online/start/3`.
5. **Session Sizes**: Collect in batches of 10 to 50 games. Stop with `Ctrl+C`, run the validator, and review metrics before proceeding.

---

## 15. Post-Collection Validation

Immediately after stopping a recording session, run the automated dataset validator:

```powershell
python -m minesweeper_rl.data.validate_human_dataset --dataset data/human
```

### Successful Validation Output
When all games meet structural and semantic requirements, the command exits with code `0` and prints:

```text
status=OK games=50 (won=0 lost=42 incomplete=8) valid=50 invalid=0 transitions=1156 (valid=1156 invalid=0) open=281 flag=338 chord=537 avg_length=23.1
```

### Key Metrics to Check
- `status=OK`: No structural or semantic errors found.
- `invalid=0`: Zero invalid games detected.
- `transitions (valid=X invalid=0)`: All transitions conform to action/observation schemas.
- `open`, `flag`, `chord`: Confirms all three action types were recorded.

---

## 16. Dataset Quality Checklist

Follow this checklist after every recording session:

- [ ] Command completed cleanly (`status=OK`).
- [ ] `games_invalid` is exactly `0`.
- [ ] `transitions_invalid` is exactly `0`.
- [ ] `open`, `flag`, and `chord` counts are all non-zero.
- [ ] Won and Lost games have `complete == true`.
- [ ] Abandoned games have `complete == false`.
- [ ] No phantom continuations reported.
- [ ] No `pre-fix recorder version` warnings.
- [ ] Dataset manifest counters match actual file counts on disk.
- [ ] `HumanDemonstrationDataset("data/human")` loads without error.

---

## 17. Troubleshooting

### Problem 1: "first action must be OPEN"
- **Symptom**: The recorder logs `rejection: first action must be OPEN`.
- **Cause**: The player right-clicked (flagged) or attempted to chord as move 1 on a fresh board.
- **Resolution**: Always perform a left-click on a hidden cell to initiate a game.

### Problem 2: "no verified fresh board yet" / "waiting"
- **Symptom**: Clicks are ignored and logged with reason `no verified fresh board yet`.
- **Cause**: The board currently has revealed cells or flags from a previous session, or the page is mid-navigation.
- **Resolution**: Click `#top_area_face` (the smiley face) in the browser to reset to an all-hidden board.

### Problem 3: A Loss Was Recorded as ABANDONED (Historic Bug)
- **Symptom**: A game where a mine was detonated appears in `metadata.jsonl` as `result: "abandoned"`.
- **Status in Current Code**: **FIXED** in recorder v3.
- **Verification**: `_pre_action_status` inspects the pointerdown face snapshot. If you encounter this, inspect `games/game_XXXXXX.json`. Verify that `recorder_version == 3`. If you are running an outdated recorder process, restart using `record_human_dataset`.

### Problem 4: OneDrive or Antivirus File Lock Warning
- **Symptom**: Terminal logs: `could not open ... for append (attempt 1/6): [WinError 32] ... retrying`.
- **Status**: The built-in exponential-backoff retry automatically handles transient file locks. If 6 attempts fail, pause OneDrive synchronization on the workspace directory or move the dataset folder outside OneDrive.

---

## 18. Example Successful Recording

A complete transcript of a successful recording session:

```powershell
PS C:\Users\SwaRa\OneDrive\Desktop\rl minesweeper> $env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
PS C:\Users\SwaRa\OneDrive\Desktop\rl minesweeper> python -m minesweeper_rl.data.record_human_dataset --dataset data/human --diagnostics
========================================
Human Minesweeper demonstration recorder
========================================
Dataset: data\human
Board: 30x16 mines=99 difficulty=expert
Existing games: 50
Next game index: 51
Play Minesweeper in the browser window; press Ctrl+C to stop.
Recorder diagnostics enabled: every cell press will be logged with its target, coordinates, isTrusted flag and accept/reject reason.
Human recording started; play Minesweeper in the browser window. Target: 9223372036854775807 completed games.
fresh board detected; ready to record a new game
recorder event | type=pointerdown trusted=True button=0 data=(15,8) action=OPEN before=hidden after=open:0 accepted=True reason=recorded
recorder event | type=pointerdown trusted=True button=2 data=(14,7) action=FLAG before=hidden after=flagged accepted=True reason=recorded
recorder event | type=pointerdown trusted=True button=0 data=(13,7) action=CHORD before=open:1 after=open:2 accepted=True reason=recorded
recorder event | type=pointerdown trusted=True button=0 data=(12,6) action=OPEN before=hidden after=hidden accepted=True reason=recorded
Game 51
Source: human
Result: LOST
Steps: 4
Duration: 12.4 seconds
Transitions recorded: 4

Dataset total:
Games: 51
Transitions: 1160
Wins: 0
Losses: 43
Abandoned: 8
^C
========================================
Recording stopped
========================================
Dataset: data\human
games=51 transitions=1160 wins=0 losses=43 abandoned=8
```

---

## 19. Before Collecting Real Data

Complete this final smoke test before conducting a dedicated data collection session:

1. **Verify Python Environment & Test Suite**:
   ```powershell
   python -m pytest -m "not live"
   ```
   *Expected: All 457 unit tests pass.*

2. **Verify Existing Dataset Integrity**:
   ```powershell
   python -m minesweeper_rl.data.validate_human_dataset --dataset data/human
   ```
   *Expected: `status=OK` with `valid=50, invalid=0`.*

3. **Verify Loader Functionality**:
   ```powershell
   python -c "from minesweeper_rl.data import HumanDemonstrationDataset; ds = HumanDemonstrationDataset('data/human'); print('Loaded games:', len(ds.game_ids()), 'Total actions:', ds.num_actions)"
   ```
   *Expected: `Loaded games: 50 Total actions: 1440`.*

4. **Live Smoke Test (1-2 Games)**:
   Launch the recorder with `--diagnostics`, play 1 short game including an OPEN, a FLAG, and a CHORD, end the game, click the smiley face to reset, and press `Ctrl+C`. Run `validate_human_dataset` and verify the new game passed validation without invalid flags.
