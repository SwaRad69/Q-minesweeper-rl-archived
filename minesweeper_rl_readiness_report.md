# MINESWEEPER RL READINESS REPORT

> Full project audit completed 2026-09-17. Repository at `c:\Users\SwaRa\OneDrive\Desktop\rl minesweeper`.
> **No files were modified during this audit.**

---

## A. Repository Architecture

### Active Source Tree

| Component | Path | Purpose | Active? |
|-----------|------|---------|---------|
| **Action types** | [`action_types.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/action_types.py) | Model-independent OPEN/FLAG/CHORD codes | ✅ Yes |
| **Config** | [`config.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/config.py) | EnvironmentConfig (30×16, 99 mines) | ✅ Yes |
| **DOM parser** | [`dom/parser.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/dom/parser.py) | Board from CSS classes | ✅ Yes |
| **Game state** | [`dom/game_state.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/dom/game_state.py) | WIN/LOST from face element | ✅ Yes |
| **Environment** | [`environment.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/environment.py) | Browser env (OPEN/FLAG/CHORD) | ✅ Yes |
| **Browser controller** | [`browser/controller.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/browser/controller.py) | Playwright lifecycle | ✅ Yes |
| **Action handler** | [`actions/handler.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/actions/handler.py) | Click dispatch | ✅ Yes |
| **Schema** | [`data/schema.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/data/schema.py) | Canonical transition records | ✅ Yes |
| **Storage (legacy)** | [`data/storage.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/data/storage.py) | Append-only JSONL store | ⚠️ Not for new recording |
| **Human dataset store** | [`data/human_dataset.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/data/human_dataset.py) | Canonical per-game JSON dataset | ✅ Yes (PRIMARY) |
| **Human recorder** | [`data/record_human.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/data/record_human.py) | Observer: DOM events → transitions | ✅ Yes |
| **Recording entry point** | [`data/record_human_dataset.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/data/record_human_dataset.py) | CLI: `python -m ...record_human_dataset` | ✅ Yes |
| **Trajectory recorder** | [`data/recorder.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/data/recorder.py) | Game lifecycle: buffer → commit | ✅ Yes |
| **Human demonstrations** | [`data/human_demonstrations.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/data/human_demonstrations.py) | Training-time dataset loader | ✅ Yes |
| **Dataset validator** | [`data/validate_human_dataset.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/data/validate_human_dataset.py) | CLI validation entry point | ✅ Yes |
| **Policy network** | [`agents/policy.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/agents/policy.py) | CNN actor-critic (OPEN_ONLY + FLAGS) | ✅ Yes |
| **Action space** | [`agents/action_space.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/agents/action_space.py) | Mask/encode/decode; `DEFAULT=FLAGS` | ✅ Yes |
| **PPO env adapter** | [`agents/minesweeper_env.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/agents/minesweeper_env.py) | Flat-action → coordinate dispatch | ✅ Yes |
| **PPO trainer** | [`agents/ppo.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/agents/ppo.py) | Rollout, GAE, clipped surrogate | ✅ Yes |
| **Checkpoint** | [`experiment/checkpoint.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/experiment/checkpoint.py) | Atomic save/load, v5 format | ✅ Yes |
| **Imitation trainer** | [`experiment/imitation_train.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/experiment/imitation_train.py) | Behavioral cloning (CE loss on 1440 actions) | ✅ Yes |
| **Policy initializer** | [`experiment/initialization.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/experiment/initialization.py) | BC checkpoint → PPO encoder+actor init | ✅ Yes |
| **Training entry point** | [`experiment/train.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/experiment/train.py) | Resumable PPO training on live site | ✅ Yes |
| **Device** | [`experiment/device.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/experiment/device.py) | auto/cuda/cpu selection | ✅ Yes |

### Archived / Inactive

| Path | Status |
|------|--------|
| `archive/` | Mirrors `data/archive/` — contains old datasets and experiments |
| `data/archive/human_v1_invalid/` | Explicitly archived as invalid (old recorder) |
| `data/archive/human_smoke_v3*/` | Earlier smoke test data, not canonical |
| `datasets/human_expert/` | Contains only a README — legacy JSONL format is rejected |
| `src/minesweeper_browser/` | Only `__pycache__/` — no Python source; dead |
| `session-ses_f50f.md` | 740 KB session log — not source code |

> [!IMPORTANT]
> No duplicate or conflicting implementations were found. The codebase has a **single canonical path** for each operation. The legacy `TrajectoryStore` (JSONL) is explicitly rejected by the canonical dataset system.

---

## B. Actual Human Recording Pipeline

### Execution Chain (traced from imports)

```
User: python -m minesweeper_rl.data.record_human_dataset --dataset data/human
  │
  ├─ record_human_dataset.main()
  │   ├─ HumanDatasetStore(root, width=30, height=16, mines=99)
  │   │   └─ Creates/loads dataset_manifest.json + games/ dir
  │   ├─ PersistentHumanRecorder(store, ...)
  │   │   └─ Implements the TrajectoryRecorder-like interface for HumanDatasetStore
  │   ├─ open_environment(url, profile, config)
  │   │   └─ Playwright browser → MinesweeperEnvironment (with page)
  │   └─ HumanTrajectoryRecorder(environment, sink, ...)
  │       ├─ install_capture()
  │       │   └─ page.add_init_script(_CLICK_CAPTURE_SCRIPT)
  │       │       └─ pointerdown listener → window.__msrlEvents[]
  │       └─ run(max_games, stop_event)
  │           ├─ drain_events() → list of JS event objects
  │           │   └─ page.evaluate("() => window.__msrlEvents.splice(0)")
  │           ├─ handle_event(event)
  │           │   ├─ Phase gating: WAITING → READY → RECORDING → TERMINAL → RESYNCING
  │           │   ├─ event.isTrusted check (rejects synthetic events)
  │           │   ├─ button check (left=OPEN, right=FLAG)
  │           │   ├─ duplicate event ID detection (token + sequence)
  │           │   ├─ _is_fresh_board() → all-hidden, correct shape, not WON/LOST
  │           │   ├─ board_before = _stable_board (authoritative, not snapshot)
  │           │   ├─ _wait_for_stable_change() → quiescence loop
  │           │   ├─ _confirm_action() → verify click had expected effect
  │           │   ├─ transition_reward() → progress + terminal bonus/penalty
  │           │   └─ recorder.record_step() → buffered transition
  │           └─ _handle_idle_board()
  │               └─ Detects new game / abandoned game / board desync
  └─ PersistentHumanRecorder.record_step()
      └─ Buffers in-memory
  └─ PersistentHumanRecorder.end_game()
      └─ Writes games/game_XXXXXX.json atomically
      └─ Appends metadata.jsonl
      └─ Updates dataset_manifest.json atomically
```

### Synchronization Model

The recorder uses a **stable-board-authoritative** synchronization model:

1. The `_stable_board` is maintained from `environment.observe()` (two-snapshot stability)
2. `board_before` is always the last stable board — never the event-time DOM snapshot
3. After a click, `_wait_for_stable_change()` polls until the board differs AND stays unchanged for `quiescence_time` (0.3s default)
4. New events arriving during settlement are deferred, not folded into the current action
5. The event-time DOM snapshot is diagnostic only — used to validate freshness, never as `board_before`

### Race Condition Analysis

| Scenario | Protection |
|----------|------------|
| Stale observation | ✅ Two-snapshot stability + quiescence wait |
| Duplicate observations | ✅ `np.array_equal` check before recording |
| Duplicate actions | ✅ Event dedup via `(token, id)` set |
| Actions without genuine state changes | ✅ `_confirm_action()` rejects no-ops |
| Clicks before DOM changes | ✅ `_wait_for_stable_change()` polls until change appears |
| Wrong coordinates | ✅ Bounds-checked; event carries `data-x`/`data-y` from DOM |
| Transitions from previous game | ✅ 5-phase lifecycle; TERMINAL → RESYNCING → fresh board required |
| Transitions after terminal | ✅ `_phase == PHASE_TERMINAL` blocks; pre-action face check |
| Phantom games | ✅ Fresh-board gating requires all-hidden board |
| Browser reload | ✅ `add_init_script` reinstalls listener; resync phase handles |
| `event.isTrusted` | ✅ Explicitly checked (line 468) — rejects synthetic events |

---

## C. Actual Dataset Pipeline

### Current Dataset: `data/human/`

```
data/human/
├── dataset_manifest.json  (format identity, counters, next_game_index)
├── metadata.jsonl         (lightweight game summaries)
├── games/
│   ├── game_000001.json
│   ├── game_000002.json
│   └── ... (44 games, 1025 transitions)
└── README.md
```

### Per-Game File Structure (verified from actual files)

```json
{
  "format": "minesweeper_human_game",
  "format_version": 2,
  "game_id": "game_000001",
  "game_index": 1,
  "source": "human",
  "outcome": "lost",
  "complete": true,
  "width": 30, "height": 16, "mines": 99,
  "action_space": "flags",
  "action_encoding": "action_type*cells+cell",
  "action_types": {"0": "open", "1": "flag", "2": "chord"},
  "num_actions": 1440,
  "cells": 480,
  "observation_representation": "flag_aware_v1",
  "observation_version": 1,
  "recorder_version": 3,
  "reward_spec": "reveal_progress_terminal_v1",
  "transitions": [
    {
      "step": 0,
      "action": {
        "action_type": 0, "type": "open",
        "x": 0, "y": 0, "cell_index": 0, "action_id": 0
      },
      "observation_before": "<base64 int8 480-byte board>",
      "observation_after": "<base64 int8 480-byte board>",
      "game_status": "running",
      "terminated": false, "truncated": false,
      "reward": 8.0,
      "opened_cells": 8, "newly_revealed": 8,
      "timestamp": 1789644627.068
    }
  ]
}
```

### Model-Agnostic Design

> [!TIP]
> The dataset is **fully model-agnostic**. Each game file is self-describing with frozen identity fields (`action_space`, `action_types`, `num_actions`, `cells`, `observation_version`, `reward_spec`). A future GNN or quantum adapter can read `observation_before` (base64 int8 board) and `action_id` without importing any Python module from this project.

The adapter pattern is correctly implemented:

```
data/human/games/game_XXXXXX.json (canonical)
     │
     ├── HumanDemonstrationDataset → CNN behavioral cloning (1440-action)
     ├── (future) GNN adapter → node/edge encoding
     └── (future) Quantum adapter → feature map encoding
```

**No CNN/PPO-specific coupling exists in the recorded data.** The `observation_before`/`observation_after` are raw `FLAGGED(-2)/HIDDEN(-1)/OPEN(0..8)` boards. The `action_id` uses the universal `action_type * cells + cell` encoding.

---

## D. Dataset Validation

### Validation Run Results

```
status=OK games=44 (won=0 lost=36 incomplete=8) valid=44 invalid=0
transitions=1025 (valid=1025 invalid=0)
open=250 flag=300 chord=475 avg_length=23.3
```

### Validation Coverage

| Check | Status |
|-------|--------|
| Corrupted JSON files | **VERIFIED** — file parse errors caught |
| Invalid board dimensions | **VERIFIED** — width/height validated per game |
| Invalid board values | **VERIFIED** — `decode_board()` validates `[-2, 8]` range |
| Invalid action indices (`0..1439`) | **VERIFIED** — `action_id` cross-checked against `action_type * cells + cell` |
| Actions on already-open cells (OPEN) | **VERIFIED** — semantic validator checks `before[y,x] == HIDDEN` for OPEN |
| Impossible state transitions | **VERIFIED** — `board_before` continuity from previous `board_after` |
| Duplicate game IDs | **VERIFIED** — `seen_ids` set check |
| Missing transitions | **VERIFIED** — `total_steps` vs actual count |
| Broken game boundaries | **VERIFIED** — phantom continuation detection (first board = previous final board) |
| Incorrect rewards | **PARTIALLY VERIFIED** — reward is a derived field; validator checks it is present but does not recompute it |
| Incorrect terminal flags | **VERIFIED** — `terminated` must match `game_status` in `{won, lost}` |
| Incorrect WIN/LOSS | **VERIFIED** — `outcome` must match last `game_status` for complete games |
| Incomplete games | **VERIFIED** — marked `complete: false`, outcome must not be terminal |
| Phantom games | **VERIFIED** — `semantic_game_result()` detects phantom continuations |
| Stale transitions | **VERIFIED** — board continuity check |
| Schema mismatches | **VERIFIED** — `format`, `format_version`, `action_space` identity checked |
| Information leakage (hidden mines) | **VERIFIED** — `FORBIDDEN_LAYOUT_KEYS` explicitly rejected |
| Recorder version | **VERIFIED** — `recorder_version < 3` flagged as unreliable |
| First action must be OPEN | **VERIFIED** — semantic validator checks |
| First board must be all-hidden | **VERIFIED** — semantic validator checks |
| CHORD on non-number cell | **VERIFIED** — semantic validator checks |
| FLAG on open cell | **VERIFIED** — semantic validator checks |
| Safe reveal ending in LOST | **VERIFIED** — semantic validator checks |
| Terminal no-op transitions | **VERIFIED** — semantic validator checks |

> [!NOTE]
> **Reward recomputation**: The validator does not independently recompute `reward = progress + terminal`. This is a **NON-BLOCKING** gap because rewards are marked as `derived_fields` in the manifest and can be recomputed from `observation_before`, `observation_after`, and `game_status`. The imitation trainer does not use rewards at all (it trains on `board → action_id` cross-entropy). PPO computes its own rewards from the live environment.

---

## E. CNN + PPO

### Architecture Verification

The **active** pipeline uses the **FLAGS** action space (not OPEN_ONLY):

```
Raw board [batch, 16, 30]  (int8: FLAGGED=-2, HIDDEN=-1, OPEN 0..8)
      ↓
board_to_tensor() → one-hot [batch, 11, 16, 30]  (11 channels)
  channel 0 → FLAGGED (-2)
  channel 1 → HIDDEN (-1)
  channels 2..10 → OPEN 0..8
      ↓
CNN encoder:
  Conv2d(11, 16, 3×3, pad=1) → ReLU
  Conv2d(16, 16, 3×3, pad=1) → ReLU
      ↓ features [batch, 16, 16, 30]
      ├── Actor: Conv2d(16, 3, 1×1) → flatten → [batch, 1440]
      │   (3 planes: OPEN[0..479], FLAG[480..959], CHORD[960..1439])
      │   → action_mask → Categorical → action
      └── Critic: AdaptiveAvgPool2d(1) → flatten → Linear(16, 1) → V(s)
```

### Tensor Flow Verification

| Property | Expected | Actual |
|----------|----------|--------|
| Observation shape | `[batch, 16, 30]` int8 | ✅ Correct |
| Encoded shape | `[batch, 11, 16, 30]` float32 | ✅ Correct |
| CNN input channels | 11 | ✅ `observation_channels` attribute |
| Actor output | `[batch, 1440]` | ✅ 3 planes × 480 cells, flattened row-major |
| Action mask shape | `[1440]` bool | ✅ `[OPEN|FLAG|CHORD]` concatenated |
| Critic output | `[batch, 1]` → `[batch]` | ✅ Reshaped in `_forward()` |
| Action indexing | `action_type * cells + cell` | ✅ `split_compound_action()` |
| Device handling | Auto CUDA/CPU | ✅ `resolve_device()` |

### PPO Algorithm Verification

| Component | Status |
|-----------|--------|
| Rollout collection (env.reset/step loop) | ✅ Standard, with action masking |
| Log probabilities (Categorical) | ✅ `distribution.log_prob(action)` |
| Old log probs stored per step | ✅ In `RolloutBuffer.log_probs` |
| Ratios: `exp(new_log_probs - old_log_probs)` | ✅ Line 427 |
| GAE(λ) computation | ✅ Reverse sweep, `non_terminal` factor |
| Returns: `advantages + values` | ✅ Line 144 |
| Clipped surrogate: `min(r*A, clip(r)*A)` | ✅ `clipped_surrogate_loss()` |
| Value loss: MSE | ✅ `(values - returns).pow(2).mean()` |
| Combined loss: `-policy + vf_coef*value - ent_coef*entropy` | ✅ Lines 433-437 |
| Gradient clipping | ✅ `clip_grad_norm_` |
| Advantage normalization | ✅ Optional, default on |
| KL early stopping | ✅ Optional `target_kl` |
| Terminal handling in GAE | ✅ `non_terminal = 1.0 - float(terminated[step])` |

> [!NOTE]
> **GAE terminal semantics**: The `terminated` flag in the rollout buffer is set to `done = terminated or truncated` (line 267). This means truncated episodes also zero the bootstrap value. For the current episode-batch training (`collect_episode`), this is acceptable because each rollout is one episode and the final transition is the actual terminal or a truncated forced-stop. This is semantically correct for the Minesweeper environment.

### Important Observation: OPEN_ONLY vs FLAGS

The `DEFAULT_ACTION_SPACE` is `FLAGS` (line 60 in [`action_space.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/agents/action_space.py#L60)). The training entry point ([`train.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/experiment/train.py#L1-L7)) is **FLAGS-only** — there is deliberately no `--action-space` option. The human dataset uses the same FLAGS action space.

> [!IMPORTANT]
> Your problem statement says the current CNN + PPO architecture should have 10 channels and 480 actions (OPEN-ONLY). **This is NOT the current repository state.** The active pipeline uses **11 channels** and **1440 actions** (FLAGS: OPEN+FLAG+CHORD). The OPEN_ONLY path still exists but is retained only for reading completed v4 checkpoints. This is architecturally consistent — the human dataset, the CNN, the PPO trainer, and the training entry point all agree on `FLAGS`.

---

## F. PPO Checkpointing

### Checkpoint Contents (v5)

| Field | Stored | Restored |
|-------|--------|----------|
| `policy` (state_dict) | ✅ | ✅ |
| `optimizer` (state_dict) | ✅ | ✅ |
| `config` (PPOConfig) | ✅ | ✅ (with mismatch warning) |
| `num_actions` | ✅ | ✅ (validated) |
| `iteration` | ✅ | ✅ |
| `episode_count` | ✅ | ✅ |
| `total_steps` | ✅ | ✅ |
| `model` metadata | ✅ | ✅ (validated on load) |
| `action_space` | ✅ | ✅ (validated on load) |
| RNG state (Python/NumPy/PyTorch/CUDA) | ✅ | ✅ |
| `metrics` (latest) | ✅ | Read-only |
| `metadata` (experiment provenance) | ✅ | Read-only |

### Checkpoint Safety

- **Atomic writes**: temp file in same dir + `os.replace` — crash-safe
- **Version validation**: v5 for FLAGS, v4 for legacy OPEN_ONLY — cross-loading rejected
- **Architecture validation**: observation channels, board dims, action count, CNN channel width — all checked
- **First-conv weight inspection**: even if metadata is missing, the serialized `encoder.0.weight` shape is checked
- **Device portability**: `map_location="cpu"` + trainer moves to target device
- **Resume**: Full training state restored; can continue from exactly where it stopped

---

## G. PPO Fine-Tuner

### What "Fine-Tuning" Actually Means

There are **two** distinct fine-tuning paths:

#### Path 1: Behavioral Cloning (Imitation Pretraining)

[`experiment/imitation_train.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/experiment/imitation_train.py)

This is **supervised learning / behavior cloning**, not RL:
```
loss = cross_entropy(policy_logits[batch, 1440], human_action_id[batch])
```

It:
- Loads `data/human/` via `HumanDemonstrationDataset`
- Splits by game (deterministic seed)
- Trains `PolicyValueNetwork(action_space=FLAGS)` with Adam + CE
- Saves its own checkpoint format (`minesweeper_imitation`, v1)
- Reports per-action-type accuracy (OPEN/FLAG/CHORD)
- The critic is never optimized (it receives gradients only through the shared encoder)

#### Path 2: PPO Initialization from Imitation

[`experiment/initialization.py`](file:///c:/Users/SwaRa/OneDrive/Desktop/rl%20minesweeper/src/minesweeper_rl/experiment/initialization.py)

This transfers BC weights into a PPO policy:
- Copies **encoder + actor** weights from an imitation checkpoint
- **Resets the critic** (value head) to zero — `V(s) = 0`
- Full compatibility validation (observation channels, dims, action count, architecture)
- The PPO training then fine-tunes with online RL (from the live environment)

### Dataset Compatibility

| Property | Human Dataset | BC Trainer |
|----------|--------------|------------|
| Board encoding | FLAGGED=-2, HIDDEN=-1, OPEN 0..8 | ✅ Same |
| Board shape | `[16, 30]` int8 | ✅ Same |
| Action encoding | `action_type * 480 + cell` | ✅ Same |
| Action count | 1440 | ✅ Same |
| Action space | `flags` | ✅ Same |

**The current human dataset (44 games, 1025 transitions) can be consumed by the fine-tuner immediately.**

---

## H. Website Integration

### DOM Structure Assumptions

| Element | Selector | Purpose | Verified By |
|---------|----------|---------|-------------|
| Board cells | `.cell[data-x][data-y]` | Cell grid | DOM parser |
| Cell state | `hd_closed`, `hd_opened`, `hd_flag`, `hd_typeN` | Observable state | `_parse_cell()` |
| Game face | `#top_area_face` | WIN/LOST detection | `classify_game_status()` |
| Win indicator | `*top-area-face-win` suffix | Game won | `_FACE_WIN` |
| Loss indicator | `*top-area-face-lose` suffix | Game lost | `_FACE_LOSE` |
| No-guessing mode | `nnhd_` prefix variants | Alternative skin | Parser supports both |

### Timing and Async Handling

- Board reads use `evaluate_all` (single synchronous DOM pass — no interleaving)
- Two-snapshot stability: two identical consecutive reads = stable
- Quiescence window: 0.3s unchanged board = settled
- Action timeout: 5s max wait for stable change
- Mine grace: 1.0s extra wait for delayed LOST status

> [!WARNING]
> **UNVERIFIED**: The actual minesweeper.online website has not been tested during this audit (no browser was opened). All DOM assumptions are traced from code. The 44 recorded games in `data/human/` that passed validation provide indirect evidence that the website integration works, but changes to the website since those games were recorded cannot be verified without a live test.

---

## I. Game Lifecycle Safety

### Phase State Machine

```
WAITING ──(fresh board observed)──→ READY
READY ──(first OPEN action)──→ RECORDING
RECORDING ──(WIN/LOST)──→ TERMINAL ──→ RESYNCING
RECORDING ──(board changes without click)──→ RESYNCING (abandoned)
TERMINAL ──(any click)──→ RESYNCING
RESYNCING ──(fresh all-hidden board)──→ READY
```

### Phantom/ABANDONED Prevention

| Scenario | Protection |
|----------|------------|
| Game A terminal → stray click on finished board | Pre-action face check; `_pre_action_status()` catches it |
| Game A terminal → Game B accidentally starts from A's board | Fresh-board gating: requires all-hidden, not WON/LOST |
| Game A transitions bleeding into Game B | Game ends on terminal → `_enter_resyncing()` clears state |
| Browser reload mid-game | `add_init_script` reinstalls; phase resets to WAITING |
| Recorder crash → partial write | Per-game JSON files are atomic; partial game is `complete: false` |
| No-transition game on rejected first action | `discard_game()` — never written to disk |
| Board changed without click during recording | Detected by `_handle_idle_board()` → `_abandon_active()` |
| File locks (OneDrive) | Retry with backoff (6 attempts, 50ms → 500ms delay) |
| Atomic file replacement | `tempfile.mkstemp` + `os.replace` — crash-safe |

> [!TIP]
> The previous phantom/ABANDONED problem is addressed. The current recorder (v3) has explicit lifecycle phases, fresh-board verification, pre-action face snapshots, and event deduplication. The 44 games in `data/human/` all pass semantic validation with 0 invalid games and 0 invalid transitions.

---

## J. Tests

### Test Suite Results

```
454 passed in 32.58s (exit code 0)
```

### Test Coverage by Category

| Category | Files | Count | Status |
|----------|-------|-------|--------|
| DOM parser | `test_dom_parser.py` | ✅ | Passed |
| Game state | `test_game_state.py` | ✅ | Passed |
| Actions | `test_actions.py` | ✅ | Passed |
| Environment | `test_environment.py`, `test_environment_observe.py`, `test_environment_reset.py`, `test_environment_step.py` | ✅ | Passed |
| Reward | `test_reward.py` | ✅ | Passed |
| Policy network | `test_policy_network.py` | ✅ | Passed |
| PPO | `test_ppo.py` | ✅ | Passed |
| PPO env | `test_minesweeper_ppo_env.py` | ✅ | Passed |
| Checkpoint | `test_checkpoint.py` | ✅ | Passed |
| Training resume | `test_train_resume.py` | ✅ | Passed |
| Training readiness | `test_training_readiness.py` | ✅ | Passed |
| Experiment | `test_experiment.py` | ✅ | Passed |
| Human recorder lifecycle | `test_human_recorder_lifecycle.py` | ✅ | Passed |
| Human recording | `test_record_human.py` | ✅ | Passed |
| Human dataset | `test_human_dataset.py` | ✅ | Passed |
| Human schema | `test_human_schema.py` | ✅ | Passed |
| Trajectory data | `test_trajectory_data.py` | ✅ | Passed |
| Imitation | `test_imitation.py` | ✅ | Passed |
| PPO pretraining | `test_ppo_pretraining.py` | ✅ | Passed |
| Storage persistence | `test_storage_persistence.py`, `test_persistence_reliability.py` | ✅ | Passed |
| Flag-aware training | `test_flag_aware_training.py` | ✅ | Passed |
| Flags-only architecture | `test_flags_only_architecture.py` | ✅ | Passed |
| Device | `test_device.py` | ✅ | Passed |
| Random agent | `test_random_agent.py` | ✅ | Passed |
| Stress | `test_stress.py` | ✅ | Passed |

### Live Website Tests (Not Run)

| File | Status |
|------|--------|
| `test_live_website.py` | **UNVERIFIED** — requires browser |
| `test_live_board_matrix.py` | **UNVERIFIED** — requires browser |
| `test_live_random_agent.py` | **UNVERIFIED** — requires browser |
| `test_live_stress.py` | **UNVERIFIED** — requires browser |
| `test_browser_controller.py` | **UNVERIFIED** — requires browser |

---

## K. Model-Agnostic Design

### Separation of Concerns

```
                    ┌──────────────────────────────────┐
                    │  data/human/games/game_XXXXXX.json  │
                    │  (canonical, model-independent)     │
                    └─────────────┬────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
  ┌─────────▼─────────┐  ┌───────▼───────┐  ┌─────────▼─────────┐
  │ CNN 11-ch one-hot  │  │ GNN nodes/    │  │ Quantum feature   │
  │ (HumanDemo...      │  │ edges adapter │  │ map adapter       │
  │  Dataset)          │  │ (future)      │  │ (future)          │
  └────────────────────┘  └───────────────┘  └───────────────────┘
```

The game files store:
- **Raw boards**: base64 int8, `[y, x]` row-major, values `{-2, -1, 0..8}`
- **Actions**: `(x, y, action_type, action_id, cell_index)` — fully decomposable
- **Game status**: `{ready, running, won, lost}`
- **Metadata**: timestamps, duration, outcome, board dims

**No neural network representation is baked into the data files.** The one-hot encoding happens in `board_to_tensor()` at training time.

### Coupling Analysis

| Component | CNN-specific? | Blocks future models? |
|-----------|--------------|----------------------|
| Board encoding (int8 values) | No | No |
| Action encoding (`type*cells+cell`) | No — universal | No |
| Reward definition | No — `progress + terminal` from observable boards | No |
| `observation_version` marker | No — versioned for compatibility | No |
| `action_space: flags` | No — describes the action vocabulary | No |
| One-hot channels | CNN-specific but applied at training time | No |

---

## L. Future QRL Compatibility

### Reusable Components

| Component | Reusable for QRL? | Notes |
|-----------|-------------------|-------|
| Human dataset (raw boards) | ✅ Yes | Observable boards in universal int8 |
| Action encoding | ✅ Yes | `action_type * cells + cell` is model-independent |
| Reward definition | ✅ Yes | `progress + terminal` from observable state |
| Terminal semantics | ✅ Yes | WIN/LOST from game status |
| Trajectories (game sequences) | ✅ Yes | Ordered transitions with board continuity |
| Human demonstrations (for offline RL) | ✅ Yes | Same data, different adapter |
| Environment facade | ✅ Yes | `reset()/step()` interface |
| Action masking logic | ✅ Yes | Board → boolean mask is model-independent |
| Validation framework | ✅ Yes | Semantic rules apply regardless of model |

### Potential Blockers

| Issue | Severity | Notes |
|-------|----------|-------|
| The `PolicyValueNetwork` is CNN-specific | Not a blocker | A QRL model would implement its own `nn.Module`; the `PPOTrainer` accepts any module with `(obs) → (logits, value)` |
| The `board_to_tensor()` function is CNN-specific | Not a blocker | QRL would implement its own encoding; the raw int8 board is the input |
| The `PPOTrainer` is PPO-specific | Not a blocker | DDQN/QRL would use their own trainer with the same environment interface |
| Checkpoint format is PPO/CNN-specific | Not a blocker | QRL would define its own checkpoint format |

> [!NOTE]
> **No changes to the human recorder are needed for future QRL/Hybrid RL.** The canonical dataset stores only observable facts. Each future model writes its own adapter to convert int8 boards into its input representation.

---

## Reproducibility

### What Is Captured

| Property | Stored? | Location |
|----------|---------|----------|
| Random seed | ✅ | Checkpoint RNG states; BC config seed |
| Board dimensions | ✅ | Config, manifest, every game file |
| PPO hyperparameters | ✅ | Checkpoint `config` field |
| Model architecture | ✅ | Checkpoint `model` metadata |
| Action space identity | ✅ | Checkpoint + dataset |
| Observation version | ✅ | Dataset manifest + game files |
| Reward definition | ✅ | `reward_spec` in dataset |
| Training iteration/episode/step | ✅ | Checkpoint counters |
| Dataset provenance | ✅ | BC checkpoint stores dataset stats |
| Experiment configuration | ✅ | `run_config.json` in experiment dir |
| Training history | ✅ | `history.json` in BC experiment |

### What Is Missing

| Property | Status |
|----------|--------|
| Git commit hash in checkpoints | ❌ Not captured |
| Python/PyTorch/NumPy version in checkpoints | ❌ Not captured |
| Hardware description in checkpoints | ❌ Not captured |
| Experiment naming convention | ⚠️ Manual (directory names) |

These are **non-blocking** for reproducibility within a single machine.

---

# READINESS CLASSIFICATION

### Human Recorder
## **READY**

The v3 recorder has:
- 5-phase lifecycle state machine
- Fresh-board gating (prevents phantom games)
- Pre-action face snapshots (prevents false ABANDONED on fast terminal)
- Two-snapshot board stability + quiescence window
- Event deduplication (token + sequence)
- `event.isTrusted` check
- Per-action confirmation (`_confirm_action()`)
- Deferred event handling (no folding of overlapping actions)
- Atomic file writes with OneDrive retry

All 454 offline tests pass. 44 human games recorded and validated with 0 errors.

---

### Human Dataset
## **READY**

44 games, 1025 transitions. All pass structural and semantic validation. Self-describing format with frozen identity. No information leakage. Phantom continuation detection. Legacy datasets explicitly rejected.

---

### Dataset Validator
## **READY**

Comprehensive structural and semantic validation covering all 22+ checks listed in the audit. Runs successfully on the current dataset (`status=OK`). The only gap is reward recomputation (non-blocking since rewards are derived fields and the imitation trainer doesn't use them).

---

### CNN + PPO
## **READY WITH CONDITIONS**

The implementation is correct and all 454 tests pass. The architecture uses **FLAGS** (11 channels, 1440 actions), not OPEN_ONLY (10 channels, 480 actions) as stated in the problem description. This is internally consistent — the recorder, dataset, CNN, PPO, and training entry point all agree on FLAGS.

**Condition**: You should be aware that your active pipeline is FLAGS-based (OPEN+FLAG+CHORD with 11 channels and 1440 actions), not the OPEN_ONLY baseline described in your problem statement. If you want OPEN_ONLY, the infrastructure supports it but the training entry point deliberately blocks it.

---

### PPO Fine-Tuner
## **READY**

The behavioral cloning pipeline (`imitation_train.py`) correctly loads the human dataset, trains with cross-entropy on the 1440-action space, and the initialization pipeline (`initialization.py`) correctly transfers encoder+actor weights to a PPO policy with value head reset. Full architecture compatibility validation exists.

---

### Future QRL Infrastructure
## **READY WITH CONDITIONS**

The dataset is model-independent. The environment interface is standard. Action encoding is universal. No neural-network-specific data is baked into the canonical files.

**Condition**: QRL/Hybrid RL models will need to implement their own:
1. Board → quantum feature map encoding
2. Trainer (not PPO)
3. Checkpoint format
None of these require changes to the human recorder or dataset.

---

# CRITICAL FINDINGS

### BLOCKING

**None.** No blocking issues were found. The pipeline is functional end-to-end.

---

### NON-BLOCKING

1. **Reward not recomputed by validator**: The validator checks rewards are present and finite but does not independently verify `reward = progress + terminal_bonus`. Since rewards are derived fields and the BC trainer ignores them, this is cosmetic.

2. **No wins in dataset**: All 44 games are losses (36) or incomplete (8). This means the BC model has never seen a winning trajectory. This is a data collection issue, not a code issue. More human games (especially wins) will improve imitation quality.

3. **Git commit / version not in checkpoints**: Experiment reproducibility across machines or time would benefit from recording the code version.

4. **`src/minesweeper_browser/` directory**: Contains only `__pycache__/` — dead code artifact that should be cleaned up (but does not affect functionality).

5. **Problem description says OPEN_ONLY, code says FLAGS**: Your audit request describes a 10-channel/480-action OPEN_ONLY architecture. The actual codebase uses 11-channel/1440-action FLAGS. Both are internally consistent. Clarify which you intend going forward (the code is clear: FLAGS is the active experiment).

---

### VERIFIED

- ✅ 454 tests pass (all offline tests)
- ✅ Dataset validation: OK (44 games, 1025 transitions, 0 errors)
- ✅ Human recorder lifecycle (5-phase state machine, fresh-board gating)
- ✅ `event.isTrusted` checking
- ✅ Phantom game prevention (fresh-board verification + continuation detection)
- ✅ Action masking (OPEN/FLAG/CHORD legality)
- ✅ CNN tensor flow (11-channel one-hot → actor 1440 logits → masked action)
- ✅ PPO algorithm (GAE, clipped surrogate, value loss, entropy, gradient clipping)
- ✅ Checkpoint atomic writes and resume
- ✅ BC trainer loads human dataset and trains on 1440-action CE loss
- ✅ BC → PPO initialization (encoder+actor transfer, critic reset)
- ✅ Model-agnostic dataset format
- ✅ OneDrive file-lock retry mechanism
- ✅ Board encoding/decoding (base64 int8, value range `[-2, 8]`)

---

### UNVERIFIED

- ❌ Live website compatibility (no browser opened during audit)
- ❌ Live recording session (requires human playing in browser)
- ❌ Website DOM class names unchanged since last recording
- ❌ Playwright version compatibility with current website
- ❌ CUDA training (no GPU available for testing)

---

# MOST IMPORTANT QUESTION

> "Can I now manually play Minesweeper, record NEW human games, validate the resulting dataset, and use that dataset with the current CNN+PPO/PPO fine-tuning pipeline, while keeping the infrastructure reusable for future QRL/Hybrid RL experiments?"

## **YES**

### Evidence:

1. **Recording**: The entry point `python -m minesweeper_rl.data.record_human_dataset --dataset data/human` is fully functional. The recorder (v3) has comprehensive lifecycle safety. 44 games have already been successfully recorded with 0 validation errors.

2. **Validation**: `python -m minesweeper_rl.data.validate_human_dataset --dataset data/human` runs and reports `status=OK`. All 22+ validation checks pass.

3. **BC Training**: `python -m minesweeper_rl.experiment.imitation_train --dataset data/human` loads the dataset, splits by game, and trains with cross-entropy on the 1440-action FLAGS space.

4. **PPO Initialization**: The `initialize_policy_from_imitation()` function transfers BC weights (encoder+actor) into a PPO policy with the value head reset.

5. **PPO Training**: `python -m minesweeper_rl.experiment.train --games N --run-dir experiments/...` trains online PPO with the same FLAGS action space. It can start from scratch or from an imitation-pretrained checkpoint.

6. **QRL Compatibility**: The canonical `data/human/` dataset stores only observable facts (raw boards, universal action encoding, observable game status). No CNN/PPO-specific representation is baked in. A future QRL adapter reads the same files.

7. **454 tests pass** covering the entire pipeline from DOM parsing through PPO training.

---

# MINIMUM NEXT STEPS

Before collecting the real human dataset:

1. **Run a live smoke test**: Record 2–3 complete games with `python -m minesweeper_rl.data.record_human_dataset --dataset data/human --games 3` and verify the validator still reports `status=OK`. This confirms the website DOM hasn't changed.

2. **Play to win**: Your current 44 games have 0 wins. If you want the BC model to learn winning behavior, you need at least some winning games in the dataset.

That's it. The infrastructure is ready.
