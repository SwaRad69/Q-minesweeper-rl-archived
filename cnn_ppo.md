# CNN + PPO — Training Guide

> **STATUS (read first): FLAGS-only.** The whole project is now **OPEN + FLAG +
> CHORD**, **1440 actions**, **11 observation channels**
> (`PolicyValueNetwork()` -> 1440 logits, `MinesweeperPPOEnv` -> 1440 actions).
> There are two supported ways to train it:
>
> 1. **Scratch** — random initialization, live PPO (`--training-mode scratch`,
>    the default).
> 2. **Human-pretrained** — behavioral-clone the CNN from fresh human games,
>    then PPO fine-tune (`--training-mode human_pretrained`).
>
> There is **no `--action-space` option anymore**: the main training path cannot
> be run OPEN-only. The completed OPEN-only baseline (480 actions, 10 channels,
> checkpoint version 4) was moved to `archive/experiments/` and is no longer
> trainable or resumable. Where the body of this guide says "OPEN-only", "480
> actions", "no FLAG/CHORD", "version 4" or passes `--action-space`, read it as
> **historical**. Current experiments are named `experiments/cnn_ppo_flags_v1/`
> and `experiments/cnn_ppo_flags_human_v1/`; the BC run is
> `experiments/cnn_bc_flags_v1/`. Human-game recording is documented in
> [`HUMAN_DATA.md`](HUMAN_DATA.md) and
> `experiments/cnn_ppo_flags_human_v1/README.md`.

Operational and research documentation for the classical CNN + PPO experiments
of this repository. Every command, path, argument and metric below was verified
against the current implementation under `src/minesweeper_rl/` at the time of
writing. Where the code and an older document disagree, the code is treated as
the source of truth and the discrepancy is called out.

> This document is self-contained. It is intended to be usable months later
> without any other context.

---

## Quick start: train a CNN + PPO model

Run every command from the repository root
(`C:\Users\SwaRa\OneDrive\Desktop\rl minesweeper`). The recipes below are the
current, copy-paste-ready path for the FLAGS-only model. The `--action-space`
flag no longer exists; any command below that shows it is historical and will
not run.

### 0. One-time setup

```powershell
python -m pip install -e ".[test,plot]"
python -m playwright install chromium
python -m pytest -m "not live"                 # expect: 413 passed

$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
```

### 1. Always start with a smoke test (5 games)

```powershell
python -m minesweeper_rl.experiment.train --games 5 `
  --action-space flags `
  --run-dir experiments/cnn_ppo_expert_flags_v1_smoke `
  --checkpoint-interval 1 --seed 42 --device auto --fresh
```

If this completes and writes checkpoints, your browser, CUDA and pipeline work.

### 2. Option A — train from scratch (flag-aware)

```powershell
python -m minesweeper_rl.experiment.train --games 2000 `
  --action-space flags `
  --training-mode scratch `
  --run-dir experiments/cnn_ppo_expert_flags_v1 `
  --checkpoint-interval 10 --seed 42 --device auto
```

### 3. Option B — train human-pretrained (recommended strong start)

This is a three-stage pipeline. Stage 3a is only needed once; repeat 3b–3c as
you collect more human games.

**3a. Record human games** (optional if `data/human/` already has games):

```powershell
python -m minesweeper_rl.data.record_human_dataset --dataset data/human
python -m minesweeper_rl.data.validate_human_dataset --dataset data/human
```

Play in the opened browser; `Ctrl+C` stops (the in-progress game is saved as
incomplete by default). See [`HUMAN_DATA.md`](HUMAN_DATA.md).

**3b. Behavioral cloning** (imitation) from the human games:

```powershell
python -m minesweeper_rl.experiment.imitation_train `
  --dataset data/human `
  --output experiments/cnn_imitation_expert_v1 `
  --epochs 20 --batch-size 64 --learning-rate 1e-3 --val-split 0.2 --seed 42 `
  --device auto
```

Useful filters: `--winning-only`, `--exclude-incomplete`, `--game game_000001`,
`--resume`.

**3c. PPO fine-tune** initialized from the BC checkpoint:

```powershell
python -m minesweeper_rl.experiment.train --games 2000 `
  --action-space flags `
  --training-mode human_pretrained `
  --pretrained-policy experiments/cnn_imitation_expert_v1/checkpoint_latest.pt `
  --run-dir experiments/cnn_ppo_expert_human_v1 `
  --checkpoint-interval 10 --seed 42 --device auto
```

The value head is reset to zero; encoder + actor are copied from BC. The
initialization is recorded in `run_config.json` and reused automatically on
resume. To keep improving: record more games -> re-run 3b on the full dataset
-> restart 3c with `--fresh` (or a new `--run-dir`).

### 4. Evaluate and compare

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"

python -m minesweeper_rl.experiment.evaluate --action-space flags `
  --checkpoint experiments/cnn_ppo_expert_human_v1/checkpoint_latest.pt `
  --games 50 --run-dir experiments/cnn_ppo_expert_human_v1_eval --device auto

python -m minesweeper_rl.experiment.compare_evaluations `
  scratch=experiments/cnn_ppo_expert_flags_v1_eval `
  human=experiments/cnn_ppo_expert_human_v1_eval
```

### 5. Stop, resume and restart

- **Stop safely:** `Ctrl+C` in the training terminal (finishes the current
  episode, writes a checkpoint and exits).
- **Resume:** re-run the **identical** `experiment.train` command — it auto-loads
  `checkpoint_latest.pt` and continues to the absolute `--games` target.
- **Restart from random weights:** add `--fresh` (archives existing artifacts
  instead of deleting them).

> **Do not** run the full 2,000-game training as part of a documentation check.
> The commands are recorded here for when you are ready.

---

## 1. Purpose

The CNN + PPO experiments train a **classical deep reinforcement-learning
agent** that plays the live `minesweeper.online` Expert board.

- The agent is a small **convolutional actor-critic** (`PolicyValueNetwork`) —
  the "CNN".
- The learning algorithm is a **Proximal Policy Optimization** implementation
  (`PPOTrainer`) — the "PPO".
- The **active default** is the **flag-aware** model: OPEN + FLAG + CHORD,
  1440 actions, 11 observation channels.
- There are two training modes:
  - **`scratch`** — random initialization, pure live PPO (the historical
    baseline protocol);
  - **`human_pretrained`** — the CNN actor is first behavioral-cloned from
    recorded human games (`data/human/`), then PPO fine-tunes it. See
    [`HUMAN_DATA.md`](HUMAN_DATA.md) and the Quick start above.
- The completed **OPEN-only** baseline (480 actions, 10 channels) remains
  reproducible with `--action-space open_only` and is the reference result.
- These models are the **classical baseline** intended for later comparison with
  a GNN/DDQN variant and with quantum/hybrid RL approaches. Any such comparison
  requires the **same evaluation protocol** (see §24).

Start from the **Quick start** above. The rest of this guide documents the
historically verified OPEN-only protocol in detail; the same mechanics
(monitoring, checkpoints, resume, evaluation) apply to the flag-aware runs.

---

## 2. Experiment Specification

Values below are the defaults used by the verified primary command
(`--seed 42 --device auto`) and match the run configuration snapshots written by
the implementation.

| Setting | Value |
| --- | --- |
| Environment | Live `minesweeper.online` driven by Playwright Chromium (persistent profile `browser_profile/`) |
| Difficulty | `expert` |
| Board dimensions | 30 wide × 16 high (`[height, width]` observation = `[16, 30]`), 480 cells |
| Mine count | 99 |
| Observation representation | Canonical `numpy.int8` board; hidden = `-1` (`HIDDEN`), revealed = `0..8` |
| Observation channels | 10 = `HIDDEN` + `OPEN 0..8` (channel 0 = hidden, channel 1 = open 0, …, channel 9 = open 8) |
| Action space | 480 discrete actions |
| Action semantics | OPEN-only; `action = y * width + x` (row-major); illegal/expired cells masked |
| Reward | `revealed(after) - revealed(before) + terminal`; `revealed = count(board >= 0)`; WIN `+100`, LOSS `-10`; no shaping, no hidden-mine information |
| CNN architecture | `Conv2d(10→16, k=3, p=1)+ReLU` → `Conv2d(16→16, k=3, p=1)+ReLU`; actor `Conv2d(16→1, k=1)` → flattened to 480 logits; critic `AdaptiveAvgPool2d(1)` → `Flatten` → `Linear(16→1)` |
| PPO hyperparameters | `gamma=0.99`, `gae_lambda=0.95`, `clip_coef=0.2`, `entropy_coef=0.01`, `value_coef=0.5`, `learning_rate=3e-4` (Adam), `num_epochs=4`, `minibatch_size=64`, `max_grad_norm=0.5`, `target_kl=None`, `normalize_advantages=True` |
| PPO rollout size | **1 episode per update** (`rollout_steps=1`, see §10) |
| Initialization | `random` |
| Seed | `42` for the primary run (`--seed`; note the CLI default is `None`, i.e. unseeded) |
| Training target | `2000` completed episodes (`--games 2000`, a **total** target) |
| Device | `auto` → CUDA when available (verified on `NVIDIA GeForce RTX 5060 Laptop GPU`, CUDA 12.8, torch 2.11.0+cu128) |

Reward constants are defined in `src/minesweeper_rl/environment.py:34-36`
(`SAFE_REVEAL_REWARD=1.0`, `MINE_REWARD=-10.0`, `WIN_REWARD=100.0`).

---

## 3. Repository and Experiment Data

Run directories (one per experiment definition; all artifacts of a run live in
its own `--run-dir`):

```text
experiments/cnn_ppo_expert_flags_v1/   # active: flag-aware scratch
experiments/cnn_ppo_expert_human_v1/   # active: flag-aware human-pretrained
experiments/cnn_imitation_expert_v1/   # behavioral-cloning checkpoint (input to the above)
experiments/cnn_ppo_expert_v1/         # historical OPEN-only baseline
experiments/cnn_ppo_expert_v1_smoke/   # smoke run
```

All artifacts for a run are written into a single `--run-dir`. The
implementation creates these files:

| File | Meaning | Written when |
| --- | --- | --- |
| `run_config.json` | Full reproducible configuration snapshot (experiment name, algorithm, architecture, observation channels, action count, action type, board, PPO hyperparameters, games target, checkpoint interval, seed, device, reward, URL, environment provenance) | Only on the **first** fresh start, once |
| `episodes.jsonl` | **Raw experimental data.** One JSON object per completed game | Every episode (append-only) |
| `updates.jsonl` | **Raw experimental data.** One JSON object per PPO update | Every episode, after the update |
| `checkpoint_latest.pt` | Newest full training state, used for resume | Every `--checkpoint-interval` episodes and at the end |
| `checkpoint_best.pt` | Best checkpoint by rolling win rate | After episode ≥ 10, only when the rolling win rate improves |
| `training_progress.png` | Multi-panel training figure | At the end of training (unless `--no-plots`) |
| `archive/<timestamp>/` | Archived previous artifacts moved aside by `--fresh` | Only on `--fresh` |

Evaluation (a separate command, §17) writes into its own run directory:

| File | Meaning |
| --- | --- |
| `episodes.jsonl` | Raw per-game evaluation records (`phase="eval"`) |
| `evaluation_results.json` | Machine-readable evaluation summary (checkpoint, device, games, win rate, etc.) |
| `evaluation.png` | Evaluation bar chart (unless `--no-plots`) |

There is **no separate `logs/` directory and no log file**; console logging is
written to the terminal (stdout) only.

### Raw JSONL schemas (verified)

`episodes.jsonl` fields:
`episode`, `phase`, `outcome`, `won`, `reward`, `steps`, `mine_hit`,
`cells_revealed`, `duration_s`, `invalid_actions`, `timestamp`,
`open_actions`, `source`, `revealed_total`.

`updates.jsonl` fields:
`iteration`, `phase`, `policy_loss`, `value_loss`, `entropy`, `approx_kl`,
`clip_fraction`, `total_loss`, `updates`, `timestamp`, `learning_rate`,
`device`.

---

## 4. Prerequisites

This repository currently uses a **system-wide Python 3.12** installation; there
is no virtualenv or conda environment checked in. If you use one, activate it
first with your normal command (e.g. `& .\.venv\Scripts\Activate.ps1`). For the
current machine no activation is required.

Verify Python and package installation (run from the repository root):

```powershell
python --version
python -c "import torch, numpy, playwright; print('torch', torch.__version__); print('numpy', numpy.__version__)"
```

Verify the GPU with the NVIDIA driver:

```powershell
nvidia-smi
```

Verify CUDA is visible to PyTorch:

```powershell
python -c "import torch; print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda); print('device_count', torch.cuda.device_count()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected on this machine:

```text
cuda_available True
cuda_version 12.8
device_count 1
gpu NVIDIA GeForce RTX 5060 Laptop GPU
```

If the Playwright browser binary is missing:

```powershell
python -m playwright install chromium
```

---

## 5. Environment Setup

Training and evaluation read the game URL from the `MINESWEEPER_URL`
environment variable (or the `--url` argument). Set it for the current
PowerShell session:

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
```

PowerShell syntax note: `$env:NAME = "value"` sets a process-scoped environment
variable. It persists for the lifetime of the terminal window and every child
process (including `python`) inherits it. It is **not** permanent; re-run it in
each new terminal, or pass `--url` explicitly.

Optional diagnostics (DEBUG logs from the environment):

```powershell
$env:MINESWEEPER_DIAGNOSTICS = "1"
```

Optional profile override (default `browser_profile`):

```powershell
$env:MINESWEEPER_PROFILE = "browser_profile"
```

---

## 6. Pre-Training Checklist

- [ ] You are in the repository root (`C:\Users\SwaRa\OneDrive\Desktop\rl minesweeper`).
- [ ] Dependencies installed: `python -m pip install -e ".[test,plot]"`.
- [ ] Playwright Chromium installed: `python -m playwright install chromium`.
- [ ] Full unit test suite passes (§7).
- [ ] CUDA verified and the RTX 5060 detected (§4).
- [ ] `MINESWEEPER_URL` set (§5).
- [ ] Training mode chosen (`--action-space flags` + `--training-mode scratch` or
      `human_pretrained`).
- [ ] Target `--run-dir` chosen deliberately (scratch:
      `experiments/cnn_ppo_expert_flags_v1`; human-pretrained:
      `experiments/cnn_ppo_expert_human_v1`).
- [ ] Seed recorded (`--seed 42`).
- [ ] Confirmed you are **not** about to overwrite an existing experiment
      (training refuses to start if `run_config.json` exists without a
      checkpoint; it resumes if `checkpoint_latest.pt` exists).
- [ ] Smoke test passed (§8) or a resume test passed (§14).

---

## 7. Full Test Suite

The non-live suite (fast, no website access):

```powershell
python -m pytest -m "not live"
```

Verified result at the time of writing:

```text
413 passed, 10 deselected
```

Live tests (these **do** access and mutate the real website) are opt-in:

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
$env:MINESWEEPER_DIAGNOSTICS = "1"
python -m pytest -m live -s
```

A successful non-live run means every unit/integration test for the DOM parser,
action space, environment, CNN policy, PPO algorithm, checkpointing, resume,
`--fresh` archiving, metric continuity and device selection passed. It does
**not** exercise the live website.

---

## 8. 3–5 Game Smoke Test

Current flag-aware CLI:

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"

python -m minesweeper_rl.experiment.train --games 5 `
  --action-space flags `
  --run-dir experiments/cnn_ppo_expert_flags_v1_smoke `
  --checkpoint-interval 1 --seed 42 --device auto --fresh
```

Historical OPEN-only CLI: the same command with `--action-space open_only` and
`--run-dir experiments/cnn_ppo_expert_v1_smoke`.

> **Important — verified behavior:** `experiments/cnn_ppo_expert_v1_smoke/`
> already contains `checkpoint_latest.pt` at **episode 8** from earlier runs.
> Because `--games` is a *total* target and the runner auto-resumes, running the
> command **without** `--fresh` would immediately see 8 ≥ 5 and do no work. To
> run a genuine fresh 5-game smoke test you must either add `--fresh` (which
> archives the old artifacts under `archive/`) or use a brand-new `--run-dir`
> name. The command above uses `--fresh`.

`--fresh` is safe: it **archives**, never deletes (§16).

What this test checks (integration only):

- The browser launches and `MINESWEEPER_URL` loads the Expert board.
- `reset()` reaches a stable, freshly opened board (controlled first move).
- The CNN receives 11-channel one-hot observations of shape `[11, 16, 30]` for
  the flag-aware model (`[10, 16, 30]` for OPEN-only).
- The action mask yields 1440 OPEN/FLAG/CHORD actions for the flag-aware model
  (480 OPEN actions for OPEN-only) and no invalid action is selected.
- Each of the 5 episodes completes, a PPO update runs, `episodes.jsonl` and
  `updates.jsonl` grow, and checkpoints are written.
- The GPU is used when `--device auto` resolves to CUDA.
- The browser is closed cleanly at the end.

> **This is an integration test, NOT a meaningful measurement of learning
> performance.** 5 games is far too few to show any learning; early win rate
> will be near zero and rewards will be mostly negative.

Inspect afterward (all under `experiments/cnn_ppo_expert_v1_smoke/`):

```powershell
Get-ChildItem experiments/cnn_ppo_expert_v1_smoke
Get-Content experiments/cnn_ppo_expert_v1_smoke/episodes.jsonl
Get-Content experiments/cnn_ppo_expert_v1_smoke/updates.jsonl
Get-Content experiments/cnn_ppo_expert_v1_smoke/run_config.json
```

Expect 5 lines in `episodes.jsonl`, 5 lines in `updates.jsonl`, a
`checkpoint_latest.pt`, and `training_progress.png`. `checkpoint_best.pt` will
**not** exist for a 5-game smoke test because best checkpoints begin at
episode ≥ 10.

---

## 9. GPU Verification During Training

In a **second** terminal while training runs:

```powershell
nvidia-smi
```

For continuous refresh (2 second interval) or a live dashboard:

```powershell
nvidia-smi -l 2
nvidia-smi dmon
```

The torch-side check during a run is printed in the training header:

```text
device selection | requested=auto | device=cuda (NVIDIA GeForce RTX 5060 Laptop GPU)
cuda available: True
cuda device: NVIDIA GeForce RTX 5060 Laptop GPU
model | observation_channels=11 actions=1440 action_space=flags parameter_device=cuda
```

**GPU utilization is likely to be intermittent, not 100%.** This is expected:

- **GPU:** the CNN forward/backward passes, the PPO loss/optimizer step, and all
  rollout/minibatch tensors live on the selected device (`PPOTrainer` owns this
  boundary).
- **CPU:** Playwright, Chromium, DOM snapshot reads, board parsing, waiting for
  board stability, reward computation, action masking, JSONL writing and
  checkpoint file I/O.

Wall-clock time is dominated by browser interaction. Each PPO update is a few
forward/backward passes over a tiny `[11, 16, 30]` input (flag-aware;
`[10, 16, 30]` OPEN-only), so the GPU spikes briefly once per episode and idles
in between. Low average utilization is not a sign of a problem.

---

## 10. Understanding Training

The training loop is in `run_training` (`src/minesweeper_rl/experiment/train.py:388`).
Each iteration does exactly three things, **once per episode**:

1. **Environment interaction** — `trainer.collect_episode()`
   - Calls `env.reset()` then steps the live browser until the game terminates.
   - `collect_episode` uses `max_steps=2000` (`ppo.py:313`). A single Expert game
     needs far fewer than 2000 legitimate OPENs (381 safe cells), so hitting the
     cap marks a stuck game as a **truncated** episode; the counter still
     advances.
2. **PPO update** — `trainer.update()`
   - Computes GAE, normalizes advantages, runs `num_epochs=4` passes over the
     episode's transitions in `minibatch_size=64` chunks, and steps Adam.
3. **Metric + checkpoint bookkeeping**
   - `metrics.log_update(...)` then `metrics.log_episode(...)`.
   - `manager.save_best(...)` is attempted every episode once `episode >= 10`.
   - `manager.save_latest(...)` is written every `--checkpoint-interval`
     episodes (see below).

**The PPO update frequency is once per episode.** In `train.py:727`, the
`PPOConfig` is built with `rollout_steps=1`, and the loop calls
`collect_episode()` + `update()` each iteration. `rollout_steps=1` is a nominal
value here; the actual rollout is always one full episode because
`collect_episode` uses `stop_at_episode_end=True`.

> **Do not confuse this with `--checkpoint-interval`.** `--checkpoint-interval`
> only controls how often a checkpoint file is written. It has nothing to do
> with how often PPO updates. PPO updates after **every** episode regardless of
> the checkpoint interval.

A checkpoint is **always** written once more at the end of the run (or after a
graceful stop), independent of the interval (`train.py:535`).

---

## 11. Starting a Training Run

Current commands for the active (flag-aware) model:

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"

# Flag-aware, from scratch
python -m minesweeper_rl.experiment.train --games 2000 `
  --action-space flags --training-mode scratch `
  --run-dir experiments/cnn_ppo_expert_flags_v1 `
  --checkpoint-interval 10 --seed 42 --device auto

# Flag-aware, human-pretrained (run BC first; see Quick start 3b)
python -m minesweeper_rl.experiment.train --games 2000 `
  --action-space flags --training-mode human_pretrained `
  --pretrained-policy experiments/cnn_imitation_expert_v1/checkpoint_latest.pt `
  --run-dir experiments/cnn_ppo_expert_human_v1 `
  --checkpoint-interval 10 --seed 42 --device auto
```

Historical OPEN-only baseline (480 actions, 10 channels; the verified original
command, still reproducible):

```powershell
python -m minesweeper_rl.experiment.train --games 2000 `
  --action-space open_only `
  --run-dir experiments/cnn_ppo_expert_v1 `
  --checkpoint-interval 10 --seed 42 --device auto
```

Argument reference (from `build_parser`, `train.py:555`):

| Argument | Effect |
| --- | --- |
| `--games 2000` | **Total** number of completed episodes for the experiment. On resume this is the absolute target, not "2000 more". |
| `--run-dir experiments/cnn_ppo_expert_v1` | Isolated output directory (metrics, config, checkpoints, plot). Default is exactly this path. |
| `--checkpoint-interval 10` | Write `checkpoint_latest.pt` every 10 completed episodes. `0` disables interim saves (a final save still happens). Default is `10`. |
| `--seed 42` | Seeds Python, NumPy and torch RNGs before the run. Default is `None` (unseeded). |
| `--device auto` | Use CUDA if available, else CPU. `cuda` fails loudly if CUDA is unavailable; `cpu` forces CPU. |
| `--action-space flags` | Action space: `flags` (default; OPEN+FLAG+CHORD, 1440) or `open_only` (historical, 480). |
| `--training-mode scratch` | Initialization: `scratch` (default, random) or `human_pretrained` (load a BC checkpoint). |
| `--pretrained-policy PATH` | Required for `--training-mode human_pretrained`; must be an imitation/BC checkpoint (`checkpoint_latest.pt` from `imitation_train`). Copies encoder+actor, resets the value head. |

Other useful flags (all optional): `--fresh`, `--checkpoint PATH`,
`--max-consecutive-failures N` (default 3), `--epochs`, `--minibatch-size`,
`--learning-rate`, `--gamma`, `--gae-lambda`, `--clip-coef`, `--entropy-coef`,
`--value-coef`, `--max-grad-norm`, `--target-kl`,
`--no-advantage-normalization`, `--no-plots`, `--diagnostics`, `--url`,
`--profile`.

`--games` semantics, verified in `run_training`:

```python
while completed_episodes(trainer) < games:
    ...
```

The loop runs until the **absolute** completed-episode counter reaches
`games`. Resuming with `--games 2000` after 750 episodes runs 1250 more. It is
safe to re-run the identical command to continue.

> Do **not** start this run as part of this documentation task. The command is
> recorded here for when you are ready.

---

## 12. Monitoring Training

### Console output

One line per episode, e.g.:

```text
episode 0042/2000 | update 0042 | steps= 17 reward= +6.00 outcome=lost    revealed=  6 open=  6 | winrate=0.0% meanR=-1.20 best=n/a | policy=+0.0001 value=68.1234 entropy=6.14 kl=-0.00001 |  92.4 games/h elapsed=0.4m device=cuda
saved checkpoint at episode 50 -> experiments/cnn_ppo_expert_v1/checkpoint_latest.pt
new best win rate 4.0% -> experiments/cnn_ppo_expert_v1/checkpoint_best.pt
```

The console reports a rolling **win rate** and **mean reward** over the last 50
episodes (`CONSOLE_WINDOW = 50`), the per-update PPO losses, approximate KL,
entropy, throughput (games/hour) and the active device.

### Raw metrics (`episodes.jsonl`, `updates.jsonl`)

See §3 for the full field lists. These are the authoritative raw records.

### Plots

`training_progress.png` is written automatically at the end of training
(unless `--no-plots`). It is a 4×2 panel figure:

1. Episode reward + rolling mean
2. Rolling win rate
3. Rolling episode length (steps)
4. Rolling mine-hit rate
5. Cells revealed per episode
6. Rolling OPEN actions
7. PPO policy/value loss vs iteration
8. Entropy / approximate KL vs iteration

### How to tell training is progressing

- `episodes.jsonl` grows by one line per episode; `updates.jsonl` matches.
- `meanR` (rolling mean reward) trends upward over hundreds of episodes.
- `winrate` (rolling win rate) eventually becomes non-zero.
- `revealed` / `revealed_total` and `open_actions` increase as the policy
  survives longer.
- `entropy` generally decreases slowly from ~`ln(1440) ≈ 7.27` (flag-aware;
  `ln(480) ≈ 6.17` for OPEN-only) as the policy sharpens.
- `approx_kl` stays small (the `--target-kl` guard is off by default).

### Metrics useful for the eventual paper

- Win rate from `episodes.jsonl` (`won`) as a rolling/aggregate learning curve.
- Mean reward (`reward`) and reward distribution.
- Coverage: `revealed_total` and `cells_revealed`.
- Episode length (`steps`) and mine-hit rate (`mine_hit`).
- PPO optimization health: `policy_loss`, `value_loss`, `entropy`,
  `approx_kl`, `clip_fraction` from `updates.jsonl`.

All of these fields **actually exist**; none are invented here.

---

## 13. Pausing / Stopping Safely

The supported interruption is **Ctrl+C** (SIGINT) in the training terminal, or
SIGTERM. This is documented in the module docstring and implemented in
`install_stop_handlers` (`train.py:130`).

Behavior on the **first** signal:

1. A stop flag (`threading.Event`) is set. The handler does not kill anything.
2. The current episode **finishes** (the stop flag is checked at the top of the
   loop, so an in-flight `collect_episode` and its PPO `update` complete).
3. Its metrics are written to `episodes.jsonl` and `updates.jsonl`.
4. `checkpoint_latest.pt` is written one final time.
5. The Playwright browser is closed by the `open_environment` context manager.
6. The process exits (return code 0 unless the run recorded errors).

A **second** Ctrl+C during shutdown force-terminates immediately via
`os._exit(130)`.

Graceful interruption vs abrupt kill:

| | Ctrl+C (once) | Abrupt kill / power loss / closing the terminal |
| --- | --- | --- |
| Current episode | finishes | lost mid-episode |
| Metrics | current episode logged | final JSONL line may be truncated (it is skipped on next load) |
| Checkpoint | final `checkpoint_latest.pt` written | last periodic checkpoint remains |
| Browser | closed cleanly | Chromium may be left open / profile locked |
| Model progress | fully preserved | up to one checkpoint interval of updates lost |

Note the model weights are only as fresh as the last checkpoint. If metrics
recorded more episodes than the checkpoint (abrupt kill between logging and
saving), a resume trusts the metric log for **episode numbering only** (to avoid
duplicate ids) and continues from the checkpoint's model state — so a few
episodes of gradient updates can be lost. This is intentional and never
overwrites existing metrics.

After an abrupt kill, if Chromium misbehaves on the next launch, close lingering
browser processes and/or remove a stale lock (the persistent profile is
`browser_profile/`):

```powershell
Get-Process chrome, chromedriver -ErrorAction SilentlyContinue | Stop-Process -Force
```

---

## 14. Resuming Training

Resume by running the **same command again**. The runner looks for
`<run-dir>/checkpoint_latest.pt` and, if present, restores it automatically:

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"

python -m minesweeper_rl.experiment.train --games 2000 `
  --run-dir experiments/cnn_ppo_expert_v1 `
  --checkpoint-interval 10 --seed 42 --device auto
```

Checkpoint discovery and restoration:

- Discovery: `<run-dir>/checkpoint_latest.pt` (or the `--checkpoint` path).
- The payload is read with `torch.load(..., weights_only=True)` and validated
  before anything is restored.
- Restored on resume (`load_checkpoint`, `checkpoint.py:352`): policy weights,
  optimizer state, `iteration`, `episode_count`, `total_steps`, PPO
  hyperparameters, and the Python/NumPy/torch (and CUDA) RNG states.
- The existing `episodes.jsonl` / `updates.jsonl` are loaded (`metrics.load_existing()`)
  and appended to; they are **not** truncated or overwritten.
- The run configuration is compared against the saved one (see §15); a mismatch
  refuses the resume.

Concrete example — 750 of 2000 episodes completed:

```text
Checkpoint found.
Resuming training.
Previous episode: 750 / 2000
Previous update: 750
Previous total steps: 12345
Continuing from episode 751...
```

The loop then runs until episode 2000, i.e. 1250 more episodes.

The resume log reports:

```text
restored | episode=750 update=750 total_steps=12345
```

`--games` (target), `--checkpoint-interval` and `--device` **may** change between
sessions. Hyperparameters and board/action/observation settings may **not** —
changing them requires a new experiment (`--fresh`).

---

## 15. Checkpoints

- **Naming:** `checkpoint_latest.pt` (resume point) and `checkpoint_best.pt`
  (best rolling win rate). With `--checkpoint PATH`, the latest path is
  overridden and the best path becomes `checkpoint_best.pt` next to it.
- **Save interval:** `--checkpoint-interval N` writes the latest checkpoint every
  N completed episodes (default 10; `0` disables interim saves). A final latest
  checkpoint is always written at the end or on graceful stop. The best
  checkpoint is considered every episode once the episode number is ≥ 10
  (`best_min_episodes = 10`) and only written when the rolling win rate
  (window 50) improves.
- **Latest checkpoint:** the most recent recoverable state; this is what resume
  loads.
- **Contents** (`build_checkpoint`, `checkpoint.py:187`):
  `format` (`"minesweeper_ppo"`), `version` (`5` for the active flag-aware
  1440-action / 11-channel model; `4` is the legacy OPEN-only 480-action /
  10-channel baseline), `policy` state dict,
  `optimizer` state dict, `config` (PPO hyperparameters), `num_actions`,
  `iteration`, `episode_count`, `total_steps`, `model` metadata (class, width,
  height, channels, observation_channels, action_space), `rng`
  (Python/NumPy/torch and, when on CUDA, CUDA states), `metrics`, and `metadata`
  (algorithm, seed, URL, profile, device, human_data, initialization, and the
  embedded `run_config`).
- **Compatibility checks** (`validate_checkpoint`, `checkpoint.py:264`):
  format string; version must equal `5` for the flag-aware model (a version-`4`
  checkpoint is accepted **only** by an OPEN-only model); `num_actions`,
  `action_space`, model class/width/height/channels must match; and
  `observation_channels` must match (11 flag-aware / 10 OPEN-only), also
  cross-checked against the first convolution's input channels. In addition,
  the embedded/run-dir `run_config.json` is compared on `STRICT_CONFIG_KEYS`
  (observation channels, action count, action type, board size, mines, and the
  PPO hyperparameters).
- **If a checkpoint is incompatible:** the process logs the specific mismatches
  and exits with code 1. The checkpoint is **left untouched** for inspection.
  Passing `--fresh` archives it (never deletes) and starts a new run.

Atomic writes: each checkpoint is written to a temp file in the same directory
and then `os.replace`-d into place, so a crash during saving cannot destroy the
previous valid checkpoint.

> **Legacy artifacts (preserved under `archive/`):** `archive/smoke_test/`
> holds a flag-aware **960-action / 11-channel** checkpoint (`minesweeper_ppo`,
> pre-v4; originally `runs/smoke_test/`) and `archive/ppo_experiment.pt` is an
> old `minesweeper_ppo_v1` file (originally in the repository root). Both are
> incompatible with the current model (version 5 flag-aware, or version 4
> OPEN-only) and will be rejected on resume. They are not part of the active
> experiment; both were moved to `archive/` on 2026-09-12.

---

## 16. Starting a Fresh Experiment

`--fresh` starts a new run from random initialization, **archiving** existing
artifacts instead of deleting them. It moves these, when present, from the
run-dir into `archive/<YYYYMMDD_HHMMSS>/`:

```text
checkpoint_latest.pt
checkpoint_best.pt
episodes.jsonl
updates.jsonl
run_config.json
```

(Plus an explicitly overridden `--checkpoint` file, if any.) Plot files, the
`archive/` directory itself, and evaluation outputs are not part of the archive
set.

Use `--fresh` when:

- You intentionally want a brand-new run in an existing directory.
- A checkpoint is incompatible and you have accepted that you are starting over.
- You are running the smoke test in a directory that already has data.
- A directory has `run_config.json` but no checkpoint (it cannot be resumed).

Do **not** use `--fresh` when:

- You want to continue an existing run — that is the default behavior; `--fresh`
  would discard your continuation.
- You are unsure whether the data matters — archive, but be deliberate: a fresh
  run produces new metrics under the same experiment name, so keep the archive
  or a copy.

Example:

```powershell
python -m minesweeper_rl.experiment.train --games 5 `
  --run-dir experiments/cnn_ppo_expert_v1_smoke `
  --checkpoint-interval 1 --seed 42 --device auto --fresh
```

---

## 17. Evaluation

The evaluation entry point loads a frozen checkpoint, selects actions greedily
(`argmax` of the masked policy) under `torch.no_grad()`, never creates or steps
an optimizer, and asserts the weights are unchanged afterward (parameter
checksum comparison). Verified CLI:

```powershell
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"

python -m minesweeper_rl.experiment.evaluate `
  --checkpoint experiments/cnn_ppo_expert_v1/checkpoint_latest.pt `
  --games 50 `
  --run-dir experiments/cnn_ppo_expert_v1_eval `
  --device auto
```

Arguments: `--checkpoint` (default `runs/train/checkpoint.pt`, which does not
exist here — always pass a real path), `--games`, `--run-dir` (default
`runs/eval`), `--device`, `--seed`, `--max-consecutive-failures`, `--no-plots`,
`--url`, `--profile`, `--diagnostics`.

Use a **separate** `--run-dir` for evaluation (as above) so eval records are not
appended to the training `episodes.jsonl`. If you do share a directory, the
`phase` field (`"train"` vs `"eval"`) distinguishes the records.

### Training vs frozen evaluation

| | Training (`experiment.train`) | Frozen evaluation (`experiment.evaluate`) |
| --- | --- | --- |
| Policy | updated | frozen, greedy argmax, `no_grad()` |
| Optimizer | Adam steps | never used |
| Writes checkpoints | yes | no |
| Writes `run_config.json` | yes (first fresh start) | no |
| Output | `episodes.jsonl`, `updates.jsonl`, `training_progress.png` | `episodes.jsonl`, `evaluation_results.json`, `evaluation.png` |
| Weight immutability | not checked (weights change) | asserted byte-for-byte unchanged |

> **Determinism caveat:** evaluation runs against the **live website**, which
> serves a **new random board for every game**. There is no fixed evaluation set
> and no fixed board sequence, so evaluation is **not perfectly deterministic**
> even with a fixed `--seed`. `--seed` only seeds torch. Reproducibility comes
> from the aggregate statistics over a sufficiently large number of games, not
> from replaying identical boards. Do not claim deterministic board-level
> reproduction.

The evaluation summary is printed and saved:

```text
evaluation summary | games=50 wins=3 losses=47 win_rate=6.0% | mean_reward=... mean_steps=... mine_hit_rate=... cells_revealed=... duration=...s | invalid=0 env_failures=0
games=50 stopped_early=False weights_unchanged=True errors=0
```

---

## 18. Experimental Data for the Paper

> **PRESERVE RAW DATA.** `episodes.jsonl` and `updates.jsonl` are the primary
> experimental record. They are append-only JSON Lines: **never manually edit
> them.** Editing corrupts the run history and breaks resume continuity.

Files to preserve for the primary run:

| File | Preserve? | Why |
| --- | --- | --- |
| `episodes.jsonl` | **Yes — critical** | Per-game raw data: win/loss, reward, steps, coverage, mine-hit, timing |
| `updates.jsonl` | **Yes — critical** | Per-update PPO optimization data |
| `run_config.json` | **Yes** | Reproducibility record (config, seed, device, provenance) |
| `checkpoint_latest.pt` / `checkpoint_best.pt` | **Yes** | Model state for re-evaluation; `best` for the headline policy |
| `training_progress.png` | Yes (derived) | Convenience figure; regenerate from raw data if needed |
| `evaluation_results.json` | **Yes** | Frozen-evaluation summary for tables |
| `evaluate` `episodes.jsonl` | **Yes** | Raw evaluation games |
| `archive/` | **Yes** | Preserves superseded runs |
| `evaluation.png` | Yes (derived) | Convenience figure |

Do not run `--fresh` on the primary directory without first copying/archiving
anything you still need.

What the raw data can later generate:

- **Learning curves:** rolling/aggregate win rate and mean reward from
  `episodes.jsonl` (`won`, `reward`) vs episode number.
- **Reward curves:** `reward` distribution and rolling mean.
- **Win-rate plots:** `won` per episode, rolling and cumulative.
- **Revealed-cell / coverage plots:** `cells_revealed` and `revealed_total`.
- **PPO optimization plots:** `policy_loss`, `value_loss`, `entropy`,
  `approx_kl`, `clip_fraction` from `updates.jsonl`.
- **Tables:** aggregate summary (games, wins, win rate, mean reward, mean steps,
  mine-hit rate, mean cells revealed) via `summarize` / `load_run`.

Quick inspection:

```powershell
python -c "from minesweeper_rl.experiment import load_run, summarize; eps, ups = load_run('experiments/cnn_ppo_expert_v1'); print(len(eps), 'episodes', len(ups), 'updates'); print(summarize(eps).format())"
```

---

## 19. Troubleshooting

### CUDA unavailable / GPU not detected

Symptom: `device selection | requested=auto | device=cpu`, or
`device 'cuda' was requested but CUDA is not available`.

- Run `nvidia-smi`. If it fails, the NVIDIA driver is missing/broken.
- Run the torch check in §4. If `cuda_available` is `False`, the installed torch
  build is CPU-only or does not match the driver.
- This machine uses `torch 2.11.0+cu128` with CUDA 12.8 — a `+cu128` build is
  required for the RTX 5060 (Blackwell). A CPU wheel (`+cpu`) will never detect
  the GPU.
- Reinstall the CUDA wheel if needed, then re-check.
- Using `--device auto` degrades gracefully to CPU; `--device cuda` fails loudly
  by design.

### Browser doesn't launch

- Ensure Chromium is installed: `python -m playwright install chromium`.
- The controller runs **headed** Chromium with the persistent profile
  `browser_profile/`. A stale profile lock after an abrupt kill can block
  launch — close lingering browsers (§13) and retry.
- If a login is required, launch once manually with a long board timeout via
  `test_browser.py` and log in; the persistent profile may retain the session.

### Website unavailable / unexpected board

- Confirm `MINESWEEPER_URL` / `--url` is `https://minesweeper.online/start/3`.
- Errors such as `Cannot reset: no Minesweeper board is available to restart`,
  `new-game control ... did not appear`, or board-shape errors usually mean the
  page did not load, the site changed, or a non-game page was served. Open the
  URL in a normal browser to check.
- Enable diagnostics for detail: set `$env:MINESWEEPER_DIAGNOSTICS = "1"` and
  look for `minesweeper_rl.environment` DEBUG lines.

### Training appears stuck

- One Expert episode can legitimately take a while under browser latency. Watch
  the console line and `nvidia-smi -l 2`.
- A game that never terminates is truncated at `max_steps=2000` and counted, so
  the loop always makes progress.
- More than `--max-consecutive-failures` (default 3) consecutive exceptions
  stops the run gracefully after saving. Check the logged exception and fix the
  underlying browser/website issue.

### Checkpoint failure

- Checkpoints are written atomically; a failed save leaves the previous valid
  file. Verify disk space and directory permissions.
- If a checkpoint will not read, the error names the file. Start a new run with
  `--fresh` (archives the bad file) or inspect it.

### Resume failure

- `refusing to resume: requested configuration is incompatible ...` — you
  changed a strict setting. Re-run with the original hyperparameters, or
  `--fresh` for a new experiment.
- `the checkpoint has no run-config snapshot` — a warning; compatibility is then
  limited to the model/action/observation checks.
- Version/format/`num_actions`/channel errors mean the file is from a different
  (legacy) revision (e.g. the old 960-action / 11-channel build). Use `--fresh`.
- `refusing to start: <dir> already contains a run configuration but no
  checkpoint` — the run-dir is not resumable; use `--fresh` or another
  `--run-dir`.

### Incorrect dimensions

- The board must be exactly `30 × 16` (480 cells) and the observation must be
  10 channels. Wrong-shape errors indicate the wrong difficulty/URL, a changed
  site layout, or a mismatched policy. The policy is constructed from the live
  environment's `width`/`height`, so a non-Expert board would silently change
  the action count — always verify the run configuration records
  `width=30, height=16, num_actions=480, observation_channels=10`.

### Storage / file-writing problems

- Metrics and checkpoints are written under `--run-dir`; ensure it is on a disk
  with free space.
- A truncated final JSONL line (from an abrupt kill) is skipped on load with a
  warning — this is expected and safe.
- Windows path quoting: wrap paths with spaces in double quotes.

### Existing experiment directory

- Training resumes automatically if `checkpoint_latest.pt` exists.
- It refuses to start if `run_config.json` exists with no checkpoint.
- `--fresh` archives instead of deleting; nothing is silently destroyed.

---

## 20. Overnight Training Checklist

- [ ] `python -m pytest -m "not live"` passes (413 passed expected).
- [ ] CUDA verified (`torch.cuda.is_available()` is `True`).
- [ ] `MINESWEEPER_URL = "https://minesweeper.online/start/3"` set.
- [ ] Smoke test passed (Quick start §1 / §8).
- [ ] Resume test passed (§14) — start a small run, Ctrl+C, re-run, confirm it continues.
- [ ] Training mode chosen: `--action-space flags` with `--training-mode scratch`
      **or** `human_pretrained` (`--pretrained-policy` supplied for the latter).
- [ ] Correct experiment directory: `experiments/cnn_ppo_expert_flags_v1`
      (scratch) or `experiments/cnn_ppo_expert_human_v1` (human-pretrained).
- [ ] Seed recorded: `--seed 42`.
- [ ] 2,000-game target confirmed: `--games 2000`.
- [ ] Checkpoint interval confirmed: `--checkpoint-interval 10`.
- [ ] Laptop connected to power (GPU + browser for many hours).
- [ ] Sleep/hibernate will **not** interrupt training:
      `powercfg /change standby-timeout-ac 0` and
      `powercfg /change hibernate-timeout-ac 0`.
- [ ] Raw data location confirmed: `experiments/cnn_ppo_expert_v1/episodes.jsonl`
      and `updates.jsonl`.

---

## 21. Complete Command Cheat Sheet

Run everything from the repository root
(`C:\Users\SwaRa\OneDrive\Desktop\rl minesweeper`).

```powershell
# --- CUDA check (driver) ---
nvidia-smi
nvidia-smi -l 2

# --- PyTorch CUDA check ---
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# --- Install dependencies + browser ---
python -m pip install -e ".[test,plot]"
python -m playwright install chromium

# --- Full test suite (non-live) ---
python -m pytest -m "not live"

# --- Environment setup ---
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"

# =====================================================================
# CURRENT (flag-aware: OPEN+FLAG+CHORD, 1440 actions, 11 channels)
# =====================================================================

# --- Smoke test (5 games; --fresh overwrites/archives an existing run) ---
python -m minesweeper_rl.experiment.train --games 5 `
  --action-space flags `
  --run-dir experiments/cnn_ppo_expert_flags_v1_smoke `
  --checkpoint-interval 1 --seed 42 --device auto --fresh

# --- Option A: flag-aware from scratch (2,000 games) ---
python -m minesweeper_rl.experiment.train --games 2000 `
  --action-space flags --training-mode scratch `
  --run-dir experiments/cnn_ppo_expert_flags_v1 `
  --checkpoint-interval 10 --seed 42 --device auto

# --- Option B: record human games + behavioral cloning ---
python -m minesweeper_rl.data.record_human_dataset --dataset data/human   # Ctrl+C to stop
python -m minesweeper_rl.data.validate_human_dataset --dataset data/human
python -m minesweeper_rl.experiment.imitation_train `
  --dataset data/human --output experiments/cnn_imitation_expert_v1 `
  --epochs 20 --batch-size 64 --learning-rate 1e-3 --val-split 0.2 --seed 42 --device auto

# --- Option B: flag-aware human-pretrained PPO (2,000 games) ---
python -m minesweeper_rl.experiment.train --games 2000 `
  --action-space flags --training-mode human_pretrained `
  --pretrained-policy experiments/cnn_imitation_expert_v1/checkpoint_latest.pt `
  --run-dir experiments/cnn_ppo_expert_human_v1 `
  --checkpoint-interval 10 --seed 42 --device auto

# --- Resume any run (identical command; auto-detects checkpoint) ---
python -m minesweeper_rl.experiment.train --games 2000 `
  --action-space flags --training-mode scratch `
  --run-dir experiments/cnn_ppo_expert_flags_v1 `
  --checkpoint-interval 10 --seed 42 --device auto

# --- Restart from random weights (archives existing artifacts) ---
python -m minesweeper_rl.experiment.train --games 2000 `
  --action-space flags --training-mode scratch `
  --run-dir experiments/cnn_ppo_expert_flags_v1 `
  --checkpoint-interval 10 --seed 42 --device auto --fresh

# --- Evaluate a frozen flag-aware checkpoint ---
python -m minesweeper_rl.experiment.evaluate --action-space flags `
  --checkpoint experiments/cnn_ppo_expert_human_v1/checkpoint_latest.pt `
  --games 50 --run-dir experiments/cnn_ppo_expert_human_v1_eval --device auto

# --- Compare scratch vs human-pretrained evaluation ---
python -m minesweeper_rl.experiment.compare_evaluations `
  scratch=experiments/cnn_ppo_expert_flags_v1_eval `
  human=experiments/cnn_ppo_expert_human_v1_eval

# ---- HISTORICAL OPEN-only baseline (480 actions, 10 channels) ----
python -m minesweeper_rl.experiment.train --games 2000 `
  --action-space open_only `
  --run-dir experiments/cnn_ppo_expert_v1 `
  --checkpoint-interval 10 --seed 42 --device auto

# --- Inspect the experiment directory (use any active --run-dir) ---
Get-ChildItem experiments/cnn_ppo_expert_flags_v1

# --- Inspect metrics (counts + summary) ---
python -c "from minesweeper_rl.experiment import load_run, summarize; eps, ups = load_run('experiments/cnn_ppo_expert_flags_v1'); print(len(eps), 'episodes', len(ups), 'updates'); print(summarize(eps).format())"

# --- Inspect raw records ---
Get-Content experiments/cnn_ppo_expert_flags_v1/episodes.jsonl -Tail 5
Get-Content experiments/cnn_ppo_expert_flags_v1/updates.jsonl -Tail 5
Get-Content experiments/cnn_ppo_expert_flags_v1/run_config.json
```

---

## 22. Experimental Integrity

The **historical, verified reference experiment** is the OPEN-only baseline:

```text
Random initialization
CNN
PPO
OPEN-only
480 actions
10 observation channels
No FLAG
No CHORD
No human warm-start
No hidden mine information
Fixed reward
```

The **active flag-aware experiment** is a separate experiment with its own
definition (a fresh `--run-dir` and its own `run_config.json`):

```text
Random init (scratch) OR human behavioral-cloning init (human_pretrained)
CNN
PPO
OPEN + FLAG + CHORD
1440 actions
11 observation channels
No hidden mine information in observations
Fixed reward
```

Never change the action space, observation channels, initialization or reward
**within one run directory**. Pick the definition up front; use a different
`--run-dir` for a different definition.

These are recorded in `run_config.json`, which is enforced on resume via
`config_mismatches` over `STRICT_CONFIG_KEYS`:

`observation_channels`, `num_actions`, `action_type`, `width`, `height`,
`mines`, `gamma`, `gae_lambda`, `clip_coef`, `value_coef`, `entropy_coef`,
`learning_rate`, `minibatch_size`, `num_epochs`, `max_grad_norm`,
`normalize_advantages`, `target_kl`.

Changing any of the following constitutes a **new experiment**, not a
modification of an existing run:

- architecture (layers, channels, heads),
- reward function or reward constants,
- observation representation or channel count,
- action space (OPEN-only → FLAG/CHORD, or board size),
- PPO hyperparameters,
- seed (for a claim of a specific run),
- training protocol (episodes per update, evaluation protocol).

Start a new experiment in a **new `--run-dir`** (or `--fresh` with a clear
intent). Do not mutate an in-progress run's protocol half-way through: the
resume guard will reject hyperparameter changes, but training-target or
checkpoint-interval changes are intentionally allowed and should be logged in
your notes.

---

## 23. What NOT To Do

- **Do not manually interact with the training browser** (clicking cells,
  restarting games, editing the DOM). It corrupts the episode the agent is
  playing and the resulting metrics.
- **Do not modify the experiment while it is running** — no hyperparameter
  edits, no code changes mid-run.
- **Do not change hyperparameters half-way through a run.** The resume guard
  rejects it; if you bypass it you have created a new experiment.
- **Do not overwrite existing experimental data.** Never run `--fresh` on a run
  you still need without archiving/copying it first.
- **Do not delete or hand-edit raw metrics** (`episodes.jsonl`,
  `updates.jsonl`). They are the paper's raw data.
- **Do not confuse `--checkpoint-interval` with the PPO update frequency.** PPO
  updates after every episode; the interval only controls checkpoint file writes.
- **Do not claim deterministic reproduction on the live website.** The site
  serves a new random board each game.
- **Do not mix experiment definitions inside one run.** FLAG/CHORD and human
  warm-start belong to the flag-aware experiment (its own `--run-dir`); adding
  them to the OPEN-only reference run, or changing them mid-run, breaks it.
- **Never add hidden-mine information** to any observation; reward shaping is a
  new experiment, not a tweak to an existing run.
- **Do not use the legacy 960-action / 11-channel artifacts** preserved under
  `archive/` (`archive/smoke_test/`, `archive/ppo_experiment.pt`).

---

## 24. Research / Paper Notes

**What the CNN+PPO training run establishes**

- A reproducible, from-scratch classical RL baseline on live Minesweeper Expert,
  with a fixed CNN actor-critic, fixed reward, OPEN-only 480-action space and
  10-channel observation.
- The sample efficiency and learning trajectory of tabula-rasa PPO on a large,
  long-horizon, partially observable combinatorial task.

**What data it produces**

- Raw per-episode outcomes and costs (`episodes.jsonl`).
- Raw per-update optimization telemetry (`updates.jsonl`).
- Reproducibility config (`run_config.json`) and model states
  (`checkpoint_latest.pt`, `checkpoint_best.pt`).
- Training-progress curves (`training_progress.png`).

**What can be reported in the paper**

- Learning curves (win rate / reward vs episodes) over the 2,000-game run.
- Coverage curves (`revealed_total`) and episode-length / mine-hit behavior.
- PPO optimization health (loss, entropy, approximate KL, clip fraction).
- The final frozen-policy evaluation summary (`evaluation_results.json`).

**What still requires a separate evaluation experiment**

- Headline win rate and confidence intervals: the training run's own win rate is
  a moving target and is biased by exploration. A dedicated frozen-policy
  evaluation over many games (`experiment.evaluate`) is required, ideally
  repeated across seeds/runs.
- Cross-algorithm comparison numbers (CNN+PPO vs GNN/DDQN vs quantum/hybrid).

**Why later comparisons must use a consistent evaluation protocol**

- The environment is a live, non-deterministic website: evaluation boards differ
  every game. Comparing agents requires the **same** number of evaluation games,
  the **same** greedy/frozen policy selection, the **same** board dimensions and
  difficulty, and the **same** reported metrics (win rate, mean reward, coverage,
  etc.).
- Report the evaluation protocol alongside every algorithm so that
  CNN+PPO / GNN-DDQN / quantum comparisons are apples-to-apples.
