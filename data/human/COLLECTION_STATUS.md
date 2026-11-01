# Collection status

> **ACTIVE — FLAGS-only collection (Recorder v3).**

`data/human/` is the **active, canonical** FLAGS-only dataset (`recorder_version: 3`, `action_space: "flags"`, `num_actions: 1440`).

### Current Status
- **Games recorded**: 50 games (monotonic IDs `game_000001` through `game_000050`).
- **Next game ID**: `game_000051`.
- **Transitions recorded**: 1156 transitions.
- **Validation status**: 100% valid (`games_valid: 50`, `games_invalid: 0`).
- **Action distribution**: `open: 281`, `flag: 338`, `chord: 537`.
- **Outcomes**: 42 lost, 8 incomplete/abandoned, 0 won.

The previous pre-fix contents (recorder v1) were archived under `data/archive/`:
- `data/archive/human_v1_invalid/`
- `data/archive/human_smoke_v3/`
- `data/archive/human_smoke_v3_interrupted/`

### Operator Workflow
1. Run smoke verification in [`../../HUMAN_SMOKE_TEST.md`](../../HUMAN_SMOKE_TEST.md).
2. Record games:
   ```powershell
   $env:MINESWEEPER_URL = "https://minesweeper.online/start/3"
   python -m minesweeper_rl.data.record_human_dataset --dataset data/human
   ```
3. Validate:
   ```powershell
   python -m minesweeper_rl.data.validate_human_dataset --dataset data/human
   ```
4. Consult [`../../HUMAN_DATA.md`](../../HUMAN_DATA.md) for full architecture and operator guidelines.
