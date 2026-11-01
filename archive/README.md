# `archive/` — Legacy / Superseded Artifacts

This directory exists to keep obsolete, legacy, and superseded project
artifacts out of the active repository tree **without deleting anything**.
Everything here is preserved for historical and debugging reference only.

- **Do not** use anything under `archive/` in the current training,
  evaluation, recording, or imitation-learning pipeline.
- **Do not** delete, convert, merge, or hand-edit archived material.
- The canonical human dataset is `data/human/`. The active experiments live
  under `experiments/`. Nothing in this directory is read by any active code,
  test, or documented command.

## Manifest

Archived on 2026-09-12. Original folder/file names are preserved.

| Original location | Archive location | Reason for archival |
| --- | --- | --- |
| `datasets/human_expert/` | `archive/human_expert/` | Legacy human-data system from the old append-only `TrajectoryStore` (`games.jsonl` / `transitions.jsonl` / `meta.json`). Old JSONL schema with incompatible semantics: string `action_type`, chord folded into OPEN, `action_index` = bare cell index, no canonical 1440-action `action_id`. Superseded by the canonical `data/human/`; `HumanDatasetStore` / `HumanDemonstrationDataset` detect and reject it explicitly. Kept unconverted as historical material. |
| `datasets/_validate/` | `archive/_validate/` | 2-game / 37-transition scratch dataset written by the same legacy `TrajectoryStore` (schema_version 1, store name `_validate`). Not referenced by any code, test, or documentation. |
| `runs/smoke_test/` | `archive/smoke_test/` | Run artifacts holding a flag-aware **960-action / 11-channel** checkpoint (`minesweeper_ppo`, pre-v4). Incompatible with the current models (version 5 flag-aware 1440-action, version 4 OPEN-only) and rejected on resume. Documented as a legacy artifact to ignore in `cnn_ppo.md` §15. |
| `ppo_experiment.pt` (repository root) | `archive/ppo_experiment.pt` | Old `minesweeper_ppo_v1` checkpoint from the 15-episode live PPO verification run. Incompatible with the current checkpoint formats and not part of any active experiment. Documented as legacy in `cnn_ppo.md` §15 / §23. |
| `data/human_smoke/` | `archive/human_smoke/` | One-off smoke-test recording of the current human recorder (6 games / 30 transitions, `recorder_version: 2`, i.e. recorded before the game-lifecycle fixes; fails the current semantic validation). Its purpose — smoke-verifying the recorder before real collection — is complete. Not referenced by any code, test, or documentation. Preserved as debugging evidence of the recorder's evolution. |

### FLAGS-only reset (2026-09-17)

| Original location | Archive location | Reason for archival |
| --- | --- | --- |
| `experiments/cnn_ppo_expert_v1/` | `archive/experiments/cnn_ppo_expert_v1_open_only/` | Completed OPEN-only baseline run (checkpoint version 4, 480 actions, 10 observation channels). Superseded by the FLAGS-only architecture; a v4 checkpoint is rejected by the v5 flag-aware model. |
| `experiments/cnn_ppo_expert_v1_smoke/` | `archive/experiments/cnn_ppo_expert_v1_smoke_open_only/` | OPEN-only smoke run (`run_config.json` records `OPEN_ONLY`, 480 actions, 10 channels). Historical only. |
| `experiments/cnn_ppo_expert_flags_v1/` | `archive/experiments/cnn_ppo_expert_flags_v1_placeholder/` | Placeholder README for the old `*_expert_flags_*` naming; replaced by the clean `experiments/cnn_ppo_flags_v1/`. |
| `experiments/cnn_ppo_expert_human_v1/` | `archive/experiments/cnn_ppo_expert_human_v1_placeholder/` | Placeholder README for the old `*_expert_human_*` naming; replaced by `experiments/cnn_ppo_flags_human_v1/`. |

The old datasets under the former `data/human/` and `data/human_smoke_v3*` were
archived separately to [`../data/archive/`](../data/archive/), because the
canonical dataset `data/human/` must now be empty and FLAGS-only.

## Notes

- Nothing was deleted, overwritten, converted, or merged during the archival;
  each item was moved as-is, preserving its original folder/file name.
- The legacy-format description (why `archive/human_expert/` must never be
  read with the canonical 1440-action semantics) is documented in
  [`../datasets/README.md`](../datasets/README.md) and
  [`../HUMAN_DATA.md`](../HUMAN_DATA.md).
- Historical documentation (`PROJECT_CONTEXT.md`) still mentions some of these
  artifacts at their original locations; it describes past states of the
  project and is intentionally left unmodified.
