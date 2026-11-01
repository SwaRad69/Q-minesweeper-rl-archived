# `data/archive/` — archived datasets

Old datasets preserved for historical/debugging reference only. **Do not** use
anything here for training, imitation learning, conversion, or merging.

| Archived item | Reason |
| --- | --- |
| `human_v1_invalid/` | The previous `data/human/` contents: 43 games, all `recorder_version: 1` (recorded before the recorder lifecycle fix). The semantic validator marks 100% of them invalid (non-fresh initial board, phantom continuation, CHORD no-op, terminal no-op, first action not OPEN, safe OPEN ending in LOST). Superseded by the fresh FLAGS-only `data/human/`. |
| `human_smoke_v3/` | `recorder_version: 3` smoke recordings made during recorder development. Not a training dataset; kept as debugging evidence. |
| `human_smoke_v3_interrupted/` | Interrupted recorder smoke run (`recorder_version: 3`). Debugging evidence only. |
| `human_smoke_v3.record.*.log` | Console logs from the smoke runs. |

The repository-level legacy archive (old JSONL TrajectoryStore datasets and
pre-v5 checkpoints) lives in [`../../archive/`](../../archive/). The canonical
FLAGS-only dataset is [`../human/`](../human/).
