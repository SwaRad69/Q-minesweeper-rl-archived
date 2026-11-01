# Collection status

> **INVALID — do not use these games for training.**

The `game_000001` … `game_000043` recordings were captured before the recorder
lifecycle fix and carry `recorder_version: 1`. The semantic validator marks all
of them invalid:

```text
games=43 valid=0 invalid=43
  pre-fix recorder version: 43
  non-fresh initial board: 22
  phantom continuation: 12
  CHORD no-op: 10
  terminal no-op: 10
  first action not OPEN: 7
  safe OPEN ending in LOST: 3
```

They are kept on disk only as historical/debugging evidence and are **never
modified**. `HumanDemonstrationDataset` excludes invalid games by default
(`include_invalid=False`) and refuses to build a dataset if none are valid, so
they cannot silently enter training.

New games (`recorder_version: 2`) recorded after the fix are valid and are the
only ones the loader will use. To inspect the invalid ones explicitly, pass
`include_invalid=True`.

See [`../../HUMAN_DATA.md`](../../HUMAN_DATA.md).
