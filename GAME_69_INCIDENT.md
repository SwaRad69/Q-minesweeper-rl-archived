# Game 69 incident investigation

Investigated and fixed on 2026-09-20. Original production game files and metadata were not changed. This workspace has no `.git` directory, so there is no commit/history attribution available.

## Finding and evidence limits

Game 69 was not terminated by a game-duration limit. Its saved reason is `unrecoverable_capture_gap`, its 15 saved transitions all say `running`, and its recorded duration is 13.302342 seconds. The recorder interpreted a failure to associate a board change with a pending press as game abandonment. That is a recording-integrity condition, not evidence that the human stopped playing.

There are two pre-fix call sites for this exact reason in `data/record_human.py`: pending-action recovery in `handle_event`, and repeated unexplained observations in `_handle_idle_board`. Both called `_abandon_active`, which explicitly passed `RESULT_ABANDONED` to `PersistentHumanRecorder.end_game`. The result was chosen before persistence.

The available artifacts are `data/human/games/game_000069.json` and line 69 of `data/human/metadata.jsonl`. No Game 69 console/event/observation log was found. Consequently, **the precise unrecorded press, which of those two call sites executed, and any intervening Playwright exception cannot be established retrospectively**. Claiming a particular exception, selector timeout, or raw mouse event caused this incident would exceed the evidence.

What can be established beyond the final label:

- The transcript already combines multiple modeled effects at step 1: a FLAG at `(3,3)` also reveals 38 cells.
- Step 10 is a CHORD at `(9,7)`, but its only board change is a new flag at `(4,8)`.
- The final FLAG at `(1,7)` also places a flag at `(1,8)` and reveals `(1,9)`, `(2,9)`, `(3,9)`. This is not a single FLAG transition under the project's action model.
- Code inspection identifies two real ways of producing those combined effects: later events drained into a local batch are invisible to the current action's wait, and new events arriving during a blocking observation are not checked before its candidate is accepted.
- Deterministic tests reproduced both ordering failures on the original code. A third original-code regression reproduced `unrecoverable_capture_gap` while the face read was temporarily unavailable: `_recover_pending` returned `None`, but idle handling continued into abandonment after two equal changed boards.

Thus the confirmed classification defect is **capture uncertainty being promoted to abandonment**. Action/observation ordering defects are demonstrated in code and consistent with Game 69's corrupted action boundaries. Which final trigger occurred in Game 69 remains unknown without the missing event log.

## Timeline

All times below are UTC on 2026-09-20; add 05:30 for India time. Stored start is 11:52:59.550745 IST, end is 11:53:12.853087 IST. These timestamps describe recorder commits, not physical pointerdown times: the old capture script did not save click timestamps.

| Event / zero-based step | UTC | Action | Open cells afterward |
|---|---|---|---:|
| Recorder start | 06:22:59.550745 | Active game allocation | — |
| 0 | 06:22:59.554366 | OPEN (0,0) | 24 |
| 1 | 06:23:01.316130 | FLAG (3,3); first observable attribution anomaly | 62 |
| 2 | 06:23:03.211258 | FLAG (6,1) | 62 |
| 3 | 06:23:03.891519 | FLAG (7,2) | 62 |
| 4 | 06:23:04.581453 | FLAG (7,1) | 62 |
| 5 | 06:23:05.334088 | FLAG (8,5), also two reveals | 64 |
| 6 | 06:23:06.197247 | CHORD (8,4) | 80 |
| 7 | 06:23:07.060223 | CHORD (9,5) | 81 |
| 8 | 06:23:07.858662 | FLAG (11,6) | 81 |
| 9 | 06:23:09.302047 | FLAG (10,7), also one reveal | 82 |
| 10 | 06:23:09.821222 | CHORD (9,7), only a flag changes | 82 |
| 11 | 06:23:11.209390 | CHORD (5,8) | 85 |
| 12 | 06:23:11.532130 | FLAG (3,8) | 85 |
| 13 | 06:23:11.954706 | CHORD (3,7) | 86 |
| 14 | 06:23:12.511631 | FLAG (1,7), two flags and three reveals in its delta | 89 |
| Finalization | 06:23:12.853087 | `abandoned / unrecoverable_capture_gap` | 89 |
| Metadata append | 06:23:12.856444 | Saved incomplete record | — |

Recorded action counts: OPEN 1, FLAG 9, CHORD 5. Commit intervals range from 0.322740 to 1.895128 seconds (mean 0.925519). End follows the last transition by 0.341456 seconds. Last persisted observation is step 14's `observation_after`; its individual acquisition timestamp, subsequent idle observations, actual intervals between physical moves, first warning, and first exception are unavailable. No long inactivity is demonstrated by the saved transitions.

## Execution paths before the fix

Confirmed common tail for Game 69:

```text
unresolved pending press + changed board judged unrelated
  handle_event recovery OR _handle_idle_board repeated-change handling
  _abandon_active("unrecoverable_capture_gap")
  PersistentHumanRecorder.end_game(RESULT_ABANDONED, termination_reason=...)
  HumanDatasetStore.write_game
  active recording -> incomplete artifact, result=abandoned
  _enter_resyncing / resync phase -> ignore further gameplay until fresh board
```

Reproduced ordering mechanisms:

```text
run drains [A, B, C] into a local list
  handle_event(A) -> _wait_for_stable_change
  JavaScript queue is empty; Python's B/C are invisible here
  observe board after B/C -> record their effects as A
  later B/C evaluated against an advanced stable board -> pending/rejection/gap

_wait_for_stable_change checks queue: empty
  observe/status operation yields to browser
  human clicks B; observed board includes B
  candidate accepted, or next iteration returns contaminated candidate
  B is then processed against the board already containing its effect
```

Reproduced independent transient-status path:

```text
action pending -> board now shows its effect -> status face read fails
  _recover_pending(board, None) returns None
  idle processing nevertheless continues
  second equal changed board -> unrecoverable_capture_gap -> ABANDONED
```

### All abandonment producers / callers

`HumanTrajectoryRecorder` below is in `src/minesweeper_rl/data/record_human.py`. State descriptions refer to the pre-fix implementation.

| Function / caller | Trigger / reason | State before -> after |
|---|---|---|
| `_commit_action` -> `_abandon_active` | `record_step` raises DataError; `transition_validation_failure` | recording -> abandoned partial record -> resync |
| `handle_event` -> `_recover_pending` -> `_abandon_active` | pending action doesn't match next pre-click snapshot; `unrecoverable_capture_gap` | recording/pending -> abandoned -> resync |
| `handle_event` -> `_abandon_active` | face already terminal and terminal action wasn't captured; `terminal_action_not_captured` | active partial transcript -> abandoned -> resync |
| `_handle_idle_board` -> `_abandon_active` | terminal face without terminal transition; same reason | active partial transcript -> abandoned -> resync |
| `_handle_idle_board` -> `_abandon_active` | verified fresh board; `fresh_board_reset` | active -> abandoned -> ready |
| `_handle_idle_board` -> `_abandon_active` | repeated nonfresh changed board; `unrecoverable_capture_gap` if pending, otherwise `unattributed_board_change` | active -> abandoned -> resync |
| `run` -> `_abandon_active` | explicit stop, positively closed page, or session exit | active -> abandoned; run returns |
| `PersistentHumanRecorder.end_game` in `human_dataset.py` | caller supplies abandoned | builds result/outcome and persists; does not infer abandonment |
| `TrajectoryRecorder.end_game` in `data/recorder.py` | caller supplies abandoned | legacy JSONL sink; not the production human dataset sink |
| `RecordingEnvironment.reset` in `data/recorder.py` | reset while a prior game is active | previous agent game -> abandoned, new game starts |
| `RecordingEnvironment.step` in `data/recorder.py` | terminated/truncated without WON/LOST | agent game -> abandoned |

The last two are agent/environment wrappers, not used by the live human CLI. Schema constants, summary-reader defaults for missing `result`, report counters, and storage validation also mention abandoned but do not terminate the live session. A zero-transition game is discarded instead of persisted by `_abandon_active`.

## Previous incidents

Ten earlier abandoned games exist in the current dataset. None retains a `termination_reason` identifying its path.

| Game | Duration (s) | Steps | FLAG transitions with more than one changed cell |
|---|---:|---:|---:|
| 2 | 27.849 | 24 | 2 |
| 4 | 1.593 | 2 | 0 |
| 5 | 81.632 | 54 | 8 |
| 6 | 206.034 | 138 | 21 |
| 7 | 3.842 | 4 | 0 |
| 8 | 1.518 | 2 | 0 |
| 38 | 42.753 | 20 | 2 |
| 44 | 5.159 | 3 | 0 |
| 65 | 7.546 | 5 | 0 |
| 66 | 130.512 | 60 | 15 |
| 69 | 13.302 | 15 | 6 |

Games 2, 5, 6, 38, and 66 share the multi-effect attribution pattern; this does not prove their ultimate termination trigger. Durations and transition counts do not cluster around a single timer or move limit.

Separate archived evidence exists:

- `data/archive/human_smoke_v3.record.err.log`, around lines 296–311: `a safe reveal cannot end in LOST`, then Game 5 ABANDONED, then `terminal status before the action`. This demonstrates a terminal-attribution path, not the saved Game 69 reason. The archived game's index 5 is not necessarily production `game_000005`.
- `data/archive/human_smoke_v3_interrupted/record.err.log`, lines 196–227: stop signal followed by `run -> drain_events -> Page.evaluate -> TargetClosedError`. That is a browser-lifecycle failure and cannot be assumed to explain Game 69.

Conclusion: multiple independently reproducible defects exist; a shared ordering defect plausibly explains several transcripts. The artifacts do not support attributing every historical abandonment to one bug.

## Timing, concurrency, and persistence audit

| Timer | Default | What it actually measures |
|---|---:|---|
| human `poll_interval` | 0.05 s | interval between attempts |
| human `action_timeout` | 5 s | initial action observation/recovery budget; also initial board-read budget |
| `quiescence_time` | 0.3 s | unchanged candidate interval after a click |
| `mine_grace` | 1 s | extra wait for delayed loss status on an unchanged hidden target |
| environment readiness / browser board wait | 10 s | startup cell availability |
| environment observation timeout | 2 s | acquire two identical complete DOM snapshots |
| environment observation poll | 25 ms | retry spacing within observation |
| environment action timeout / poll | 5 s / 25 ms | autonomous action execution; not a total human-game timer |
| environment new-game timeout | 30 s | autonomous navigation/reset readiness |
| storage retry | 6 attempts, 0.05 s initial, 0.5 s cap | transient file-open/replace locks |

No human total-duration deadline, inactivity timer, TTL, stale-game timer, heartbeat watchdog, move limit, or maximum failed-observation counter exists. No independent observer, detector, checkpoint writer, or finalizer task exists either: the Python recorder uses synchronous Playwright on one thread. Browser JavaScript input runs concurrently with blocking Python observations; that is the relevant race. Stop handlers set a threading.Event; they do not finalize in a competing thread.

Transitions remain in memory during play. There is no per-action checkpoint write in the canonical sink. First-game allocation updates the manifest; finalization writes a game file, metadata, and manifest. Game 69's reason is selected before those writes, so a rename/file-lock failure cannot itself be the reason recorded in this artifact. No Game 69 persistence exception was supplied/found.

The previous sink did preserve a terminal transition in memory after failed persistence, but the CLI unwound and closed the browser. A failure after the game file was committed also made a retry look like a duplicate. Both are addressed in the scoped finalization changes.

## Changes

- Keep the whole event batch visible during each action's wait. The next press's atomic pre-click board **and face** bound the previous transition, rather than an already advanced observation/current terminal face.
- Recheck the queue after board/status reads, including the final iteration, and after idle status detection. An unreadable queue is unknown, never empty.
- Use acknowledged event reads instead of destructive `splice(0)`. A lost Playwright reply leaves unacknowledged presses available for retry; acknowledgment is document-token scoped.
- Keep pending actions through observation/status outages. Do not overwrite unresolved actions with later presses. Recover a losing mine/chord even when its normalized board is unchanged.
- Treat missing face DOM as observation failure instead of silently inferring READY/RUNNING. Catch expected Playwright/runtime/validation read failures; programming assertions are not swallowed.
- Suspend an irrecoverably inconsistent transcript while the game remains active. At a verified terminal/reset/closure boundary save it with the already-supported canonical `result="incomplete"` and specific reason. Genuine reset, cancellation, page/context closure, and disconnect still produce `abandoned` when capture was intact. No partial transcript is relabeled WON/LOST without its terminal transition.
- Freeze result/time/reason before finalization and retry transient persistence faults. Resume only an identical game payload, repair interrupted metadata tail, avoid repeated metadata append, and reconcile counters. Permanent non-lock filesystem errors propagate explicitly. A simultaneous stop cannot overwrite a pending terminal result.
- Add timestamped rotating CLI audit logs at `DATASET/recorder.log` (override `--log-file`), event IDs/document tokens/capture times, capture-suspension details, last successful observation time, and explicit finalization reasons.
- Keep canonical format/version/observation/action encodings unchanged. Extend the legacy sink's accepted results with `incomplete` so it can represent the same distinction.

## Tests and validation

Before changes, all three tests in the initial `test_human_capture_ordering.py` reproduction failed: contaminated FLAG, early terminal assignment from a batched later move, and false capture-gap abandonment during unavailable status. The existing lifecycle baseline passed (37 tests), showing those cases were previously uncovered.

New/updated regression coverage includes normal and long WIN/LOSS, ten virtual days of thinking, 242 transitions, repeated DOM/status failures, ordered input during observation/status reads, real-browser batched clicks, lost evaluate reply after JS execution, actual detached board/face recovery, unchanged-board mine loss recovery, recovery when a later press's historical face is absent, ending an unrecoverable transcript at an actual terminal boundary without looping, explicit page/context/browser closure, cancellation versus terminal finalization, write failure before game creation and before/after metadata append and during manifest replacement, idempotent finalization, and propagating programming errors. Existing short-game, storage-lock, lifecycle, and validation tests remain exercised. There is no independent watchdog to fake; the action-observation deadline is exercised through timeout/recovery.

Commands and results:

```powershell
# Reproduction before fix: 3 failed
python -m pytest tests/test_human_capture_ordering.py -q -p no:cacheprovider --basetemp=.test-tmp-incident69-repro

# Focused recorder/persistence suite after fixes: 124 passed
python -m pytest tests/test_human_capture_ordering.py tests/test_record_human.py tests/test_human_recorder_lifecycle.py tests/test_human_dataset.py tests/test_storage_persistence.py -q -p no:cacheprovider --basetemp=.test-tmp-incident69-expanded

# Final complete non-live suite, including offline Chromium integrations:
# 489 passed, 11 deselected in 46.63 seconds
python -m pytest -m 'not live' -q -p no:cacheprovider --basetemp=.test-tmp-incident69-validated

# Existing production dataset validator: exit 0
python -m minesweeper_rl.data.validate_human_dataset --dataset data/human
# status=OK games=69 (won=0 lost=58 incomplete=11) valid=69 invalid=0
# transitions=1464 (valid=1464 invalid=0) open=359 flag=432 chord=673

# Live smoke: 1 passed in 6.29 seconds; isolated profile and dataset
$env:MINESWEEPER_URL='https://minesweeper.online/start/3'
$env:MINESWEEPER_PROFILE=Join-Path (Get-Location) '.test-tmp-incident69-live-profile'
python -m pytest tests/test_live_human_recorder.py -m live -q -s -p no:cacheprovider --basetemp=.test-tmp-incident69-live-approved
```

The initial pytest command without `--basetemp` failed in fixture setup because the sandbox could not use the default system temp directory (9 passed, 100 setup errors). All subsequent runs used dedicated workspace temp directories. One initial offline integration run had an overly short 50 ms fixture observation budget; the fixture now allows 500 ms, without changing production timeouts. The first live attempt was blocked by `ERR_NETWORK_ACCESS_DENIED`; the approved network-enabled retry passed.

## Remaining limits

- The exact final raw event/call site of Game 69 is unavailable; new audit logs improve future attribution but cannot recreate missing historical evidence.
- The existing validator does not reject all multi-action board effects. Independent decoding found multi-cell FLAG deltas in 35 of the 69 historical games. Its OK result is not a certification that those old action labels are trustworthy. Historical artifacts were preserved; automatic reconstruction would invent missing actions. Review those games before using them for training.
- A long physical-human Expert session has not been replayed. The live smoke is a short 9x9 session with trusted Playwright input; long-duration/many-transition coverage is deterministic simulation. Missing-face recovery is covered by real offline Chromium and the final full suite. The final historical-face/terminal-boundary guard is covered by deterministic regression tests.
- A document navigation can permanently destroy presses not yet transferred from its old JS context. Lost history cannot be fabricated; a confirmed capture gap now suspends the transcript and yields incomplete recording at a real boundary, rather than declaring an active game abandoned.
- There is still no durable per-move journal or multiprocess writer lock. Process/power loss can lose in-memory actions, and simultaneous independent collectors against the same dataset are not supported by this change. Persistence retries keep finalization pending during prolonged sharing locks; disk-full and other permanent non-lock errors surface as errors.

## Follow-up verification on 2026-09-20

The user requested verification of the remaining steps after the earlier implementation. The implementation and regression coverage were inspected, and the following checks were rerun without changing production code or the production dataset.

```powershell
python -m pytest -m 'not live' -q -p no:cacheprovider --basetemp=.test-tmp-incident69-reverify-20260920
# 489 passed, 11 deselected in 55.03s; includes offline Chromium integrations.

python -m minesweeper_rl.data.validate_human_dataset --dataset data/human
# Exit 0: status=OK games=69 (won=0 lost=58 incomplete=11)
# valid=69 invalid=0 transitions=1464 (valid=1464 invalid=0)
# open=359 flag=432 chord=673 avg_length=21.2
```

The isolated live test initially could not navigate because the sandbox returned `ERR_NETWORK_ACCESS_DENIED`. The network-enabled retry used these settings and command:

```powershell
$env:MINESWEEPER_URL='https://minesweeper.online/start/3'
$env:MINESWEEPER_PROFILE=Join-Path (Get-Location) '.test-tmp-incident69-reverify-live-profile'
python -m pytest tests/test_live_human_recorder.py -m live -q -s -p no:cacheprovider --basetemp=.test-tmp-incident69-reverify-live-approved-20260920
```

**That retry failed: 1 failed in 13.92s.** The collector logged `capture suspended game=game_000001 steps=2 phase=recording reason=unrecoverable_capture_gap pending=3 last_observation_at=1789914711.206835`. The test exhausted its 81-click loop with `sink.active_game_id == 'game_000001'`, failing at `tests/test_live_human_recorder.py:86`. No ABANDONED artifact was produced. This is an unresolved live capture/finalization failure, not evidence that the original false-abandonment classification returned.

Two diagnostic retries with the following command passed in 5.63s and 5.62s:

```powershell
python -m pytest tests/test_live_human_recorder.py -m live -q -s -p no:cacheprovider --basetemp=.test-tmp-incident69-reverify-live-diagnostic --log-file=.test-tmp-incident69-live-diagnostic.log --log-file-level=INFO
```

To investigate the intermittent failure, eight additional isolated runs were made, preserving separate DEBUG logs and stopping if a test failed:

```powershell
for ($attempt = 1; $attempt -le 8; $attempt++) {
    python -m pytest tests/test_live_human_recorder.py -m live -q -s -p no:cacheprovider "--basetemp=.test-tmp-incident69-repeat-$attempt" "--log-file=.test-tmp-incident69-repeat-$attempt.log" --log-file-level=DEBUG
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

All eight passed (5.61s, 6.34s, 6.46s, 5.91s, 5.59s, 5.65s, 5.67s, 5.55s); each saved a complete LOST result in its isolated dataset. Overall, the network-enabled live checks were **10 passes and 1 failure**. Repetition was prompted by the failure; it does not resolve or invalidate it.

Inspecting `.test-tmp-incident69-repeat-4.log` revealed a further limitation of the live test: events 2, 3, and 4 were rejected with `a safe reveal cannot end in LOST`; later event 5 was recorded as the terminal action. The test still passed because it checks the final result and existing dataset validator, not whether every effective live click was attributed correctly. The available log does not contain the raw DOM snapshots needed to establish why these contradictions occurred. The test's click loop also does not execute the production idle/recovery loop between every move, so the failed finalization cannot yet be attributed solely to production behavior.

**Verification conclusion:** all non-live checks and existing production dataset validation pass. Live validation is not clean: an intermittent capture failure and suspicious rejected-action logs remain to investigate. A physical-human long Expert session and improved live event/DOM diagnostics are still needed. No claim of complete live capture reliability is made by this follow-up.
