# Incremental sparse-surface histories

## Purpose and status

`StreamingShockHistory` processes each surface catalog when it arrives rather
than requiring the entire saved sequence in memory. It reuses the frozen
grouping, maximum-five event optimizer, continuation-segment policy, group
statistics and both delayed-support variants. It changes execution order,
not physical estimators or ancestry rules.

Saved-data replay is validated for unequal two bubbles at 128³ and three
bubbles at 64³. The component is **not yet connected to a hydro callback**.
No new GPU simulation, physical convergence result or tracking-accuracy
improvement is claimed by this execution-workflow comparison.

## API and emitted records

Create `StreamingShockHistory()`, call `push(catalog)` in strictly increasing
actual-time order, write/discard its returned delta, then call `finish()`.
The default explicitly selects `HistorySettings(max_partners=5)`; the original
batch/default settings and prior results are unchanged.

Required catalog fields are positions, normals, time, grid spacing, Mach,
Mach-valid flags, source positions and upstream/downstream source-overlap
flags. This bounded version requires fixed grid spacing and source positions.
Source coordinates describe existing statistics; they do not determine the
geometry grouping or temporal event association.

Each `push` returns the current partition, original-statistic group nodes,
Mach/count/quality rows, event accounting, adjacent optimizer records, newly
available delayed-support records and buffer measurements. Shared
continuations inherit labels. Combinations/separations and distance-specific
alternatives retain the same explicit graph representation as the offline
history. The component does not store all prior nodes/events or emitted deltas.

On arrival of C, it emits accepted B→C events immediately and the delayed
support assessment for A→B, with C's event/time recorded as availability.
Conditional support stays separate and assigns no inherited identity.
`finish` emits four pending terminal records (two gates × two variants), not
four physical relationships. It invents no future observation and releases
retained geometry and adjacent-result state. Closing and input validation
are explicit; a failed push does not partially commit the event.

## Memory and ownership

Between calls the component retains at most two owned geometry catalogs:
positions, normals and group labels. It uses a third while processing A–B–C,
then releases A. Mach and quality statistics are emitted immediately instead
of retained as complete catalog sequences. Only the previous adjacent
interval's hypotheses and current group-to-segment mapping remain in state.

Input arrays are copied so a caller's views cannot keep full hydro arrays
alive through this state. Emitted partition/hypothesis records are detached;
mutating them cannot corrupt the next event. A caller must still write and
discard output deltas rather than accumulating them in memory.

The bound is a number of recent catalogs, not a constant byte/RSS guarantee.
Surface population and active-group complexity can grow. Geometry `nbytes`
exclude Python hypotheses, caller input, output copies, source metadata and
temporary matching/statistics arrays. Worker RSS includes imports and
temporary allocations and is recorded as a high-water at post-push checkpoints.

## Completed replay comparison

| Case | Events | Group observations | Exact adjacent results | Exact delayed results |
| --- | ---: | ---: | ---: | ---: |
| Three bubbles, 64³ | 17 | 45 | 32 | 64 |
| Unequal two bubbles, 128³ | 21 | 65 | 40 | 80 |

All partitions, complete adjacent hypotheses/decisions, graph events, segment
labels and existing node statistics agree exactly with the frozen references.
Both support variants match their full offline result dictionaries, including
unresolved and pending cases. Availability is checked separately. Independent
sorted-value quantile, source-flag and original-row checks cover statistics;
the three-bubble table also reproduces all existing rows exactly.

Ten new tests and 23 focused regressions pass. Tests include batch equivalence,
delayed emission, gaps/empty catalogs, close behavior, invalid inputs, recovery
after algorithm failure, input/output isolation and release of old arrays.
A 40-event equal-size test retains constant geometry bytes after the second
event and only the current mapping/previous interval; it does not estimate
general physical tracking accuracy.

Each saved sequence was replayed in a separate fresh CPU process without
loading reference histories. Observed push/finish compute totals are 10.382 s
and 132.617 s, respectively. Each is one correctness run, without warmup or
repetitions; input/output and independent validation are excluded from those
compute totals. They are not a new comparative performance benchmark.

Results, memory measurements and limits:
`outputs/streaming_history_replay_v1/README.md`.

## Reproduce

Use fresh output subdirectories and the NumPy/SciPy analysis environment:

```sh
python -m experiments.astrolink.replay_streaming_history \
  --directory outputs/three_bubbles_iwr_compgpu7_n064 \
  --output-dir outputs/streaming_history_replay_v1/three_n064

python -m experiments.astrolink.replay_streaming_history \
  --directory outputs/two_bubbles_detection_extension_iwr_n128 \
  --output-dir outputs/streaming_history_replay_v1/unequal_two_n128

python -m experiments.astrolink.check_streaming_history

python -m unittest tests.experiments.test_streaming_history \
  tests.experiments.test_delayed_support tests.experiments.test_event_history \
  tests.experiments.test_group_history_table
```

The next implementation boundary is callback integration and a small
online/offline GPU test. Matching remains the dominant CPU work; buffering
alone does not establish efficient many-bubble scaling.
