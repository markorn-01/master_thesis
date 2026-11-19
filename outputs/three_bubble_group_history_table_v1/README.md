# Verified three-bubble group-history table

`group_history.csv` contains one row per detected retained group and event:
45 observations across 17 saved times, organized into 15 existing geometric
continuation segments. These are observations/segments, not physical shock
counts. Every statistic is recomputed from original sparse samples and checked
against the existing history. No grouping or association is rerun.

Columns identify the node, actual event time, event-local group, continuation
segment and inheritance status, sample count, valid/invalid Mach counts,
median Mach, sample p16/p84 and upstream/downstream source-overlap counts.
Hashes identify the original catalog rows and positions. Catalog row numbers
are local indices, not persistent moving points. Missing statistics are blank
in CSV and null in JSON, never fabricated zeros.

`event_accounting.csv` separately includes all 17 events, including the empty
initial state. Retained plus unretained counts equal all detected samples.
Every retained sample contributes to exactly one group row. The log-scale
sample-count display uses retained group counts, not triangle counts or area.

`tracking_events.json` preserves all shared accepted and distance-specific
event explanations, each becoming available at its child event. Only shared
continuations retain segment labels; combinations start new segments.

`conditional_support.csv` / `.json` contain one consolidated additional
relationship: e0013:g9 → e0014:g4, supported by the continuation to e0015:g3.
It is available only at event 15, actual time approximately 0.15. Both delayed
variants and both gates support it. It never changes group counts, Mach
statistics or inherited labels. In particular, the B-group statistic already
uses all B samples regardless of which earlier fragments have tracking links.

`visibility.json` records which observations, events and conditional edges
may be shown at each playback event. The extra edge is absent at events 13
and 14 and visible from event 15. This is an offline display of saved results
with causal availability, not an online integration of the general tracker.

Readback reproduces all 45 CSV/JSON rows. An independent sorted-value
calculation checks 135 quantiles. All 17 sample-accounting checks and 250
hash-entry checks pass. Five new tests cover group accounting, invalid Mach
values, empty events, changed inputs and hiding delayed support until C.
See `verification.json` and `qa.json` for evidence.

Reproduce with a fresh directory:

```sh
python -m experiments.wind_bubble.build_group_history_table \
  --output-dir outputs/three_bubble_group_history_table_v1
```

The synchronized movie is in `../three_bubble_group_demo_v1/`.
