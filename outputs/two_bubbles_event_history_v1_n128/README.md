# Automatic two-bubble event history

Generated from the validated, fixed event selections in
`outputs/two_bubbles_event_tracking_v1_n128/`; no simulation rerun or parameter
adjustment was used.

- `event_history.png`: all 53 group observations, 32 shared continuations,
  one shared combination and two distance-specific combination alternatives.
  These produce 21 continuation segments, including isolated observations.
- `segment_statistics.png`: median valid Mach and surface sample count for
  each segment with at least three observations. Symmetric curves overlap.
- `graph.json`: every observation and selected event, preserving branching
  parent/child sets and the distances at which each event was selected.
- `group_histories.csv`: all node statistics and continuation segment labels.
- `verification.json`: input/source hashes and summary counts.

Solid graph links agree at both distances; dashed links at t=0.10→0.11 are
wide-distance-only alternatives with weakly distinguished parent subsets.
Only shared continuations propagate G labels. Branches connect histories
without forcing one parent identity onto the child. G labels are not physical
shock IDs, and starts/ends do not establish physical births/deaths.

See `experiments/wind_bubble/EVENT_HISTORY.md` for methods, limitations,
verification and reproduction commands. Thirteen focused history/segment tests
passed. Graph readback checked observation accounting, edge adjacency and label
inheritance; validated inputs were hash-checked and remained unchanged.
