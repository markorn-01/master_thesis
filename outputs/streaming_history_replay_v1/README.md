# Incremental tracking replay: completed checkpoint

We now have a component that accepts one sparse surface catalog at a time.
When C arrives, it evaluates delayed support for A→B and releases A. The
whole saved sequence is no longer required by the component. This is a
replay validation, not yet a hydro-callback implementation.

## What matched

| Saved case | Events | Group observations | Exact adjacent comparisons | Exact delayed results |
| --- | ---: | ---: | ---: | ---: |
| Three bubbles, 64³ | 17 | 45 | 32 | 64 |
| Unequal two bubbles, 128³ | 21 | 65 | 40 | 80 |

Adjacent counts include two matching gates. Delayed counts include both
gates and both support variants, including pending final windows; these are
not counts of physical shocks. All 38 partitions, full adjacent optimizer
results, 85 graph event explanations, segment assignments and 110 group
observations agree with the prior offline references. Conditional support,
unresolved outcomes and quality flags are preserved, with no early emission.
The three-bubble group-history table matches exactly. Quantiles, source flags,
row joins and per-event sample accounting are independently checked.

The final transition in each case has four pending records (two variants ×
two gates), with no availability event/time. No C observation is fabricated.
The existing accepted last-interval events still exist. Conditional support
does not establish physical ancestry or change inherited identities.

## Working memory and observed runtime

Both workers retained at most **two** geometry catalogs between arrivals and
used at most **three** while processing. Closing released all geometry and
adjacent-interval state. `buffer_check.png` shows the bound alongside changing
payload size. Growing surfaces can increase bytes even with a fixed buffer.

| Case | Maximum retained geometry bytes | Maximum processing geometry bytes | Recorded worker RSS high-water | Push/finish compute |
| --- | ---: | ---: | ---: | ---: |
| Three bubbles, 64³ | 389,312 | 565,712 | 98,549,760 bytes | 10.382 s |
| Unequal two bubbles, 128³ | 1,473,584 | 2,151,744 | 295,976,960 bytes | 132.617 s |

Geometry payload counts only owned positions, normals and group labels.
It excludes hypotheses, temporary arrays, caller input, source metadata and
output deltas. RSS is the process high-water observed at post-push checkpoints,
including interpreter/imports and prior output work; it is not live geometry
memory. Fresh workers load no reference histories. The separate validator
loads references, so its memory is outside these worker measurements.

These are single fresh-process correctness replays with no warmup/repetitions.
Compute timings exclude catalog I/O, output writes and reference validation.
Do not infer a speedup over batch matching, a total-memory reduction relative
to a measured baseline, or general many-bubble scaling from this table.

## Files and checks

- `summary.csv`: combined equivalence/resource summary.
- `buffer_check.png`: retained catalog counts and owned geometry payload.
- Each case directory: emitted `groups_XXXX.npz` partitions,
  `event_XXXX.json.gz` incremental records, pending `finish.json`, and
  `resource_history.csv`. Files preserve outputs as they arrive rather than
  collecting them inside the tracker.
- Each worker `verification.json`: fresh-process scope, timings, inputs and
  code/output hashes. The top-level `verification.json` records equivalence
  against frozen offline results. `qa.json` records final readback and review.

Ten new streaming tests and 23 focused regressions pass (33 total). A 40-event
equal-size input test confirms stable retained geometry size; weak-reference
checks confirm release of old arrays. Caller mutations of input/output records
cannot alter retained state. A failed push can be retried without advancing
event/segment counters. All 334 recorded hash entries pass; the buffer figure
was visually reviewed.

No original simulation, detector, estimator, partition, history or visualization
was modified. The next bounded task is connecting this tested component to the
simulation callback and validating one small GPU case. The CPU matcher cost
still needs consideration before larger many-bubble applications.

Implementation and commands:
`../../experiments/astrolink/STREAMING_HISTORY.md`.
