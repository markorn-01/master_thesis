# Fixed checkpoint before multi-frame tracking integration

## Purpose

The next algorithm change must answer a bounded question: does one later
observation help describe earlier fragmented shock fronts, without silently
changing accepted events or turning geometric support into physical ancestry?
This checkpoint freezes the inputs, adjacent maximum-five baseline, evaluation
rules and stopping point. It implements no new tracker and needs no GPU run.

## Fixed comparison

Synthetic inputs contain nine existing adjacent cases and seven existing
three-state cases. Each has three predefined distance gates: 48 configurations
in total. These are known development/regression cases, not held-out validation.
They cover ordinary motion, resampling, approaching surfaces, separation,
fragment combination, disappearance, future-motion evidence, indistinguishable
ancestries and competing families. Truth is used only to calculate errors;
the adjacent matcher receives only the first two geometry/label catalogs.

Real-data inputs are the complete unequal two-bubble 128³ sequence (21 states,
20 intervals) and three-bubble 64³ sequence (17 states, 16 intervals), at the
existing two physical matching distances. Their 72 interval/distance results
are reused from verified maximum-five archives. Both saved graphs are
reassembled exactly from those archives and fixed spatial partitions.

All intervals belong to the comparison, including easy continuations and
terminal intervals. Focus transitions are recorded for visual review, not
used to exclude other results. No physical ancestry labels exist for these
simulations, so changed hydro links are not called true/false links.

## Candidate and ablation

Compare the adjacent baseline with two separate, one-event-latency support
layers: static delayed families, and delayed families using the existing
normal-motion estimate. Derive future continuation seeds from the adjacent
tracker; synthetic expected continuations must not be supplied as truth seeds.
Use the existing 50% family coverage and 0.1 score margin. Keep physical
estimators, spatial groups and event thresholds fixed. Source locations and
display triangles do not determine association.

Keep three evidence channels: accepted adjacent events, conditional family
support, and unresolved cases. A supported fragment may add information
without earning an inherited identity. Do not invent a future observation at
the end of a sequence. Such support decisions remain pending there.

## Measurements and stopping point

- Synthetic: false and missed accepted links, exact configurations, and
  ambiguous/unmatched counts, separated by case and identifiability. Measure
  family coverage separately from ancestry correctness.
- Hydro: preserve original accepted events; list added support, unresolved
  groups, distance sensitivity and source-overlap flags. Conflicts remain
  explicit rather than silently overriding the frozen graph.
- Runtime: rerun methods together with one untimed warmup and three measured
  repeats, separating adjacent matching from the support-layer cost. Exclude
  file I/O and independent audits. The first-pass timing column in this
  checkpoint is explicitly unbenchmarked and cannot support a speed claim.
- Integrity: unchanged physical arrays and spatial partitions, exact graph
  accounting, independent geometry/selection checks and row-order checks.

Evaluate the two frozen delayed variants once across the full suite. Fewer
misses count as an ancestry improvement only without additional false links
on identifiable controls. A family-support improvement is reported separately.
Indistinguishable observations remain explicitly nonidentifiable. If neither
variant improves the declared task, retain the baseline and report the result;
do not automatically tune thresholds or launch another GPU simulation.

## Completed baseline

| Existing synthetic suite | Configurations | Exact | False links | Missed links |
| --- | ---: | ---: | ---: | ---: |
| Adjacent regression | 27 | 23 | 0 | 6 |
| Three-state development | 21 | 6 | 0 | 36 |

These are counts summed over chosen cases and gates, not shock counts or a
population accuracy estimate. Zero false links alongside many misses shows
the baseline's conservative behavior; it does not establish complete tracking.
The new comparison reproduces known limitations rather than discovering a new
physical result. All 48 synthetic configurations pass independent geometry
and exact occupancy-DP audits and reversed-row link checks. Twenty-one focused
tests pass, including future-information isolation and unknown-truth accounting.

Artifacts: `outputs/multiframe_tracking_baseline_v1/`. `protocol.json` contains
the frozen comparison contract; NPZs contain synthetic observations; truth is
in the separate case description. Real inputs are referenced by paths and
hashes, avoiding duplicate simulation files. Existing histories stay intact.

```sh
python -m experiments.astrolink.freeze_multiframe_baseline \
  --output-dir outputs/multiframe_tracking_baseline_v1

python -m unittest tests.experiments.test_multiframe_baseline \
  tests.experiments.test_expanded_event_tracking \
  tests.experiments.test_tracklet_fragments \
  tests.experiments.test_event_history
```

Use a fresh output directory. NumPy/SciPy are sufficient for matching; the
existing audit module also requires the project's usual plotting imports.
