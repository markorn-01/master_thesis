# Bounded full-sequence five-partner validation

Scope: the completed 128³ unequal-wind sequence through t=0.20, all 21 saved
states/20 adjacent intervals, two fixed matching distances (1.5/64 and 2.5/64).
This is an offline comparison on identical spatial groups, not a new hydro run.

The baseline is the committed three-partner history. The candidate extension
is the separate experimental tracker allowing up to five partners in either
combination or separation events. Signed normals, coverage, exclusivity,
distance penalty, score, conflict constraints and tie tolerance remain fixed.
The original tracker/history files are not modified.

## Checks

1. Verify saved-data readback, prior experiment/source hashes and group-array
   alignment at every event. Read original 40 interval/distance records.
2. Compute all 40 expanded results. Assert every original candidate, including
   its score and coverage fields, is preserved exactly. If no new candidate
   is possible, require the full candidate list and optimizer selection to
   remain identical.
3. Independently recompute sparse geometry/coverage/score for every admitted
   event and exact group-occupancy dynamic programming for every optimizer
   choice and exclusion gap.
4. Record every added/removed accepted event, its score gap and minimum
   coverage/exclusivity. Compare agreement between distances separately.
   Preserve unresolved and distance-specific choices.
5. Assemble a separate whole-sequence graph and compare it with the baseline.
   Retain source-overlap flags in group statistics. Review changed intervals
   and the graph; do not infer physical false-link rates from hydro data that
   have no independent ancestry truth labels.

The preceding synthetic comparison covers 60 configurations and records known
failure modes. Unit/regression tests remain separate from the real-data
optimizer audit. A favorable result establishes bounded geometric consistency,
not general turbulent-box accuracy, physical ancestry or scalable complexity.
The cap of five and bounded candidate counts are still explicit limitations.

Results and interpretation: `outputs/expanded_full_sequence_validation_n128/`.

```sh
python -m experiments.wind_bubble.validate_expanded_history \
  --output-dir outputs/expanded_full_sequence_validation_n128
```

Use a fresh output directory. No GPU required. Source catalogs, saved spatial
partitions and original graph remain immutable; candidate comparisons and the
experimental graph are written only to the new directory.
