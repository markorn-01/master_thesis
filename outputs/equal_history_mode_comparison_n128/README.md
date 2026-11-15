Equal-wind 128³ additional configuration check using the reusable workflow.
No new hydro simulation, spatial threshold or event criterion was introduced.

- 30/32 interval/distance selections are unchanged.
- Only 0.10→0.11 changes, at both matching distances.
- Two shared five-parent combinations replace the old distance-specific
  three-parent subsets. Parent sets: [2,3,4,5,6]→1 and [9,10,11,12,13]→14.
- All 32 shared continuations and the original outer combination persist.
- Distance agreement increases from 15/16 to 16/16 intervals.
- Each selected set covers 77.98% / 89.89% of its later group at narrow / wide
  gates. Score gaps are ~0.02651 / ~0.01008; these are not probabilities.
- Each inner surface had SEVEN earlier retained patches; only five are linked.
  The omitted sets are [1,7] and [8,14]. This is partial geometric association,
  not recovery of every physical predecessor.
- Each later 1,108-sample inner group has 32 upstream endpoint source overlaps.
  Measurements/flags are preserved; association does not validate source-
  affected Mach or heating reconstruction.

All spatial partition arrays and measured group quantities match exactly
between modes. All original candidates are retained. Both complete graphs
reassemble from their records. All 64 geometry/optimizer checks from the two
runs are verified, with recorded source/input hashes. Changed combinations
remain geometrically local to the respective source; nearest-source descriptors
are inspection only and were not inputs to tracking. No physical false-link
rate can be measured without independent truth labels.

`history_comparison.png`: unchanged baseline versus experimental graph.
`changed_events.json`: all changes, coverages, source descriptors and gaps.
`verification.json`, `interval_comparison.csv`, `distance_agreement.csv`: checks.
34 focused unit/regression tests pass. No thesis text was changed.

```sh
python -m experiments.wind_bubble.compare_tracked_histories \
  --baseline outputs/equal_history_baseline_validation_n128 \
  --candidate outputs/equal_history_expanded_validation_n128 \
  --output-dir outputs/equal_history_mode_comparison_n128
```

Use a fresh output directory. The command also works for other completed
outputs of `track_shock_history` with the same input catalogs/settings.
