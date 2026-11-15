# Bounded full-sequence validation: outcome

All 20 adjacent intervals of the saved unequal-wind 128³ sequence through
0.20 were compared at both fixed matching distances. Spatial grouping and
all event criteria except maximum arity remain unchanged. The experiment
admits at most five partners instead of three.

## Results

- 36 of 40 interval/distance configurations have identical accepted events.
- Changes occur only at 0.10→0.11 and 0.19→0.20, at both distances.
- All 47 shared continuation events and the shared outer combination persist.
- Distance agreement increases from 18/20 intervals to 20/20.
- Shared geometric combinations increase from one to three. The two old
  distance-specific subset combinations are replaced by shared choices.
- Extra high-arity candidates occur only in the four changed configurations.
  Elsewhere the full original candidate lists and solver selections match
  exactly, not only the plotted graph.

| Transition | Selected earlier groups | Later group coverage, narrow / wide | Exclusion gaps, narrow / wide |
| --- | --- | --- | --- |
| 0.10→0.11 | 2,3,4,5,6 of seven inner fragments | 78.20% / 89.85% | 0.03804 / 0.01029 |
| 0.19→0.20 | All five weak-source fragments, 2,3,4,5,6 | 99.18% / 100% | 1.13143 / 1.10497 |

Group numbers are event-local; identical numbers at the two transitions do
not mean the same physical shocks. Exclusion gaps are objective-score margins,
not probabilities or physical confidence intervals.

## Important earlier-transition limitation

At 0.10 seven inward-oriented patches were retained near the stronger source.
The expanded optimizer chooses FIVE, not all seven. Groups 1 (32 samples) and
7 (60 samples) are left without links to the later inner group. No observation
is removed from the graph to hide that omission.

An independent dense probe of all seven reproduces their union coverage of
92.48% / 100%, consistent with the earlier conditional family test. However,
full-seven exclusive contributions are:

- Narrow gate: [1.88,6.39,15.79,15.79,15.79,15.79,12.41]%.
- Wide gate: [0,0,12.03,12.03,12.03,12.03,10.15]%.

Thus even raising the cap to seven would not admit the full set under the
unchanged 10% per-parent exclusivity rule. This event model and the earlier
family-support model have different semantics: a fragment may support a
family without contributing a uniquely identifiable footprint.

Both omitted groups DO participate in other admissible <=5-partner events.
They are not simply disqualified individually. Alternatives containing group
1 score lower by 0.04827 / 0.01029; alternatives containing group 7 score lower
by 0.03804 / 0.02357. These are current-model preferences, not evidence that
the omitted patches are physically unrelated. The resulting five-parent link
is an explicit partial geometric association, not complete ancestry recovery.

## What passed, and what cannot be claimed

All 40 optimizations pass independent exact occupancy DP and exclusion-gap
checks; 241 admitted hypotheses pass independent sparse geometry/coverage/score
checks. Every original candidate remains byte-equivalent as a dictionary.
All existing saved products pass established readback. The preceding 60-case
configuration comparison is hash-verified. 29 tracking/history/family tests
pass. The whole-sequence comparison figure was visually reviewed.

No changes outside the two fragment-rich transitions were observed. There
are no independent physical ancestry labels for the hydrodynamic data, so
this is NOT a measured zero-false-combination rate. Agreement between distances
is not truth. Parent multiplicity influences event scores; the cap of five,
10% exclusivity condition and bounded enumeration remain limitations.
No general turbulent-box scalability or accuracy claim is made.

Source-overlap flags and all Mach/heating measurements are unchanged. In
particular, 72 of the final 488 weaker-source samples have upstream positions
inside an injection region; better association does not validate those states.

## Artifacts and use

- `full_sequence_comparison.png`: baseline and expanded graphs, with changed
  intervals shaded. Solid lines: shared choices; dashed: distance-specific.
- `graph.json`: a separate full-sequence experimental graph.
- `tracking_results.json.gz`: complete expanded hypotheses/decisions for all
  40 comparisons, including rejected high-arity combinations.
- `changed_events.json`: every added/removed accepted event and its quality
  criteria and score gap.
- `interval_comparison.csv`, `distance_agreement.csv`, `group_statistics.csv`.
- `earlier_seven_fragment_probe.json`: independent dense earlier-family audit.
- `verification.json`: checks, counts, scope limits and hashes.

The five-parent option is suitable for further EXPLICITLY EXPERIMENTAL use
on this saved case, preserving the partial-parent limitation. The original
production tracker and histories remain unchanged. There is enough evidence
to report the bounded result; another GPU run is not needed for this validation.

Reproduce with a fresh output directory:

```sh
python -m experiments.wind_bubble.validate_expanded_history \
  --output-dir outputs/expanded_full_sequence_validation_n128
python -m experiments.wind_bubble.probe_earlier_fragment_support \
  --validation outputs/expanded_full_sequence_validation_n128
```

See `experiments/wind_bubble/FULL_SEQUENCE_ARITY_VALIDATION.md` for the protocol.
