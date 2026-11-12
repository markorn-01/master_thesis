# Frozen tracking validation on unequal winds

The completed GPU run is `outputs/two_bubbles_unequal_iwr_compgpu9_n064/`.
Its two source strengths are 1 and 0.5 at x=0.35 and x=0.65. Source locations,
injection radius, ambient state, wind speed, resolution and output schedule
match the earlier equal-wind 64³ case. The total prescribed mass/mechanical-
energy injection is 75% of the equal case; this is deliberately not a fixed-
total-injection comparison. Online/offline equivalence passed 17 events.

## What stayed fixed

- Grouping: radius 1.5 native cells, signed normal angle 60 degrees, minimum
  retained group population 20 samples. Smaller components remain recorded.
- Event tracker: identical controlled source hashes and all previous default
  parameters, including the three-parent cap, coverage/exclusivity thresholds,
  distance penalty and numerical tie tolerance.
- Association gates: 1.5/64 and 2.5/64 domain units.
- Conditional family test: same code, 50% fragment and union coverage, score
  margin 0.1, normal angle 60 degrees and distance weight 0.25. Static and
  backward-predicted variants are both evaluated. Inputs use float64 arithmetic,
  as in the frozen equal-wind family experiment; original arrays are unchanged.

No threshold was adjusted to obtain a desired number of inner groups. The
window scan examines all triples with a shared later one-to-one continuation
whose earlier group is not already linked by a shared continuation. Empty
earlier populations are retained as no-evidence windows. The scan does not
choose groups using source ownership or normal orientation.

## Results compared with equal winds

| Observation | Equal winds, 64³ | Unequal winds, 64³ |
| --- | --- | --- |
| Shared outer combination | t=0.07→0.08 | t=0.08→0.09 |
| Combined outer history continues | Through 0.16 | Through 0.16 |
| Group observations over all events | 39 | 28 |
| Continuation segments, including isolated observations | 15 | 5 |
| Final retained groups | 3 | 2 |
| Distance settings agree | 16/16 intervals | 16/16 intervals |

Combination intervals refer to detected groups at saved times, not an exact
physical collision instant. Agreement across two distances is not physical
truth. The equal-case inner-combination subsets have tiny objective gaps;
their solid graph connections must not be interpreted as validated ancestry.

Unequal segments G00/G01 follow the two early outer groups. G02 begins at 0.09
after their combination and continues through 0.16. G03 is an isolated
20-sample inward-oriented patch at 0.13. G04 is a larger inward-oriented group
followed at 0.14, 0.15 and 0.16. Its final centroid lies near the stronger
source, but this geometric description is not an ancestry assignment. The
pipeline does not retain a second inner group through the end; that does not
establish physical absence of a second reverse shock.

At 0.12, 64 samples in nine small components are excluded by the frozen cutoff.
At 0.13, 64 samples in sixteen small components are excluded in addition to
the retained 20-sample patch. These populations remain in the original
catalogs, saved partition/raw-component files and grouping accounting table.
No missing detection is relabeled as zero physical shock strength.

## Conditional support outcomes

The automatic scan has three eligible windows: starts 0.00, 0.08 and 0.13.
Both gates and both prediction variants yield 12 independently checked
configurations.

- Window 0.00/0.01/0.02: no earlier shock groups, therefore no earlier support.
- Window 0.08/0.09/0.10: the two earlier outer groups jointly support the
  combined group. Narrow-gate union coverage is 99.74% static and 100%
  predicted; wide-gate coverage is 100%. This corroborates the selected
  geometric combination but does not establish material ancestry.
- Window 0.13/0.14/0.15: the earlier 20-sample patch has a preferred later
  group, but covers only **16.67%** of it at the narrow gate and **25%** at
  the wide gate. Both variants reject the family because coverage is below
  50%. G03 is therefore not connected to G04. Later observations cannot
  supply the missing earlier coverage under this fixed rule.

This unequal test exposes a detection/retention limitation rather than
supporting a forced two-family interpretation. The method handles the outer
history without tuning and explicitly leaves an insufficiently supported
inner predecessor relationship unresolved.

## Quality, verification and artifacts

At t=0.16, the outer group has 5,084 samples (median Mach 6.468), the inner
group 340 (median Mach 1.544). Four inner samples have upstream endpoints
inside an injection region. The flags are retained and these reported medians
use the unchanged finite-positive-Mach population; no quality-based estimator
change is introduced. There are no unknown endpoints. Minimum detected-point
boundary clearance is 0.10033 domain units, or 6.42 cells, across the run.

Readback reconstructs histograms, source flags and population accounting.
All 32 distance/interval optimizations pass an independent occupancy DP;
50 hypotheses pass independent sparse geometry/score checks. All 12 family
configurations pass independent dense geometry checks. Graph readback checks
observation completeness and edge adjacency. All animation point joins are
checked, and all 17 GIF frames decode. Twenty-seven focused tests pass.

`outputs/two_bubbles_unequal_history_v1_n064/` contains:

- `unequal_history.gif`: 17 events at 2 FPS; full surfaces, cutaway and graph.
- `event_0008.png`, `event_0009.png`, `event_0013.png`, `event_0014.png`,
  `event_0016.png`: reviewed still frames.
- `equal_unequal_comparison.png` / `equal_comparison.json`: short comparison.
- `graph.json`, `tracking_results.json.gz`: graph and complete event decisions.
- `fragment_windows.json` / `fragment_families.csv`: all conditional tests,
  including empty evidence and rejected support.
- `groups.csv`, `groups_*.npz`: retained/excluded populations and raw components.
- `group_statistics.csv`: original sample-weighted statistics and quality flags.
- `verification.json`: fixed settings, independent checks and hashes.

```sh
python -m unittest tests.experiments.test_unequal_history tests.experiments.test_tracklet_fragments tests.experiments.test_event_history tests.experiments.test_event_tracking
python -m experiments.wind_bubble.analyze_unequal_history --output-dir outputs/two_bubbles_unequal_history_v1_n064
```

Use a new output directory. CPU postprocessing only; no new simulation needed.
