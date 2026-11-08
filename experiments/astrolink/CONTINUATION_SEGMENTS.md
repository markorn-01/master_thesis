# Provisional histories from clear local continuations

This step connects unique local candidates across adjacent saved times.
It deliberately stops at uncertainty. Segment numbers are labels for
short history stretches, not established physical shock identities.

## Rule

At both tested distances (1.5/64 and 2.5/64 domain units), the parent must
have the same accepted target, the target must have exactly one eligible
parent, and support must be nonredundant. Intersect these candidate pairs
across the two settings. Only those pairs inherit the previous segment
number. Every other retained child starts a new segment. No deleted group
is recovered across a gap, and no multiple-parent ancestry is declared.

This changes neither matcher acceptance nor saved physics. It is a
conservative interpretation layer beside the original geometric links.
Agreement at two settings is not proof of correct physical correspondence;
a false unique match could still persist under both settings.

## Real-data result

Using all 17 saved events, original cutoff-20 native graph groups and
original physical normals:

- 53 group observations form 25 provisional segments;
- 28 unique continuation links connect adjacent observations;
- 7 segments have multiple observations; 18 contain a single observation.

The three longest segments are shown in `provisional_histories.png`:

| Segment | Time interval | Observations | Description from original grouping |
| --- | --- | ---: | --- |
| S000, S001 | 0.01–0.07 | 7 each | Two early large detected groups |
| S002 | 0.08–0.16 | 9 | Later large outer detected group |

The early histories overlap because the setup and measurements are
symmetric. The change from two groups to one breaks the early segments;
S002 is not assigned the physical ancestry of S000 or S001. That is an
unresolved representation/identity transition, not a declared shock merger.
Inner-group histories also break at multiple-parent support rather than
silently absorbing small patches.

The left plot counts detected surface samples, not surface area or gas
mass. The right plot takes the median of finite positive Mach measurements
with `mach_valid=True` in each group; each valid sample has equal weight.
Neither is a new physical estimator or a measurement of individual gas
particle trajectories. Full per-group observations, including short segments,
remain in `group_histories.csv`; the figure shows only the three longest.

## Checks and known losses

Twenty-seven focused tests pass, covering the status layer, support controls,
new segment inheritance, splits, multiple-parent events and empty-event gaps.
The nine existing synthetic cases are additionally evaluated at three
distances (.1, .2, .35 in toy units), with intersection across all three.
Translation within the gate and fresh sampling retain the stipulated link.
The nearby disappearing-sheet false extra link is not inherited because
the result has multiple parents; the true link is also withheld there.

There are no false inherited pairs in these nine particular controls, but
several stipulated true links are missed: large motion, approaching sheets,
split/fragment cases and the nearby-sheet disappearance. Even opposite
normals do not rescue a link when one tested distance is too small to reach
the target. This is the explicit cost of the conservative rule, not a
claim that general physical tracking has been solved.

Saved catalog hashes remain unchanged. CSV readback independently recomputes
sample counts and Mach medians. Every inherited edge is checked against
both original support records for acceptance, unique parentage and
nonredundancy; histories are checked contiguous with no gap bridging.
Source and input hashes are saved in `verification.json`.

## Reproduce and stopping point

```sh
python -m experiments.wind_bubble.build_continuation_histories --output-dir outputs/two_bubbles_provisional_histories_n128
python -m unittest tests.experiments.test_continuation_segments tests.experiments.test_continuation_status tests.experiments.test_support_evidence tests.experiments.test_controlled_tracking
```

CPU postprocessing only; NumPy, SciPy and Matplotlib are sufficient. The
output directory must not already exist. Results are `group_histories.csv`,
`continuation_links.csv`, `segments.csv`, `controlled_checks.json`, the
example figure and `verification.json`.

Stop at this checkpoint to inspect the histories. There is no automatic
resolution of ambiguous identities, no physical merger tree, no new
simulation, no mesh dependence and no additional algorithm search in this
step. These histories can be discussed as measured group evolution with
explicit breaks, while uncertain collision transitions remain visible.
