# Matched 128/256-cubed collision profiles

## Result

The 256-cubed flow-only run completed, and all 321 copied strips match their
recorded hashes. The corresponding 321 strips from the 128-cubed run were
also verified. Both runs use the same requested time schedule, source positions,
source strengths, physical injection radius, prescribed mass/energy budgets,
solver timestep cap, and interpolated physical ray y=0.375, z=0.5. Shared
hydrodynamic-source fingerprints match. Recorded source differences concern
the finder, capture runner, and analysis; the 256-cubed run omits finder masks.

**Higher resolution improves the outer density profile, but the original 10%
sampled-state control still does not pass.** The original collision-gap question
therefore remains unresolved by this check. This is not evidence of shock absence.

## Correcting the control identity

The original GPU assessment named every pressure ridge in the noncentral
x windows `outer_control`. Those windows also contain inner wind structures.
The two tracks passing briefly at t=0.128–0.132 lie at x approximately 0.295
and 0.705, with pressure normals opposite to the true outermost fronts.
At t=0.13 the outermost fronts are near x=0.163 and 0.837. Consequently the
two passing inner tracks cannot establish that the outer positive control passed.

The review now selects the outermost pressure ridge on each side at each event
and checks its ambient-to-downstream pressure orientation. A failed or
unavailable outermost candidate cannot be replaced by an accepted inner feature.
Ties remain unclassified. Passing both controls at the same time is recorded
separately; an isolated pass at another time cannot validate the central layer.
The original region names remain for compatibility, with explicit control flags
added to the observations. No detector formula or threshold changed.

The corrected review finds **zero persistent outermost controls at either
resolution**. The 18 passing observations on the two inner tracks are retained
as noncentral compatible features. The copied GPU report and raw strips are
unchanged; corrected reviews are saved separately as `review_n128` and
`review_n256` under the comparison output root.

## Density improvement

The following illustrative comparison uses upstream sampling six cells ahead
of the outer pressure crest and the largest downstream density within eight
cells. The theoretical ratio comes from the pressure ratio at that density
maximum; no theoretical state is substituted into the gas.

| Time | 128-cubed density ratio | 256-cubed density ratio | RH prediction, 128 / 256 | Shortfall, 128 / 256 |
|---|---:|---:|---:|---:|
| 0.08 | 2.475 | 2.936 | 3.815 / 3.843 | 35.1% / 23.6% |
| 0.12 | 2.582 | 3.046 | 3.781 / 3.796 | 31.7% / 19.8% |
| 0.13 | 2.623 | 3.081 | 3.739 / 3.781 | 29.8% / 18.5% |

Shortfall is one minus the measured density ratio divided by the pressure-based
prediction. The denser, narrower peak at 256 cubed supports a resolution effect,
but two resolutions do not establish convergence or identify its unique cause.
The central density and pressure profiles also remain resolution-dependent.

## Speed and sampling sensitivity

At each time and resolution, the outermost left ridge was checked with 9-,
17-, and 33-event two-sided fits using the original endpoint distances.
The 256-cubed run was additionally checked with a 33-event fit and endpoint
distances doubled in cell units, to match the original 128-cubed physical
separations. This gives 21 declared comparisons.

None passes the original 10% gate. Six 256-cubed comparisons at 0.12/0.13
pass the existing 20% sensitivity gate: 17/33-event native-distance tests and
the matched-physical-distance test. These are sensitivity results; the
acceptance threshold was not relaxed to claim validation. Longer fits average
motion over longer intervals and are not assumed to give exact instantaneous speeds.

Neither resolution has a late central track passing all checks persistently.
Because the outer positive control remains unsuccessful, this cannot establish
that the center is shock-free. The 256-cubed run records no production masks,
so it also cannot establish a production detection or miss.

## Figures and records

All products are under `outputs/collision_front_resolution_comparison_n128_n256`.

- `comparison/outer_profiles.png`: density, pressure, and x velocity at the
  three times. Each resolution is shifted to its own outer pressure crest;
  horizontal distances are physical domain units, not cell widths.
- `comparison/central_profiles.png`: unshifted central density, pressure,
  temperature proxy, and x velocity on the same physical ray. The first three
  rows use logarithmic axes. The dotted line is the midplane, not a shock label.
- `comparison/control_identity.png`: the complete ray at 0.13, distinguishing
  the passing inner features from the outermost control fronts.
- `comparison/outer_windows.csv`, `sampled_pairs.csv`, `density_examples.csv`,
  and `central_midplane.csv`: the declared numerical comparisons.
- `comparison/verification.json`: integrity checks, matched setup, scope,
  and per-run corrected outcomes.

Thirty-one targeted software tests pass, including the paired full-mask/flow-only
evolution check, control-identity regressions, and data-integrity/setup checks.
Raw inputs were checked before and after comparison. No simulation was rerun.

## Reproduce

```sh
python -m experiments.wind_bubble.compare_front_motion_resolution
python -m unittest tests.experiments.test_front_motion_resolution_comparison tests.experiments.test_collision_front_motion_runner tests.experiments.test_collision_front_motion tests.experiments.test_outer_control_calibration
```

Use a fresh comparison output directory. The command creates corrected reviews
if needed and refuses to reuse a review from an older analysis-source version.
