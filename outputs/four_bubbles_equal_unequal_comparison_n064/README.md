# Equal versus unequal four-source interaction

Completed comparison of the copied 64³ GPU runs. The source layout, initial
conditions, injection radius, wind speed, timestep limit, 17 analysis times
and detector/tracker settings match. Strengths change from `[1,1,1,1]` to
`[1.5,1,0.75,0.75]`, in the existing source order. Total prescribed mass and
mechanical-energy coefficients match; this is not a measurement of equal
net gas-energy deposition under the EI conversion.

## What changed

| Measurement | Equal strengths | Unequal strengths |
|---|---:|---:|
| Group observations, summed over all saved times | 89 | 71 |
| Continuation segments | 40 | 25 |
| Shared continuation relationships | 49 | 46 |
| Shared combination relationships | 5 | 5 |
| Detected samples at t=0.16 | 9,812 | 10,187 |
| Retained group sizes at t=0.16 | 8,512; 325; 325; 325; 325 | 9,005; 553; 333; 127; 127 |
| Median Mach at t=0.16 | 6.4769 | 6.4455 |
| Median direct sampled density ratio at t=0.16 | 1.6353 | 1.6264 |
| Minimum detected-surface boundary clearance, in cells | 8.15 | 6.70 |

The unequal run visibly breaks the equal run's symmetry. Its group records
also evolve at different times. At t=0.10, an accepted two-parent combination
is recorded in the unequal case; the equal case's accepted four-parent
combination appears at t=0.11. These are times of geometric association, not
independently measured physical collision times.

At t=0.13, the equal run has 25 retained groups, while the unequal run has
eight. At t=0.15, the counts are five and eleven. Both end with five retained
groups, but their sizes differ. Fewer total observations or segments do not
show that tracking became more accurate. Similarly, ending with five groups
does not prove the same five physical shocks or a source-owner assignment.

The global Mach and sampled-density-ratio histories remain similar despite
the geometric differences. This illustrates why global histograms and spatial
views provide different information. Distribution counts weight surface
samples equally, not by physical surface area.

## Sampling limitations retained in the comparison

No endpoint positions are unknown. Both runs have downstream endpoints
inside injection regions at t=0.01–0.03. The unequal run has upstream overlaps
at t=0.14 (44 samples), t=0.15 (46) and t=0.16 (46); the equal run has 24 at
t=0.15 and 36 at t=0.16. Flags are advisory and remain in the displayed
statistics. Geometric grouping does not resolve those state-sampling issues.
Early positive density ratios below one are also retained; these are direct
sampled ratios and are not forced to follow a theoretical shock relation.

## Figures

- `event_0009_comparison.png`, `event_0013_comparison.png`,
  `event_0016_comparison.png`: original Mach-colored 3D surface points plus
  density and pressure slices. Each column shares color scales across both
  runs and all three times. Camera and spatial bounds are fixed. Hollow stars
  project source positions onto the z=0.5078125 slice; none lies in that plane.
- `histories.png`: global median/percentile diagnostics, sample counts, retained
  groups, samples excluded by grouping, and endpoint-overlap counts. Percentile
  shading represents the population spread, not uncertainty.
- `final_histograms.png`: saved Mach and density-ratio bin counts at t=0.16.
- `matched_events.csv`: all history fields and group/sample accounting at every
  event, including invalid and out-of-range counts.

These are static inspection figures, not new smooth meshes or animations.
No regrouping, retuning, new relationships or hydrodynamic evolution is added.

## Verification

All 180 original files in each run were read and fingerprinted. All 25
recorded source fingerprints in each run match the local checkout. Local
checks repeated the 21-field online/offline comparison, exact histogram
counts, sparse catalogs, slices, full tracking records/partitions, row
fingerprints, histogram reconstruction, sampling flags, and the independent
whole-sequence batch reference. Both runs passed. Four targeted comparison
tests reject mismatched geometry, timing, input totals and strength ordering.

The GPU reports final gas-state agreement; complete gas states are not saved,
so that comparison cannot be independently repeated from these copied files.
`verification.json` preserves this distinction and hashes the original inputs,
comparison source and generated products. It does not claim independent
validation of physical ancestry, complete shock detection or turbulence.

Reproduce into a fresh output directory:

```sh
python -m experiments.wind_bubble.compare_four_bubble_strengths \
  --equal outputs/four_bubbles_iwr_compgpu2_n064 \
  --unequal outputs/four_bubbles_unequal_iwr_compgpu2_n064 \
  --output-dir outputs/four_bubbles_equal_unequal_comparison_n064_repeat
```

## Bounded conclusion

The existing workflow executes consistently for this unequal four-source
interaction and exposes changes in geometry and association history without
assuming symmetry. This completes the planned equal/unequal comparison.
Additional random spacing, source activation times, higher resolution or
turbulent-box performance are separate decisions, not prerequisites for
reporting this result.
