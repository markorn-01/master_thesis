# Imported 128³ two-bubble run: comparison with 64³

## What was checked

The completed GPU run `outputs/two_bubbles_budget_matched_iwr_compgpu3_n128`
contains 17 events over t=0–.16. It uses the same source geometry, nominal
injection radius, ambient medium, gamma and requested/actual analysis times as
the existing 64³ reference. Its mass-rate parameter is .00791011235955 per
source, giving the same discrete mass rate and prescribed mechanical-energy
coefficient as the reference.

All saved history, histogram, surface and slice products were read back for
both resolutions. Checks cover exact histogram reconstruction and accounting,
percentile summaries, sampled density ratios and validity flags, per-source
endpoint flags, surface exclusion from both sources, finite geometry, times,
slice shapes/coordinates and boundary clearance. All 128³ recorded source hashes
match the local code. The historical 64³ runner hash differs because the budget-
matching option was added afterward; its analysis/source/configuration hashes
remain consistent. That expected difference is recorded, not silently ignored.

This is saved-product verification. It does not repeat the 128³ analysis against
stored full gas states: the higher-resolution run did not retain them. The
earlier 64³ online–offline validation remains a separate result.

## Main spatial finding

At t=.12, using a graph radius of 1.5 native cells and a 60° signed-normal limit:

- 64³ has an outer component of 4112 samples and ten inner patches of 14–16
  samples. The 20-sample cutoff hides all ten inner patches.
- 128³ has an outer component of 17152 samples and two large inner components
  of 1108 samples each, plus smaller residual components totaling 40 samples.

The saved fields and point clouds therefore support a more connected inner-
surface representation at higher resolution in this experiment. They do not
prove that every inner component is an independently identified physical shock,
or that the higher-resolution solution is converged.

Smaller components still matter. At .13 the high-resolution native graph has
two additional 32-sample components, and at .14 two additional 52-sample
components. These pass a fixed cutoff of 20, giving five retained groups.
Their persistence shows that resolution alone does not remove all grouping
ambiguity.

## Neighborhood and cutoff controls

The comparison separates three choices:

1. **Native cells:** radius 1.5 cells and minimum 20 samples at each resolution.
2. **Native cells, scaled count:** keep radius 1.5 cells but scale minimum count
   by N²: 20 at 64³, 80 at 128³.
3. **Common physical radius:** use radius 1.5/64 code units (1.5 cells at 64³,
   3 cells at 128³), with the scaled sample cutoff.

The N² count scaling is a surface-sampling heuristic, not a measured physical
area threshold. Curvature, orientation and sample selection affect the relation.
Raw components remain saved under every setting; exclusions are not erasures.
Both 30° and 60° angle limits are evaluated over all 17 events.

Retained counts for the illustrated 60° limit:

| Resolution / choice | t=.12 | t=.13 | t=.14 | t=.16 |
| --- | ---: | ---: | ---: | ---: |
| 64³, native radius, cutoff 20 | 1 | 11 | 3 | 3 |
| 128³, native radius, cutoff 20 | 3 | 5 | 5 | 3 |
| 128³, native radius, cutoff 80 | 3 | 3 | 3 | 3 |
| 128³, common physical radius, cutoff 80 | 3 | 3 | 3 | 3 |

Equal retained counts do not imply equal sample membership. At .14 the native
128³/cutoff-80 setting excludes 104 samples in small components. Increasing
the radius to the common physical scale connects those samples into larger
components, leaving zero excluded samples. The raw labels and sizes distinguish
these outcomes even though both displays contain three groups.

## Statistical diagnostics

At the final saved event t=.16:

| Measurement | 64³ | 128³ |
| --- | ---: | ---: |
| Selected surface samples | 6168 | 25136 |
| Median pressure-profile Mach | 6.58848 | 6.66252 |
| Median directly sampled density ratio | 1.53660 | 2.42942 |

The Mach medians differ by about 1.12%, while the sampled density-ratio medians
differ by about 58.1%. Consequently similar Mach histories do not establish
agreement of all shock diagnostics. These are count-weighted populations of
detected samples, not one-to-one matched surface elements or area-weighted
measurements. Differences in populations, state sampling, flow resolution and
the source profile can all contribute; this comparison does not isolate a
single cause or establish which density ratio is physically accurate.

Ratios below one are retained as before. Source-overlap flags remain advisory
and do not remove samples from these histories. At .16 the 128³ catalog has 24
upstream endpoints inside a source region, while the 64³ catalog has none.
The endpoint-fraction panel shows the full time dependence, including larger
overlap fractions at early times.

Minimum detected-surface boundary clearance across saved events is .100318
(6.42 cells) at 64³ and .104028 (13.32 cells) at 128³. This verifies a geometric
margin for the detected samples, not the absence of all boundary influences.
No unknown sample endpoint positions occur in either dataset.

## Figures and data

Results: `outputs/two_bubbles_resolution_comparison_n064_n128/`.

- `event_0004_comparison.png`, `event_0008_comparison.png`,
  `event_0012_comparison.png`, `event_0013_comparison.png`,
  `event_0014_comparison.png`, `event_0016_comparison.png`: paired pressure,
  density and Mach-colored refined samples. Color limits are shared across
  both resolutions and all six events.
- `inner_patch_sensitivity.png`: grouping choices at .12/.13/.14.
- `diagnostic_histories.png`: Mach/density-ratio percentile histories, sample
  counts and endpoint source-overlap fractions.
- `grouping_summary.csv`, `raw_component_sizes.csv`, `groups_*.npz`: raw and
  retained graph information for each event and setting. Group IDs are local.
- `diagnostic_histories.csv`: paired scalar histories.
- `verification.json`: import and analysis checks, hashes and display limits.
- `analysis_readback.json`: saved graph-accounting and selected order-invariance checks.
- `SHA256SUMS`: output checksums, excluding the manifest itself.

The gas slices are at z=.5078125 and .50390625, respectively. The point overlay
uses the same physical slab |z−.5|≤.0234375 at both resolutions. The overlaid
points are projected 3D samples and do not all lie in the saved gas plane.
No pixelwise error or direct sample correspondence is inferred. Stars mark
sources for display and are not clustering ownership labels.

Prescribed source totals match, but the resolved injection mask and spatial
profile still differ. The existing EI pressure update is state-dependent, so
equal mechanical-energy coefficients do not guarantee equal cumulative net
gas-energy deposition. This remains a limitation of the physical comparison.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/astrolink-mpl \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.wind_bubble.compare_two_bubble_resolutions \
  outputs/two_bubbles_iwr_compgpu9_n064 \
  outputs/two_bubbles_budget_matched_iwr_compgpu3_n128 \
  --output-dir outputs/two_bubbles_resolution_comparison_n064_n128_new
```

NumPy, SciPy, Matplotlib and the existing graph helper are sufficient; JAX and
FuzzyCat are not imported. The output directory must be new. Four focused tests
cover metadata/budget/schedule rejection, graph-scale definitions and histogram
accounting; existing graph unit tests also apply.

## Next decision

Use the higher-resolution saved sequence to test whether the large inner groups
have more stable candidate correspondence through .12–.16, retaining small
components, endpoint flags and competing matches in the audit. Do not promote
these links to physical ancestry solely because the display has three groups.
Investigate representative density-sampling paths before interpreting their
strong resolution dependence as a physical density contrast. General physical
tracking and smooth mesh visualization remain separate unfinished steps.
