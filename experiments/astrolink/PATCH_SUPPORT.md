# Inner patches and pooled temporal support

This experiment inspects the gap exposed by the first association study. It
reads existing 64³ two-bubble catalogs on the CPU. It changes neither the shock
finder nor the hydrodynamics, and is not integrated into the online runner.
No persistent physical shock IDs or tracked energetics are assigned.

## Two distinct causes of the gap

The graph already detects small inner patches at t=.12. At the illustrated 60°
normal threshold, there are ten inner components of 14–16 samples, plus the outer
component. The association experiment retains groups of at least 20 samples,
so the inner patches are excluded. This is an analysis cutoff, not evidence that
the inner shocks physically form at t=.13.

At t=.13, each of the ten patches has 20 samples and enters the association
population. At t=.14 the main inner samples lie in two 152-sample components;
eight additional isolated samples remain below the cutoff. A small patch can
have all of its own samples supported by a larger group while covering less
than 25% of that group. The earlier pairwise rule consequently rejects several
plausible patch-to-group links.

## The experimental pooled-support rule

1. Use the same distance and signed-normal compatibility checks as before.
2. For each parent patch, measure the fraction of its samples covered by each
   child group. Count a parent sample once per group pair.
3. A parent is eligible only if its best target covers at least 50% of it and
   beats the runner-up by the declared margin. This is an absolute difference
   in coverage fractions, not a relative percentage. Ties remain unresolved.
4. For each child, collect the eligible parents that favor it. Form the **union**
   of their supported child-sample rows. Overlap counts once.
5. Accept a candidate pool only if it covers at least 50% of the child. Preserve
   the parent lists, individual support fractions, union row IDs, and rejected
   decisions for inspection.

This rule is asymmetric: it addresses fragments contributing to a larger group.
It does not replace a general split/merge model. It does not enforce conserved
ancestry or verify that each parent contributes indispensable new support.
Nearby unrelated patches with compatible normals can still create false links.
The dominance condition reduces one ambiguity; it does not prove ownership.
Fractions are weighted by sample counts, not physical surface area.

## Scope of the comparison

The spatial graph remains fixed: radius 1.5 cells, normal angles 30° or 60°,
and a 20-sample association cutoff. Raw components below the cutoff are retained
in the saved archives and inspection table, but do not enter this matching test.

We compare five adjacent transitions: .07→.08, .09→.10, .12→.13, .13→.14,
and .14→.15. The first two are controls around outer-surface connection.
Three displacement gates (1.5, 2.5, 3.5 cells) and dominance margins .10/.25
give 60 comparisons including both spatial partitions. The association normal
angle remains 60°. Parent and pooled-child coverage thresholds remain .50.
These are a bounded sensitivity test, not tuned or optimal thresholds.

Source locations, Mach, density and physical shock categories are not matching
features. Endpoint overlap with injection regions is counted in the component
table for context, but is not a filter or an ownership label.

## Results

For the .13→.14 transition, five eligible patches favor each of the two larger
inner groups. The illustrated 60° partition and .25 dominance margin give:

| Displacement gate | Best single patch covers | Unique pooled support covers | Samples covered per child |
| --- | ---: | ---: | ---: |
| 1.5 cells | 21.1% | 89.5% | 136 / 152 |
| 2.5 cells | 31.6% | 100% | 152 / 152 |
| 3.5 cells | 42.1% | 100% | 152 / 152 |

The same pooled acceptance occurs for both spatial angle settings and both
tested dominance margins in this transition. At 2.5 cells, individual patch
support counts sum to 168 per child, but their union contains only 152 samples.
At 3.5 cells the sum is 232, while the unique count remains 152. Summing supports
would therefore overstate coverage.

The supported outer two-parent connection is retained in the corresponding
control transitions. The .14→.15 inner associations also remain supported.
However, the .12→.13 inner links are still absent because the earlier patches
are excluded by the minimum-size cutoff. We have not silently relaxed that
cutoff to make the temporal sequence appear complete.

This recovers geometric support lost by the pairwise child-coverage threshold.
It is not independent evidence of five physical shocks merging into one:
the five patches may be fragments of a single partially sampled surface.
No ground-truth physical segmentation is available for these inner structures.

## Files and checks

Results are in `outputs/two_bubbles_patch_support_n064/`:

- `patch_sequence.png`: .12/.13/.14 spatial sections. Gray shows components
  below the association cutoff; colors are local to each event.
- `pooled_support.png`: single-patch versus unique union coverage.
- `raw_components.csv`: all raw group sizes, centroids and source-overlap counts.
- `parents.csv`: best/runner-up target scores and eligibility decisions.
- `children.csv`: pool membership, unique coverage, overlap accounting and acceptance.
- `links.csv`: proposed patch-to-group links, including rejected pools.
- `support_*.npz`: supported child row indices for every pool and configuration.
- `event_*.npz`: original geometry and raw/retained labels. Rows are local to
  their event; they are not persistent material or surface-point IDs.
- `summary.csv`, `metadata.json`, `verification.json`, `SHA256SUMS`.

Seven new synthetic tests cover collective support, overlap deduplication,
competing targets, distant patches, opposite normals, row reordering and empty
inputs/threshold validation. Readback verification independently reconstructs
support via radius-neighbor enumeration, checks parent choices and union row
sets, and verifies hashes. These are software checks, not physical validation.

```bash
MPLCONFIGDIR=/tmp/astrolink-mpl \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.check_patch_support \
  outputs/two_bubbles_iwr_compgpu9_n064/online \
  --output-dir outputs/two_bubbles_patch_support_n064_new

/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.verify_patch_support \
  outputs/two_bubbles_patch_support_n064
```

The runner requires a new output directory. No GPU or new simulation is needed.

## Next decision

Inspect sensitivity to the minimum group size separately, retaining tiny patches
as low-support evidence rather than calling them new physical shocks. Establish
false-link controls before integrating any pooled rule into a persistent tracker.
FuzzyCat evaluation and mesh visualization remain separate follow-ups; this
experiment neither executes FuzzyCat nor assigns temporal physical identities.
