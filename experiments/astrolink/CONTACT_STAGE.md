# Contact-stage grouping study

This CPU postprocessing experiment checks whether the first AstroLink results
survive changes in input order and a bounded set of parameters. It reads the
existing 64³ two-bubble catalogs; it does not evolve the gas or change the shock
finder. It does not yet perform FuzzyCat matching or temporal tracking.

## Methods

- Seven saved events: t = 0.04, 0.06, 0.07, 0.08, 0.09, 0.10, 0.16.
- Original row order and two shuffled orders (seeds 42 and 7).
- AstroLink 0.5.4: normal weights 0, 1, 4; density neighborhoods 20, 40;
  significance setting `auto`, 2.0. This gives 252 runs. Positions are in
  grid-cell units, normals are normalized, adaptive feature rescaling is off,
  and intrinsic dimension is fixed at 2. See the earlier feature experiment.
- A proximity graph: connect points within 1.5 or 2.5 grid cells if their signed
  normal angle is at most 30° or 60°. Connected components form groups. This
  gives 84 runs. Components smaller than 20 samples are gray in displays;
  their raw memberships remain saved. No components are discarded from the
  archive.

Neither method receives source ownership, source coordinates, Mach number,
gas density, or time as clustering features. Source stars appear only in plots.
Signed normal compatibility prevents directly linking oppositely oriented
surfaces. However, the graph has no tangent-plane or curvature test and can
still bridge distinct nearby surfaces with compatible normals.

AstroLink output retains its nested hierarchy. Figures flatten it to the
deepest non-root assignment. Zero display groups means no selected non-root
groups, not zero shocks. ARI compares partitions without depending on numeric
label names; ARI = 1 with all points unassigned is trivial agreement.

## Findings

Four declared example settings give the following group counts in original
input order. They illustrate behavior; they were not optimized to obtain an
expected number of physical shocks.

| Setting | .04 | .06 | .07 | .08 | .09 | .10 | .16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AstroLink w=1, k=20, S=auto | 2 | 10 | 0 | 0 | 0 | 0 | 3 |
| AstroLink w=4, k=20, S=auto | 2 | 10 | 2 | 17 | 2 | 2 | 3 |
| Graph r=1.5 cells, angle=30° | 2 | 2 | 2 | 2 | 2 | 1 | 3 |
| Graph r=2.5 cells, angle=60° | 2 | 2 | 2 | 1 | 1 | 1 | 3 |

All tested graph partitions agree under the three input orders (ARI = 1).
Their threshold-dependent merging shows that order stability alone does not
establish physical accuracy. At t=0.08, the w=4 AstroLink example changes from
17 groups in original order to 2 with shuffle seed 42. Its coverage also changes.
This order sensitivity occurs without a reported fitting warning in that run.

Four AstroLink runs emitted a prominence-model convergence warning: position-only,
k=20, both significance settings, at t=.09/seed7 and t=.10/seed42. The installed
library continues using its initial fit parameters when optimization fails.
These outputs are retained and explicitly flagged in `summary.csv` and the
archives; treat them as unsuccessful fits, not reliable calibrated detections.
Both significance settings repeat the same underlying fitting problem, so four
warnings do not represent four independent numerical failure modes.

At t=.16 the illustrated settings recover an outer group and two inner groups,
consistent with the earlier spatial inspection. This is not independent proof
of shock identities, and colors do not establish correspondence between events.
No ground-truth segmentation is available for the interacting stage.

## Reproduce and inspect

Run from the repository root in the existing AstroLink environment:

```bash
MPLCONFIGDIR=/tmp/astrolink-mpl NUMBA_CACHE_DIR=/tmp/astrolink-numba \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.check_contact_stage \
  outputs/two_bubbles_iwr_compgpu9_n064/online \
  --output-dir outputs/two_bubbles_contact_sensitivity_n064_new

/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.verify_contact_stage \
  outputs/two_bubbles_contact_sensitivity_n064
```

The runner refuses an existing output directory. Each event archive contains
catalog positions, normals, eligible row indices and all configurations' labels.
AstroLink also saves hierarchy intervals, original-catalog ordering, IDs,
significances, point-density values, fit parameters and library messages.
`summary.csv` reports each run; `order_stability.csv` compares each pair of orders.
The verifier checks input/source hashes, reconstructs display labels from the
saved hierarchy, recomputes graph partitions and validates summary/ARI accounting.
It is a bookkeeping check, not a new physical validation.

## Next decision

Inspect which graph edges connect surfaces near contact using the saved gas
slices, normals and geometry. Establish what a group should represent after
collision before assigning persistent identities. FuzzyCat can then be evaluated
for correspondence; it should not be used to conceal unstable spatial grouping.
Higher resolution and mesh display remain separate follow-ups.

Method references: [AstroLink API](https://astrolink.readthedocs.io/en/latest/api.html),
[SciPy proximity queries](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_pairs.html),
[SciPy connected components](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.connected_components.html).
This graph is a simple baseline, not an implementation of the full
[PCL region-growing method](https://pointclouds.org/documentation/tutorials/region_growing_segmentation.html).
