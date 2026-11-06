# First AstroLink experiment on two-bubble shock samples

This is CPU postprocessing of the verified 64³ GPU dataset. It does not rerun
hydrodynamics, change sample selection, infer source ownership, or track shocks.

## Question

Can AstroLink produce useful groups of detected surface samples before contact,
near contact, and after interaction? Do the results survive point reordering
and an explicit change in the influence of the local normals?

AstroLink returns a density hierarchy, including a root containing the entire
input. It is not automatically a segmentation into every disconnected physical
shock surface. Root-only output must not be interpreted as no shock detections.
Reference: https://astrolink.readthedocs.io/en/latest/api.html

## Controlled comparison

Input events 0004, 0008 and 0016 correspond to t=0.04, 0.08 and 0.16.
All eligible catalog points are used, without Mach filtering or subsampling.
Both feature modes use the same finite positions and finite nonzero normals.

Position features are (x-0.5)/dx, retaining isotropic spatial distances. Normal
features are w*n/|n|. Thus the metric is

    distance² = |delta position / dx|² + w² |delta unit normal|².

The tested weights are w=0 (positions only), 1, and 4. AstroLink adaptive=0
prevents per-feature rescaling from undoing the chosen weight. The intrinsic
dimension is set to two as an initial surface-manifold assumption. This
assumption is not a measured local dimension and may need sensitivity checks.
Other settings: k_den=20, S='auto', h_style=1, workers=1, working_memory=256 MiB.
Source positions and radii do not enter the clustering features.

Each configuration is run twice: original catalog order and an exact permutation
from NumPy default_rng(42). Values and geometry do not change. The 18 runs are
three events, three metrics and two orderings. These are not repeated hydro runs
or independent physical realizations.

## Preserve hierarchy and index meaning

NPZ outputs contain features, the input permutation, eligible original catalog
rows, AstroLink ordering, original-catalog ordering, hierarchy intervals/IDs,
significances and per-point density. Interval membership is
catalog_ordering[start:end]. Nested memberships remain available.

For plots only, each point takes its deepest non-root hierarchy entry. Gray
means no such assignment, even though the point belongs to the root. Invalid
geometry has label -2. These partitions are a display convention, not persistent
physical IDs. Cluster colors may swap between runs and times. Partition stability
uses adjusted Rand agreement, which ignores arbitrary label renumbering.
An agreement of one for two root-only outputs is trivial and not evidence of
successful physical segmentation.

## Reproduce

Use the existing astrolink-env environment (first run used AstroLink 0.5.4,
NumPy 2.4.6). No JAX or GPU is needed:

```bash
python -m experiments.astrolink.check_two_bubble_clustering \
  outputs/two_bubbles_iwr_compgpu9_n064/online \
  --events 4 8 16 --normal-weights 0 1 4 --order-seeds 0 42 \
  --output-dir outputs/two_bubbles_astrolink_n064_repeat
```

Inspect occluded inner groups through a thin section, without reclustering:

```bash
python -m experiments.astrolink.plot_shock_cluster_sections \
  outputs/two_bubbles_astrolink_n064 --event-index 16 \
  --output outputs/two_bubbles_astrolink_n064/event_0016_sections.png
```

The section selects |z-0.5| <= 1.5 dx for display only. Clustering always uses the
complete eligible 3D point set. Bottom panels in the 3D comparison show normalized
point density in feature space, not gas density. Raw wall times include the first
Numba compilation and are not performance benchmarks.

## Findings and next decision

See outputs/two_bubbles_astrolink_n064/README.md for the first results. A moderate
normal weight separates the early bubbles, and the late frame contains one large
outer and two small inner groups. Near contact, the tested settings are incomplete
or depend on point ordering. This does not yet validate AstroLink as a general
shock segmentation method, and it does not rule out improved settings.

Next isolate the near-contact behavior using neighboring events and a small
controlled neighborhood/significance sensitivity test. Compare against a simple
geometry/normal-connectivity baseline rather than selecting settings solely to
obtain two clusters. Resolve what constitutes a physical group before attempting
time correspondence. FuzzyCat's membership comparisons still require a meaningful
common object representation; moving Eulerian cell indices are not physical IDs.
