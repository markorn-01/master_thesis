# AstroLink first test on two-bubble shock surfaces

18 local CPU runs using AstroLink 0.5.4 and the saved GPU 64³ surface catalogs.
No hydrodynamic states, detector outputs or original catalogs were changed.
Three event times, three feature metrics, two input orderings. Five feature and
index-mapping tests passed, and verify_results.py reconstructed all 18 saved
feature arrays, permutations, hierarchy mappings and display partitions.

## Main findings

The table gives deepest non-root display-group counts as original/shuffled order.
It does not count persistent physical shocks or count the root as a detection.

| Time | Positions only | Positions + normals, w=1 | Positions + normals, w=4 |
| --- | --- | --- | --- |
| 0.04, separated | 0 / 0 | 2 / 2 | 2 / 0 |
| 0.08, near contact | 1 / 1 | 0 / 0 | 17 / 2 |
| 0.16, interacting | 3 / 3 | 3 / 3 | 3 / 3 |

- At t=0.04, w=1 separates two equal groups of 664 points, corresponding
  spatially to the two bubbles. Membership is unchanged by the tested shuffle.
  Positions alone produce only the all-data root. With w=4, changing the
  point order changes whether the two non-root groups pass the hierarchy selection.
- At t=0.08, positions alone assign only 14.7% of points to a non-root group,
  while w=1 gives only the root. At w=4, original order gives 17 groups covering
  61.8% of points, whereas shuffled order gives two groups covering all points.
  Adjusted Rand agreement is about 0.587 for those two partitions.
- At t=0.16, every tested metric/order gives the same partition up to label
  renumbering: an outer group of 5608 points and two inner groups of 280 points
  each. The outer median Mach is about 6.62 and inner medians about 1.53.
  The section view reveals the inner groups hidden behind the outer shell in 3D.
  Their positions are consistent with outer/inner populations but this is not
  independent physical validation or temporal identity tracking.

Conclusion: useful structure is present, but this initial configuration is not
reliable across the contact stage. Feature weighting and input order matter.
Do not select a preferred run solely because it produces two groups. A root-only
result means no selected non-root density groups, not zero surface detections.
Stable root-only partitions yield adjusted Rand agreement of one trivially.
The narrow initial test does not establish universal algorithm failure or success.

## What the files contain

- metadata.json: package versions, metric/settings, input hashes, analysis source
  hash, event times and interpretation limits.
- summary.csv: sample counts, hierarchy/display counts, assigned fractions,
  fitted S and k_link, exploratory elapsed times.
- hierarchy.csv: raw entry IDs, counts, significances and centroids. Root
  significance is infinite by the library convention. Entries may be nested.
- order_stability.csv: adjusted Rand agreement after mapping back to the same
  original catalog rows. Colors themselves are not the metric.
- event_*_order*.npz: full hierarchy/index mappings and display labels.
- event_*_comparison.png: original-order 3D labels and ordered point-density plots.
- event_*_sections.png: original versus shuffled labels in a thin z section,
  rendered without reclustering. All three section views and 3D views were inspected.
- verification.json and verify_results.py: independent saved-array accounting.
- SHA256SUMS: preservation manifest.

The same input geometry is used for every metric. Positions are measured in cell
units and normals are normalized, with weights w=0,1,4. S is automatic, k_den=20,
adaptive=0, intrinsic dimension=2, h_style=1 and workers=1. See
experiments/astrolink/TWO_BUBBLE_CLUSTERING.md for reasoning and reproduction.

Recheck from the repository root:

```bash
PYTHONPATH=. python outputs/two_bubbles_astrolink_n064/verify_results.py
```

Next investigate adjacent contact-stage events and controlled settings before
attempting temporal tracking. No FuzzyCat experiment or triangle mesh is included.
