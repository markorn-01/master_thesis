# Local mesh fragmentation check

This CPU-only diagnostic reads the saved event-16 mesh, including raw
ball-pivot proposals. It does not rerun the simulation, move vertices, change
normal vectors, relax guards, or modify any physical diagnostic.

Two deterministic patches on event-local inner group 1 are compared. The
anchor maximizes projection along either `(0,1,0)` or `(-1,1,1)/sqrt(3)`.
Triangle centroids within three grid spacings of the anchor are selected;
points within five spacings provide visual context. These are illustrative
locations, not a statistical survey of the whole surface or persistent IDs.

| Patch | Raw proposals | Accepted | First tangent rejection | First face-normal rejection |
| --- | ---: | ---: | ---: | ---: |
| Axis direction | 59 | 59 | 0 | 0 |
| Diagonal direction | 59 | 18 | 39 | 2 |

The tangent guard requires each triangle edge to lie within 20 degrees of
both endpoint tangent planes. The diagonal patch's maximum departure per
triangle has median 21.08 degrees (16th–84th percentiles 15.36–26.75).
Its maximum pairwise saved-normal angle has median 8.63 degrees
(5.86–10.09), below the separate 45-degree normal-agreement guard.
Thus, saved normals can agree with one another while disagreeing with the
triangle geometry built from the original refined sample positions.

Among 28 diagonal core samples, one is unused by any raw group triangle;
11 are unused after filtering. Median nearest-sample separation is 0.815
grid spacings. The axis patch has 26 core samples, all used both before
and after filtering, and median nearest-sample separation 0.974 spacings.
Unused-point counts use all group faces, avoiding patch-cropping artifacts.
These results identify filter-induced fragmentation locally, without
establishing that all rejected triangles are physically correct or that
all gaps elsewhere have the same cause. Reported reasons are the first
failed guard in the existing ordered checks; failures can overlap.

The complete saved event's accepted unordered face set is reproduced
before plotting. Three known-geometry tests distinguish normal agreement
from edge/tangent disagreement and reject zero-length edges. Input hashes
are recorded and checked unchanged after rendering.

## Reproduce

Using a Python environment with NumPy, SciPy and Matplotlib:

```sh
python -m experiments.wind_bubble.inspect_mesh_patch --direction axis --output-dir outputs/two_bubbles_mesh_patch_n128
python -m experiments.wind_bubble.inspect_mesh_patch --direction diagonal --output-dir outputs/two_bubbles_mesh_patch_diagonal_n128
python -m unittest tests.experiments.test_mesh_patch
```

The output directories must not already exist. Each contains a per-triangle
CSV, summary JSON, before/after/rejected triangle figure and angle histograms.
Black points are original refined positions, not necessarily cell centers.
Two-dimensional projections can overlap triangles; the figures are not
intersection or physical-hole tests.

## Next decision

Before animating meshes, compare local geometric normals estimated from
neighboring positions against the saved physical normals on these same
patches. That comparison should remain a display experiment, with all
original physical measurements preserved. Do not silently relax the
tangent guard: the existing close-sheet counterexample still demonstrates
that geometric reconstruction can connect distinct surfaces in one group.
