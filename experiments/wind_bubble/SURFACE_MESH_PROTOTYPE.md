# One-event triangle display prototype

## What this adds

A display-only triangle mesh for the saved 128³ two-bubble event at t=.16.
All 25,136 vertex positions, normals and Mach values are copied unchanged from
the sparse catalog. The original histories, histograms, shock measurements,
physical surface-area estimator and temporal candidates are not modified.
This connects existing samples; it does not smooth or relocate them, create
new shock detections, or solve physical shock identity.

The comparison figure shows original points beside triangle faces, with a full
view above and a z<=.5 cutaway below. Both use the same camera, bounds and Mach
scale. Cutaway faces are shown only if all three vertices satisfy z<=.5; this
creates an artificial open rim, not a physical discontinuity. Gray marks show
unused vertices. Inner surfaces can be visible through mesh holes; this does
not represent physical transparency of the gas.

## Experimental method

Triangulate separately inside each saved event-local spatial group (native
radius 1.5 cells, normal limit 60°, minimum count 20). The selected event has
three retained groups: sizes 21,840, 1,648 and 1,648. These labels constrain
triangle construction; they are not independently validated physical shock IDs.

For each sample:

1. Find same-group neighbors within the maximum edge distance.
2. Require compatible signed normals (45° maximum) and near-tangent chords
   at both endpoints (20° maximum departure from the tangent plane).
3. Project the neighbors onto the local tangent plane defined by the saved
   normal, and use 2D Delaunay triangulation to propose incident triangles.
4. Retain a triangle only when all three of its vertices propose that same
   unordered triplet. Map the triangle back to the **original 3D positions**.
5. Check all edges, normal compatibility, positive area, minimum interior angle
   12°, and triangle-normal agreement within 30° of every vertex normal.
   Orient winding consistently with the saved normal direction.
6. Reject all faces incident to an edge used by more than two candidate faces.

Delaunay is supplied by SciPy/Qhull. Its API does not guarantee inclusion of
every point, and degenerate configurations can fail. The implementation skips
failed/duplicate projected neighborhoods and records the counts, rather than
requiring every input vertex to belong to a triangle. See the
[SciPy Delaunay documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html).
This local proposal/consensus/filter combination is our experimental construction,
not a validated named reconstruction algorithm.

Regular grid samples have equivalent diagonals. Projected triangle proposals
use a deterministic 1e-7-cell perturbation derived from lexicographic coordinate
rank to resolve these ties. This affects only the temporary proposal coordinates:
**exported 3D vertices are never perturbed**. There is no random Qhull joggle,
vertex smoothing, interpolation of new geometry, or explicit global hole filling.
Row-permutation consistency is tested on the regular planar control; arbitrary
real-data perturbation stability is not established.

The prototype uses installed NumPy/SciPy/Matplotlib rather than adding another
mesh package. Open3D documents alternative methods such as ball pivoting and
Poisson reconstruction; no such algorithm is implemented here. Its documentation
also discusses Poisson extrapolation into low-support regions, reinforcing the
need to inspect unsupported reconstructed geometry rather than treating a closed
appearance as validation. See the
[Open3D reconstruction tutorial](https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html).

## Actual-event results

| Maximum edge | Triangles | Used original vertices | Boundary edges |
| --- | ---: | ---: | ---: |
| 1.5 cells | 29,912 | 22,936 / 25,136 | 20,328 |
| 2.0 cells | 32,856 | 23,432 / 25,136 | 17,944 |

The figure and PLY use the 2.0-cell setting. Its longest accepted edge is about
1.7411 cells; the smallest triangle angle is about 13.5954°. There are no
cross-group triangles, over-subscribed edges or inconsistent shared-edge
windings in either saved mesh.

There are many open boundaries, so this is **not a watertight mesh or a finished
smooth visualization**. Local-chart disagreement intentionally rejects many
faces. Relaxing that agreement or filling holes requires further checks for
incorrect connections and overlapping geometry. More used vertices or fewer
holes do not by themselves establish better physical reconstruction.

Triangles have flat colors equal to the mean of their three original Mach
values. This is a display convention, not a new Mach estimator or an
area-weighted diagnostic. Physical area and thermalization are not recalculated
from these faces. No conservation or physical accuracy follows from a mesh's
visual appearance.

## Controls and known limitation

Five constructed cases are saved in `controls.json`:

- A regular 7×7 planar grid produces 72 triangles using all 49 vertices.
- Two patches in the same group separated by a gap wider than the edge limit
  remain disconnected (24 faces; no cross-gap face).
- Two nearby sheets in different groups stay separate (144 faces).
- Two nearby sheets in the same group with opposite normals stay separate
  (144 faces).
- **Counterexample:** two close aligned sheets assigned to the same group are
  bridged. All 168 retained faces mix the stipulated sheet labels.

The counterexample is preserved as a failure. The geometric information and
input labels are insufficient for this method to distinguish those sheets.
Thus the guards bound locality and prevent cross-group connections; they do
**not** guarantee correct physical topology or absence of all gap bridging.
A wrong or overly merged input group can still yield wrong triangles. No
synthetic truth labels are provided to the mesher.

Global self-intersections, overlapping faces and vertex-manifoldness are not
fully checked. At-most-two-face edge incidence is only an edge-level condition.
The prototype should be reviewed before animation or use on many bubbles.

## Outputs and checks

`outputs/two_bubbles_surface_mesh_n128_event0016/` contains:

- `surface_mesh_comparison.png`: points versus triangles, full/cutaway views.
- `mesh_edge_1p5.npz`, `mesh_edge_2p0.npz`: original arrays, original-index faces,
  event-local labels, face groups, time and grid spacing.
- `surface_mesh.ply`: standard ASCII PLY of the 2.0-cell mesh, with original
  positions, per-vertex Mach values and triangle indices for external inspection.
- `controls.json`, `metadata.json`, `readback.json`, `SHA256SUMS`.

Both saved meshes pass independent checks of exact vertex/normal/Mach readback,
indices, unique faces, group isolation, edge bounds, tangent/normal/angle guards,
edge incidence and PLY vertex/face readback. Six focused tests pass, including
the explicit aligned-sheet failure. Inputs retain their original hashes.
These checks validate the construction constraints, not physical topology.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/astrolink-mpl \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.wind_bubble.prototype_surface_mesh \
  outputs/two_bubbles_budget_matched_iwr_compgpu3_n128/online/surfaces_0016.npz \
  outputs/two_bubbles_resolution_comparison_n064_n128/groups_n128_event0016.npz \
  --output-dir outputs/two_bubbles_surface_mesh_n128_event0016_new

/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.wind_bubble.verify_surface_mesh \
  outputs/two_bubbles_surface_mesh_n128_event0016_new

/opt/anaconda3/envs/astrolink-env/bin/python -m unittest \
  tests.experiments.test_surface_mesh
```

Use a new directory. No GPU, new simulation, JAX or Open3D installation is
required. The next decision is whether this deliberately incomplete display
is useful, or whether an alternative reconstruction should be compared on
known curved/gapped surfaces before any mesh animation is attempted.
