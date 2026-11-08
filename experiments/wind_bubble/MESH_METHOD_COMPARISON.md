# Reconstruction comparison on controls and one saved bubble event

## Outcome

Compared the local tangent-plane prototype with Open3D 0.19.0 ball pivoting,
using a single radius and a three-radius variant. Single-radius ball pivoting
is the preferred **display candidate among these tested settings** on the
controlled curved cases. It was then applied to the same saved 128³ two-bubble
event at t=.16. The bubble improvement is modest and the mesh remains open.
This does not validate physical topology, smoothing, identity tracking or
mesh-based physical diagnostics.

## Methods and fair comparison

- **Local:** previous tangent-plane Delaunay/three-vertex-consensus prototype.
- **Ball single:** Open3D ball pivoting with radius 1.0 times the reference spacing.
- **Ball multi:** radii .8, 1.0 and 1.4 times that spacing.

Ball pivoting constructs triangles from an oriented point cloud with a chosen
ball radius; see the official
[Open3D tutorial](https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html#ball-pivoting).
All versions retain original 3D vertices. No normals are estimated from a
reconstructed mesh: known control normals or saved shock normals are supplied.
The wrapper verifies that Open3D returns vertices in the original order without
moving them, and maps triangle indices back to the full catalog.

Each method works separately inside each supplied group. The same geometric
guards apply afterward: maximum edge 2 reference spacings, compatible signed
normals within 45°, chords within 20° of both endpoint tangent planes,
face-normal agreement within 30°, positive nondegenerate area and minimum
triangle angle 12°. Duplicate/cross-group faces are rejected. All faces on an
edge with more than two incident faces are removed. The local method additionally
requires three-vertex proposal agreement; ball pivoting has its own construction
rather than that agreement rule. Thus this compares guarded display pipelines,
not unfiltered raw algorithms.

The sphere uses 1600 deterministic Fibonacci samples, with reference spacing
set by the median nearest-neighbor distance. The cap case retains the same
spacing. Plane/sheet controls use unit grid spacing. The actual bubble uses
its recorded dx=1/128. Radius factors and guards are not tuned on the bubble.

## Controlled inputs and measurements

Five inputs are saved:

1. Unit sphere, analytic area 4π.
2. Sphere with samples above z=.6 removed; analytic remaining area 3.2π.
3. Planar grid with a central disk removed.
4. Two very close aligned sheets with separate supplied groups.
5. The same sheets incorrectly assigned one group, with separate construction
   truth labels retained only for evaluation.

Triangle area is calculated **only for these known-geometry controls** to assess
missing or excess reconstruction. It is not the thesis shock-area estimator and
is not used to recompute physical bubble area/heating. Triangle normals, edge
incidence, used vertices and boundary edges are recorded as additional evidence.

Known-hole checks sample barycentric locations on each triangle. The spherical
check flags radial projections with z>.62, inside the removed cap with a .02
margin. The planar check flags samples inside the radius-2.5 disk, inside the
radius-3 removed region. These test hole interiors with margins: they are not
exact continuous triangle-hole intersection tests, nor proof that every part
of the nominal opening remains unchanged. A synthetic triangle closing each
known opening is explicitly detected by the metric tests.

Cross-sheet faces are identified using construction labels that the mesher does
not receive. Group labels and true sheet labels are deliberately separate in
the difficult fifth control.

## Results and selection

| Measurement | Local | Ball single | Ball multi |
| --- | ---: | ---: | ---: |
| Sphere triangle area / analytic area | .99305 | .99805 | .99805 |
| Sphere boundary edges | 32 | 0 | 0 |
| Removed-cap area / analytic remaining area | .97156 | .97703 | .97703 |
| Cap-case boundary edges | 83 | 55 | 55 |
| Sampled hole-interior intrusion faces (both hole cases) | 0 | 0 | 0 |
| Cross-sheet faces, separately labeled sheets | 0 | 0 | 0 |
| Cross-sheet faces, sheets incorrectly in one group | 288 | 288 | 288 |

Ball multi also has one inconsistently wound shared edge in the separately
labeled sheet control, so it is excluded by the recorded selection rule.
Selection first rejects extra sampled hole intrusions relative to local,
cross-sheet connections when supplied groups are separate, and edge-topology
violations. Among remaining candidates it minimizes the sum of absolute sphere
and cap area-fraction errors relative to one. The rule and scores are saved
in `selection.json`; the choice is frozen before bubble reconstruction.

**The same-group counterexample remains a failure for every method.** All 288
faces mix the two stipulated sheets. It is retained as a separate unresolved
ambiguity control, not silently included as a successful reconstruction. The
selection rule is conditional on useful supplied groups; it does not certify
correctness when two surfaces have already been incorrectly combined.

The controls are selected ideal geometries, not an independent exhaustive
benchmark. No point-order or noise robustness claim is made for ball pivoting.
Global self-intersection/overlap and vertex-manifold tests remain incomplete.

## Applying the choice to the saved bubble

The input is unchanged: 25,136 samples in three event-local groups at t=.16.
The physical meaning of those groups is not established by meshing them.

| Bubble measurement | Local | Ball single |
| --- | ---: | ---: |
| Retained triangles | 32,856 | 35,408 |
| Used original vertices | 23,432 | 22,915 |
| Boundary edges | 17,944 | 17,758 |
| Cross-group faces | 0 | 0 |
| Nonmanifold edges | 0 | 0 |
| Inconsistently wound shared edges | 0 | 0 |

More triangles do not imply more samples are represented: ball pivoting uses
slightly fewer vertices here. Its improvement in continuity is limited; both
meshes have many holes. The ball-pivot candidate initially has 49,949 faces;
the guards reject 13,281 for nontangent edges, 928 for face-normal disagreement,
240 for vertex-normal disagreement and 92 for small angles. Those constraints
explain why a dense raw mesh is not retained as an apparently complete result.
Weakening them solely to make the image look closed would need separate checks.

`bubble_mesh_comparison.png` shows unchanged samples, local triangles and the
selected ball-pivot mesh with fixed camera/bounds/Mach scale. The bottom row is
an artificial z<=.5 cutaway; it is not another physical shock surface. Face
color is the mean of original vertex Mach values. Seeing inner samples through
mesh holes does not represent physical gas transparency.

No positions are smoothed or relocated. No new points, physical identities,
Mach estimates, histograms, shock areas or heating rates are introduced. This
is not yet a finished smooth animation. The comparison supports retaining the
single-radius candidate for further display assessment, while preserving the
point view and explicitly showing unresolved gaps.

## Verification and files

Results: `outputs/two_bubbles_mesh_method_comparison/`.

- `controlled_mesh_comparison.png`, `bubble_mesh_comparison.png`.
- Five input NPZ files, fifteen control face arrays, and two bubble NPZ meshes.
- `summary.csv`, `construction.json`, `selection.json`, `metadata.json`,
  `readback.json`, `SHA256SUMS`.

All 15 control meshes and both bubble meshes pass independent original-index,
vertex-preservation and geometric-guard readback. Control area is also checked
with Heron's formula, independently of the cross-product area calculation.
The saved selection is recomputed from the control table. Original input hashes
are unchanged. Fourteen focused software tests pass, including native Open3D
smoke tests and explicit hole/bridge-metric controls. The known same-group
bridging failure is not converted into a successful physical test.

## Environment and reproduce

Open3D is optional and pinned in `mesh_comparison_requirements.txt`. It was
installed in `/tmp/thesis-open3d-comparison`, with missing Python support
packages in `/tmp/thesis-open3d-support`, leaving the installed project/JAX
environments unchanged. The first support-package download failed; its retry
completed. Package versions are recorded in output metadata. This was CPU
postprocessing; no GPU, remote simulation, or FuzzyCat execution was needed.

In a separate environment with the optional dependencies installed:

```bash
python -m pip install -r experiments/wind_bubble/mesh_comparison_requirements.txt

MPLCONFIGDIR=/tmp/astrolink-mpl python -m experiments.wind_bubble.compare_surface_meshes \
  --output-dir outputs/two_bubbles_mesh_method_comparison_new

python -m experiments.wind_bubble.verify_mesh_comparison \
  outputs/two_bubbles_mesh_method_comparison_new

python -m unittest tests.experiments.test_ball_pivot_mesh \
  tests.experiments.test_mesh_benchmark tests.experiments.test_surface_mesh
```

Use a new output directory. For this session's isolated temporary installation,
prefix Python calls with
`PYTHONPATH=/tmp/thesis-open3d-comparison:/tmp/thesis-open3d-support` and use
`/opt/anaconda3/envs/astrolink-env/bin/python`. The core shock finder and normal
online workflow do not require Open3D.
