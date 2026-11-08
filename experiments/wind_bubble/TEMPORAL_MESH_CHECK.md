# Fixed-setting three-event mesh check

## Purpose and decision

Apply the control-selected single-radius ball-pivot display to t=.08, .12 and
.16 of the same saved 128³ two-bubble run. Check geometric constraints, separation,
open boundaries and consistency without changing parameters between events.
No new hydrodynamic simulation or GPU run was needed.

The fixed settings produce readable surface views, and distinct retained groups
are geometrically separated in the two later events. However, the meshes remain
highly fragmented. This is not yet a finished smooth surface animation. Retain
the original points as the measurement view and do not interpret mesh gaps,
mesh pieces or their changes as physical shock openings/identities/events.

## Frozen settings

- Open3D 0.19.0 ball pivoting, one radius equal to dx=1/128.
- Existing spatial grouping: radius 1.5 cells, 60° normal limit, minimum 20 samples.
- Same postconstruction guards: edge <=2 cells, signed-normal agreement 45°,
  tangent departure <=20°, face-normal agreement 30°, minimum angle 12°,
  and removal of faces on over-subscribed edges.
- Original vertex coordinates, Mach values and normals copied unchanged.
- Camera elevation/azimuth 25°/−65°, spatial bounds .1–.9, Mach limits 0–16.
- Cutaway at z=.5, only for display. Faces crossing the cutaway plane are removed
  rather than capped. There is no geometry interpolation or temporal prediction.

Radius and guard choices are inherited from `MESH_METHOD_COMPARISON.md`; no
per-event tuning is introduced. Meshing works separately inside each saved group.
No persistent physical identities, statistics, shock areas or heating rates are
created or changed.

## Results

| t | Original samples | Retained groups | Excluded samples | Triangles | Used vertices | Boundary edges | Edge-connected mesh pieces |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| .08 | 11,424 | 1 | 0 | 15,947 | 10,149 | 7,961 | 391 |
| .12 | 19,408 | 3 | 40 | 26,556 | 17,452 | 13,938 | 622 |
| .16 | 25,136 | 3 | 0 | 35,408 | 22,915 | 17,758 | 830 |

There are no cross-group triangles, over-subscribed edges or inconsistent
shared-edge windings in the saved meshes. The t=.16 triangles reproduce the
previous single-event result exactly.

Mesh pieces are defined by triangles sharing a complete edge, not merely a
vertex. These are reconstruction fragments, **not physical shocks or input
surface-group counts**. For example, the one retained group at .08 has 391
edge-connected mesh pieces. At .12 each inner candidate group has 57 pieces;
at .16 the two inner candidates have 155 and 157 pieces. Counts include many
singleton triangles. They demonstrate that apparently surface-like shaded
views should not be treated as continuous reconstructed physical surfaces.
Counts at different events involve different sample populations and do not
establish mesh convergence or a physical fragmentation rate.

## Separation check

At .12 and .16, all three pairs of distinct retained groups pass a sufficient
geometric nonintersection bound. Let d be the closest distance between any
vertices of the two groups, and L_A/L_B the largest retained triangle edges in
each group. If d > L_A + L_B, triangles in those groups cannot intersect: every
point of a triangle lies within its largest edge length of each vertex.

All six tested group-pair bounds pass and are independently checked using
chunked dense pair distances. For example at .12, the smallest outer/inner
vertex separation is about .11223 code units, versus a summed edge bound about
.02728. At .08 only one group is retained, so a cross-group separation result is
not applicable. This does not verify correct topology **inside** that group.

The known control where close aligned sheets are incorrectly assigned the same
group still fails. Neither the separate-group bound nor the mesher solves that
problem. Global self-intersections and physical shock identities remain unvalidated.

## Reading the two figures

`temporal_mesh_comparison.png` has one row per event and four columns:
original points, mesh, original points with the upper half removed, and mesh
with the upper half removed. Gray indicates original samples not used by faces,
including the 40 size-excluded samples at .12. Mesh holes may reveal inner
surfaces through the outer surface; this is not physical transparency.

`central_mesh_sections.png` has two columns:

- Left: original samples within ±1 cell of z=.5, projected onto xy.
- Right: actual intersections of reconstructed triangles with z=.5.

These are different selections, so line/sample appearance need not agree exactly.
Colors indicate local group labels and are reassigned independently at each
event. Black stars indicate wind sources for orientation only. Line gaps can
result from missing/rejected triangles and must not be read as physical openings.
Coplanar triangles would be counted and skipped; none occur in these three events.

Many visible gaps persist at every event. The checks establish local geometric
constraints and later cross-group separation, not recovery of every true opening
or continuous coverage. Three snapshots cannot determine flickering over all
17 events, nor whether a complete mesh animation improves interpretation.

## Verification and outputs

Results: `outputs/two_bubbles_temporal_mesh_n128/`.

- `mesh_0008.npz`, `mesh_0012.npz`, `mesh_0016.npz`: unchanged original arrays,
  labels, raw and accepted faces, time/dx, and plane-intersection segments.
- Both figures, summary/group CSVs, metadata, readback and checksums.

Independent readback verifies all face geometry and original values, agreement
with the frozen settings, original-index mapping, raw-to-accepted face membership,
CSV accounting and all six cross-group separation bounds. The 1,539 saved plane
segments are checked against their triangles using barycentric coordinates.
The final-event mesh matches the earlier reconstruction exactly. Eight focused
tests pass, including plane intersections, sufficient/inconclusive separation,
edge-connected components and native ball-pivot smoke tests. Input hashes remain
unchanged. Passing these tests does not validate physical surface topology.

## Reproduce

With the same optional Open3D setup as the previous comparison:

```bash
PYTHONPATH=/tmp/thesis-open3d-comparison:/tmp/thesis-open3d-support \
MPLCONFIGDIR=/tmp/astrolink-mpl \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.wind_bubble.check_temporal_mesh \
  --output-dir outputs/two_bubbles_temporal_mesh_n128_new

PYTHONPATH=/tmp/thesis-open3d-comparison:/tmp/thesis-open3d-support \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.wind_bubble.verify_temporal_mesh \
  outputs/two_bubbles_temporal_mesh_n128_new

PYTHONPATH=/tmp/thesis-open3d-comparison:/tmp/thesis-open3d-support \
/opt/anaconda3/envs/astrolink-env/bin/python -m unittest \
  tests.experiments.test_temporal_mesh tests.experiments.test_ball_pivot_mesh
```

Use a new directory. No additional dependency download, GPU, JAX or FuzzyCat is
needed for this session. The next mesh decision should address the observed
fragmentation before presenting the triangles as a smooth continuous surface.
Any later animation should keep a point-cloud comparison and label missing mesh
coverage explicitly.
