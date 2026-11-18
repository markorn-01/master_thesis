# Frozen adaptive display at three saved times

`check_adaptive_mesh_times.py` applies the already checked adaptive local
implicit display to events 12, 13 and 14 of the saved 64³ three-source run.
These times surround the previously reviewed fragment association. The
retained group populations are 1, 8 and 4 respectively. Event 16 remains the
earlier static reference; it is not refitted for this check.

This is CPU postprocessing of existing catalogs and partitions. No new
simulation, shock detection, grouping or temporal association is performed.
The saved graph supplies consistent display colors, not reconstruction inputs.

## Frozen checks

The reference result must be conditionally eligible. Its recorded source,
input and output hashes and the saved history's validation report are checked
before processing. Catalog times, group populations and history labels must
agree. Every source/input hash is checked again after rendering.

The existing half-cell extraction, two-cell support, kernel, normal limits,
local-hull trimming and conservative defect removal are reused. Refinement
still permits at most two halvings of extraction spacing. The target remains
at least 95% of each retained group's samples within a quarter cell of its
**own group's** triangles. Whole-scene nearest-surface distances do not replace
this per-group check. Physical arrays and partitions are read back exactly.

Native self-intersection and vertex/edge-manifold/winding measurements apply
to the combined candidate mesh. A global geometry failure suppresses all mesh
faces in that frame's default display. Otherwise only groups meeting the fixed
per-group accuracy criterion are meshed. Failed groups and samples outside
retained groups are displayed as original gray points. Candidate faces remain
saved for inspection; no threshold is relaxed to make a frame pass.

`verification.json` distinguishes successful provenance/readback (`passed`)
from all selected events meeting display eligibility
(`all_selected_events_display_eligible`). A failed group is a reported result,
not an excuse to drop the event. `summary.csv` reports candidate geometry
metrics separately from the number of approved display faces and fallback
points. `group_accuracy.csv` gives own-group distances and source-overlap flags.

## Views and scope

`three_time_comparison.png` compares original samples and the approved display
at the three times. Camera, bounds and continuation-segment colors are fixed.
Both columns use a z <= 0.5 cutaway; black stars mark wind sources. The cutaway
and mesh approval affect display only. The full candidate faces and original
arrays are preserved in `mesh_0012.npz`, `mesh_0013.npz` and `mesh_0014.npz`.

`three_state_preview.gif` shows those three saved states at one frame per
second. Every state is fitted independently; no intermediate states, temporal
smoothing or point trajectories are added. This is a short preview, not a
full simulation animation or a quantitative flicker test.

The known unresolved same-group two-sheet counterexample still applies.
Good sample agreement and mesh topology checks do not prove physical topology,
shock ancestry or completeness. Injection-overlap flags remain measurements
to interpret; they are not removed by display reconstruction.

## Run

Use the optional CPU environment described in `SURFACE_DISPLAY_RESEARCH.md`:

```sh
python -m experiments.wind_bubble.check_adaptive_mesh_times \
  --output-dir outputs/three_bubble_adaptive_mesh_times_n064

python -m unittest tests.experiments.test_adaptive_mesh_times
```

Use a new output directory. The result directory's README records the observed
outcomes. The three new tests check that failed groups remain as points, a
global intersection suppresses mesh display, and a cross-group face is rejected.

## Observed issue and orientation-only correction

The frozen fit meets every per-group accuracy target at all three times. The
unmodified display nevertheless fails global winding checks at 0.12 and 0.14:
each has two inconsistent shared edges caused by one reversed triangle. The
0.13 mesh passes without a correction. These original results are retained in
`outputs/three_bubble_adaptive_mesh_times_n064/`, including the all-points
fallback where global orientation failed.

`orient_display_mesh.py` adds a separate orientation pass using the documented
[Open3D orientation operation](https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html#open3d.geometry.TriangleMesh.orient_triangles).
Repair requires orientability and manifold geometry. It chooses the component
sign that minimizes changes relative to the original face orientation, rather
than using wind-source ownership. Vertex arrays and each triangle's unordered
vertex membership must remain exactly unchanged. Edge/boundary counts are
checked unchanged, followed by an independent winding check.

`check_display_winding.py` applies this pass to the saved candidate meshes;
it does not refit the implicit field. All geometry, per-group accuracy and
physical-array readback checks are repeated. The previously passing event-16
reference returns exactly the same face array, providing a no-op regression.

The corrected results are in `outputs/three_bubble_adaptive_mesh_oriented_n064/`.
One face changes winding at 0.12, none at 0.13 and one at 0.14. All retained
groups then pass the display checks: 1/1, 8/8 and 4/4. The whole-scene fractions
of retained samples within a quarter cell of candidate triangles are 99.87%,
99.95% and 99.90%; per-group targets are checked separately. No self-intersections,
nonmanifold vertices/edges or inconsistent winding remain in the native checks.
Samples outside retained groups remain gray points (162, 122 and 50).

These results support a full-sequence preview with the orientation pass. They
do not validate flicker, prove physical surfaces or remove the original
same-group two-sheet ambiguity. Five new targeted tests pass across the display
approval and orientation helpers.

```sh
python -m experiments.wind_bubble.check_display_winding \
  --previous outputs/three_bubble_adaptive_mesh_times_n064 \
  --output-dir outputs/three_bubble_adaptive_mesh_oriented_n064

python -m experiments.wind_bubble.plot_mesh_time_review \
  --directory outputs/three_bubble_adaptive_mesh_oriented_n064

python -m unittest tests.experiments.test_adaptive_mesh_times \
  tests.experiments.test_display_mesh_orientation
```
