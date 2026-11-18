# Three-time display check with triangle winding corrected

Start with `review.png` for the original samples beside the smooth display,
or `three_state_preview.gif` for a three-frame preview. Both columns use a
z <= 0.5 cutaway. Camera, spatial bounds and continuation-segment colors stay
fixed; black stars mark the wind sources. Gray points are outside approved
display groups and remain present in the output.

## Finding and correction

The frozen fitter passes all group-accuracy checks at 0.12, 0.13 and 0.14.
Its original candidate at 0.12 and 0.14 nevertheless has one reversed triangle,
causing two inconsistent shared edges at each time. Those original failures
remain in `outputs/three_bubble_adaptive_mesh_times_n064/`.

An explicit orientation-only pass reverses one triangle at each affected time.
It changes no vertex coordinate, adds/deletes no candidate triangle, changes
no boundary edge and leaves every triangle's unordered vertex membership
exactly unchanged. It does not refit or smooth the surface. The 0.13 candidate
is unchanged. The earlier 0.16 static reference's complete face array also
remains bitwise unchanged under this pass.

| Time | Retained/approved groups | Faces reversed | Samples within 0.25 cell of candidate triangles | Samples outside retained groups |
| --- | ---: | ---: | ---: | ---: |
| 0.12 | 1/1 | 1 | 99.87% | 162 |
| 0.13 | 8/8 | 0 | 99.95% | 122 |
| 0.14 | 4/4 | 1 | 99.90% | 50 |

All groups meet the fixed per-group target separately; the table's agreement
percentages are whole-scene measurements over retained samples. At all three
times, native checks detect no self-intersections, nonmanifold vertices/edges
or inconsistent shared-edge winding after correction. No parameter or
accuracy threshold is relaxed to achieve that result.

The meshes have 2, 24 and 13 edge-connected pieces respectively. Piece counts
are not physical shock counts. A retained group may have several display
pieces because unsupported geometry is left open. The graph's group/segment
identification is unchanged, including the previously unresolved ancestry.

Source-overlap flags are preserved: 160, 132 and 94 upstream samples overlap
an injection region; downstream overlaps are zero at these times. A successful
display does not validate source-affected Mach or heating measurements.

## What passed and what remains

- All fitted triangle geometry and physical arrays are checked exactly against
  the saved candidates/catalogs, with source/input hashes rechecked afterward.
- Candidate topology and own-group sample distances are measured again after
  orientation. Five targeted tests cover approval/fallback and orientation.
- The GIF has three decoded frames. `review.png` was visually inspected.
- This supports a full-sequence preview with explicit winding correction.
  Three states do not quantify flicker or validate the remaining saved times.
- The same-group two-sheet ambiguity remains. These checks establish display
  consistency with the detections, not correct physical topology or ancestry.

`verification.json` reports both successful readback and all selected events
meeting display eligibility. Its `faces_added_or_removed: false` refers to
the candidate mesh geometry; approved display visibility changes from the
previous all-points fallback. `summary.csv`, `group_accuracy.csv` and
`checks.json` contain the measurements. `review.verification.json` records the
readable figure's provenance. Existing production defaults remain unchanged.

Reproduction commands are in `experiments/wind_bubble/ADAPTIVE_MESH_TIME_CHECK.md`.
This is CPU postprocessing; no new GPU simulation is required.
