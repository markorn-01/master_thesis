# Original three-time check: retain the failures

This applies the previously checked adaptive fitter, unchanged, to saved
three-source events 12, 13 and 14 (t = 0.12, 0.13, 0.14). No hydrodynamic run
or retracking is performed. The fitting/support/refinement parameters are
frozen from `outputs/three_bubble_surface_display_adaptive_n064/`.

All retained groups meet the per-group criterion: at least 95% of their
original samples lie within a quarter cell of their own group's triangles.
However, the combined candidate mesh at 0.12 and 0.14 each has two inconsistent
shared-edge windings. Both frames therefore fail the global display gate and
use original points in the default view. The 0.13 frame passes.

| Time | Retained groups | Approved without winding correction | Inconsistent shared edges | Candidate sample agreement within 0.25 cell |
| --- | ---: | ---: | ---: | ---: |
| 0.12 | 1 | 0 | 2 | 99.87% |
| 0.13 | 8 | 8 | 0 | 99.95% |
| 0.14 | 4 | 0 | 2 | 99.90% |

`verification.json` has `passed: true` for completed provenance/readback checks
and `all_selected_events_display_eligible: false` for the display outcome.
The distinction is intentional. `candidate_faces` preserves the fitted
triangles for inspection; `faces` contains only approved display faces. All
original physical arrays, partitions and continuation-segment labels are
saved exactly. `summary.csv` geometry measurements refer to the complete
candidate; the approved display-face count is a separate column.

This evidence is preserved rather than overwritten by the subsequent
orientation-only correction. Use the corrected review in
`outputs/three_bubble_adaptive_mesh_oriented_n064/`. It reverses one triangle
at each affected time without changing any fitted geometry or accuracy target.

The preview GIF contains only three saved states, at one frame per second.
Its verification is an encoding/readback check, not a temporal-flicker test.
No physical topology, shock ancestry or injection-affected measurement is
validated by the smooth display. See `experiments/wind_bubble/ADAPTIVE_MESH_TIME_CHECK.md`.
