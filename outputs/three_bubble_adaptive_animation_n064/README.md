# Full three-source adaptive surface animation

Open `adaptive_surface_evolution.gif`. It contains all 17 saved events,
including the empty initial state, from simulation t = 0 to 0.16. Playback is
one frame per second (17 seconds) and loops. No intermediate simulation states
are added. This is CPU visualization of existing catalogs, not a new GPU run.

## Panels

- Left: original detected surface samples.
- Right: approved adaptive mesh groups, with original gray points for rejected
  meshes and samples outside retained groups.

Both panels use the same fixed camera, bounds and z <= 0.5 cutaway. Black
stars mark wind sources. Colors follow geometric continuation-segment labels;
combinations receive new labels/colors. Colors do not measure Mach number or
prove wind ownership. Source-overlap flags and the lowest per-group agreement
are shown for each event. Each fit is independent in time.

## Result

Fifteen of the sixteen populated events pass all mesh display checks.
Event 5 (t ≈ 0.05) uses gray original points because its combined candidate
mesh has a detected self-intersection. The movie states the reason explicitly.
Its three groups have 100% quarter-cell sample agreement, illustrating why
sample agreement alone is insufficient. No event is removed, interpolated
over or made to pass by weakening fitting/support limits.

All 45 retained-group observations across the sequence meet the fixed own-group
accuracy target; the lowest agreement is 95.238%. These are repeated geometric
observations, not 45 physical shocks. Samples outside retained groups remain
points even when all retained groups at an event are approved.

`display_checks_over_time.png` shows candidate sample agreement and approved
versus retained group counts. The drop in approved groups at 0.05 is the
geometry fallback, not evidence that physical shocks disappeared. Agreement
is undefined for the empty initial event.

Example stills include the empty event, rejected event 5, events 6/8, the
fragment transition at 13/14 and final event 16. Every surface vertex and
scientific measurement remains as stored in the checked mesh/catalog outputs;
rendering changes only visibility and appearance. Source overlaps remain
flagged, not removed or declared physically valid by a successful mesh.

## Verification

- All 17 GIF frames are decoded, with the expected one-second durations.
- Every decoded header is matched against all expected rendered headers;
  saved-event order matches exactly, allowing GIF palette quantization.
- Original positions, normals, Mach values and overlap flags are read back
  against the catalogs before rendering. All recorded input/source hashes
  are checked afterward.
- Seven example PNGs and the quality-history PNG are decoded. The rejected
  and final-state stills and the history plot were visually reviewed.
- Two frame-encoding tests pass, including intentional header-order rejection.
  An empty/first-populated-state smoke run and explicit build resume also pass.

The complete meshes and diagnostics are in
`outputs/three_bubble_adaptive_sequence_n064/`. Events 12, 13, 14 and 16 retain
their exact previously checked candidates. Fitting settings, physical
diagnostics, spatial grouping and temporal associations remain unchanged.

This movie supports visual review. It does not quantify temporal flicker,
establish physical surface topology/ancestry, resolve the same-group close-sheet
counterexample, or validate arbitrary turbulent geometry or higher resolutions.
The known 0.05 candidate is preserved for inspection in `mesh_0005.npz` under
`candidate_faces`; its approved `faces` array is empty.

See `experiments/wind_bubble/ADAPTIVE_SURFACE_ANIMATION.md` for reproduction.
