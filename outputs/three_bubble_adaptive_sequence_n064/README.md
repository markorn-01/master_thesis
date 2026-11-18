# Checked meshes for all 17 saved three-source events

This CPU postprocessing run covers t = 0–0.16 of the existing 64³ three-source
simulation. It does not rerun hydrodynamics, shock detection, grouping or
tracking, and does not change any scientific measurement.

## Outcome

- Event 0 is the empty initial state (`no_detections`), with no measured
  sample-to-surface distance statistic.
- Fifteen of the sixteen populated events pass every mesh display check.
- Event 5 (t ≈ 0.05) uses original points because the combined candidate mesh
  has a detected self-intersection. It is explicitly `geometry_fallback`.
- All 45 retained-group observations meet the own-group criterion: at least
  95% of samples within a quarter cell of their own candidate triangles.
  The lowest group agreement is 95.238%. These are observations across times,
  not 45 distinct physical shocks.
- The rejected event's three groups each have 100% quarter-cell sample
  agreement. Thus close sample fit alone does not certify valid geometry.
  The intersection is not diagnosed as physical shock crossing by this check.

`verification.json` has `passed: true` for completed construction/readback and
`all_events_with_retained_groups_approved: false` for the display outcome.
The fallback is preserved rather than removed, interpolated over or made to
pass by weakening parameters. Samples outside retained groups also stay as
raw points, separately from this rejected event.

## Frozen method and exact references

The fixed local implicit field, support limits, local-hull trimming,
conservative defect removal, bounded extraction refinement and orientation-only
pass are those already checked. No vertex is relocated by the orientation
pass. Display approximations remain separate from the original physical arrays.

Events 12, 13 and 14 reuse the exact checked oriented meshes. Event 16 reuses
the exact static adaptive mesh. Physical arrays, labels, times, vertices and
complete candidate face arrays are checked; every reused candidate is also
remeasured. Other states are built independently. No temporal smoothing or
other event's geometry enters a fit. Source/input hashes are verified again
after completion.

The sequence includes 86 triangle reversals in new builds. The earlier
orientation corrections in reused references are already included in those
saved face arrays and are not performed again. Triangle reversal changes
winding only, not surface geometry or physical shock normals.

## Files

- `mesh_0000.npz` through `mesh_0016.npz`: original arrays, candidate geometry,
  approved display faces, group labels and approved-group IDs.
- `event_*.json`: source/input hashes, construction/refinement/orientation
  details, group checks and mesh checksum for each event.
- `summary.csv`: whole candidate metrics and separate approved/fallback
  populations. `group_accuracy.csv`: own-group distances and source overlaps.
- `metadata.json`: complete status, frozen configuration and resume manifest.
- `verification.json`: complete saved-event coverage, reuse and outcome flags.

`candidate_faces` retains rejected geometry for inspection. `faces` contains
only the approved default display geometry, and is empty at event 5. All
original samples remain under `original_*`, including source-overlap flags.
Native geometric checks are numerical tests, not proofs of physical topology.
The previously demonstrated same-group close-sheet ambiguity remains.

The full movie is in `outputs/three_bubble_adaptive_animation_n064/`.
Reproduction and explicit resume commands are in
`experiments/wind_bubble/ADAPTIVE_SURFACE_ANIMATION.md`.
