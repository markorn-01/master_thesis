# Checked meshes for the unequal four-source sequence

CPU postprocessing of the existing 64³ GPU run: all 17 saved events, t=0 to
t=0.16. No simulation, detection, grouping or association is rerun here.

The existing implicit oriented-plane fit, masked marching cubes, support
restrictions, conservative cleanup, bounded grid refinement and orientation
checks are reused with their existing settings. Each approved group meets the
target of at least 95% of original samples lying within 0.25 grid cell of its
own mesh. This measures agreement with detections, not physical truth.

All retained groups pass at every populated event except t=0.11. At that
event, the 31-sample group g2 has no adequate reconstruction and remains raw
points; the other two groups pass. The initial empty event remains empty.
The existing final combined-cleanup branch was not needed in this sequence.

At the final event all five retained groups pass. The displayed mesh contains
155,510 triangles, while the 42 samples excluded by grouping remain points.
Those counts refer to display geometry and detected samples, not physical
shock objects or measured surface areas.

`mesh_*.npz` preserves the original positions, normals, Mach values, labels,
continuation segments and overlap flags alongside separate display vertices
and faces. `event_*.json` records group accuracy, cleanup, orientation and
fallback decisions. `tables/` preserves the verified callback histories,
statistics and causal relationship availability. `environment.json` records
the separate CPU display environment; the JAX environment was not changed.

The table join has 71 group observations, 25 continuation segments and one
additional conditional relationship. The latter becomes available at event
15 (t=0.15); it does not change IDs or statistical continuity.

Smooth appearance does not establish correct physical topology. Very close
sheets in one group remain a known limitation. Each frame is fitted
independently; no hole filling or temporal smoothing is performed.

The complete movie is in
`outputs/four_bubbles_unequal_animation_n064/four_bubble_evolution.gif`.
Commands and panel explanations are in
`experiments/wind_bubble/MULTI_SOURCE_ANIMATION.md`.
