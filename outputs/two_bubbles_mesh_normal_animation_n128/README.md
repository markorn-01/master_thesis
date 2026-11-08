# Experimental mesh comparison animation

Open `mesh_normal_comparison.gif`: 17 saved events, t=0–0.16, 2 FPS,
8.5 seconds per loop. Original shock normals are on the left; experimental
two-cell geometric display normals are on the right. Same refined positions,
Mach values, camera, scales and z≤0.5 cutaway in both panels.

See `experiments/wind_bubble/MESH_NORMAL_ANIMATION.md` for method, results,
checks and commands. `summary.csv` reports whole-mesh metrics. NPZs retain
original and candidate normals separately. `verification.json` records
encoding checks, mesh metrics and source/input hashes.

The fitted reconstruction does not improve every event or measure. Gaps
remain; physical holes, within-group topology and temporal flicker are not
validated. Some cross-group separation bounds are inconclusive. This does
not change physical diagnostics, add persistent IDs, or replace the
production renderer.
