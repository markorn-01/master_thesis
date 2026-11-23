# Why are detections missing between the lower two bubbles?

This diagnostic addresses Tobias's question about the central interaction in
the three-bubble movies. It follows the path from the gas state to the exported
shock samples, before any grouping, tracking, or mesh reconstruction.

## What the existing saved products establish

`inspect_collision_slice.py` reads the original density/pressure slices and
surface catalogs. In the box x=0.45–0.55, y=0.335–0.415, z=0.42–0.58, there
are no exported surface samples from t=0.07 through 0.16. This late gap is
already present before grouping and smoothing. Earlier detections at t=0.05
and 0.06 can fall outside the displayed half-cell slab.

Across the dense central layer, density rises while P/rho falls. The in-plane
temperature/density gradient dot product is negative throughout the central
slice ROI from t=0.06. That is a reason to check the alignment criterion, but
the saved slices omit the z-gradient contribution and the velocity field.
They cannot establish the full 3D rejection reason or classify the layer as a
contact or an unresolved shock.

Preliminary products: `outputs/three_bubble_collision_slice_check_n064/`.

## Matching replay

`audit_collision_detection.py` reuses the original three-source geometry,
source-budget metadata, float32 event schedule, and solver timestep cap. It
checks the saved reference's source fingerprints before integrating. Tracking
is disabled for this diagnostic; it does not alter the gas or primary shock
products. No detection rule, threshold, Mach estimator, or source model changes.

The observer saves physical 3D states at events 4, 6, 8, 12, and 16 (t=0.04,
0.06, 0.08, 0.12, 0.16). These are deliberate validation snapshots, about 25 MiB
in total, rather than a change to ordinary compact output.

The audit evaluates the actual production functions and checks:

1. The full 3D alignment mask equals `grad(P/rho) dot grad(rho) > 0`.
2. The minimum-jump mask equals the intersection of its pressure and
   temperature tests, with the existing Mach-1.3 thresholds and boundary rule.
3. The shock-zone mask equals compression AND alignment AND both jump tests.
4. Selected surfaces are inside the zone, and their refined positions and
   grid indices reproduce the replay's exported catalog after source exclusion.
5. All 17 replay surface catalogs, histograms, and gas slices agree with the
   original GPU run using the existing comparison tolerances. A mismatch is
   reported as a failed replay; it is never silently accepted or used as an
   explanation of the original movie.

Use a free, working GPU and the compatible Python environment:

```bash
export CUDA_VISIBLE_DEVICES=<free-index> XLA_PYTHON_CLIENT_PREALLOCATE=false

"$HOME/.venvs/master-thesis-py312/bin/python" -m unittest \
  tests.experiments.test_collision_detection_audit \
  tests.experiments.test_detection_audit

"$HOME/.venvs/master-thesis-py312/bin/python" \
  -m experiments.wind_bubble.audit_collision_detection \
  --require-gpu \
  --output-dir outputs/three_bubble_collision_detection_audit_iwr_n064
```

Choose a fresh output directory. The runner will not overwrite earlier results.
The default reference is
`outputs/three_bubbles_online_tracking_iwr_compgpu8_n064`; the directory name
is historical and does not select a GPU. There is one integration, not an
additional full offline simulation.

## How to read the output

All audit products are under the new run's `audit/` directory:

- `verification.json`: replay agreement, mask/export invariants, thresholds,
  source/input/state hashes, and central-slice counts.
- `collision_stage_counts.csv`: independent pass/fail counts, sole-failure
  counts, and mutually exclusive four-test patterns for both the central
  3D volume and the saved slice. Pattern bits from low to high are compression,
  alignment, pressure jump, temperature jump. Several tests can fail in one
  cell; independent failure counts must not be added together.
- `production_masks_*.png`: six panels show compression, alignment, pressure
  jump, temperature jump, combined zone, and selected surface. Teal means
  passing/selected; density contours locate the gas structure. The central
  counted box is drawn explicitly, with a wider surrounding view.
- `collision_profiles_*.png` and `cross_collision_profiles.csv`: resolved
  density, pressure, P/rho, velocity, sound speed, entropy proxy P/rho^gamma,
  divergence, full gradient dot product and its x/y/z terms, and local jumps.
  The line is at the nearest y cell to 0.375 and the saved central z plane.
  Lab-frame velocity is not a shock-frame Mach estimate.
- `collision_crop_*.npz`: local 3D primitive states, coordinates and production
  arrays, including both flanks of the layer; `state_*.npy`: the five full
  physical states used to compute those arrays.

Grid-center ROI counts and refined-position catalog counts are distinguished.
The box is outside the injection spheres and does not claim to cover every
possible interacting shock front.

## What this check can conclude

A passing replay can identify exactly which production tests remove cells
in the checked region, and whether surface selection/export causes further
losses. Profiles then show whether the rejected resolved structure contains
compression and opposing thermodynamic gradients. This is a detector audit,
not an independent proof of physical shock/contact identity. If the structure
is unresolved or overlapping, a physical accuracy claim requires a separate
controlled or resolution study. Do not weaken a criterion merely to fill the
visible gap.

Local controls use an exact-state compressive planar shock and an isobaric
density transition with retained converging velocity. They verify that the
audit retains the former and exposes alignment rejection of the latter;
they do not substitute for the matching three-bubble replay.

## Local preparation result

The focused controls and existing detection-audit tests passed (10 tests).
An end-to-end 64-cubed CPU check completed and its production-mask/export
invariants passed. Its strict comparison to the original GPU products failed
on surface normals (first reported maximum absolute mismatch about 4.06e-7).
The tolerance was not relaxed. Its output directory explicitly records that
it is preliminary: `outputs/three_bubble_collision_detection_audit_cpu_n064/`.
The matching GPU replay remains required for the original-run explanation.

Commit the audit code, tests and instructions before pulling them remotely;
the local CPU full-state files are not needed for that run.
