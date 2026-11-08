# Original versus fitted-normal mesh animation

`outputs/two_bubbles_mesh_normal_animation_n128/mesh_normal_comparison.gif`
compares all 17 saved events of the 128³ budget-matched two-bubble run,
from t=0 to 0.16. At two frames per second it lasts 8.5 seconds and loops.
Each displayed frame is one saved state, without interpolation.

## Panels

- Left: original saved physical shock normals used for ball pivoting and
  geometric triangle guards.
- Right: experimental display normals fitted to same-group neighbors
  within two grid spacings, then used for the same reconstruction and guards.

Both panels preserve the original refined sample positions and physical
Mach values. They share a fixed camera, spatial bounds and Mach color scale
0–16. Face colors are the mean of vertex Mach values. The z≤0.5 cutaway
removes upper faces for display only. Gray points are unused or excluded
samples, not gas density or transparent gas. The initial empty frame is
retained. No fit or reconstruction uses information from another event.

The labels below each panel report whole-mesh statistics, not just the
visible cutaway: used samples, edge-connected pieces, open mesh edges and
accepted faces. Mesh pieces are not physical shocks or persistent IDs.

## Sequence-level observations

Excluding the initial empty event, the fitted reconstruction has:

| Measurement | Increased | Decreased | Unchanged |
| --- | ---: | ---: | ---: |
| Used samples | 12 events | 4 events | 0 |
| Edge-connected mesh pieces | 5 events | 11 events | 0 |
| Open mesh edges | 14 events | 2 events | 0 |
| Accepted triangles | 11 events | 5 events | 0 |

Thus the three selected-time improvements do not extend uniformly to all
events. In particular, fitted normals do not reliably reduce mesh openings.
More represented points or a larger connected piece do not prove correct
topology. At t=0.10, 0.12, 0.13 and 0.14, respectively, 72, 56, 48 and 16
retained samples have invalid plane fits and explicitly keep saved normals.
The physical normals themselves are never overwritten.

This animation supports visual inspection of frame changes; it does not
quantitatively validate temporal flicker, assign persistent identities, or
establish physical shock merging. The known same-group close-sheet false
bridge remains unresolved. No physical Mach, density, heating or tracking
diagnostic is altered, and no production default is replaced.

## Checks and reproduction

All 34 mesh outputs pass the independent geometric checker using their
actual normal sets. Saved output positions, physical normals, Mach values
and group labels are read back exactly against the source data. At t=0.08,
0.12 and 0.16, both modes reproduce the prior checked faces and normals
exactly. All meshes have zero nonmanifold edges and inconsistent shared-edge
winding, without claiming global self-intersection or within-group validity.
Of 288 recorded cross-group distance/edge bounds, 276 establish separation;
12 are inconclusive at t=0.06, 0.07, 0.10 and 0.14, in both modes. A failed
sufficient bound is not evidence of an intersection, but separation cannot
be claimed for those group pairs from this check alone.

The GIF is decoded and every frame's size, 500-ms duration and order checked
against its rendered saved-event/time header, allowing palette quantization.
This verifies encoding, not physical or temporal correctness. Fifteen focused
tests pass, including empty initial events and entirely excluded input groups.
Input and source hashes are recorded in `verification.json`.

```sh
python -m experiments.wind_bubble.animate_mesh_normals --output-dir outputs/two_bubbles_mesh_normal_animation_n128 --fps 2
python -m unittest tests.experiments.test_patch_normals tests.experiments.test_mesh_patch tests.experiments.test_temporal_mesh
```

Requires the optional Open3D environment; no GPU or hydrodynamic rerun is
needed. The output directory must not already exist. Outputs include 17
paired mesh NPZs, the GIF, four example PNG frames, a 34-row CSV and the
verification report. Candidate display normals are stored separately from
saved physical normals. The images are display experiments, not replacement
physics products.
