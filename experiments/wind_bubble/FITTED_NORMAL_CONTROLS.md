# Known-geometry tests of fitted display normals

Tested the original normal directions and covariance-plane fits with
neighborhood radii of two and three cell widths. All positions, ball radius
(one cell width), group labels and geometric guards stayed fixed.

Two stages distinguish effects: filtering the original raw triangle proposals
with each normal set, and rerunning ball pivoting with each normal set before
filtering. Five controlled inputs, three modes and two stages give **30
checked combinations**. Truth sheet labels are used only for evaluation;
the reconstruction sees only the supplied group labels.

## Results

Both stages and all three normal modes gave the following aggregate metrics:

| Input | Accepted triangles | Sampled hole intrusions | Cross-sheet triangles |
| --- | ---: | ---: | ---: |
| Complete sphere | 3,196 | not applicable | 0 |
| Sphere with missing cap | 2,503 | 0 | 0 |
| Plane with central hole | 444 | 0 | 0 |
| Nearby sheets with separate group labels | 255 | not applicable | 0 |
| Nearby sheets sharing one group label | 288 | not applicable | **288** |

The complete sphere area is 99.805% of its analytic area, with zero boundary
edges. The cap surface area is 97.703% of its reference area, with 55 boundary
edges. All combinations have zero nonmanifold edges and zero inconsistent
shared-edge winding. These aggregate agreements do not assert identical
triangle connectivity between all fitted-normal modes.

At the cap boundary, 12 points have fewer than six neighbors at radius two
cells, so their fitted normals are explicitly invalid and the saved normals
are retained. The three-cell fit has no invalid points. Other controls have
no invalid fits. The saved-normal mode does not perform a fit.

The same-group close-sheet case remains a known reconstruction failure:
every accepted triangle mixes vertices from the two stipulated surfaces.
Neither the original normals nor either fitted setting separates them.
Zero nonmanifold edges therefore does not establish correct physical
topology. Separately labeled sheets remain separate because reconstruction
and fitting both operate within each supplied group.

## Checks and scope

All five original single-radius baseline face sets reproduce the earlier
method comparison. Every combination is checked against the geometric
limits with an independent checker. Areas are cross-checked using Heron's
formula. Reopened artifacts preserve original positions and saved normals
exactly, and all metrics are recomputed from the saved faces. Reference
hashes and source hashes are recorded in `readback.json`. The eight local
normal/patch tests also pass.

Hole intrusion checks sample ten barycentric locations per triangle. The
cap check uses normalized probe height above 0.62; the planar hole checks
radius below 2.5. These are the existing conservative test interiors, not
exact continuous intersection tests or proof that every hole boundary is
recovered. No global self-intersection or vertex-manifold proof is provided.

The controls show no measured regression from these normal fits. They do
not demonstrate better reconstruction on ideal geometry, or resolve nearby
surfaces sharing a group. Together with the real-patch improvement, they
justify a limited real-data display comparison, not replacement of the
physical shock normals or a claim of a general reconstruction solution.
The production method, Mach values, heating and temporal associations are
unchanged.

## Reproduce

In a Python environment with the optional Open3D requirements installed:

```sh
python -m experiments.wind_bubble.check_fitted_normal_controls --output-dir outputs/two_bubbles_fitted_normal_controls
python -m unittest tests.experiments.test_patch_normals tests.experiments.test_mesh_patch
```

The output directory must not already exist. `summary.csv` reports both
stages; the figure shows full ball-pivot reruns. NPZ files retain original
and fitted normals, fit flags, raw proposals and accepted faces.

Next: compare the original and two-cell display-normal reconstruction on
the full real event, with explicit gap and topology measurements. Keep the
three-cell option as a sensitivity comparison rather than selecting a
global setting from one favorable patch.
