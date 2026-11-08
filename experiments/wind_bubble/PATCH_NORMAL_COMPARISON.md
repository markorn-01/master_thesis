# Local geometric normals: display experiment

The previous diagnostic found that the tangent guard removes most proposed
triangles from one diagonal-facing inner-surface patch, but none from an
axis-facing patch. This experiment asks whether normal directions estimated
from neighboring positions explain that difference. No simulation rerun or
GPU is required.

## Method

For each point of event-local group 1 at t=0.16, select same-group neighbors
within either two or three grid spacings. Subtract their centroid and form
the position covariance matrix. Its smallest-eigenvalue eigenvector is the
normal to the least-squares plane. Orient its sign toward the saved shock
normal. Require at least six points, two resolved spatial directions and
a distinct smallest eigenvalue; otherwise retain the saved direction and
mark the fit invalid. The numerical eigenvalue checks do not guarantee a
physically meaningful local surface or exclude mixed sheets.

Keep the original refined positions and original ball-pivot triangle
proposals fixed. Apply the same geometric guards to these proposals with
each set of normals. Filtering and nonmanifold-edge removal use the whole
original event before cropping the two patches. Other groups retain saved
normals. This isolates the filtering effect; ball pivoting itself has not
been rerun using the fitted normals.

## Results

| Patch | Original normals | Plane fit, radius 2 cells | Plane fit, radius 3 cells |
| --- | ---: | ---: | ---: |
| Axis-facing: accepted out of 59 | 59 | 59 | 59 |
| Diagonal-facing: accepted out of 59 | 18 | 32 | 26 |

The diagonal core's median angular difference from the original normals is
4.50 degrees for the two-cell fit and 3.42 degrees for the three-cell fit.
The median maximum edge departure from endpoint tangent planes decreases
from 21.08 degrees to 19.02 and 19.61 degrees, respectively; the guard is
20 degrees. Both fit sizes have valid numerical fits at every core point.
Small changes in normals can therefore move triangles across the guard
threshold. The remaining gaps show that replacing normals alone does not
solve this patch's reconstruction problem. These two patches do not establish
which neighborhood size is best globally.

## Validation and interpretation

The saved-normal case reproduces the complete original accepted face set.
Brute-force neighbor counts and independent SVD plane normals check all
1,648 inner-group points at both radii (3,296 valid fits). Saved positions
and physical normals are read back exactly, and the input file hash is
checked unchanged. Five focused tests cover planar geometry, rotations,
normal sign, separate groups and unresolved neighborhoods. Together with
the three prior patch-angle tests, all eight tests pass.

The output NPZ stores explicitly named candidate normal arrays and validity
flags alongside unchanged input positions and saved normals. It is a
diagnostic artifact, not a replacement surface catalog. The comparison
image projects the original accepted triangles into each patch view;
overlaps in projection are possible.

Neither positions nor the physical shock estimator were smoothed. Mach,
density jumps, heating estimates and temporal associations are unchanged.
Group labels remain event-local, with no new physical identities assigned.
Local plane fits can mix nearby sheets belonging to one group; the existing
close-sheet counterexample remains unresolved. More accepted triangles do
not establish correct topology or justify using their area for physics.

## Reproduce

```sh
python -m experiments.wind_bubble.compare_patch_normals --output-dir outputs/two_bubbles_patch_normals_n128
python -m unittest tests.experiments.test_patch_normals tests.experiments.test_mesh_patch
```

Use a Python environment with NumPy, SciPy and Matplotlib. The output
directory must not already exist.

Next, evaluate any display-normal candidate on the existing known-geometry
controls, including holes and close sheets, before rerunning ball pivoting
on real surfaces. Keep the current production reconstruction unchanged
until those comparisons are reviewed.
