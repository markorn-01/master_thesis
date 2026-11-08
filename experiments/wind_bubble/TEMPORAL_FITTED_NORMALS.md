# Three-time check of fitted display normals

Compared the original reconstruction with a two-cell-radius covariance-plane
normal fit at t=0.08, 0.12 and 0.16. Each fit and ball-pivot reconstruction
uses that event alone, without temporal interpolation or access to another
event. Ball radius remains one cell width and geometric guards are unchanged.
The original three-time mesh outputs provide baseline faces and positions.

## Results

Arrows below mean original normals → fitted display normals.

| Time | Used samples | Accepted faces | Edge-connected pieces | Open mesh edges |
| --- | ---: | ---: | ---: | ---: |
| 0.08 | 10,149 → 10,181 | 15,947 → 15,805 | 391 → 375 | 7,961 → 8,271 |
| 0.12 | 17,452 → 17,837 | 26,556 → 26,975 | 622 → 606 | 13,938 → 14,647 |
| 0.16 | 22,915 → 23,389 | 35,408 → 36,497 | 830 → 688 | 17,758 → 18,053 |

More points are represented and total piece counts decrease at all three
times. However, open-edge counts increase at every time, and the early
event loses 142 accepted triangles. At t=0.12 singleton pieces increase
from 402 to 482; the outer group's piece count increases from 508 to 516
while its largest piece grows. Thus, the candidate does not uniformly
improve connectivity or remove gaps. Piece counts are mesh properties,
not counts of physical shocks.

At t=0.12 the two inner retained groups each decrease from 57 to 45 pieces.
At t=0.16 they decrease from 155/157 to 88/90. This supports the earlier
observation that fitted normals can help the inner-surface display.
Group labels are assigned within each event and do not identify persistent
physical shocks across these three times.

All retained points have numerically valid fits at t=0.08 and 0.16. At
t=0.12 there are 56 invalid fits, which explicitly retain saved normals.
The 40 samples excluded by the original grouping remain excluded; they
are distinct from the 56 retained samples with invalid fits. Numerical fit
validity does not establish correct surface membership or physical accuracy.

## Validation

- Original positions, physical normals, Mach values and labels are read back
  exactly for every output. Baseline face arrays are copied and checked
  exactly; event-16 fitted normals and faces reproduce the prior full-event
  comparison exactly.
- Independent geometry checks cover all six accepted meshes. All have zero
  nonmanifold edges and zero inconsistent shared-edge winding.
- Of 700 deterministic sampled neighborhoods (100 per retained group and
  event), 694 have valid fits and their directions match independent SVD
  plane estimates. Neighbor counts are checked by brute-force distances.
- At t=0.12 and 0.16, all three pairwise group-separation bounds pass for
  each mode. At t=0.08 only one group exists, so this test is inapplicable.
  These bounds exclude intersections between distinct groups, not false
  bridges or intersections within one group.
- Thirteen focused normal-fit, patch-angle and temporal-geometry tests pass.
  Input and source hashes are recorded in `readback.json`.

No physical diagnostic, shock normal, Mach estimator, tracking association,
source prescription or point position was changed. The close same-group
sheet counterexample remains unresolved. Open mesh boundaries are not
validated physical holes. Three snapshots do not establish temporal flicker
or the correctness of intervening animation frames.

## Reproduce and outputs

```sh
python -m experiments.wind_bubble.check_temporal_fitted_normals --output-dir outputs/two_bubbles_temporal_fitted_normals_n128
python -m unittest tests.experiments.test_patch_normals tests.experiments.test_mesh_patch tests.experiments.test_temporal_mesh
```

Requires the optional Open3D environment; CPU postprocessing only. The output
directory must not already exist. Outputs contain six explicitly named mesh
NPZs, aggregate and per-group tables, a readback report, fixed-camera Mach
cutaways, and actual z=0.5 triangle-plane sections. Black/colored point
positions are never smoothed. Gray points in cutaways include unused samples
and size-excluded groups. In saved-normal mode `fit_valid` is false because
no fit was attempted.

Next, a short side-by-side animation of all saved events can inspect temporal
appearance, keeping the original and candidate labeled separately and
retaining the quantitative gap/topology limitations. It should be presented
as an experimental visualization, not an established general shock-surface
reconstruction or physical tracking solution.
