# Bounded delayed fragment-support experiment

## What question this answers

At 128³, the inner detection changes from fourteen small groups at t=0.10 to
two larger inner groups at t=0.11. At 64³ the analogous change is ten patches
at t=0.13 to two groups at t=0.14. The adjacent-event prototype could choose
only three parents per combination; almost interchangeable triples had tiny
score differences. Choosing a preferred triple is an unnecessarily restrictive
description if the task is to follow a fragmented observed surface.

This experiment adds **conditional geometric support families**, separate from
the frozen event graph and from claims of unique physical ancestry. Each later
coherent group must already have a shared one-to-one continuation into the
third state. Earlier fragments are compared with that family, allowing any
number of fragments. Already shared earlier continuations (the outer surface)
are excluded from this focused review.

## Primary references and actual adaptation

1. Reid (1979), *An Algorithm for Tracking Multiple Targets*, IEEE TAC
   24(6), 843–854, DOI 10.1109/TAC.1979.1102177.
   [Primary paper](https://graphics.stanford.edu/courses/cs428-03-spring/Papers/readings/CollaborativeProcessing/Reid_MHT_ieee_trans_ac_1979.pdf).
   Supports deferring associations and using later observations. We do not
   implement its probabilistic MHT, Kalman filtering or hypothesis tree.
2. Jaqaman et al. (2008), *Robust single-particle tracking in live-cell
   time-lapse sequences*, Nature Methods 5, 695–702, DOI 10.1038/nmeth.1237.
   [Author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC2747604/).
   Supports a second stage linking already formed track segments and treating
   merges/splits explicitly. Our surface-support test is not their LAP
   optimizer; particle intensity conservation is not transferred to shock
   samples, whose counts are not conserved quantities.
3. Schnorr et al. (2020; online 2018), *Feature Tracking by Two-Step
   Optimization*, TVCG 26(6), 2219–2233, DOI 10.1109/TVCG.2018.2883630.
   [Authors' description](https://www.graphics.rwth-aachen.de/publication/02168/).
   Informs the existing event-conflict approach. Its adjacent-time optimizations
   do not by themselves supply new evidence for identical parent subsets.
4. Pfister et al. (2000), *Surfels: Surface Elements as Rendering Primitives*,
   SIGGRAPH, 335–342, DOI 10.1145/344779.344936.
   [Primary publisher page](https://doi.org/10.1145/344779.344936).
   Informs a separate display experiment using oriented disks without
   connectivity. We do not implement their full renderer, octree, visibility
   splatting or EWA filtering.

These are references for specific design ideas, not evidence that this original
adaptation inherits their published accuracy. Existing forward centroid/RMS
prediction experiments in `motion_prediction.py` and `expansion_prediction.py`
remain unchanged. Here the direction is retrospective: a later continuation
conditions earlier fragment support.

## Fixed three-state procedure

For times t0<t1<t2, take shared group continuations B(t1)→C(t2).
For each B sample, find its nearest normal-compatible C sample within the
association gate. The median displacement projected onto B's normals is s.
Both B and C need at least 50% gated sample coverage to use the model.
Predict each B position backward using

`x_pred(t0) = x_B - [(t1-t0)/(t2-t1)] s n_B`.

This assumes approximately constant normal motion over this short window;
it is not a hydrodynamic propagation solution or material-point trajectory.

For each earlier fragment and predicted family:

- Signed normal angle <=60 degrees, same physical distance gates as before.
- At least 50% of fragment samples must have compatible nearby family samples.
- Score = coverage × (1 - 0.25 × mean covered distance / gate).
- Prefer a family only if its score exceeds the next eligible family (or the
  unmatched score zero) by >=0.1. This is an algorithmic margin, not probability.
- All preferred fragments are collected, without a three-parent cap. Require
  their union to cover >=50% of the predicted child samples. Overlapping support
  does not establish exclusive ancestry or conserved area/energy.

`static_family` uses the same later-conditioned families and thresholds but
does not move B backward. This ablation separates the changed family definition
from any benefit of the motion estimate. Both methods require the later
continuation, so even the static test is delayed. Nothing is written back into
the original accepted tracking graph. Online use would have one-event latency;
that integration is not implemented here.

## Controlled results: seven cases, three gates, three methods

| Method | Exact / 21 | False stipulated links | Missed stipulated links |
| --- | ---: | ---: | ---: |
| Adjacent event prototype | 6 | 0 | 36 |
| Static family support | 9 | 10 | 2 |
| Delayed normal-motion family support | 13 | 6 | 2 |

This is a small targeted suite, not a general accuracy ranking. The eight-
fragment case tests the cap limitation. Two motion cases test retrospective
information; the smallest gate rejects their future seed continuation. Two
controls have identical observed data but different stipulated ancestry. Both
family methods include both supported parents there: the six false links
demonstrate why these outputs must not be claimed as physical ancestry.
Coincident competing families remain ambiguous; opposite-normal distractors
are rejected. Forty-two independent dense geometry/score/union checks and
forty-two shuffled-order checks cover the family methods.

## Saved hydro windows: result and ablation

| Resolution / times | Fragments per inner family | Union coverage at narrow gate | Wide gate |
| --- | ---: | ---: | ---: |
| 64³, 0.13 / 0.14 / 0.15 | 5 and 5 | 89.47% each | 100% each |
| 128³, 0.10 / 0.11 / 0.12 | 7 and 7 | 91.70% each | 100% each |

All ten/fourteen inner patches receive the same two-family membership at both
gates and in both methods. **Static and motion-predicted family memberships
are identical on these real windows.** Therefore the measured improvement in
description comes from accepting full fragment families, not demonstrated
benefit of normal-motion prediction. The real data have no independent
physical ancestry labels. Each of eight real method/window/gate combinations
is independently checked using dense distances. Original fields are unchanged.

## Display and scope

`outputs/inner_fragment_family_review_v1/` shows the original points and a
separate oriented-disk rendering of exactly the same family samples. Disks
have radius 0.45 grid cells and ten polygon vertices, centered on the original
refined positions in the plane perpendicular to each saved normal. No points
move, no cross-sample triangles are created, and physical diagnostics are not
recomputed. The display still has holes; it is not a completed smooth mesh or
a topology reconstruction. Disk overlap alone is not evidence of continuity.

## Current decision and next validation

Keep this as a labeled support layer, not a replacement physical-ancestry
tracker. Freeze these parameters for an unequal-wind test with source strengths
1 and 0.5, identical source locations/injection radius and shared wind speed.
See `experiments/wind_bubble/UNEQUAL_BUBBLES.md`. Remote collision validation
is pending; a short CPU injection/online-offline smoke test is separate evidence.

```sh
python -m experiments.astrolink.check_tracklet_fragments --output-dir outputs/tracklet_fragment_support_v1
python -m experiments.wind_bubble.plot_fragment_families --output-dir outputs/inner_fragment_family_review_v1
python -m unittest tests.experiments.test_tracklet_fragments tests.experiments.test_surfel_display
```

Use new output directories. Reproduction uses saved catalogs only and requires
NumPy, SciPy and Matplotlib, not JAX or a GPU. The Overleaf draft and bibliography
are in `outputs/tracklet_fragment_support_v1/thesis_fragment_support.tex` and
`tracking_references.bib`. Do not describe the unequal collision as completed.
