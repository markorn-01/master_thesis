# What blocks early weaker-source inner detections?

Input: GPU selected-state audit at 128³, unchanged source strengths 1:0.5,
source budgets, detection thresholds and 17-event schedule.

## Demonstrated findings

The production detection threshold M_min=1.3 gives local pressure ratio
>=1.8625 AND temperature ratio >=1.2921782544. Both must be met at the same
cell, together with compression and aligned gradients.

For local inward cells outside the discrete injection mask that already pass
compression AND gradient alignment:

| Time | Eligible cells | Maximum local P ratio | Maximum local T ratio | Pass both jumps |
| --- | ---: | ---: | ---: | ---: |
| 0.10 | 568 | 1.26873 | 1.02293 | 0 |
| 0.12 | 1480 | 1.43267 | 1.15319 | 0 |
| 0.14 | 2400 | 1.68044 | 1.23778 | 0 |
| 0.15 | 2976 | 2.04408 | 1.35790 | 12 |
| 0.16 | 3272 | 2.45817 | 1.47380 | 56 |

Maxima are separate summaries, not necessarily values from the same cell.
At the first three times both maxima are below their thresholds. In fact,
no cell anywhere in the inspected weaker-source sphere passes the local
pressure test at those three times. The sphere has radius twice the fixed
injection radius. The inward subset is a geometric inspection population,
not physical source ownership or a complete shock label.

For this local population, every zone cell at 0.15/0.16 becomes a surface cell;
no detected surface is removed by the source-export exclusion. Thus later
surface selection and catalog source exclusion do not explain these missing
early exported samples. Spatial grouping subsequently removes the 12-sample
patch at 0.15 and retains only 24 of the 56 selected samples at 0.16, as
established in the previous component audit.

This identifies the **detection-stage local jump conditions** as the immediate
bottleneck. It does not prove that a physical shock above Mach 1.3 was missed.
Possible physical/numerical explanations for the measured weak local jumps
remain distinct from an implementation error. The final profile-aware Mach
estimator was not changed and is not what rejects these early cells.

## Offline sampling-separation check

`stencil_sensitivity.csv` repeats ONLY the local jump measurement at one,
two, three and four cell widths on each side, keeping the same eligible cells,
directions and threshold ratios. This does not regenerate zones/surfaces,
run a new simulation, or define a replacement shock finder. SciPy multilinear
sampling at one-cell separation is first checked against the production JAX
log jumps (rtol=1e-4, atol=3e-6).

No eligible weak-source cell at 0.10, 0.12 or 0.14 passes both jumps at any of
these separations. Widening the local stencil alone therefore does not recover
the early detections in this controlled probe.
At 0.15 the joint counts are [12,40,0,0]; at 0.16 [56,216,108,0]. Wider sampling
is not monotonically better: it can cross nonuniform/source-affected gas.
The CSV separately reports endpoint exclusion and all eight interpolation
corners versus the actual discrete source mask, plus candidate-center exclusion.
These geometric checks do not guarantee unperturbed upstream/downstream states.
No sensitivity result is promoted into the existing tracking graph.

## Figure

`detection_gate_profiles.png` shows one illustrative +x ray near the weaker
source. It uses nearest saved y,z grid centers rather than an interpolated
exact-axis profile. The ray cannot represent the whole surface. Pressure,
pseudo-temperature, local log jumps and velocity divergence are shown at all
five times; the last panel uses full 3D eligible-cell counts.
Gray indicates discrete source injection; dotted vertical line is geometric
source exclusion. Red lines are unchanged jump thresholds. Divergence <0 is
compression. These local jumps are not physical shock strengths or a final
Mach reconstruction.

## Checks and limits

- Full-state hashes and recorded source hashes match.
- Ten local crops agree with corresponding full-state cells exactly.
- Production mask intersections and all 40 source/region count rows agree.
- All 17 sets of online surfaces, histograms and slices agree with the preceding
  GPU run (floating tolerances 1e-6/1e-8; exact integer/string fields).
- Input hashes, one-cell sampler cross-check and conclusions are recorded in
  `verification.json`. Four focused review/probe tests pass.
- Figure inspected. No physical estimator, production grouping/tracking rule,
  source prescription or simulation state was modified.

The next controlled experiment should distinguish a developing/poorly separated
inner transition from detection incompleteness. Following the unchanged problem
slightly longer while monitoring source separation and boundaries is a direct
option. A threshold reduction, different clusterer or larger stencil is not
yet validated as a solution. Do not force a two-inner-history result.

Reproduce with a new output directory (requires full states on disk):

```sh
python -m experiments.wind_bubble.review_detection_audit \
  --output-dir outputs/two_bubbles_detection_gate_review_n128
```
