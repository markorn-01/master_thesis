# Completed bounded review of the collision-layer detection gap

## Conclusion for the tested simulations

The central density ridge is not established as one missed, resolved shock.
At both 64^3 and 128^3 it remains a dense layer cooler than its hot surroundings,
with bulk contrasts that do not form a clean single-shock thermodynamic jump.
The production gap originates in shock-zone detection, before grouping,
tracking or smooth-surface rendering.

The independent tests did not establish a missed above-threshold front in
the checked central gap. This is a conclusion about the inspected runs and
sampling choices, not a proof that no physical shock can exist inside an
unresolved structure, or that the detector is complete in general.

## Verification

The 128^3 simulation completed, and its audit was recovered from the saved
states after the wrapper argument error. It was not reintegrated. The copied
state hashes, saved primary-product hashes, reference-input hashes, and the
recorded postprocessing source hashes all match. The intentional wrapper
repair is documented in the run's audit report; no production physics changed.

The prescribed source budgets, physical injection radius and source positions,
event schedule and timestep cap match the 64^3 case. All five captured times
were checked. Two grid resolutions do not establish convergence.

At the later captured times, the central regions have the following masks:

| Time | 64^3 central-slice zone cells | 128^3 central-slice zone cells | 64^3 central-volume zone cells | 128^3 central-volume zone cells |
| --- | --- | --- | --- | --- |
| 0.06 | 0 | 0 | 56 | 360 |
| 0.08 | 0 | 0 | 0 | 0 |
| 0.12 | 0 | 0 | 0 | 0 |
| 0.16 | 0 | 0 | 0 | 0 |

Counts refer to sampled grid centers in a fixed physical box; they are not
counts of physical shocks. At t=0.06 there are still detections at other depths.
At t>=0.08 the empty central-volume mask shows that the late gap is not merely
a choice of slice. The pressure-jump condition fails throughout that volume
at these later captured times. Alignment also fails throughout at t=0.08 and
0.16; some cells pass alignment at t=0.12 but still fail pressure jump.

## Where the apparently compatible pairs occur

Every post-contact pair passing the 20% diagnostic compatibility level in
the wider axial or pressure-normal scans was located: 124 pairs in total.
The 5%, 10% and 20% levels are sensitivity checks on state-pair consistency,
not physical detection thresholds or verified counts of shocks.

1. Ten axial pairs pass the 10% level at t=0.16. They connect cells at
   x=0.43359375 and 0.44140625, or their reflected counterparts. These are
   near the wind sources, outside the original central x interval 0.45–0.55.
   Every proposed downstream cell is already present in the exported shock
   catalog. These cases are not evidence of a missing central front.
2. Eight pressure-directed pairs pass the 10% level at t=0.08. Their centers
   are at x=0.45703125 or 0.54296875, y=0.35546875 or 0.39453125, and
   z=0.42578125 or 0.57421875. They are away from the saved central plane.
   Their sampled P2/P1 is about 1.256, corresponding to a pressure-based
   single-shock reference Mach estimate of about 1.098. The estimate inferred
   from the mass jump is about 1.022. Both are below the configured Mach floor
   1.3. These marginal fits pass only one of the tested sampling half-widths
   (2 cells), so they do not independently identify an isolated physical shock.
3. The looser 20% level was checked as well, because the outer-forward-front
   comparison often needs that tolerance at these resolutions. Of the 124
   pairs, 54 have sampled pressure jumps reaching the Mach-1.3 reference.
   **All 54 are spatially covered by the existing detector:** 46 axial pairs
   contain an exported surface grid cell on the same sampled segment, and
   8 pressure-ray pairs at t=0.06 have an exported cell at the exact ray center.

This avoids selecting only the tighter fits while ignoring the control's
limited accuracy. Spatial coverage does not identify a unique physical
shock object or establish complete recovery of all possible embedded fronts.

The pressure-based Mach values here are diagnostic estimates under a
single-shock assumption, not independently measured physical Mach numbers.
Short samples can contain only part of a broadened transition; a small
sampled jump does not prove a complete physical shock is weak or absent.

## Figures and data

- [Same physical axis at both resolutions](matched_axis_profiles.png):
  density, pressure and P/rho at t=0.08 and 0.16, sampled at identical physical
  y=0.375 and z=0.5 by multilinear interpolation. Gray shading marks the
  checked central x interval. The ridge remains relatively cool at both grids.
- [Candidate locations](candidate_locations.png): representative locations
  of the tightest marginal fits, with the actual exported samples. Each
  overlay is a half-cell-thick slab, not a full-depth projection. White boxes
  mark the original audit region; cyan points are exported detections and
  green crosses are the reviewed sample-pair centers.
- `candidate_review.csv`: all 124 reviewed post-contact pairs, exact export
  membership/segment coverage, mask values, and inferred jump estimates.
- `common_axis_profiles.csv`: the numerical values used for the common-axis figure.
- `review.json`: verification, scope, summary and immutable input hashes.

All reviewed state-pair metrics were recomputed from the captured primitive
states and matched the supplied tables. Exact grid-index membership was used
to establish detector coverage; nearest-neighbor distance alone was not used
as proof that a candidate was detected. All input hashes were rechecked after
the review, and both final figures were visually inspected. No new simulation,
production estimator or threshold change was needed for this review.

## Suggested explanation for Tobias

> I repeated the case at 128^3 with the same prescribed source budgets and
> inspected the gas profiles and detector masks. The bright central ridge
> remains dense but cooler than the surrounding gas; it does not show a clean
> single-shock jump. The finder rejects it through the temperature–density
> alignment and directed pressure-jump conditions, before grouping or smoothing.
> I also checked the marginal shock-like comparisons independently: those
> reaching our reference pressure-jump threshold are already covered by detected
> surfaces elsewhere around the layer. I have therefore not found a verified
> missed above-threshold front in the central gap in these two runs.

The bounded check is complete. It supports reporting this result without
forcing the dense ridge into the shock catalog. It does not justify calling
the entire layer a pure contact, proving physical shock absence, or claiming
physical convergence or general shock-finder completeness.
