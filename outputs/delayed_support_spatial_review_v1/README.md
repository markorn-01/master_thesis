# See what the delayed-support layer added

Start with `unequal_two_n128_delayed_support.png`. Then open
`three_n064_delayed_support.png`.

## Read either figure from left to right

1. **A: earlier fragments.** Teal samples belong to fragments already in the
   accepted combination. Orange/magenta samples are the additional fragments
   supported by the delayed layer. Every plotted sample is an original
   detected position, not a new fitted surface vertex.
2. **B: later detected group.** The question is whether the highlighted earlier
   fragments support this observed surface. Blue indicates its role in this
   review, not material ownership or an inherited identity.
3. **C: extra observation.** The original tracker already accepts a continuation
   from B to C at both matching distances. This supplies the later evidence
   used by the delayed layer. It does not prove each A fragment's ancestry.
4. **Bottom diagram.** Solid links are existing accepted geometric decisions.
   Dashed orange/magenta links are additional conditional support. They are
   deliberately separate from the accepted graph. The diamond joins the
   parents of the existing combination; it is not an additional detection.

Group numbers are local to each event. A:g1 and B:g1 are different observations;
their repeated number alone does not establish identity. Point counts are
samples at each time, not conserved particles. Colors show evidence roles.

## The two concrete changes

| Case | Existing combination | Added conditional fragments | Resulting support family |
| --- | --- | --- | --- |
| Unequal two bubbles, 0.10→0.11, evidence at 0.12 | 5 fragments | A:g1 (32 points), A:g7 (60 points) | 7 fragments |
| Three bubbles, 0.13→0.14, evidence at 0.15 | 4 fragments | A:g9 (20 points) | 5 fragments |

Both tested methods and both matching distances support these memberships.
The narrow/wide geometric union coverage is 92.48%/100% for the unequal case
and 89.53%/100% for the three-source case. These are sample-coverage fractions,
not confidence probabilities, physical surface completeness or ancestry truth.

The extra fragments were already detected. This step creates pictures of
their additional relationships; it does not create new shock detections,
change the algorithm or promote the dashed links into accepted identities.

## Sampling quality and display scope

The three upper panels use the same local bounds and camera for each case.
All family samples are shown. Nearby nonfamily samples inside the box are
gray; distant fronts outside the local zoom are not shown. Black stars are
wind sources. There is no smoothing, point movement or invented point path.

The upstream source-overlap count is the number of family samples whose
upstream sampling endpoint lies inside an injection region. It is not the
number of surface positions inside a source. Counts are 112/8/0 over A/B/C
for the unequal case, and 42/38/22 for the three-source case. Of the added
fragments, A:g7 has 24 flags and A:g9 has 16; A:g1 has zero. Those flags remain
visible because geometric support does not validate Mach/heating sampling.

## Verification and reproduction

Six catalog joins match original row indices, coordinates, group labels and
saved populations exactly. Every family is checked against both variants and
both gates; its B→C seed must be a saved shared continuation. Source-overlap
counts reproduce the original group statistics. Both figures were visually
reviewed. Four new review tests and twelve delayed-support tests pass (16
total). All 247 recorded hash entries pass.

`review.json` records the six observations, bounds, camera, roles and coverage.
The two `*_original_points.npz` files retain exact family positions and source
catalog row indices for independent readback. `verification.json` and
`qa.json` record provenance and checks. The accepted graphs and the earlier
smooth-mesh visualizations are unchanged.

```sh
python -m experiments.wind_bubble.review_delayed_support \
  --output-dir outputs/delayed_support_spatial_review_v1

python -m unittest tests.experiments.test_delayed_support_review \
  tests.experiments.test_delayed_support
```

Use a fresh output directory. This is a completed visual checkpoint on saved
data; it needs no GPU simulation or new tracking fit.
