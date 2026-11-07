# Controlled experiment: translation and isotropic expansion

## Why this step

Centroid translation cannot describe a symmetric bubble that grows about a
fixed center. This experiment adds an observed size-change estimate while
retaining the previous predictors as baselines. It uses synthetic catalogs
with supplied groups; no hydrodynamic simulation or production tracker changed.

## Model and information available

The first two catalogs at t0/t1 initialize motion using the existing unique,
nonredundant one-to-one geometric bootstrap (gate .20 in toy units). No truth
links, source locations, true centers, true radii or future positions initialize
the predictor. The future requested time t2 is known.

For each initialized group calculate its sample centroid c and RMS extent R:

    c = mean(sample positions)
    R = sqrt(mean(|sample position - c|²))
    v = (c1 - c0) / (t1 - t0)
    R_dot = (R1 - R0) / (t1 - t0)
    R_pred = R1 + R_dot * (t2 - t1)
    x_pred = c1 + v * (t2 - t1) + (R_pred / R1) * (x1 - c1)

R is a moment of the detected sample cloud. It is not generally a physical shock
radius. The prediction translates and scales the whole group uniformly; it does
not reconstruct local curvature, anisotropic growth, rotation or deformation.
Positive uniform scaling leaves the normal directions unchanged, so the observed
normals are reused. No future normals are used to construct the prediction.

Motion initialization failure retains the marked geometric fallback from the
previous experiment. Unresolved extent (R0 or R1 <= 1e-12) or a nonpositive
predicted extent retains the translation-only prediction and records the
fallback reason. It does not clamp the size or invent expansion. A fallback is
not a guarantee of a correct link.

Three final gates (.08/.20/.35) and the same matching rule are applied to all
three methods: geometry, centroid translation, and translation plus isotropic
expansion. Normal limit 60°, parent/pooled-child coverage thresholds .5 and
parent dominance margin .25 remain fixed. Gates are arbitrary toy distances,
not hydrodynamic grid widths. Actual nonuniform time intervals are used.

## Results

All ten earlier motion cases are retained. Three controls add a translating,
expanding sphere with 26 symmetric directions; slowing spherical expansion;
and growth along x only. Their normals are appropriate to the curved surfaces.
The inherited six-point symmetric-expansion control is a minimal point-cloud
case with its original constant normals, not a resolved physical sphere.

At the illustrative final gate .20:

| Case | Geometry | Translation | Translation + expansion |
| --- | --- | --- | --- |
| Fixed-center constant expansion | Missed | Missed | Correct stipulated link |
| Translating and expanding sphere | Missed | Missed | Correct stipulated link |
| Slowing expansion | Correct | Correct | Missed: overshoot |
| Changing sample coverage | Correct | Missed | Missed |

The joint translation/expansion control reproduces the constructed future
positions to floating-point precision. This establishes consistency for a case
that satisfies the model assumptions, not accuracy for arbitrary shocks.

Directional growth is not recovered exactly: the anisotropic control has
construction-paired position residuals about .10–.19 toy units after uniform
scaling. It is missed at gate .08, but its group-level link passes at .20 and .35
for all three methods. Thus an accepted group link does not establish accurate
surface prediction. Construction-paired positions are available only in these
toy inputs; the predictor itself does not receive point identities.

The other known problems remain explicit: balanced splits cannot be expressed
by the dominant-target rule; ambiguous bootstrap cannot supply motion; nearby
disappearance can generate a false link; and sample coverage changes can mimic
both translation and size change. Increasing model complexity has not solved
physical identity. Bubble expansion commonly slows or becomes nonspherical,
so the constant scalar extent-rate assumption needs assessment on saved data.

Do not choose a larger gate merely to hide prediction errors. The selected
synthetic cases do not supply an overall physical tracking accuracy estimate.

## Verification

Outputs: `outputs/controlled_expansion_prediction/`.

- Thirteen three-catalog input NPZ files with saved translation/expansion
  predictions, plus full results and summary CSV.
- `expansion_prediction_comparison.png`: false/missed stipulated links by
  method and final gate. Zero false links alone does not mean successful tracking.
- Provenance, independent readback and checksums.

All 117 configurations (13 × 3 methods × 3 gates) preserve the same group-level
results under independent row shuffling. Moment reductions agree within 1e-12;
small floating-point changes are recorded rather than called exact equality.
An independent dense-distance bootstrap and matching check reproduces all 117
final comparisons. An independent pair-distance identity for RMS extent checks
all thirteen moment predictions. Thirty-three focused tests pass, including
eight new tests for expansion recovery, future-information exclusion, known
failures, and explicit invalid-extent fallback.

No physical tracker, simulation catalog, estimator or previously accepted
bubble association was changed. Verification passing means the calculations
are consistent, not that every constructed correspondence succeeds.

## Reproduce and next decision

```bash
MPLCONFIGDIR=/tmp/astrolink-mpl \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.check_expansion_prediction \
  --output-dir outputs/controlled_expansion_prediction_new

/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.verify_expansion_prediction \
  outputs/controlled_expansion_prediction_new

/opt/anaconda3/envs/astrolink-env/bin/python -m unittest \
  tests.experiments.test_expansion_prediction \
  tests.experiments.test_motion_prediction \
  tests.experiments.test_controlled_tracking \
  tests.experiments.test_patch_support \
  tests.experiments.test_support_evidence
```

Use a new output directory. NumPy, SciPy and Matplotlib suffice; no GPU, JAX or
FuzzyCat execution is required.

The next useful step is an observational prediction audit on the saved 128³
bubble sequence: estimate from two earlier events, compare against the next
held-out event, and retain geometry as the baseline. Record bootstrap failures,
changes in sample coverage, residual distances and association changes without
promoting any result to physical identity. Such an audit can reveal whether
this assumption is useful on real evolving groups before modifying tracking.
