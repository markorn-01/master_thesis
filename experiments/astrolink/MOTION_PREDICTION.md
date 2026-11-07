# Controlled experiment: motion prediction from observed surface groups

## Purpose

Test whether motion estimated from earlier observations improves the geometric
baseline. This is an isolated synthetic experiment, not integration into the
bubble tracker. It uses supplied groups, not clustering. No truth links, true
velocities or future positions are supplied to the prediction function.

## Model

Three catalogs are available at strictly increasing times t0, t1, t2. Matching
between t0 and t1 uses the existing geometric rule with distance gate .20 in toy
units. Only unique accepted one-to-one links with nonredundant support initialize
a velocity. Such geometric links are candidates, not guaranteed physical truth.

For an initialized group, use its observed sample centroids:

    v_est = (centroid(t1) - centroid(t0)) / (t1 - t0)
    predicted_position = position(t1) + v_est * (t2 - t1)

The whole group is translated with that velocity; normals remain unchanged.
Times can be unequal: most cases have t0=0, t1=.2, t2=1.2. This tests correct
use of the measured time intervals rather than assuming uniform cadence.

Uninitialized groups keep their observed positions and are marked as geometric
fallback, with velocity recorded as null. They are not silently assigned a
measured velocity of zero. There is no future-frame fitting, truth-based
initialization, rotation, deformation, split logic or gap bridging.

Both methods then use the same pooled rule at the same final distance gate,
.08, .20 or .35. The bootstrap gate stays fixed at .20. Normal limit 60°, coverage
thresholds .5 and dominance margin .25 remain unchanged. Geometry has no motion
prediction; its gate measures separation from observed positions. With prediction,
the same gate measures separation from extrapolated positions. Toy units have
no numerical equivalence to the hydrodynamic grid.

## Ten constructed cases and findings

- **Constant translation:** observation-derived motion recovers a sheet that
  moves .30 during the final interval, beyond the .20 geometric search gate.
- **Passing sheets:** nearest-position geometry links each parent to the wrong
  sheet at the .20 gate. Prediction recovers the stipulated two correspondences.
  These sheets passing through each other are a constructed ancestry test,
  not a hydrodynamic collision solution.
- **Approaching sheets:** prediction works at residual gate .08, but .20/.35
  again admit both nearby targets and produce ambiguity. Motion does not
  remove sensitivity to the residual search scale.
- **Abrupt stop:** the sheet stops after t1. Geometry succeeds at .20; prediction
  overshoots and misses. A wider .35 gate recovers it in this toy case.
- **Catalog split:** a balanced one-to-two split remains ambiguous. Translating
  a parent cannot fix the rule's single-dominant-target restriction.
- **Fragment pooling:** two stationary fragments linking to one catalog group
  succeeds with either method. This is representation change, not proof of a
  physical merger.
- **Nearby disappearance:** close aligned sheets already make bootstrap
  ambiguous, so neither obtains a velocity. The fallback still falsely attaches
  the disappearing sheet to the survivor at .20/.35. No truth initialization is
  used to make this case look successful.
- **Changing sample coverage:** a stationary sheet is only half sampled at t1.
  Its centroid shifts, even though the physical surface does not translate.
  The predictor invents motion and misses the next group at all tested gates,
  while geometry succeeds. Centroid displacement is not necessarily shock speed.
- **No initialization:** the first displacement already exceeds the bootstrap
  gate. Neither velocity nor ancestry is inferred from truth; geometric fallback
  remains explicit. Both methods miss at .08/.20 and recover at .35.
- **Symmetric expansion:** six symmetric samples move radially outward about
  a fixed center. The centroid stays fixed, so translation predicts no expansion
  and fails at .08/.20, just like geometry. This is a minimal geometric control,
  not a resolved sphere or physical wind-bubble solution.

The expansion result is particularly relevant to the intended application:
closed wind-bubble shocks can grow while their centers barely move. Whole-group
centroid translation therefore cannot be adopted as the general motion model.
It can also be corrupted by fragmentation or changes in sampling coverage.

These selected cases do not provide a measured accuracy rate for turbulent
hydrodynamics. They justify retaining an experimental baseline and developing
an expansion-aware prediction test before using motion in the real bubble
associations. Split/disappearance uncertainty remains a separate requirement.
No currently accepted simulation links or physical shock identities changed.

## Verification and outputs

`outputs/controlled_motion_prediction/` contains ten three-catalog NPZ inputs,
saved predicted positions, full group-level results, summary CSV, the comparison
figure, provenance, readback and checksums. All 60 configurations (10 cases ×
3 final gates × 2 methods) preserve group-level results under independent row
permutations. Truth is used only to count false/missed stipulated links.

An independent dense distance/dot-product calculation checks all ten bootstrap
initializations and all 60 final matches, and reproduces saved velocities and
predicted positions. Twenty-five focused software tests pass, including eight
new tests. These explicitly retain the known motion failures and check that
changing truth labels or future positions cannot change the prediction.
Passing verification means the experiment was calculated correctly, not that
prediction succeeds on every case or physical tracking has been validated.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/astrolink-mpl \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.check_motion_prediction \
  --output-dir outputs/controlled_motion_prediction_new

/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.astrolink.verify_motion_prediction \
  outputs/controlled_motion_prediction_new

/opt/anaconda3/envs/astrolink-env/bin/python -m unittest \
  tests.experiments.test_motion_prediction \
  tests.experiments.test_controlled_tracking \
  tests.experiments.test_patch_support \
  tests.experiments.test_support_evidence
```

Use a new directory. NumPy, SciPy and Matplotlib suffice; no GPU, JAX or FuzzyCat
execution is required. Earlier controlled benchmarks and animation files are
separate changes and have not been overwritten.
