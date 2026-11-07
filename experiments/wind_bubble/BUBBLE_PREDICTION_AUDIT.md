# Observational prediction audit on the saved 128³ two-bubble sequence

## Question

Do centroid translation or isotropic expansion improve prediction of the next
observed surface catalog? This applies the controlled predictors to the existing
17-event GPU run without changing simulations, grouping, accepted tracking
results, or physical identities. No new GPU run was needed.

## Procedure

For every three consecutive events, use only the first two to estimate motion
and extent change, then compare with the third. There are 15 such triplets,
with target times .02–.16. Each triplet is initialized independently: this is a
one-step prediction audit, not a recursively maintained tracker.

The saved graph uses radius 1.5 native cells and a 60° signed-normal limit.
Both minimum sample cutoffs are retained: 20 and 80. Bootstrap/final distance
gates are 1.5/64 and 2.5/64 code units (3 and 5 cells at 128³). Bootstrap uses
the same gate as final matching. No parameters are fitted to the future event.
Unique accepted, nonredundant one-to-one bootstrap links initialize the predictors.

For each triplet and setting compare:

1. **Geometry:** unchanged positions at the middle event.
2. **Translation:** extrapolate the observed group centroid.
3. **Expansion:** extrapolate centroid and scalar RMS extent, as defined in
   `../astrolink/EXPANSION_PREDICTION.md`.

There are 180 comparisons: 15 triplets × 2 cutoffs × 2 gates × 3 methods. Motion
initialization failure leaves the explicitly marked geometry fallback; invalid
extent prediction leaves translation. Reused normals remain at the middle
observation. Split/merge ambiguity and absence of physical truth remain open.

The target event is withheld from predictor construction. However, it belongs
to an already inspected simulation with previously chosen graph settings;
this is not an independently reserved statistical test dataset.

## Measurements and interpretation

For each retained middle-event sample, measure distance from its predicted
position to the nearest retained position at the target event. Save the median,
84th and 95th percentiles per group, in grid-cell units. The plotted statistic
is the median of these group medians: each retained group contributes one value,
not one value per sample. It can change sharply when small groups enter the
population. No persistent point-row correspondence is assumed.

Nearest-position distances ignore normal direction and physical shock identity.
A closer point may belong to another surface. Consequently they measure
geometric proximity, not true prediction error for identified physical shocks.
The reverse direction (target to predicted samples) is also saved, along with
forward sample-weighted summaries. These distances are not surface-area weighted.
Missing targets produce missing summaries with zero finite distances, not a
measured zero separation.

Candidate links are evaluated separately with the original signed-normal and
pooled-support rule (coverage thresholds .5, dominance margin .25). Save all
parent/child choices, ambiguity and redundant support. More accepted links
would not itself establish higher physical accuracy.

Size-excluded components do not enter either matching or the distance target
population; their counts remain recorded. Source-overlap samples remain included
and their per-group counts are saved as advisory information. Models operate
on detected sample-cloud moments, not prescribed source centers or source IDs.

## Findings

**There is no demonstrated improvement in accepted correspondence.** All
three methods produce exactly the same accepted link pairs in all 60
triplet/cutoff/gate combinations. This is stronger than merely matching link
counts. It does not establish that those shared links are physically correct.
The current geometric baseline should remain unchanged.

Prediction has mixed geometric effects. For example, at target t=.08 with
cutoff 80 and the narrower gate, the median of group-median nearest distances
decreases from about 1.532 cells for geometry to .428 cells for expansion.
Translation is about 1.540 cells. Expansion is helpful for geometric proximity
in this early example, not proven physical ancestry.

Later groups respond differently. At t=.11,.12→.13, cutoff 80 and the narrower
gate, local groups at the middle event .12 give:

| Middle-event local group | Samples | Geometry median distance (cells) | Expansion median distance (cells) |
| --- | ---: | ---: | ---: |
| 0, largest outer-surface candidate | 17152 | 1.301 | .519 |
| 1, inner candidate | 1108 | .124 | 1.045 |
| 12, inner candidate | 1108 | .124 | 1.045 |

Thus expansion improves the largest group's proximity while worsening the two
inner groups. Translation gives about 1.039 cells for each inner group. The
sample count for each inner group remains 1108 between .11 and .12: unchanged
count does not imply unchanged sample coverage or centroid. Their observed
centroid shifts yield opposite estimated x velocities of about ±2.046 code
units per unit time. These are sample-centroid rates, not independently
measured physical shock propagation velocities. This audit does not isolate
the cause of the shifts.

The final matching gates span 3 or 5 grid cells, so changes in sub-gate distance
need not change accepted links. The wider gate also retains the previously
reported ambiguous/redundant small-fragment cases. No settings are retuned to
make prediction appear successful.

Near appearance or changes in grouping, initialization is limited. For instance,
the cutoff-80 inner groups at middle time .11 lack unique accepted parents from
.10, so no velocities are inferred for them. At .15→.16 with cutoff 20, only
the largest group initializes uniquely from .14; the two inner groups have
fragment-pooling predecessors and use the marked geometric fallback. A clean
plot does not imply every group has an available motion model.

## Decision and remaining scope

Keep geometry as the baseline and keep both predictors experimental. The toy
success of expansion does not establish an overall improvement on this real
sequence. Better geometric proximity for some surfaces is insufficient grounds
to promote links to persistent physical IDs or select a new production rule.

Before changing the tracker, investigate how late inner-group sample coverage
and centroids change, and retain explicit uncertainty for appearance, splitting,
pooling and disappearance. Smooth mesh visualization can proceed as a separate
display task without altering the measurements or claiming solved tracking.
No additional simulation resolution is justified by this audit alone.

## Outputs and verification

`outputs/two_bubbles_prediction_audit_n128/` contains:

- `prediction_audit_gate_1p5.png`, `prediction_audit_gate_2p5.png`: proximity and
  candidate-link histories. Link curves overlap because the link sets agree.
- `summary.csv`, `group_diagnostics.csv`: group counts, initialized/fallback
  counts, exclusion counts, source-overlap counts, sample-count ratios and
  per-group distances.
- `results.json.gz`: bootstrap links, model parameters, final candidate decisions,
  forward/reverse distance summaries. Dense predicted grids are unnecessary:
  saved model parameters and original input positions reconstruct the prediction.
- `metadata.json`, `readback.json`, `SHA256SUMS`.

All 60 original geometry baselines and bootstrap records are reproduced. All
180 saved predictions and distance summaries are checked, including CSV readback.
For 36 selected configurations covering the contact stage and later inner
surfaces, an independent per-point query verifies parent decisions, child
unions and exclusive support. Another 180 brute-force nearest-distance checks
agree with the KD-tree calculation. Original input hashes remain unchanged.
Three focused tests check empty/missing populations and directional distances.

These checks establish computational consistency, not physical tracking accuracy.
The source-budget matching scope, finite resolution, surface sampling and missing
identity truth remain the limitations described in the earlier resolution report.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/astrolink-mpl \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.wind_bubble.audit_bubble_predictions \
  outputs/two_bubbles_temporal_comparison_n064_n128 \
  --output-dir outputs/two_bubbles_prediction_audit_n128_new

/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.wind_bubble.verify_bubble_predictions \
  outputs/two_bubbles_prediction_audit_n128_new

/opt/anaconda3/envs/astrolink-env/bin/python -m unittest \
  tests.experiments.test_bubble_prediction_audit
```

Use a new output directory. NumPy, SciPy and Matplotlib suffice; no GPU, JAX or
FuzzyCat execution is required.
