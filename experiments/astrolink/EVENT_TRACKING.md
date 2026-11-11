# Event-aware geometric tracker, first controlled comparison

## Goal and current scope

Track geometrically coherent shock-front structures through continuation,
combination and separation, rather than requiring a single inherited ID
through every transition. This stage implements actual event selection
and measures correspondence errors on stipulated synthetic catalogs.
It does not change the existing bubble results, shock finder or physics.

The formulation is informed by feature-event conflict graphs. Schnorr et
al., *Feature Tracking by Two-Step Optimization*, TVCG 26(6), 2219–2233,
DOI [10.1109/TVCG.2018.2883630](https://doi.org/10.1109/TVCG.2018.2883630),
describe one-to-one graph matching followed by optimization over competing
event explanations. Their [author-hosted abstract](https://www.graphics.rwth-aachen.de/publication/02168/)
was consulted. **Our implementation is an original one-stage prototype,
not a reproduction or empirical evaluation of their published method.**

## Fixed rules, before using this on the bubble sequence

1. A pair of samples can support a correspondence if within distance gate
   d and with signed normal angle at most 60 degrees. Negative group labels
   are excluded. Sample row identity is never propagated.
2. A continuation hypothesis must cover at least 50% of both groups.
3. A combination hypothesis has two or three parents and one child. At
   least 50% of every parent is covered, and their union covers at least
   50% of the child. Each parent must exclusively cover at least 10% of
   all child samples beyond support from the other parents.
4. Separation is the reversed combination rule: one parent, two or three
   children, with each child exclusively supporting at least 10% of the
   parent's samples. This avoids calling redundant nearby alternatives
   a separation or combination.
5. Choose a globally consistent set: a group can participate in at most
   one event in that adjacent-time comparison. Events can contain multiple
   groups; this restriction does not force one-to-one tracking.
6. Leave optimizer ties unresolved rather than choosing an arbitrary
   ancestry. Groups without an accepted event remain unmatched/ambiguous;
   these labels do not establish physical appearance or disappearance.

The 50% coverage threshold follows the earlier baseline. The 10% exclusive
fraction, distance weight below and arity cap are explicit new heuristic
choices, fixed across these tests. They have not been optimized or shown
appropriate for real shocks. The tiny-fragment stress case deliberately
exposes the exclusive-fraction limitation. Only distance sensitivity is
measured in this first comparison.

## Objective and solver

For each candidate event and participating group g, let c_g be its fraction
of samples covered by at least one compatible sample in the partner group
or union of partner groups. Let q_g be the mean nearest-compatible distance
of its covered samples, divided by d. The score is

    score(event) = sum_g c_g - 0.25 * sum_g (c_g * q_g).

Every group contributes separately, so complementary two-to-one or
one-to-two events can explain more supported groups than a partial
continuation. Among otherwise equally covering events, closer geometry
scores higher. This preference can be wrong for physical ancestry.
Neither coverage nor score is mass, area, energy or a probability.

Binary event variables maximize total score subject to at most one event
per parent or child group (weighted set packing). SciPy's mixed-integer
solver must finish to optimality; timeouts raise an error, not a silent
approximation. For each selected event, solve again forbidding it. If the
objective loss is at most 1e-8, the event is unresolved and not accepted.
This is a numerical tie test, not statistical confidence. Exclusion gaps
and ambiguous alternatives remain in the saved results.

This bounded prototype permits events with up to three partners, raises
an error above 5,000 generated hypotheses, and excludes many-to-many events,
multi-frame motion, gap recovery and physical propagation estimates.
It is not yet a scalable turbulent-box tracker or a temporal graph builder.

## Results

Compared against the unchanged pooled-support baseline at distances 0.10,
0.20 and 0.35 in toy units. These are synthetic point catalogs, not hydro
simulations; their truth labels are withheld from reconstruction and used
only for evaluation. No parameter is selected automatically from truth.

| Suite | Method | Exact configurations | False links | Missed stipulated links |
| --- | --- | ---: | ---: | ---: |
| Existing nine cases × three distances | Pooled baseline | 16/27 | 3 | 14 |
| Existing nine cases × three distances | Event prototype | 23/27 | 0 | 6 |
| Three added stress cases × three distances | Pooled baseline | 3/9 | 6 | 0 |
| Three added stress cases × three distances | Event prototype | 0/9 | 3 | 9 |

The prototype recognizes the existing representation split as a separation
and fragments forming one group as a combination at all three distances.
It preserves the correct approaching-aligned-sheet correspondences at the
wide distance, where the pooled baseline becomes ambiguous. In the existing
nearby-disappearing-sheet control it rejects redundant combination evidence
and selects the closer continuation, avoiding the previous false extra link.

Remaining six misses on existing cases occur when the correct displacement
lies outside the distance gate. This method has no motion prediction.

**Stress failures are part of the result:**

- *Same geometry, different ancestry:* uses precisely the same observable
  catalogs as the nearby-sheet case, but stipulates the other sheet as the
  survivor. The closest-distance preference chooses the wrong ancestry at
  every distance (three false and three missed links). No geometry-only
  deterministic rule can recover both contradictory truths from identical
  inputs. Additional evidence is required for a physical distinction.
- *Equal-distance alternatives:* the solver finds equivalent explanations.
  The prototype accepts neither, avoiding extra false links but missing the
  stipulated true one at every distance. This is deliberate abstention.
- *Tiny complementary fragment:* a 1/16 contribution is below the fixed
  10% exclusive threshold. The larger continuation is selected and the
  small stipulated link is missed at every distance.

The prototype improves the established controls but is not uniformly
better across the stress suite. These cases were designed during method
development, not a held-out statistical validation set. No general accuracy
claim or physical shock-merger validation follows from these numbers.

## Verification and next checkpoint

All 36 prototype configurations are independently checked using dense
pair distances and dot products: candidate event completeness, coverage
rules and scores. Exhaustive feasible-subset enumeration checks the optimal
objective and the events common to all near-optimal explanations. All 36
are also repeated after deterministic input-row shuffling. Saved metrics
are read back. Ten dedicated tests cover event types, three partners,
reverse separation, ambiguity, global conflicts, distance misses, empty
catalogs, invalid parameters and unchanged input arrays.

Outputs: `outputs/event_tracking_controls_v1/`, with original synthetic
inputs, per-case results, comparison and aggregate CSVs, a false/missed-link
figure and source hashes in `verification.json`.

```sh
python -m experiments.astrolink.check_event_tracking --output-dir outputs/event_tracking_controls_v1
python -m unittest tests.experiments.test_event_tracking tests.experiments.test_controlled_tracking tests.experiments.test_support_evidence
```

Requires NumPy, SciPy with `optimize.milp`, and Matplotlib. No GPU or new
hydrodynamic run. Use a new output directory.

Next, after reviewing these explicit tradeoffs, apply the fixed algorithm
to the existing two-bubble sequence and inspect its selected continuation,
combination and separation events. Keep the pooled baseline and old
candidate histories intact for comparison. Do not tune thresholds simply
to force a desired collision ancestry.
