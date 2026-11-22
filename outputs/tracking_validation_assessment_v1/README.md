# Tracking and visualization: consolidated validation assessment

**Step 2 is complete.** This assessment consolidates existing results. It adds
no simulation, optimizer run, new association rule or threshold change. The
bounded evidence audit verifies 11 saved reports and 721 unique input/code/
artifact paths, and independently reaggregates the saved comparison tables.
See `assessment.json` for the extracted results and `verification.json` for
the checks and provenance.

## Main conclusion

We have an implemented and execution-validated workflow for online geometric
shock-surface grouping, continuation/event histories, original-sample statistics
and synchronized checked surface visualization on controlled two- and
three-bubble cases. The evidence supports reproducible geometric association
and explicit uncertainty. It does not establish unique physical ancestry for
every collision fragment or general turbulent-box accuracy/scalability.

Three different questions must stay separate:

| Question | Evidence | Assessment |
| --- | --- | --- |
| Does moving analysis/tracking online change the tested results? | Matching-time GPU online/offline checks and batch comparisons. | Passed for the tested runs. |
| Do the geometric association rules behave as expected on controls? | Constructed cases with stipulated truth, independent geometry/optimizer checks, and row-order checks. | Useful but incomplete performance; identifiable and indistinguishable cases must be reported separately. |
| Are all hydro associations physically correct? | Real simulation surfaces have no independent physical ancestry labels. | A hydro false-link rate and complete physical ancestry recovery are not established. |

This assessment concerns tracking and display. It does not replace the separate
physical validation of shock detection, Mach estimation or thermalization.

## 1. Online execution is validated on the selected cases

| GPU callback case | Events | Group observations | Complete tracking records vs offline |
| --- | ---: | ---: | --- |
| Two equal sources, 32 cubed | 17 | 25 | Exact |
| Two equal sources, 64 cubed | 17 | 39 | Exact |
| Three equal sources, 64 cubed | 17 | 45 | Exact |

All three runs report agreement for 21 scalar-history fields per event, with
maximum finite scalar difference zero, exact histogram counts, matching sparse
catalogs/slices, batch tracking agreement and final gas-state agreement at the
declared tolerances. Full gas states were not retained in these copied output
folders for another local final-state comparison; that check is the GPU
runner's recorded result.

The incremental component retains at most two geometry catalogs between
events, uses at most three while processing a window, and releases them on
close. This is a bound on recent geometry, not constant total process memory.
Surface populations, Python hypotheses and temporary arrays can grow.

The earlier saved-data replay also preserves the complete unequal two-bubble
128-cubed history: 21 events and 65 group observations. That is saved-catalog
replay validation, not a new 128-cubed callback simulation.

Evidence: the three `online_streaming...`/`three_bubbles_online_tracking...`
run metadata and equivalence files listed in `assessment.json`, and
[`streaming_history_replay_v1`](../streaming_history_replay_v1/README.md).

## 2. Known-answer controls reveal benefits and failure modes

The fixed experiment contains 48 distinct case/gate configurations: 27
two-state regressions, 15 identifiable three-state development configurations,
and six three-state configurations whose observations cannot distinguish the
stipulated histories. Applying two support variants produces 96 result records.
These are known development/regression cases, not a blinded held-out sample.

Both support variants preserve all 27 two-state regression configurations.
The conservative baseline retains six missed links on those controls; added
support does not resolve them.

For the 15 identifiable three-state development configurations:

| Diagnostic link set | False links, summed | Missed links, summed |
| --- | ---: | ---: |
| Accepted adjacent baseline | 0 | 30 |
| Baseline plus static conditional support | 4 | 2 |
| Baseline plus normal-motion conditional support | 0 | 2 |

**The last two rows are hypothetical unions used for evaluation.** The real
implementation does not promote conditional links into accepted ancestry or
inherited IDs. Counts sum links across chosen cases and gates; they are not
independent physical shocks or population accuracy percentages.

Normal-motion support performs better than static support on these identifiable
controls. The remaining two misses occur when the smallest gate fails to admit
the needed future continuation seed. No truth seed is injected to bypass it.

In the six indistinguishable-observation configurations, either support variant
would add six false ancestry links if its conditional evidence were treated as
confirmed truth. Two different stipulated histories can have the same observed
geometry. Those observations alone cannot select the correct history in both
cases. This is why conditional evidence remains a separate channel.

Evidence: [`multiframe_support_comparison_v1`](../multiframe_support_comparison_v1/README.md),
particularly `synthetic_comparison.csv` and `synthetic_summary.csv`. The audit
reaggregates the detailed rows and checks the archived summary.

## 3. Real collision cases support partial geometric histories

After the explicit maximum-five-partner extension, event-set agreement between
the two fixed association gates is:

| Saved case | Intervals agreeing at both gates |
| --- | ---: |
| Equal two bubbles, 128 cubed | 16 / 16 |
| Unequal two bubbles, 128 cubed | 20 / 20 |
| Three bubbles, 64 cubed | 15 / 16 |

These are parameter-agreement results. They do not establish physical correctness.
The gates remain 1.5/64 and 2.5/64 in domain units, not automatically adjusted
for resolution or analysis cadence.

Two fragment-rich transitions illustrate the remaining distinction:

| Transition | Accepted combination | Family with added conditional support | Later-group coverage, narrow / wide |
| --- | --- | --- | --- |
| Unequal two bubbles, t = 0.10 to 0.11 | Five of seven fragments | All seven | 92.48% / 100% |
| Three bubbles, t = 0.13 to 0.14 | Four of five fragments | All five | 89.53% / 100% |

The unequal case gains conditional relationships for two previously unlinked
fragments; the three-bubble case gains one. These are three unique local added
relationships. Method/gate CSV rows repeat their evaluation and must not be
counted as additional physical relationships.

Static and normal-motion variants have identical support memberships across all
72 real interval/gate configurations. Motion prediction has not demonstrated an
additional membership benefit on these hydro sequences. All 144 support results
preserve the accepted adjacent events and report zero target conflicts.

The five-partner cap is not the only reason a full ancestry is missing. In the
earlier unequal seven-fragment transition, some parents contribute too little
exclusive support under the fixed 10% rule. Raising the cap to seven alone
would not admit the complete family. Family support and uniquely contributing
parents are different geometric criteria.

Two other three-bubble continuations remain distance-specific. Their attempted
single-parent families cover 42.35% at the narrow gate, below the fixed 50%
requirement. They remain dotted alternatives; the threshold was not lowered
to make the diagram look complete.

Evidence: [`expanded_full_sequence_validation_n128`](../expanded_full_sequence_validation_n128/README.md),
[`equal_history_mode_comparison_n128`](../equal_history_mode_comparison_n128/README.md),
[`three_bubble_history_expanded_n064`](../three_bubble_history_expanded_n064/README.md),
and the delayed-support comparison's `families.csv`, `support_links.csv` and
full result archive.

## 4. Display consistency and causal timing are checked

The completed callback-driven three-bubble movie has 17 decoded frames, 45
group observations and 15 continuation segments. Surface colors, table rows,
original sample coordinates and graph identities are joined and checked.
Every populated event in the final mesh sequence passes the fixed geometry
and own-group agreement checks. The lowest group fraction within 0.25 cell
of its own mesh is 95.24%, above the fixed 95% target.

The additional G08-to-G13 support refers to the t = 0.13 to 0.14 transition,
but first appears when event 15 supplies C. It does not assign an inherited
identity or join group-statistic curves. The last interval remains pending
for delayed support because no later observation exists.

Smoothness and small sample-to-mesh distances do not prove physical topology.
The controlled same-group close-sheet counterexample exposes invented
intermediate surfaces in the tested reconstruction families. Preserve this
negative result alongside the successful checked display sequence. Original
sample statistics and physical normals remain the analysis products.

Evidence: [`three_bubble_online_tracking_demo_n064`](../three_bubble_online_tracking_demo_n064/README.md),
[`three_bubble_adaptive_sequence_final_n064`](../three_bubble_adaptive_sequence_final_n064/README.md),
and [`three_bubble_surface_method_research_n064`](../three_bubble_surface_method_research_n064/README.md).

## 5. Physical-state quality and computational scope remain explicit

The unequal-case added fragment group 7 contains 24 upstream sampling positions
inside a prescribed injection sphere. The three-case added fragment group 9
contains 16. Improving a geometric association does not validate the sampled
Mach or thermalization measurement in such regions. These source flags are
retained as quality descriptors, not used as ancestry truth.

Detection, spatial grouping, the minimum size of 20 samples, temporal cadence
and resolution influence which groups exist. A graph's first appearance or
disappearance is not automatically a physical shock birth or death. Some
observations remain unretained, unmatched, distance-sensitive or conditional.

The matcher uses bounded partner enumeration and conflict optimization. Existing
synthetic/real checks exercise that bounded model; general many-bubble scaling
is unestablished. Prior paired CPU matching timings and the older GPU shock-
analysis benchmark do not measure the complete newly integrated workflow.
Step 3 must measure the full cost with tracking enabled and disabled.

## Supported thesis claim and fixed next step

> The implementation performs online geometric grouping and temporal association
> of detected shock-surface samples, preserves explicit continuation and
> combination/separation relationships, and provides synchronized original-sample
> statistics and checked surface visualization. Execution equivalence and bounded
> association behavior are demonstrated on the specified controlled cases;
> ambiguous fragment ancestry remains explicit.

Do not claim unique physical ownership/ancestry, zero hydro false-link rate,
complete tracking of every fragment, universally correct surface topology,
general convergence, or efficient turbulent-box scaling from this evidence.

The existing method and thresholds stay frozen. No automatic search for another
algorithm follows from this assessment. Step 3 is the remaining bounded technical
work item: measure complete-workflow runtime/memory cost. Broader science claims
would need separately scoped evidence, not a reinterpretation of these checks.

## Reproduce this evidence audit

From the repository with the NumPy/SciPy analysis environment, use a fresh
destination:

```sh
python -m experiments.wind_bubble.assess_tracking_validation \
  --output-dir outputs/tracking_validation_assessment_v1_repeat
```

This reads and hashes saved evidence, reaggregates tables and validates the
stated numerical summaries. It runs no simulation or optimizer, changes no
archived result and does not rerun the earlier unit suites. The 65-test result
belongs to the Step 1 implementation checkpoint.
