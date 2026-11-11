# Fixed event tracker on the saved two-bubble sequence

Applied the controlled prototype to all 16 adjacent intervals of the 17-event
128³ budget-matched run, at the original two association distances, 1.5/64
and 2.5/64 domain units. Original native grouping, signed-normal threshold
60 degrees and cutoff 20 are preserved. No scientific parameter was tuned.

## Result

The selected event sets agree across distances for **15/16 intervals**.
This includes the empty initial-parent comparison (and 14/15 comparisons
with detected parents). Agreement measures distance sensitivity only, not
physical correctness.

| Distance | Continuation events | Combination events | Separation events |
| --- | ---: | ---: | ---: |
| 1.5/64 | 32 | 1 | 0 |
| 2.5/64 | 32 | 3 | 0 |

Counts sum adjacent-time decisions, not unique physical shocks or identities.
No separation was selected; this does not establish that none physically
occurred. Unmatched catalog groups remain explicitly recorded.

### Reviewed transitions

- **t=0.07→0.08:** the algorithm automatically selects a combination of
  parent groups 0 and 1 into child group 0 at both distances. This reproduces
  the earlier candidate branch without hand-selecting its arrows. It is a
  detected-group combination, not confirmed physical shock merger ancestry.
- **t=0.13→0.14:** five continuations are selected at both distances,
  including the small patches. The previous matcher accepted five at the
  narrow distance but withheld two at the wide distance because coverage
  alone gave tied targets. Distance-aware global event selection now prefers
  the patch-to-patch correspondences. Real physical truth is unavailable.
- **t=0.14→0.15:** three large-group continuations are selected at both
  distances. Two small parents remain unmatched. Their exclusive child
  contributions are too small for the fixed 10% combination criterion;
  they must not be described as proven physical disappearances.

### Distance-sensitive t=0.10→0.11

The narrow distance selects only the outer continuation, leaving 14 parent
patches and two children unmatched. The wide distance adds two three-parent
combinations into inner child groups 1 and 14. Eight other parents remain
unmatched. There are 61 generated hypotheses at the wide distance.

The extra combinations are only slightly preferable to alternatives:
forbidding them lowers total objective by about **7.37e-8** and **3.76e-8**.
These gaps exceed the fixed numerical tie tolerance 1e-8, so the algorithm
selects them, but they are not strong physical evidence or probability
estimates. Tiny differences in geometry or parameters may change those
particular parent subsets. Report the interval as distance-sensitive and
weakly distinguished, not as validated ancestry. Event arity remains capped
at three, so this also does not explain every contributing fragment.

## Numerical issue discovered and corrected

An independent exact group-occupancy dynamic program found that the first
MILP result at the 61-hypothesis interval was below the true objective by
1.11e-7, despite the solver reporting optimal termination. That exceeded
the declared score-tie tolerance. The preliminary apparent 16/16 agreement
was therefore not accepted as the result.

The solver objective is now multiplied by 1e6 to reduce the impact of its
absolute numerical objective tolerance. The scientific scores, candidate
generation, coverage, exclusivity, distance penalty and tie threshold are
unchanged. Regression tests check selected score gaps above and below the
tie threshold. The complete controlled comparison was rerun separately:
all 72 baseline/prototype metric rows exactly match the original controls.
Original v1 control files remain intact; the application checks current
source hashes against `event_tracking_controls_solver_validation`.

This correction produces the final 15/16 agreement above. Objective scaling
is not a general exact-arithmetic guarantee: the independent checks below
are the evidence for these specific comparisons.

## Verification

- Independent sparse-distance-matrix construction checks coverage, nearest
  distance costs, exclusivity and scores of all **130 generated hypotheses**
  across the 32 comparisons. This differs from the tracker's per-point
  neighborhood-query implementation.
- Exact group-occupancy dynamic programming independently checks every
  optimum and selected-event exclusion gap, **32/32 comparisons**.
- Full subset enumeration additionally checks 31 small hypothesis sets;
  the 61-hypothesis case uses the independent dynamic program.
- Saved JSON is reopened; no selected events share a parent or child.
  Source catalog, partition, archive and control-report hashes are unchanged.
- **21 focused tests pass**, including the added solver-precision regression.

These are software/numerical checks. No independent physical ancestry labels
exist for this run. Added and removed pairs relative to the pooled baseline
are disagreements, not measured false or missed real-data links. This stage
does not assign persistent IDs, generate a complete multi-time graph, add
motion prediction, or validate arbitrary interacting shocks.

## Outputs and commands

`outputs/two_bubbles_event_tracking_v1_n128/` contains:

- `selected_event_review.png`: four inspected transitions, both distances.
- `summary.csv`: event and unmatched/ambiguous counts for every comparison.
- `selected_events.csv`: selected group sets, scores and exclusion gaps.
- `distance_agreement.csv`: event-set sensitivity by interval.
- `results.json.gz`: every hypothesis, selection and baseline pair difference.
- `verification.json`: checks, fixed parameters and source/input hashes.

```sh
python -m experiments.astrolink.check_event_tracking --output-dir outputs/event_tracking_controls_solver_validation
python -m experiments.wind_bubble.apply_event_tracking --output-dir outputs/two_bubbles_event_tracking_v1_n128
python -m unittest tests.experiments.test_event_tracking tests.experiments.test_controlled_tracking tests.experiments.test_support_evidence
```

Use new output directories. NumPy, SciPy with MILP and Matplotlib suffice;
no GPU or hydrodynamic rerun is needed. Recorded CPU times are diagnostic
tracker timings, not controlled performance benchmarks.

The next bounded deliverable is assembling these selected events into an
automatic multi-time group graph, with the weak distance-sensitive interval
kept explicit rather than selecting a preferred ancestry to hide it.
