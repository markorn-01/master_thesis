# Five-fragment association: bounded offline arity experiment

## Result

Allowing one event to contain all five earlier weak-source fragments resolves
this geometric association at both unchanged matching distances.

| Gate | Maximum parents | Accepted weak-source parents | Full five child coverage | Full five exclusion gap |
| --- | ---: | --- | ---: | ---: |
| 1.5/64 | 3 | None (subset ambiguity) | Not representable | — |
| 1.5/64 | 5 | 2,3,4,5,6 | 99.1803% | 1.13143 |
| 2.5/64 | 3 | 3,4,6 only | Not representable | — |
| 2.5/64 | 5 | 2,3,4,5,6 | 100% | 1.10497 |

All five parents have 100% parent coverage at both gates. Exclusive fractions
of the later group are [14.7541,14.7541,14.7541,14.7541,30.3279]% at the narrow
gate, and [10.6557,10.6557,10.6557,10.6557,14.7541]% at the wide gate. Every
parent exceeds the unchanged 10% unique-contribution requirement. Coverage,
normal-angle, distance penalty, score definition, tie tolerance and global
conflict constraints are unchanged. No threshold was lowered to force links.
The objective gap is a model score margin, not a statistical probability.

## What was changed, and what was not

`experiments/astrolink/expanded_event_tracking.py` is a separate offline
experiment. It calls the frozen tracker for the original <=3-partner candidates,
adds four/five-partner hypotheses under the same criteria, and invokes the
same conflict optimizer. Both admitted and rejected high-arity hypotheses are
retained. Maximum arity and candidate/audit counts remain bounded (5 and 5000).
The symmetric separation case is supported and tested.

The existing `event_tracking.py`, the original history and all production
simulation/finder outputs are byte-unchanged. No physical shock estimator or
source prescription changes. `experimental_graph.json` is a new optional
history in which only the last interval uses the expanded candidate model.
Earlier segment IDs remain unchanged. At that final interval, the two previous
distance-specific choices become a shared five-parent geometric combination.
This does not establish five independent physical shocks merging.

## Controls and verification

- 9 established controlled cases at gates 0.2 and 0.35: baseline candidate lists
  and selections are unchanged when no high-arity event is possible.
- 6 additional cases at gates 0.04 and 0.07: five-way combination, five-way
  separation, coincident ambiguous parents, opposite-normal fragment,
  out-of-gate motion and a small unique fragment.
- Two caps per configuration give 60 controlled comparisons. No false links
  occur in this selected suite. Out-of-gate motion remains missed. The small
  unique fragment remains below the 10% rule, leaving one expected link missing.
  Coincident parents are left unassigned; the optimizer's ambiguity list need
  not enumerate every interchangeable parent. This is not a universal ancestry
  guarantee or broad tracking benchmark.
- All 60 controlled and 4 real optimizations pass independent occupancy DP and
  sparse geometry/score checks. Two additional dense cdist checks independently
  reproduce full-five coverage, unique fractions and score.
- The real cap-three candidate lists and solver selections match the committed
  old results exactly. Source/input hashes confirm old results remain unchanged.
- 29 unit/regression tests pass, including input preservation, ambiguity,
  direction filtering, invalid arity, five-way combination/separation and
  existing tracking/history/family tests.

## Artifacts

- `association_comparison.png`: actual old subset versus shared all-five event.
- `five_fragment_exclusivity.png`: each fragment's unique contribution versus
  the unchanged 10% criterion at both distances.
- `real_results.json`: complete candidates, rejected/admitted high-arity audit,
  solver alternatives and independent dense check for the final interval.
- `experimental_graph.json`: optional revised final event; original graph intact.
- `controlled_results.json`, `controlled_summary.csv`, `real_summary.csv`,
  `verification.json`: complete evidence and hashes.

## Limits and next use

This is a tested solution to a specific representation limitation in the
observed five-fragment transition. A cap of five remains a cap. Enumeration
can become expensive with many candidate groups; larger turbulent flows
require a scalability study, not unrestricted combinations. Fragment-count
contributions affect the score, and geometric support alone cannot distinguish
all possible material ancestries. Upstream source-overlap flags remain
unchanged (72 of the final 488 samples); association does not validate their
Mach or heating estimates. No complete general many-bubble tracking claim.

A reasonable next step is to use this explicitly labeled experimental graph
in the saved surface visualization and review the resulting history. Further
GPU simulation is not required for that step. A production/default change
should remain separate from this recorded offline comparison.

Reproduce in a new output directory:

```sh
python -m experiments.wind_bubble.check_five_fragment_events \
  --output-dir outputs/five_fragment_arity_review_n128
```
