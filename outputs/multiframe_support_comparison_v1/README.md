# Completed bounded delayed-support comparison

The layer now runs automatically over complete saved sequences. It obtains
future continuation seeds from the existing tracker, assesses earlier
fragments with one-event latency, and preserves the accepted event graph.
It adds conditional geometric evidence, not inherited physical identities.
No detection, grouping, physical estimator or mesh was changed.

## Main real-data finding

| Transition | Accepted combination context | Conditional family | Union coverage, narrow / wide |
| --- | --- | --- | --- |
| Unequal two bubbles, 128³, 0.10→0.11 | Five of seven fragments | All seven | 92.48% / 100% |
| Three bubbles, 64³, 0.13→0.14 | Four of five fragments | All five | 89.53% / 100% |

The new support includes the two previously unlinked raw groups 1 and 7 in
the unequal case and raw group 9 in the three-source case. Both matching gates
support the same additional relationships. The table counts unique local
relationships; the CSV totals count each relationship once per gate. Group
numbers are event-local, not physical source or shock identities.

Static and normal-motion variants produce identical support memberships on
all 72 real interval/gate configurations. The added normal-motion model has
not demonstrated extra membership benefit on these hydro sequences. There
are no conflicts with accepted adjacent targets. Every original accepted
event is preserved exactly in both variants.

Sampling quality remains visible: unequal-case added group 7 has 24 upstream
source-overlap samples; three-case added group 9 has 16. Added geometric
support does not validate their Mach/heating measurements. Two other
three-source continuations remain distance-dependent: at the narrow gate,
their attempted single-parent families cover only 42.35% of their targets,
below the unchanged 50% requirement. No threshold was relaxed to include them.
The final transition in each sequence remains pending for delayed support
because its later observation does not exist; its baseline events are kept.

## Synthetic finding and interpretation

All 27 two-state regression configurations stay unchanged. In the 15
identifiable three-state development configurations, the following diagnostic
counts result if conditional support is hypothetically combined with baseline
links and compared with the stipulated truth:

| Diagnostic union | False links | Missed links |
| --- | ---: | ---: |
| Adjacent baseline | 0 | 30 |
| Static support plus baseline | 4 | 2 |
| Normal-motion support plus baseline | 0 | 2 |

The remaining two misses occur where the future displacement exceeds the
smallest gate, so no future continuation seed is accepted. No truth seed is
supplied to bypass this failure.

The six identical-observation configurations are reported separately. Both
support variants would introduce six false ancestry links if promoted into
the graph, despite covering the stipulated true links. Their observations
cannot distinguish the contradictory ancestries. Thus the diagnostic table
does not justify converting conditional support into accepted physical IDs.
These are known development controls, not held-out population accuracy.

## CPU cost

Median seconds from one untimed warmup and three paired repeats:

| Saved sequence | Adjacent matching | Added static layer | Added motion layer |
| --- | ---: | ---: | ---: |
| Unequal two bubbles, 128³, 21 states | 131.873 | 1.246 | 1.352 |
| Three bubbles, 64³, 17 states | 10.285 | 0.148 | 0.156 |

These are CPU postprocessing costs for complete histories. They are separate
from the earlier GPU hydro/online-analysis benchmark. One adjacent stage is
shared by the variants within each repeat; combined trial time is adjacent
plus the respective support stage. File I/O and independent audits are outside
the timers; method order alternates. There is no resolution-scaling claim
because the cases differ. The adjacent matcher remains the dominant cost.

## Checks and artifacts

The comparison covers 96 synthetic method/configuration results and 144 real
method/interval/gate results. All have audit records, including explicit
empty/protected/pending cases; nonempty families are independently recomputed
with blocked all-pairs distances. All 96 synthetic results retain categorical
decisions under row reversal. All 144 real accepted-event lists match their
baseline. Every timed adjacent computation reproduces the complete archived
result, and support repeats reproduce the audited results exactly.

Thirty-three focused tests pass. Saved result readback and 486 hash-entry
checks pass. The readable figure was visually reviewed. The float32-aware
independent oracle has a regression test; it does not relax the family rules.

- `support_comparison_readable.png`: preferred overview; short labels and
  pending terminal values masked. `support_comparison.png` is the original
  unedited experiment render, retained for provenance.
- `synthetic_comparison.csv`, `synthetic_summary.csv`: complete per-case and
  identifiability-separated metrics. `synthetic_results.json.gz`: full results.
- `real_comparison.csv`, `families.csv`, `support_links.csv`: every real window,
  candidate family, support relationship and source-overlap descriptor.
- `real_results.json.gz`: immutable adjacent events beside conditional support,
  conflicts and unresolved rows. No inherited IDs are added.
- `distance_comparison.csv`: method support-set sensitivity at the two gates.
- `runtime_trials.csv`, `runtime_summary.csv`: all paired timings and medians.
- `verification.json`, `plot_verification.json`, `qa.json`: provenance,
  readback and review checks.

The frozen experiment is complete. Retain this as a conditional support layer
beside the current event graph. No automatic parameter search or further GPU
run follows from this result. A useful next review is to display these three
newly supported fragment relationships alongside their original surfaces,
using a distinct visual style from accepted tracking events.

Implementation and reproduction:
`../../experiments/astrolink/DELAYED_SEQUENCE_SUPPORT.md`.
For the readable figure:

```sh
python -m experiments.astrolink.plot_delayed_support \
  outputs/multiframe_support_comparison_v1
```
