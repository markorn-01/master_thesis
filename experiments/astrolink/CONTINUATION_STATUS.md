# Advisory continuation status: bounded checkpoint

`continuation_status.py` adds descriptions beside existing geometric links.
It does not change matching thresholds or acceptance, assign persistent
IDs, propagate identities, or infer physical shock mergers. It requires
`support_evidence` results so exclusive-support diagnostics are explicit.

## Status meanings

| Status | Condition | Interpretation |
| --- | --- | --- |
| `continuation_candidate` | Accepted target has one eligible parent, with nonredundant support | Simple local correspondence candidate; not validated physical identity |
| `ambiguous` | Parent has competing targets under the existing dominance rule | Do not force a selected identity |
| `multiple_parent_support` | Accepted child has more than one eligible parent | Preserve all support evidence and each parent's exclusive contribution; not proof of physical merger |
| `unmatched` | No accepted usable correspondence | No usable match under this rule; does not establish shock appearance/disappearance |

Parents with an eligible target but insufficient pooled child coverage are
unmatched. Child rows report accepted incoming parents; a child without an
accepted incoming link is unmatched even when an ambiguous parent has a
possible geometric association with it. Best target and actually selected
target are separate columns. A selected target can still fail pooled
acceptance, so `matcher_accepted` remains explicit.

Redundant small-parent links remain `multiple_parent_support`, with
`redundant_support=True` and zero exclusive child samples. They are not
silently deleted or promoted to independent ancestry. `distance_sensitive`
is a separate warning when a parent's target, advisory status or redundancy
changes across the compared distances. Coverage changes alone do not set
this warning. It applies only to the distances actually compared.

## Controlled checks

Evaluated nine existing synthetic cases at three distances: 27
configurations, each also checked under shuffled catalog row order.
Every matcher-accepted pair remains reported as accepted. Translation and
resampling can yield continuation candidates; the split example remains
ambiguous; complementary fragments yield multiple-parent support. The
known disappearing-nearby-sheet example still accepts a false extra link,
now explicitly accompanied by redundant multiple-parent support. The
counterexample is not solved, and statuses do not measure accuracy.

Ten focused status tests additionally cover unmatched distant motion,
empty parent catalogs, insufficient pooled child coverage, preservation
of matcher data and invalid/missing evidence. Together with the existing
support-evidence and controlled-tracking tests, **20 tests pass**.

## Real collision review

Applied to the same three snapshots t=0.13, 0.14, 0.15, cutoff 20, native
1.5-cell graph grouping, normal limit 60 degrees. Distances are 1.5/64 and
2.5/64 domain units. All 20 parent records reproduce the prior review:

- 10 continuation candidates, 2 ambiguous and 8 multiple-parent records;
- 8 records additionally warn of distance sensitivity;
- 16 child records describe accepted incoming support.

These counts include repeated observations at two distances, not unique
physical shocks. The prior review grouped distance-sensitive records first,
giving eight stable unique-local records, eight sensitive records and four
stable multiple-parent records. Both descriptions agree; the new table keeps
local status and distance sensitivity in separate columns.

The prior review independently checked point-neighborhood support. Its
input hashes are checked here, and accepted flags, exclusive contributions
and review classifications are reproduced. CSV readback checks labels and
flags. Source and input hashes are in `verification.json`.

## Outputs and reproduction

```sh
python -m experiments.wind_bubble.check_continuation_status --output-dir outputs/two_bubbles_continuation_status_n128
python -m unittest tests.experiments.test_continuation_status tests.experiments.test_support_evidence tests.experiments.test_controlled_tracking
```

CPU postprocessing with NumPy/SciPy; use a new output directory. Outputs:
`parent_status.csv`, `child_status.csv`, `control_summary.csv`,
`control_details.json` and `verification.json`.

**Stopping point:** the advisory table is implemented and checked. Review
these statuses alongside the existing collision figures before deciding
whether to assign any persistent IDs. No renderer changes, additional
algorithm search, higher resolution or new GPU run is part of this step.
