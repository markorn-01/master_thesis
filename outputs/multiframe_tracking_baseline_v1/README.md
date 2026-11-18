# Frozen baseline for the next multi-frame tracking experiment

This checkpoint records what the existing maximum-five adjacent tracker does
before integrating delayed fragment support. It does not implement a new
tracker, rerun hydrodynamics, change grouping or modify earlier results.

## Fixed inputs and result

| Existing synthetic suite | Configurations | Exact | False links | Missed links |
| --- | ---: | ---: | ---: | ---: |
| Adjacent regression | 27 | 23 | 0 | 6 |
| Three-state development | 21 | 6 | 0 | 36 |

These are selected, previously used development cases at three gates each,
not a held-out accuracy estimate. They reproduce known limitations. The
baseline avoids false links here but misses many stipulated relationships.
Truth labels are separate from the observations and never enter matching.

The real comparison includes the full unequal two-bubble 128³ sequence
(21 events) and three-bubble 64³ sequence (17 events). Their 72 saved
interval/distance results are referenced without recomputation or duplicate
simulation files. Both graphs reconstruct exactly from those selections.
There is no real-data ancestry truth, so hydro false/missed-link rates are
not reported. All intervals are included, with difficult transitions flagged
for review and final transitions explicitly lacking a future observation.

## What to open

- `protocol.json`: frozen methods, thresholds, evidence channels, measurements
  and stopping rule for the next implementation.
- `synthetic_baseline.csv`: per-case errors and unresolved group counts.
- `synthetic_cases.json` and sixteen NPZ files: case meanings, evaluation-only
  truth and exact observations. Expected synthetic future continuations are
  evaluation-only; the next method must derive its own seeds.
- `synthetic_baseline.json.gz`: full baseline hypotheses and decisions.
- `real_cases.json` and `real_baseline.csv`: sequence references and current
  results, without inventing real-data truth scores.
- `verification.json` and `qa.json`: provenance and readback checks.

All 48 synthetic configurations pass independent geometry and exact
occupancy-DP audits and reversed-row link checks. Readback reproduces all
48 summary rows and sixteen input catalogs. All 228 recorded hashes pass.
Twenty-one focused tests pass. Single-pass observed times in the CSV are
explicitly unbenchmarked; a paired timing comparison remains part of the
next experiment.

Next: implement one-event-delayed conditional support, comparing static
families with the existing normal-motion variant. Preserve accepted adjacent
events, conditional support and unresolved cases as separate evidence. No
physical ancestry or turbulent-box tracking success is established here.

The readable method and reproduction commands are in
`../../experiments/astrolink/MULTIFRAME_TRACKING_CHECKPOINT.md`.
