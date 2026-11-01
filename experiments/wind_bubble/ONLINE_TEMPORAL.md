# Online single-bubble tracking and cumulative energy

The optional temporal products consume the current callback's physical surface
samples and the existing area-weighted powers. They do not read sparse catalogs
back from disk, retain gas snapshots, change the detector, or change callback
timing. Each online/reference writer has its own independent temporal state.

## Run

From the repository root, use a new output directory:

```bash
python -m experiments.wind_bubble.run_online_bubble \
  --num-cells 32 --num-analysis 5 --t-end 0.20 \
  --save-surfaces --track-shocks --integrate-energy --verify-offline \
  --output-dir outputs/online_temporal_validation_n032
```

For IWR, check/select a free GPU and add `--require-gpu`. Offline verification
intentionally retains reference snapshots and remains limited to 64³. The
three output flags are independent and off by default. Existing 41-field
`history.csv`, histogram definitions and sparse-catalog schema are preserved.
The old performance measurements exclude these additional optional products.

## Step 1: temporal candidate tracking

`--track-shocks` writes `online/tracks.csv`, with two rows per event, one for
`reverse_shock` and one for `forward_shock`. These are persistent **candidate
labels for the controlled single-bubble geometry**. No arbitrary connected
object matching, surface-point correspondence or mesh tracking is added.

The tracker reuses the offline radial split, band statistics, continuity
function and heuristic confidence checks. It uses the primary profile-aware
Mach values for the band statistics and quality checks. The historical offline
driver also has baseline/alternative-estimator fields; these extra estimator
outputs are not added online.

After two detections define a radial velocity, predict the next radius using
the actual elapsed time since the last detection. Compare the residual with
grid spacing plus the current and previous radial percentile half-widths.
No fixed maximum displacement per analysis event is imposed.

Key output fields:

| Fields | Meaning |
| --- | --- |
| `time`, `event_index`, `shock_kind`, `track_id` | Event and radial-candidate identity |
| `detected`, `track_status`, `continuity_ok` | Detection and continuity status |
| `radius_median`, `radius_p16`, `radius_p84` | Radius and detected radial spread |
| `radial_velocity`, `elapsed_time`, `radius_change` | Motion since the last detection |
| `predicted_radius`, `prediction_residual`, `allowed_radius_change` | Prediction and residual tolerance |
| `median_radial_alignment`, `surface_cell_count`, Mach statistics | Current band properties |
| `confidence_score`, `confidence_label` and component checks | Heuristic quality indicators, not probabilities |

Statuses are `not_yet_detected`, `initialized`, `continued`, `missing`,
`reacquired`, and `discontinuous`. Missing detections preserve the last detected
candidate state. Reacquisition uses elapsed time across the gap. As in the
offline driver, **every detection updates the candidate state**, including a
flagged discontinuity. Discontinuity does not silently discard a measurement
or assign a new ID. Radial ordering cannot establish identity in general
interacting flows, and quality flags must remain visible in interpretation.

## Step 2: cumulative energy

`--integrate-energy` writes `online/energy.csv`, with one row per event.
The input is the existing per-band, area-weighted thermalization rate, not a
sum of raw point fluxes. A constant-size streaming accumulator reproduces the
offline `_cumulative_trapezoid_over_detections` rule:

- Before the first finite power measurement: cumulative energy is NaN.
- At the first finite measurement: cumulative energy is zero, defining its origin.
- Two consecutive finite measurements: add their trapezoidal integral over
  the actual elapsed time and mark `integration_interval_valid=True`.
- Missing endpoint: add no contribution and mark the interval invalid.
- After a gap: keep the previously accumulated value; do not bridge the gap.

Each band also records `first_power_detection_time` and `integrated_duration`.
The latter adds only durations of valid integration intervals, making incomplete
time coverage explicit. `combined_cumulative_dissipated_energy` sums the
available band accumulations. It does not integrate a partially available total
power, and does not imply complete energy coverage since injection began.

Continuity and confidence flags remain advisory: **no new continuity-based
energy filtering** is introduced. This preserves the existing offline finite-
power integration rule. A finite value during a flagged discontinuity can
contribute to the energy series; inspect `tracks.csv` alongside `energy.csv`.
Changing this policy would be a separate scientific/validation decision.

## Verification and plots

With `--verify-offline`, the reference writer processes retained snapshots
after integration, with its own new tracker and energy accumulators. The
comparison checks status strings, identities, counts and flags exactly, and
floating temporal fields with rtol=1e-6, atol=1e-8 and matching NaNs. Energy
curves and valid-interval flags are additionally checked against the original
offline batch trapezoidal integrator applied to the reference powers.

The runner writes `online_temporal_histories.png`, showing radius, radial
velocity, per-band power and accumulated energy. Missing power samples break
the plotted cumulative curve, even though the stored accumulated value carries
forward. Neither numerical agreement nor plausible tracks independently
validates the physical shock reconstruction.

```bash
python -m unittest -v tests.experiments.test_online_temporal \
  tests.experiments.test_online_bubble tests.experiments.test_sparse_surfaces \
  tests.experiments.test_shared_shock_analyzer tests.experiments.test_online_benchmark
```

Tests cover irregular cadence, missing detections, reacquisition, both radial
bands, flagged discontinuities, zero and missing power, no gap bridging,
agreement with the offline energy integrator, partial band coverage, unchanged
primary products and detection of altered temporal products. All 24 focused
tests and 19 existing offline tracking/energy tests passed locally. A 32³,
five-event CPU comparison also passed, including the offline batch energy
reference. Evidence is in
[`outputs/online_temporal_cpu_validation_n032/README.md`](../../outputs/online_temporal_cpu_validation_n032/README.md).
GPU validation remains outstanding. CPU and GPU evidence should be recorded
separately before updating thesis claims. Revisit Sections 5.4, 5.5 and 5.7
after accepting the implementation.
