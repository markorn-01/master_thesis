# Online temporal diagnostics: CPU validation

Local engineering validation on 2026-11-01. GPU validation of these new
optional temporal products remains outstanding. The earlier sparse-output GPU
validation does not validate this later change.

## Implementation under test

```bash
python -m experiments.wind_bubble.run_online_bubble \
  --num-cells 32 --num-analysis 5 --t-end 0.20 \
  --save-surfaces --track-shocks --integrate-energy --verify-offline \
  --output-dir outputs/online_temporal_cpu_validation_n032
```

Use a fresh directory to repeat this command. This run used Python 3.12.4,
JAX/JAXlib 0.6.2, Equinox 0.12.2, NumPy 1.26.4 and CPU execution in a temporary
isolated environment. The source was uncommitted; recorded source hashes,
rather than the Git revision alone, identify the implementation under test.
All five recorded source hashes matched the local files at verification.

The online path retains no gas-snapshot series. The offline verification run
intentionally retains reference states in memory. Raw gas arrays are not saved.

## Checks

- All 24 focused tests passed in 44.453 s: new temporal behavior, online writer,
  sparse catalogs, shared-analyzer regression and benchmark accounting.
- All 19 existing temporal-tracking and shock-energy integration tests passed
  in 0.914 s. New tests include synthetic two-band cases and missing detections.
- The 32³ simulation completed five events and passed online/offline comparison:
  all 41 history fields, histogram arrays, sparse arrays, 10 tracking rows and
  five energy rows.
- Cumulative energy and valid-interval flags additionally matched the existing
  offline batch trapezoidal integrator applied to reference powers.
- Complete final gas states agreed within rtol=1e-6 and atol=1e-8.
- Independent CSV readback found history, tracking and energy files exactly
  equal between online and offline paths.
- Primary histories, histograms and sparse arrays matched the earlier
  `online_sparse_cpu_validation_n032_compiled_reference` run exactly.

See `equivalence.json` for runner checks and `local_checks.json` for readback.
The reference and online paths intentionally share the analysis and temporal
prescriptions. These checks establish execution consistency, not independent
physical accuracy or general shock identity matching.

## Outputs and interpretation

- `online/history.csv`: unchanged 41 primary scalar fields.
- `online/tracks.csv`: 35 fields, two radial-candidate rows per event.
- `online/energy.csv`: 13 fields, one row per event with per-band powers,
  cumulative energies, interval flags, start times and covered durations.
- `online_temporal_histories.png`: motion, power and cumulative-energy curves.
- Existing histogram archives and optional surface catalogs remain present.

The forward candidate is initially `not_yet_detected`, then `initialized` at
about t=0.05, and `continued` at t=0.10, 0.15 and 0.20. The reverse candidate
is not detected in this coarse demonstration, so it has no plotted curve and
its cumulative energy remains NaN, not zero. Its legend entry identifies the
candidate series and does not indicate that a reverse shock was measured.

Forward accumulated energy starts at zero at the first finite power detection
(t approximately 0.05). At t approximately 0.20 it is 0.007315374872956007 in
code-energy units, covering about 0.15 simulation-time units. The unresolved
initial interval is not included. This is not the full dissipated energy since
injection began or a complete wind-energy budget.

Continuity flags are advisory, matching the offline driver. Every detected
radial band updates its candidate state, even if flagged discontinuous. Energy
integration uses consecutive finite powers without an extra continuity filter;
a flagged but finite measurement can therefore contribute. No point matching,
new shock IDs on discontinuity, interacting-bubble tracking, or mesh tracking
is implemented. No performance claim is made for the new optional outputs.

## Next validation

Commit/sync the implementation, select an available IWR GPU, run the focused
and existing tracking/energy tests, then repeat the small comparison with
`--require-gpu` and a new IWR output directory. Preserve and inspect the remote
results before accepting broader thesis claims or increasing resolution.
