# Three-bubble saved-catalog profile and support reuse

Completed 26 November 2026 on the local macOS ARM CPU, Python 3.10.20,
NumPy 2.2.5 and SciPy 1.15.3. No new hydrodynamic simulation was run.

## Result

| Mode | Median replay time | Min--max, three trials |
| --- | ---: | ---: |
| Baseline | 11.038 s | 10.959--11.077 s |
| Reused support | 5.802 s | 5.799--5.995 s |

Median time decreased by **47.4%**, a **1.90x** replay speedup. Both modes
received one untimed warmup; measured trial order alternated. Input catalogs
were loaded before timing. Tracking writer construction, event processing,
compression/file writing and close are included. Verification and profiling
are outside the measured trials.

The baseline spends about 97% of receive time in adjacent-event association
in a separate instrumented replay. The expanded matcher built the same
distance/normal-compatible point support twice. It now reuses that support
for four/five-partner candidates and selects only the final event model.
No geometric rule or physical estimator was changed.

## Correctness

- All 17 saved events, 45 group observations and final pending records match.
- Every paired replay has exactly equal decoded tracking products, including
  complete scores/selection/ambiguity records, delayed evidence, partitions,
  group statistics, CSV values, fingerprints and buffer accounting.
- The original saved GPU archive agrees using established scalar tolerances
  (rtol 1e-6, atol 1e-8), retaining exact categorical and integer comparisons.
- 243 full old/new synthetic-result comparisons pass, covering both adjacent
  intervals where available, declared gates, and partner caps 3/4/5.
- 66 focused unit tests pass, including a five-way event requiring just one
  support tree and one final selection, and instrumented-output equivalence.
- Input and source hashes were unchanged throughout the comparison.

`verification.json` records the reference Git revision, environment, scopes,
phase-call counts, medians and input/source/output fingerprints. Its frozen
matcher sources are in `reference_sources/`. The original input directory's
name is preserved; the historical three-bubble run was reported as executed
on compgpu2 GPU 7 after a host change.

## Files

- `trials.csv`: unprofiled paired measurements and decoded-product digests.
- `summary.csv`: median/min/max execution times.
- `event_phases.csv`: instrumented receive stages and workload per event.
- `function_profile.csv`: separate cProfile call counts and times (includes
  product decoding; nested cumulative times must not be added together).
- `tracking_replay_comparison.png`: selected phase costs and measured times.
- `baseline_products.json.gz`, `optimized_products.json.gz`: decoded products,
  with partition dtypes/shapes/values, excluding container timestamps.
- `reference_sources/`: the pre-change matchers loaded from the pinned revision.

## Limits and reproduction

This is a CPU replay, not an end-to-end GPU benchmark. Hydro integration,
shock detection, GPU transfers, input reads and display rendering are excluded.
Trials share a process/caches and write without fsync. Full input preloading
is a replay convenience, not a new production memory policy. Exact execution
agreement does not validate physical ancestry or turbulent-box performance.
The previous GPU benchmark remains unchanged.

See `experiments/wind_bubble/TRACKING_REPLAY_PROFILE.md` for the reproduction
command and the bounded stopping point. Steps 1--2 end here; a many-source
pilot is a separate decision.
