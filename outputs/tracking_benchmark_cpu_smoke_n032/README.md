# Step 3 benchmark harness: CPU smoke check

This is a local correctness check of the benchmark harness, not the requested
GPU performance measurement. It uses JAX 0.6.2 on CPU, 32-cubed cells, five
analysis events through t = 0.04, one untimed warmup and two measured repeats
per mode. No claim about GPU tracking overhead or turbulent-box scaling follows
from these timings.

The diagnostics and tracked modes run in separate fresh worker processes.
Each uses the same three-source geometry and discrete source-budget reference.
Measured execution includes trial/writer creation, synchronized integration,
all callback computation/export and terminal close. Compilation/warmup and
independent verification are separate.

## Checks completed

- All six warmup/measured executions have the same exact final-state hash and
  solver iteration count.
- Primary histories, histograms, sparse catalogs and slices match across modes
  for both measured trials.
- The two tracked trials reproduce the same tracking products; 12 group
  observations are recorded per trial.
- The first tracked trial matches batch assembly for all five events, eight
  adjacent results and 16 delayed/terminal results.
- Source versions, setup, hardware/backend, schedules and event counts agree
  across modes. Source fingerprints match after execution.
- Construction/integration/finalization sum to each measured total, and saved
  output-byte accounting was recomputed from disk.
- Missing GPU memory measurements remain null rather than zero.
- All 74 focused local tests pass, including five new benchmark-fairness/
  accounting tests and four existing sampler/compiler/accounting regressions.

`benchmark.json` contains timings, settings, environment, warmup and trial
records, memory scope and verification results. `summary.csv` contains measured
medians/ranges. Per-mode logs, sample files and trial products are retained.
`qa.json` records the final readback checks. GPU benchmarking is still pending.

## Reproduce this smoke check

Use a compatible JAX environment and a fresh destination:

```sh
JAX_PLATFORMS=cpu python -m experiments.wind_bubble.benchmark_tracking_workflow \
  --num-cells 32 --num-analysis 5 --t-end 0.04 --repeats 2 \
  --output-dir outputs/tracking_benchmark_cpu_smoke_n032_repeat
```

The actual Step 3 measurement uses 64 cubed, 17 events through t = 0.16,
three measured repetitions and `--require-gpu` on one free GPU. See
`experiments/wind_bubble/TRACKING_BENCHMARK.md` for the fixed protocol,
comparison definitions and stopping point.
