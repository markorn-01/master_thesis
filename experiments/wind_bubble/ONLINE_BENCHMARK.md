# Controlled benchmark for online shock analysis

## Workflows

The benchmark runs the same hydrodynamic bubble, fixed injection radius 0.0625,
explicit event times and timestep cap in three separate child processes:

| Mode | During integration | After integration | Persisted data |
| --- | --- | --- | --- |
| `solver_only` | Scheduled timestep boundaries, no shock analysis | Final-state checks | No simulation data |
| `online` | Primary shock analysis and compact exports at each event | Final-state checks | History CSV and histograms |
| `stored_offline` | Complete gas snapshots retained on device | Same primary shock analysis, compact exports and raw snapshot writes | Compact products plus uncompressed `.npy` snapshots |

The baseline preserves diagnostic event boundaries so that all workflows use
the same sequence of solver steps. It measures analysis/export overhead
**conditional on that schedule**, not the cost of imposing a schedule on an
otherwise unconstrained simulation.

Online and offline use the same `diagnostic_payload` and CPU reducer. The
benchmark evaluates the primary profile-aware estimator, not the plateau and
extrapolation alternatives in the full historical offline script. The offline
workflow is therefore an efficient primary-diagnostic reference, not a timing
of every legacy diagnostic. It handles snapshots sequentially after integration;
it does not copy the entire stored series to CPU at once.

## Compilation, warm-up and measured execution

The harness explicitly lowers and compiles the existing integration core with
dynamic state, parameters and helper data. The offline per-snapshot analysis
executable is also compiled beforehand. No time-integrator or physics code is
modified for the benchmark. The core wrapper additionally returns the final
time and iteration count for consistency checks; setup and public-wrapper
validation are outside the timed execution.

Each mode runs one complete untimed warm-up, then the requested number of
measured repetitions using those executables. The warm-up includes actual
exports in a private temporary directory which is removed afterward. Measured
trial directories are preserved. The initial state is reused without donation.

The phase definitions are:

- `compile_*_seconds`: explicit integration/analysis lowering and compilation.
  Common problem setup and its initialization are outside these phase timers.
- `integration_seconds`: executable dispatch through completed solver work,
  synchronized with `block_until_ready` and `effects_barrier`. For online mode,
  this includes shock analysis, diagnostic transfer, CPU reduction, compact
  file writes and callback logging.
- `offline_analysis_seconds`: offline per-event device analysis, diagnostic
  transfer, CPU reduction and compact exports, summed across events.
- `snapshot_transfer_write_seconds`: transfer and uncompressed raw snapshot
  writes for the storage baseline, charged separately.
- `total_workflow_seconds`: sum of the three measured execution phases above.
  Compilation, warm-up, directory/writer construction, plots, final validation
  readback and result metadata are excluded.
- `process_wall_seconds`: whole child-process wall time, including startup,
  setup, compilation, warm-up and measured execution. It must not be confused
  with warmed workflow time.

CSV writes are flushed per event; `.npz`/`.npy` files are closed after writing.
The benchmark does not call `fsync`, clear filesystem caches or measure raw
storage-device throughput. Repeats share a process and allocator cache within
each mode. Modes run in fixed order, so this first benchmark does not eliminate
all temperature, filesystem or shared-machine load effects.

## Memory measurements

Three distinct metrics are recorded:

1. `sampled_peak_gpu_process_mib`: maximum `nvidia-smi` allocation observed for
   the exact worker PID. Other users' processes are excluded. The controller
   polls throughout the worker lifetime, including initialization, compilation,
   warm-up and measured runs. Driver context and retained allocator caches are
   included. This is a sampled lower bound on process framebuffer usage, not
   an exact peak of live arrays during the timed region. Short-lived peaks may
   be missed. The requested sampling interval excludes query latency.
2. `worker_peak_host_rss_bytes`: OS high-water host RSS across the whole worker,
   including compiler/initialization/warm-up memory; converted to bytes on both
   macOS and Linux. This is not a per-trial increment.
3. `compiled_*_memory`: compiler-planned argument, temporary, output and alias
   sizes. Planned total is arguments + temporaries + outputs - aliases. These
   numbers do not include all runtime/driver allocations or host callback
   memory, and are not substituted for observed GPU peaks.

Missing GPU samples are `null`/empty CSV cells, not zero memory. The controller
does not initialize JAX or allocate a CUDA context. Each child disables JAX
preallocation and requires a single visible GPU when using a GPU backend. An
interrupt terminates only the controller's own current worker.

## Output sizes and correctness

`compact_bytes` counts the actual CSV and histogram files. Snapshot file bytes
include `.npy` headers; array payload bytes are also recorded separately.
`output_payload_bytes` sums compact and snapshot files. Logs, metadata,
compilation reports, memory samples and deleted warm-up products are excluded.
The storage comparison is specifically against uncompressed `.npy` snapshots;
it is not a claim about every possible compressed snapshot format.

Before writing a successful benchmark summary, the controller checks:

- identical solver iteration counts across modes and repeats;
- agreement of final per-variable minima, maxima and means within tolerance;
- matching online/offline compact fields and exact histogram counts per trial.

Full final-state hashes are also recorded and exact hash agreement is reported
separately. A differing hash is not treated as an error if the numeric summary
and diagnostic checks pass; these summaries alone do not establish pointwise
field equality. The separate online/offline validation runner remains the
stronger final-field correctness check.

## Run locally for an engineering smoke test

```bash
python -m experiments.wind_bubble.benchmark_online_analysis \
  --num-cells 32 --num-analysis 3 --t-end 0.02 --repeats 2 \
  --output-dir outputs/online_benchmark_cpu_validation
```

Use a new output directory. This short CPU case tests the harness and artifact
accounting, not production GPU performance. Grid size is limited to 64³ here
because the storage baseline deliberately retains full snapshot histories.
Raw benchmark snapshots are ignored by Git; compact products and reports can
be reviewed without committing duplicate dense state files.

## IWR run after syncing the changes

Check GPU availability and JAX GPU execution first, following the existing
one-command-at-a-time IWR workflow. Run the longer benchmark inside `tmux`.

```bash
python -m experiments.wind_bubble.benchmark_online_analysis \
  --num-cells 64 --num-analysis 20 --t-end 0.20 --repeats 3 \
  --require-gpu --output-dir outputs/online_benchmark_iwr_n064
```

The controller writes `summary.csv`, `benchmark.json`, and per-mode logs,
metrics, sampled memory traces and measured trial products. Failures preserve
partial results and the worker log; they do not produce a successful summary.

Tests:

```bash
python -m unittest -v tests.experiments.test_online_benchmark
```

Interpret GPU results only after the device selection, correctness checks,
repeat variation and memory-sample availability have been inspected. No GPU
runtime overhead or memory improvement is established by local CPU validation.
