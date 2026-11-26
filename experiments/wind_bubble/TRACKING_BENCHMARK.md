# Step 3: complete-workflow cost of online tracking

## Completed GPU measurement

The three-bubble 64-cubed benchmark completed on compgpu6, GPU 0 (RTX 2080 Ti).
The copied products passed local revalidation. Median execution increased from
3.847 s without tracking to 26.866 s with the complete grouping/tracking package:
23.020 s added, 598.5% overhead. Primary output was unchanged; tracking added
75,534 bytes per measured run. Compilation and one warmup are excluded.

See [results, figure and limitations](../../outputs/tracking_benchmark_iwr_compgpu6_gpu0_n064/README.md)
and its `copy_checks.json` for the preserved evidence. Step 3 is complete for
this configuration; the next activity is thesis writing, not another automatic
simulation or optimization cycle.

The later user-requested [CPU replay profile and support-reuse optimization](TRACKING_REPLAY_PROFILE.md)
measures a separate local replay. Its improvement does not update these
historical GPU timings; a new GPU workflow measurement would be required.

## Question and fixed comparison

How much runtime, memory and saved output does grouping/tracking add to the
already working online shock-analysis workflow?

Use the established three-source 64-cubed case, 17 analysis events through
t = 0.16, equal source strengths and the existing discrete source-budget
reference. Compare just these two modes:

| Mode | Work performed |
| --- | --- |
| `diagnostics` | Evolve the gas; detect/reconstruct shocks; transfer diagnostic grids; write histories, histograms, sparse catalogs and background slices. |
| `tracked` | The same work, plus spatial grouping, temporal association, both delayed-support variants, group statistics and tracking products. |

The first mode is already a complete diagnostic workflow. This comparison
measures the additional cost of enabling tracking, not the total cost of
shock analysis relative to a solver without diagnostics. The older single-
bubble benchmark addressed a different comparison.

The implementation wraps the existing integration core for explicit compilation
and reuses the production `CollisionWriter` and diagnostic payload. It changes
no solver, source, detector, Mach estimator, grouping or association criterion.
The two modes share the same analysis schedule and timestep cap. No complete
gas-state sequence is stored in either measured mode.

## Warmup and three measured runs

Each mode runs in a separate fresh child process, sequentially:

1. Build the same initial state and budget-matched three-source configuration.
2. Lower and compile the integration executable once. Record this startup cost.
3. Execute one complete untimed warmup, including output and terminal close.
   Only its privately created temporary outputs are removed afterward.
4. Execute three measured runs from the same initial state, using the compiled
   executable. Preserve the measured output directories.

Tracking dependencies are imported in both modes before timing. State donation
is disabled so the initial state can be reused safely. A stable host receiver
changes its writer only after all prior callbacks have completed.

The default comparison uses three measured repetitions. The 32-cubed local
smoke test uses fewer/smaller events to check the harness; its CPU timings are
not the requested GPU performance result.

## What the time includes

`total_workflow_seconds` starts before trial-directory and writer construction
and stops after integration, callbacks and writer close have completed.

- `construction_seconds`: trial directory, primary writer and optional tracker.
- `integration_seconds`: compiled evolution, GPU shock analysis, existing
  device-to-host transfer, CPU reduction, optional grouping/tracking and saved
  event products. `block_until_ready` and `effects_barrier` synchronize it.
- `finalization_seconds`: terminal pending-support records, metadata updates
  and writer flush/close. This cost is included in the total.

Compilation, the warmup, plotting, final-state transfer/hash, output-size scans
and correctness checks are outside the measured execution. Whole-worker wall
time is reported separately and includes startup, setup, compilation, warmup
and trial validation. The first short run can therefore take substantially
longer than a warmed trial. There is no animation-rendering time in this
comparison; rendering is optional postprocessing rather than a hydro callback.

The tracking increment is:

```text
added seconds = median tracked time - median diagnostics time
overhead (%)  = 100 × added seconds / median diagnostics time
```

For illustration only: 2 seconds without tracking and 3 seconds with tracking
would mean 1 added second and 50% overhead. Actual results must come from the
GPU run; compilation and warmup do not enter this ratio. Per-phase medians
need not sum exactly to the median total.

## Memory and output

- GPU allocation is sampled with `nvidia-smi` for the exact child PID, excluding
  other processes. The peak covers compilation, warmup and measured work, with
  driver context and allocator caches. It is a sampled lower bound that can
  miss brief peaks, not an exact live-array peak.
- Host RSS is the OS high-water captured after the trials, including imports,
  setup, compilation, warmup and callback working memory. It is not an isolated
  per-trial or tracker-only allocation measure.
- Compiler-planned memory is recorded separately; it does not substitute for
  observed GPU or host memory and excludes CPU callback allocations.
- Output bytes include the primary CSV, sparse catalogs, histogram files and
  slices; tracked mode additionally counts all tracking files, including its
  settings/status metadata needed to interpret them. Logs, benchmark reports,
  memory-sample files and deleted warmup data are excluded.

Tracking is expected to add saved products relative to this diagnostics
baseline. Neither mode saves full gas snapshots. This is a different storage
question from the earlier online-versus-full-snapshots comparison.

## Correctness checks outside timing

The controller requires identical setup, source versions, environment, device
kind, hostname, event times and repetition counts. It checks every warmup and
measured final-state hash and solver iteration count across both modes.
It then compares primary scalar histories, histograms, sparse catalogs and
slices for each trial using the established exact/tolerant definitions.
Tracked repeats must reproduce each other's tracking products, and the first
tracked run must agree with independent whole-sequence batch assembly.

There is no performance-success threshold. If tracking adds substantial cost,
that is the measured finding to report. A favorable number is not required for
the benchmark to complete. A correctness mismatch invalidates the comparison.

## Run on a free GPU

Commit/push the harness first, then pull it into `.venv-iwr`. Select one free
GPU using `CUDA_VISIBLE_DEVICES` and check it with `nvidia-smi`. The controller
requires one visible GPU when using the GPU backend and disables preallocation
in its workers.

The completed compgpu6 run used `$HOME/.venvs/master-thesis-py312/bin/python`
(Python 3.12.16), because the older `.venv-iwr` symlink selected system Python
3.14 there and could not import its Python 3.12 packages. The commands below
assume `python` already selects a compatible, checked environment.

```sh
python -m unittest tests.experiments.test_tracking_benchmark \
  tests.experiments.test_online_benchmark

python -m experiments.wind_bubble.benchmark_tracking_workflow \
  --num-cells 64 --num-analysis 17 --t-end 0.16 --repeats 3 \
  --require-gpu --output-dir outputs/tracking_benchmark_iwr_n064

cat outputs/tracking_benchmark_iwr_n064/summary.csv
python -c 'import json; d=json.load(open("outputs/tracking_benchmark_iwr_n064/benchmark.json")); print(json.dumps(d["verification"], indent=2))'
```

Use a fresh output directory and keep the controller in the foreground until
the shell prompt returns. Console messages report the active mode; each worker's
analysis messages go to its `worker.log`. A `running` file alone does not prove
an OS process is actively executing if it has been paused or killed.

## Scope and stopping point

Modes run in fixed order on a shared machine, and repetitions share each
worker's allocator/filesystem caches. File writes are closed/flushed without
`fsync`; this is not a physical disk-throughput test. One resolution, source
layout and cadence do not establish general turbulent-box scaling. GPU peaks
are whole-worker samples and host peaks include compiler/warmup allocations.

Step 3 is complete after the GPU benchmark passes correctness checks, its
runtime/memory/output results are read back, and their interpretation is
recorded. No automatic optimization, algorithm search or higher-resolution
experiment follows from the size of the measured overhead. Afterward, freeze
the implementation and move to the thesis write-up.
