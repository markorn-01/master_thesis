# Benchmark harness: local CPU validation

Engineering validation used a 32³ wind bubble, t_end=0.02, three explicit event
times, and two measured repetitions per workflow after one full warm-up. Each
workflow ran in its own process. This small CPU experiment is not evidence for
production GPU speed or memory improvement.

Checks passed:

- All modes and measured repeats used 11 solver iterations.
- All measured final-state hashes were identical across modes and repeats.
- Online/offline history fields agreed within rtol=1e-6, atol=1e-8.
- Histogram counts matched exactly in both measured trials.
- The 24 focused benchmark/accounting, shared-analyzer and callback tests passed
  in 50.906 seconds.

Each online trial wrote 5,818 bytes of compact diagnostics. The storage baseline
wrote those compact products plus 1,966,464 bytes of raw `.npy` snapshot files
(three snapshots, including file headers). These storage numbers apply to this
specific uncompressed baseline and cadence only. Logs and result metadata are
excluded from payload sizes. Raw benchmark snapshot files are ignored by Git.

`summary.csv` contains compilation and warmed execution phases separately,
repeat ranges, output sizes and host RSS. GPU memory cells are empty because
this validation ran on CPU; missing GPU data must not be interpreted as zero
GPU memory. Compiler-planned allocations and whole-process host memory differ
between modes, and include different effects from actual live GPU buffers.

`benchmark.json` records source hashes, environment, trial details and
correctness checks. Per-mode `result.json` and `worker.log` provide additional
measurement definitions and run information. Warm-up products were temporary
and removed; measured products persist.

Methodology, limitations and the remote command are documented in
`experiments/wind_bubble/ONLINE_BENCHMARK.md`. The next step is an isolated
single-GPU run at 64³, t_end=0.20, 20 analysis times and three repetitions.
No GPU benchmark has been run yet.
