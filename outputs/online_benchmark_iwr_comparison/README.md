# IWR online shock-analysis benchmark comparison

## Evidence and provenance

Measurements were reported by the user from compgpu11, physical H100 GPU 3,
using `.venv-iwr`, JAX 0.6.2 and disabled JAX preallocation. Each configuration
used 20 analysis events, t_end=0.20, fixed injection radius 0.0625, one warmup
and three measured repetitions per workflow in fresh worker processes.
The 64³ run followed sync to 806d58e; the 128³ run followed sync to 375e9be.
Full remote source hashes/environment metadata have not yet been imported.

`reported_summary.csv` transcribes all six summary rows supplied in the
conversation without rounding. `reported_verification.json` transcribes both
reported verification objects, with an added provenance note. These are
transcribed evidence, not copies of the original remote reports. Full per-trial
products, memory traces, logs and original JSON reports remain on the remote.

Both reported verifications passed: three trials per resolution, exact
histogram counts, scalar rtol=1e-6 and atol=1e-8, and exact final-state hashes.

## Artifacts

- `online_benchmark_comparison.pdf`: vector figure for Overleaf.
- `online_benchmark_comparison.png`: preview.
- `thesis_benchmark.tex`: draft text and figure environment; requires graphicx.
- `reported_summary.csv`, `reported_verification.json`: supplied measurements.

Reproduce from the repository root with a compatible matplotlib environment:

```bash
python -m experiments.wind_bubble.plot_online_benchmark_comparison
```

## Interpretation

| Grid | Online overhead vs scheduled solver | Online saving vs stored/offline | Online output | Stored/offline output |
| --- | ---: | ---: | ---: | ---: |
| 64³ | 49.0% | 26.8% | 36.1 kB | 104.9 MB |
| 128³ | 16.2% | 23.9% | 36.9 kB | 838.9 MB |

Online overhead is 100*(online/solver-1); saving is
100*(1-online/stored_offline). Storage uses decimal units. GPU memory is MiB
in the reports and GiB in the figure; host RSS uses decimal GB in the figure.
Min–max runtime bars show the three observed trials, not confidence intervals.
Do not add component medians to reconstruct the median total: each component
can have a different middle trial.

The baseline enforces the same analysis-event boundaries; it does not measure
an unconstrained solver. Timings exclude compilation, warmup, setup and final
validation readback. The stored workflow includes raw snapshot transfer and
uncompressed file writes. Writes do not use fsync or clear filesystem caches.
Memory metrics cover the whole worker lifetime, including compilation/warmup;
GPU peaks are sampled lower bounds, not exact live-array peaks. Online host
RSS is higher at both resolutions. Modes run in fixed order on a shared GPU
server. Only two resolutions and three repetitions were measured; do not infer
physical convergence or general performance scaling.

See `experiments/wind_bubble/ONLINE_BENCHMARK.md` for the full timing contract.

## Preserve original remote evidence

Run this on compgpu11 from ~/master_thesis to archive both result trees while
excluding the large raw snapshot arrays. It preserves original benchmark and
summary reports, worker logs, memory traces, compact diagnostics and per-worker
results. Leave the raw snapshot files in their existing remote directories.

```bash
tar --exclude='*/snapshots/*.npy' -czf ~/online_benchmark_iwr_reports_n064_n128.tar.gz outputs/online_benchmark_iwr_n064 outputs/online_benchmark_iwr_n128
```

Transfer that archive to the local machine before labeling the original
reports as imported or independently checked. Compare imported CSV/verification
fields to these transcriptions and inspect source hashes and GPU-sample counts.
Do not overwrite the transcribed provenance with a claim of direct import.
