# Sampling-quality GPU validation: compgpu9, 64³

Completed on compgpu9, physical GPU 3 selected with CUDA_VISIBLE_DEVICES=3
(session evidence), with Python 3.12.3, JAX 0.6.2 and GPU backend. The GPU
model is not recorded. Revision: edc0b93af8248c8b8defdc61edc44d224e606c9e.
Results were copied and checked locally on 2026-11-02.

## Run

```bash
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m experiments.wind_bubble.run_online_bubble \
  --num-cells 64 --num-analysis 21 --t-end 0.20 \
  --save-surfaces --track-shocks --integrate-energy \
  --verify-offline --require-gpu \
  --output-dir outputs/sampling_quality_iwr_compgpu9_n064_repeat
```

Use a fresh directory. The injection radius is 0.0625; all 21 requested and
actual times agree. Online output saves no full gas-state sequence. The
verification path intentionally retains reference states during execution.

## Passed checks

The remote equivalence report passes 21 events, 41 primary fields per event,
all histograms and sparse fields, 21 quality rows, 42 tracking rows, 21 energy
rows, the offline batch-energy reference and final gas-state agreement. Maximum
finite primary scalar difference is zero. Floating tolerances are rtol=1e-6,
atol=1e-8 with matching missing values.

Independent local checks in local_checks.json verify:

- Source hashes match the local tested files; run status and backend are correct.
- Histories, tracks, energy rows, quality rows and sparse arrays match exactly
  between the new online/offline paths after transfer.
- Endpoint radii/overlap flags and all per-band quality counts are recomputed
  from the saved catalogs and match the CSV table.
- History, tracking and energy CSV contents and all histogram arrays are exactly
  unchanged from the earlier 21-event compgpu3 GPU run, before these flags existed.

| Time | Reverse samples | Unknown positions | Upstream inside injection | Downstream inside injection |
|---|---:|---:|---:|---:|
| 0.12 | 48 | 0 | 48 | 0 |
| 0.15 | 336 | 0 | 0 | 0 |
| 0.20 | 608 | 0 | 0 | 0 |

These flags remain advisory: no measurements, sample selection or energy
integration rules are changed. Agreement establishes execution consistency,
not accurate resolved physical states. Full gas states were not saved, so
the remote final-state check cannot be repeated from the imported products.

Repeat local checks from the repository root with NumPy:

```bash
python outputs/sampling_quality_iwr_compgpu9_n064/verify_import.py
```

The verifier requires current source hashes and the earlier preserved GPU
reference folder. Later source changes may make its hash assertion fail.
SHA256SUMS is a local preservation manifest, not an independent remote manifest.
The new transfers and optional products have not been benchmarked; older
compact-only timings do not cover them.
