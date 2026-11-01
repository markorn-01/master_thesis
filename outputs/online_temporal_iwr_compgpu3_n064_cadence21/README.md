# 64³ online temporal diagnostics: 21 analysis events

Run on compgpu3, physical GPU 0, in `.venv-iwr`, with Python 3.12.3 and JAX 0.6.2.
The GPU model is not recorded. Source revision: `34642eb1c70dddbe3fcea641149024630a212707`.
The host and GPU selection are recorded from the session, not inferred from metadata.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m experiments.wind_bubble.run_online_bubble \
  --num-cells 64 --num-analysis 21 --t-end 0.20 \
  --save-surfaces --track-shocks --integrate-energy \
  --verify-offline --require-gpu \
  --output-dir outputs/online_temporal_iwr_compgpu3_n064_cadence21_repeat
```

The physical injection radius is 0.0625. Use a new output directory.
The reference simulation retains states for offline verification; the saved
products contain no full gas snapshots. Two sets of console analysis messages
are expected: online events and offline reference processing.

## Independent import checks

Copied results were checked locally on 2026-11-01 using NumPy:

- All 41 primary history fields match exactly at 21 events.
- All 42 track rows and 21 energy rows match exactly online/offline.
- All sparse arrays and histogram arrays match exactly.
- Histograms reconstructed from catalog samples match the saved counts.
- Energy independently recomputed from saved powers matches the energy CSV,
  including interval-validity flags and accumulated time coverage.
- Requested times equal actual times, and all five recorded source hashes
  matched local source files at import.

See `local_checks.json` and `equivalence.json`. Repeat local checks with:

```bash
python outputs/online_temporal_iwr_compgpu3_n064_cadence21/verify_import.py
```

The verifier requires NumPy. Later source changes may fail the source-hash check.
The remote runner also passed its offline batch-energy comparison and final
state comparison (rtol=1e-6, atol=1e-8). Full states were not saved, so their
comparison cannot be repeated from this import. `SHA256SUMS` records local
preservation hashes; no independent remote checksum manifest was supplied.
These are same-backend execution-path checks, not CPU/GPU comparisons.

## Interpretation

The reverse candidate is first detected at t approximately 0.12,
with 9 detected events. All detected reverse continuity flags pass.
Its final radius is 0.0997968598. Its cumulative thermalization
energy is 0.00132976923904 in code-energy units, covering
0.08000000 time units since the first finite power sample.
The forward candidate is detected at 20 events, beginning at
approximately t=0.01.

Energy is zero at its first finite sample by definition; earlier heating is
not measured. Missing endpoints are not bridged. Continuity flags are advisory
and do not filter energy integration. Radial identities assume this single-source
geometry. Inward normals and repeated detections support the reverse-candidate
interpretation, but do not independently establish resolved thermodynamic jumps.
Quality labels are heuristic. Small changes in radius compared with dx=1/64
make finite-difference velocities sensitive to sampling and grid effects.

The companion comparison is in
[`../online_temporal_iwr_n064_comparison/README.md`](../online_temporal_iwr_n064_comparison/README.md).
No performance claim is made for these optional products; earlier benchmarks
measured compact-only output. Dense diagnostic arrays still transfer to the host.
