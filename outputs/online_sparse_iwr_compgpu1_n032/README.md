# Sparse shock-surface output: IWR GPU validation

Validated on compgpu1, physical GPU 0 selected with CUDA_VISIBLE_DEVICES=0,
in `.venv-iwr` using Python 3.12.3 and JAX 0.6.2. The run records revision
`49734c9a76c5b9a22e68aed5237a5237618e537b`. The GPU model is not recorded in
these run metadata; do not infer it from the earlier compgpu11 benchmarks.

The 15 focused tests passed remotely in 100.716 s, as reported in the session.
These cover sparse selection and export, compact accounting, shared-analyzer
regression, and benchmark accounting. The test console log was not copied.

## Simulation and output

```bash
python -m experiments.wind_bubble.run_online_bubble \
  --num-cells 32 --num-analysis 5 --t-end 0.20 \
  --save-surfaces --verify-offline --require-gpu \
  --output-dir outputs/online_sparse_iwr_compgpu1_n032
```

Use a new output directory when repeating the run. The injection radius is
0.0625. Times are 0, 0.05, 0.10, 0.15 and 0.20 within float32 representation.
The online catalogs contain 0, 200, 392, 560 and 872 samples outside injection.
The verification simulation intentionally retains full reference gas states
in memory, but neither workflow writes raw gas snapshots.

The remote plotter loaded the final catalog and displayed all 872 points with
no invalid Mach values. The copied figure is `online/surfaces_0004_mach.png`.
It uses only sparse output, not full gas snapshots.

## Results and local verification

The folder was copied from compgpu1 with scp and verified locally on 2026-11-01.
Original remote reports, histories, histogram archives, catalogs and figures
are preserved. `local_checks.json` records the independent local readback:

- All five events and 41 history fields per event agree exactly.
- All histogram arrays agree exactly between online and offline output.
- Every catalog array agrees exactly, including positions, normals, Mach,
  thermalization flux, sampled densities and validity flags.
- Histograms reconstructed from both catalog series reproduce the saved counts.
- Sample counts match the history, points have finite positions outside injection,
  and stored radii agree with the saved positions and domain center.
- All three recorded analysis-source SHA256 hashes matched the local files at
  the time of import verification.

The remote runner also checked final gas states pointwise using rtol=1e-6 and
atol=1e-8 (`equivalence.json`). Full final gas arrays are not saved here, so this
check was **not** repeated locally. `verify_import.py` reproduces the local
checks from the repository root with NumPy and flags changed source hashes.
`SHA256SUMS` records hashes of the local files after import; it is not a
comparison against an independently obtained remote checksum manifest.

The five online catalogs occupy 60178 bytes in total, in addition to histories
and histograms. This is an observed file-size result for this five-event 32³
case; it is not a benchmark of runtime, memory or storage scaling.

## Scope

This establishes the first sparse-output implementation on GPU for the tested
32³ case. The catalogs are point samples, not a mesh or persistent shock objects.
Dense diagnostic grids still transfer from device to host. Positive sampled
density ratios below one remain visible; numerical agreement does not establish
physical accuracy or resolved compression. The older performance figures cover
compact-only output and do not include the optional catalog writes.

Next: connect and validate temporal continuity tracking, then cumulative
energy integration. Neither is part of this validated sparse-output change.
