# Reverse-candidate normal paths: 128³ GPU inspection

Run on compgpu9, physical GPU 3 (selected in the session), with Python 3.12.3,
JAX 0.6.2, GPU backend, at revision edc0b93. GPU model is not recorded.
Imported and checked locally on 2026-11-02. The setup uses injection radius
0.0625, 21 requested events over 0–0.20 and dt_max=0.20/21. This inspects a
new simulation at t=0.20; it does not alter any physical estimator.

All 2624 valid reverse samples were inspected. Independent interpolation
matches selected endpoint pressures/densities exactly (remote maximum absolute
difference 0.0). Pressure candidates independently reproduce selected distances.
All selected endpoints lie outside the Boolean zone and injection region.
This does not establish that their surrounding physical fields are flat.

The original normal_path.png shows an illustrative point chosen nearest the
population median density-RH residual. Normal velocity is in the simulation
frame, and positive signed distance follows the upstream direction. The
representative point index is not a persistent identity; it can differ between
CPU and GPU demonstrations due to small floating differences and nearly tied
residuals. The plotted np.int64 wrappers in its title are only a representation
of the grid-index integers; scientific values are unchanged.

NumPy readback checks passed: source hashes, requested/actual times, recorded
endpoint values, radii, zone flags, indices and the defining pressure-Mach
relation. The latter is a consistency check, not independent validation of Mach.
See readback_checks.json. Full gas arrays were not saved, so no local final-state
comparison is claimed. This inspection does not perform online/offline verification
at 128³; that is a separate run documented in the quality-validation folder.

Reproduce with a fresh output directory and an available GPU:

```bash
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m experiments.wind_bubble.inspect_reverse_sampling_paths \
  --num-cells 128 --require-gpu \
  --output-dir outputs/reverse_paths_iwr_compgpu9_n128_repeat
```

Repeat readback from the repository root with NumPy:

```bash
python outputs/reverse_path_diagnostics/verify_paths.py outputs/reverse_paths_iwr_compgpu9_n128
```

All path data, candidates and original selected values are in normal_paths.npz;
per-point checks are in endpoint_checks.csv. No full gas sequence is saved.
The comparison and interpretation are in
[../reverse_path_diagnostics_iwr_compgpu9/README.md](../reverse_path_diagnostics_iwr_compgpu9/README.md).
SHA256SUMS is a local preservation manifest.
