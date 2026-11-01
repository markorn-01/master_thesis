# Sparse surface output: CPU validation, 2026-10-31

This is a local engineering validation, not an IWR GPU benchmark or a new
physical-convergence result. GPU validation in `.venv-iwr` remains outstanding.

## Run

From the repository root, in an isolated temporary environment with Python
3.12.4, JAX/JAXlib 0.6.2, Equinox 0.12.2, NumPy 1.26.4 and Orbax 0.11.15:

```bash
python -m experiments.wind_bubble.run_online_bubble \
  --num-cells 32 --num-analysis 5 --t-end 0.20 \
  --save-surfaces --verify-offline \
  --output-dir outputs/online_sparse_cpu_validation_n032_compiled_reference
```

The existing directory is preserved: choose a new output directory to rerun.
Online integration retains no gas-snapshot series. Offline verification
intentionally retains reference states in memory, but writes no gas snapshots.
Source hashes and environment/configuration are in `metadata.json`. The source
was uncommitted; the Git revision alone does not identify these changes.

## Results

- All 15 focused tests passed (sparse selection, writer accounting, shared
  analyzer regression and benchmark accounting).
- Five events produced 0, 200, 392, 560 and 872 samples outside injection.
- All 41 scalar fields per event matched exactly (maximum difference 0).
- Mach and density histogram counts matched exactly.
- All compared sparse geometry and primary property arrays matched exactly.
- The final full gas states agreed pointwise at rtol=1e-6, atol=1e-8.
- Reconstructing histograms from the saved catalogs reproduced the saved counts.
- Empty initial catalogs and their plot were handled without inventing a surface.

`equivalence.json` records the runner checks; `catalog_checks.json` records
per-event sample counts, file sizes and sparse-field maximum differences.
The five online catalogs occupy 68618 bytes in total, **in addition** to the
existing history/histogram output. The final catalog has 872 samples and occupies
25119 bytes. PNG figures, metadata and reference products are not included in
those catalog-only byte counts. No runtime or peak-memory claim is made.

## Figures

- `online/surfaces_0004_mach.png`: refined points colored by profile-aware Mach.
- `online/surfaces_0004_thermalization_flux.png`: the same geometry colored by
  local thermalization flux, not surface-integrated power or cumulative energy.
- `online/surfaces_0004_density_jump.png`: sampled density contrast.
- `online/surfaces_0000_mach.png`: explicit empty initial detection.

These are point clouds, not reconstructed meshes or tracked physical objects.
All 872 final points are displayed. Colors use each catalog's value range;
compare numerical colorbar values when viewing different events. Density ratios
below one reflect the existing coarse-resolution sampling limitations and are
retained; workflow agreement is not independent physical validation.

## Compilation-mode finding

An initial strict comparison failed for individual sampled densities because
the online primary payload was compiled while those offline samples were
computed eagerly. Re-evaluating identical inputs in both modes reproduced the
same differences. See `../online_sparse_cpu_validation_n032/compilation_audit.json`.
The final reference compiles the primary sampling payload and retains the host
analyzer for the Mach/flux reference fields. All checks then passed without
relaxing tolerances or changing shock physics.

## Evening checkpoint

Implemented: optional catalogs, standalone 3D plots, tests and CPU equivalence.
Still pending: IWR GPU validation, temporal identity tracking, cumulative energy
integration and any benchmark of the additional output cost. Existing 64³/128³
benchmark figures refer to compact-only output and must retain that scope.
