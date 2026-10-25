# Shared single-state analysis

`analyze_state.analyze_shock_state(state, config, registered_variables, helper_data)`
accepts one **unpadded, Cartesian 3D hydrodynamic state** and returns the existing
NumPy diagnostic dictionary. `center` optionally sets the origin of the radial
diagnostics. It does not change detection. The single-bubble offline runner now
calls this function for each saved state.

The extraction preserves both estimator families explicitly:

- `mach_numbers` and `thermal_energy_flux`: original baseline estimator;
- `profile_aware_mach_numbers` and `profile_aware_thermal_energy_flux`: improved
  profile-aware estimator, with consistent upstream sampling;
- plateau, extrapolation and sampling-quality fields: retained for compatibility.

The host adapter still constructs dense arrays and evaluates diagnostic
alternatives. `analyze_shock_state_device` exposes the shared JAX-only core
for online callbacks; CPU conversion and reduction occur separately.
No changes are made to detection, sampling, thermalization, solver timing,
tracking, or output cadence. The online single-bubble runner now streams compact histories and histograms.
Sparse export and transfer/runtime optimization remain later work.

## Reproducible extraction check

From the repository root, with the project dependencies installed:

```bash
python -m experiments.shock.check_analyzer_equivalence
```

The check reads only the historical `analyze_shocks_3d` function from local Git
commit `c6cda81c189d7d8c2fb9e8349664741681272aae`. It compares all returned
arrays (including shapes, dtypes and NaN locations) against both the shared
analyzer and the updated offline wrapper. Git history containing that commit
is required; no network or historical code checkout is performed.

Cases: a Mach-5 body-diagonal planar ramp, the uniform initial gas, and the final
state of a small 32³ wind bubble at t=0.20 with physical injection radius 0.0625.
The bubble is evolved once and the same state is passed to each analyzer. The
check does not compare different simulations or claim resolution convergence.
Only the final hydrodynamic state is retained during this small example run.

Products under `outputs/shared_analyzer_validation/`:

- `equivalence.json`: environment, reference commit and comparison results;
- `shock_surface_and_mach_histogram.png`: full detected bubble surface and
  cell-weighted profile-aware Mach histogram (forward/reverse samples combined).

Additional regression tests:

```bash
python -m unittest -v tests.experiments.test_shared_shock_analyzer
```

The small CPU check is evidence for extraction equivalence. The explicit
hydrodynamic suite must also be run in `.venv-iwr` before accepting a production
GPU workflow. Callback timing and online/offline simulation equivalence are
separate, outstanding milestones.
