# Reverse-candidate normal-path inspection

Purpose: investigate why density-Rankine–Hugoniot consistency at t=0.20 differs
between 64³ and 128³. This diagnostic reruns the same hydrodynamic setup with
21 requested events, dt_max=0.20/21, injection radius 0.0625, and examines the
last gas state. It does not change detection, the pressure-Mach estimator,
sampling choices, tracking, or energy integration.

## Run

Use an available GPU and a fresh directory:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m experiments.wind_bubble.inspect_reverse_sampling_paths \
  --num-cells 64 --require-gpu \
  --output-dir outputs/reverse_paths_iwr_n064
```

Repeat at 128³ with `--num-cells 128` and a different output directory.
Omitting `--require-gpu` permits a separate CPU demonstration. The initial
local checks use CPU reruns, not saved GPU gas states. Full gas states were
not preserved by the earlier inspection outputs, so new integration is needed.

## What is inspected

For every numerically valid reverse-candidate sample at t=0.20:

- Reconstruct the normal path from the refined surface position with the same
  fixed local normal as the estimator. Positive signed distance is upstream;
  negative distance is downstream. Distances are in grid-cell units.
- Record interpolated pressure, density, simulation-frame normal velocity,
  nearest-neighbor shock-zone membership, and distance from the source.
- Independently interpolate the exact selected endpoints and check them
  against the shared analyzer's selected pressures and densities.
- Reproduce first zone exits on the original quarter-cell search grid and
  all six pressure candidates at 0.25–1.5 cells beyond each exit. Verify that
  the selected distances correspond to the upstream pressure minimum and
  downstream pressure maximum. This is an implementation consistency check,
  not an independent validation of recovered physical states.
- Measure local changes in pressure/density at s−0.5, s and s+0.5 cells around
  each endpoint. The full range divided by the central value detects both
  gradients and curvature; it is a diagnostic, not a calibrated error bound.

Leaving the Boolean shock zone does not guarantee a flat physical state.
Conversely, a physical wind can have background gradients even outside a
broadened shock, so a large local variation does not itself establish a bug.
A one-cell interval has different physical width at different resolutions.

## Products and scope

`normal_paths.npz` contains all inspected paths, original pressure candidates,
selected samples, geometry and indices. `endpoint_checks.csv` contains
per-sample distances, source radii, zone flags, pressure-candidate indices and
one-cell endpoint-to-endpoint relative changes. `summary.json` summarizes
those quantities. The downstream candidate index runs from 0 (nearest exit)
to 5 (farthest). No full gas-state sequence is saved.

`normal_path.png` shows one illustrative sample chosen closest to the median
per-sample density-RH residual, without choosing a particularly clean path.
The pressure and density panels mark candidate positions and selected
endpoints. Shading shows the Boolean shock zone and injection region where
present. Normal velocity is in the simulation frame, not the shock frame;
no mass/momentum conservation test using an assumed stationary shock is made.

`outputs/reverse_path_diagnostics/derive_checks.py` recomputes full-range
endpoint variations from saved paths, including curvature. It also evaluates
hypothetical downstream choices among the existing six candidates while holding
the original selected upstream state fixed. It does not alter measurements,
select a candidate to minimize the RH residual, or establish a new estimator.
The original selected candidate's residual is checked against the saved output.

Use `verify_paths.py` in that result directory for NumPy-only readback checks.
Six focused geometry/plot/interpolation tests passed locally. See the result
README for the completed cases, findings and limitations.
