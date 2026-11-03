# Extended single-bubble validation to t=0.30

64³ wind-bubble run, 31 requested events at intervals of 0.01, fixed injection
radius 0.0625. Executed on compgpu10, physical GPU 6 (user-reported selection),
with JAX 0.6.2, Python 3.12.3, GPU backend, revision recorded in metadata.json.
No GPU model is inferred. Earlier t=0.20 validation and benchmarks are preserved.

## Reproduce

In the compatible GPU environment, with a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m experiments.wind_bubble.run_online_bubble \
  --num-cells 64 --num-analysis 31 --t-end 0.30 \
  --save-surfaces --track-shocks --integrate-energy --verify-offline --require-gpu \
  --output-dir outputs/online_temporal_iwr_compgpu10_n064_t030_repeat
```

## Checks and observations

- Remote equivalence passed at all 31 events: 41 primary history fields,
  exact histogram counts, sparse catalogs, sampling-quality records,
  62 tracking rows, energy and offline batch-energy checks. Finite scalar
  maximum absolute difference was zero; final gas states matched within
  rtol=1e-6 and atol=1e-8 on the remote run.
- verify_import.py independently reads back saved products, checks source
  hashes, reconstructs histograms and checks finite-endpoint trapezoidal energy.
  Online/offline CSVs and NPZ arrays match exactly locally. Full gas states
  were not preserved and cannot be rechecked after import.
- Minimum detected-surface clearance across events is 0.143797, or 9.203 cells.
  This measures selected surface points, not all flow perturbations, and does
  not by itself exclude boundary influence.
- Forward tracking starts at t=0.01 and continues for 29 subsequent events;
  reverse tracking starts at t=0.12 and continues for 18 subsequent events.
  Both final quality labels are high; these are heuristic radial-track labels.
- Final event: 8160 samples (7368 forward, 792 reverse), all sampling positions
  known and numerically valid, no selected endpoint inside the injection region.
  Early reverse-source overlap remains; a clean final event does not remove it.
- Final median radii: forward 0.352279, reverse 0.114942. Population-wide Mach
  median is approximately 5.0168 and sampled density-ratio median is 2.18.
- Final estimated accumulated energies: forward 0.0311129 over t=0.01–0.30;
  reverse 0.00593820 over t=0.12–0.30, in baseline code units. These are
  finite-interval heating estimates, not the bubble's stored energy or a full budget.

The longer run is a fresh integration, not a restart from the old final state.
The runner sets dt_max=t_end/num_analysis, so its timestep limit changes slightly
relative to the previous 21-event run. No exact-prefix agreement is assumed.
This is workflow validation and visualization, not a physical convergence study
or a new performance benchmark. Original compact-only cost measurements remain unchanged.

## Local readback

```bash
PYTHONPATH=. python outputs/online_temporal_iwr_compgpu10_n064_t030/verify_import.py
```

local_checks.json records checks of the copied files. SHA256SUMS records their
preservation hashes. The three sibling single_bubble_*_iwr_compgpu10_n064_t030
folders contain offline visualizations at 2 fps, 31 frames and 15.5 seconds per loop.
