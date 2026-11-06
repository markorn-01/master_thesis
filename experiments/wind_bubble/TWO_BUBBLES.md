# First controlled two-bubble experiment

This stage builds a dataset for clustering and temporal-correspondence tests.
It does not implement AstroLink/FuzzyCat, shock ownership, mesh reconstruction,
or general shock-surface power integration.

## Setup

- Hydrodynamics in the unit cube, open boundaries, uniform ambient rho=1 and P=0.01.
- Stationary equal sources at (0.35, 0.5, 0.5) and (0.65, 0.5, 0.5).
- Fixed injection radius 0.0625, gamma=5/3.
- Each source uses the existing EI prescription with nominal mass rate 0.01
  and wind velocity parameter 10 (nominal mechanical luminosity 0.5 each).
  EI adds mass and thermal energy, without imposing a free-wind velocity.
- First GPU target: 64³, 17 events over 0 <= t <= 0.16.

The sources initially produce separated bubbles and subsequently interact.
Inspect pressure/density slices and surface samples to locate the interaction;
do not identify a collision time from a radius formula alone. Source positions
are experimental choices, not labels for physical shock ownership after contact.
The shorter final time reduces boundary exposure compared with copying the
off-center setup to t=0.30. Check actual clearance on every new run.

### Injection discretization

WindConfig.source_positions configures disjoint stationary sources for 3D
Cartesian finite-volume EI. An empty tuple retains the old centered calculation.
The same source function also enters the existing source-aware timestep estimate.
Other solver modes and overlapping/truncated injection spheres are rejected.

The original source uses an analytic spherical volume and a cell-center mask
with radius R_inj - dx/2. This prescription is retained, including its discrete
normalization. Consequently the actual injected mass/energy can be smaller than
the nominal rates and can change with resolution or source placement relative
to cells. metadata.json records cell counts and discrete rates per source.
Equal rates between sources are checked, but this is not yet a convergence
comparison with the centered single-source experiments. Do not silently renormalize
or compare resolutions without examining these records.

## Outputs

- metadata.json: geometry, per-source nominal and discrete rates, software,
  source hashes, requested/actual times and completion status.
- online/history.csv: global counts and Mach/density-ratio summaries, union
  source-overlap counts, unknown sampling positions and surface boundary clearance.
- online/histograms_NNNN.npz: original Mach/density bins and cell-count weighting.
- online/surfaces_NNNN.npz: schema 3, refined positions, normals, Mach, density,
  thermalization flux and validity fields, endpoint positions, per-source
  overlap flags and geometric distances to BOTH sources. No radial labels or
  persistent object IDs. Source flags are advisory and do not filter endpoints.
- online/slice_NNNN.npz: density and pressure at z=(N//2+0.5)dx. For even N this
  is the cell-center plane just above z=0.5, not an interpolated midplane.
- Optional offline_reference/ and equivalence.json for matching-time verification.

Surface selection excludes refined positions within either injection sphere.
It does not discard the interaction midplane or assign samples to their nearest
source. Positive sampled density ratios below one remain in the statistics.
The old single-source animation loaders reject schema 3; use the dedicated
viewer until a multi-source animation is implemented.

## Run in the compatible GPU environment

Select a currently free GPU first and run check_gpu_environment. Then:

```bash
python -m experiments.wind_bubble.run_two_bubbles \
  --num-cells 64 --num-analysis 17 --t-end 0.16 \
  --verify-offline --require-gpu \
  --output-dir outputs/two_bubbles_iwr_n064
```

The output directory must be new. Online retains no gas-state sequence. Offline
verification intentionally retains snapshots and is restricted to N <= 64.
Expect two sequences of log messages, for online and stored-state analysis.

Inspect one event:

```bash
python -m experiments.wind_bubble.plot_two_bubbles \
  outputs/two_bubbles_iwr_n064/online --event-index 16 \
  --output outputs/two_bubbles_iwr_n064/event_0016.png
```

The viewer reconstructs saved histograms as a consistency check. Color scales
are event-specific, so its static images are not a fixed-scale animation.

## Validation so far

Eight focused tests cover explicit-center equivalence, two-source superposition,
mirror symmetry, discrete source budgets, rejected configurations, both-source
exclusion, endpoint flags, unknown positions and output-comparison sensitivity.
Sixteen existing shared-analyzer, online-product and sparse-catalog tests passed.

The local CPU 32³ trial used nine events to t=0.16. Online/offline verification
passed for all events, all 21 history fields, exact histogram counts, sparse
catalogs, background slices and final gas states. Finite scalar differences were
zero. Representative separate and interacting stages were rendered and inspected.
This is low-resolution workflow validation, not physical convergence. GPU validation
and the full compatible remote regression suite remain outstanding.

## Next after GPU verification

Select separated, contact and interacting events. Compare clustering based on
positions alone and positions plus normals. Reuse the earlier AstroLink toy
experiments, then evaluate the representation needed for FuzzyCat across events.
Keep clustering, temporal correspondence and display meshes as separate stages.
