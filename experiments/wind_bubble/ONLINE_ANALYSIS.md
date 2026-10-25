# First online wind-bubble analysis

## What runs during integration

`run_online_bubble.py` uses the existing hydrodynamic `build_problem` setup and
explicit analysis times. At each callback it removes ghost cells, calls the
shared JAX-only `analyze_shock_state_device`, and transfers diagnostic arrays to
an ordered CPU callback. That callback reduces them immediately and writes:

- `online/history.csv`: 41 fields per analysis time, including surface counts,
  Mach and sampled density-jump percentiles, histogram validity/range counts,
  and forward/reverse radial-band geometry and thermalization statistics;
- `online/histograms_XXXX.npz`: Mach and density-jump counts with fixed bin edges
  and the event time;
- `metadata.json`: physical setup, requested/actual times, source revision and
  source hashes, estimator, units, and run status;
- `online_histories.png`: basic scalar-history visualization.

The online run does not allocate or retain the full gas snapshot series. It
returns a final gas state. The host reducer retains only static grid geometry
and scalar rows; each event's full diagnostic grids are discarded after use.

## First correctness implementation, not a transfer optimization

The callback currently transfers **fixed-shape diagnostic grids**, including
masks, normals, offsets, Mach, flux and sampled densities. These are compressed
on the CPU; no dense diagnostic fields or gas snapshots are written to disk.
Device-to-host transfer still scales with grid volume. No peak-memory saving,
runtime overhead or performance improvement has been measured. Optional sparse
surface catalogs and device-side packing are future work.

The improved profile-aware Mach and matching thermalization flux are the
primary online fields. Plateau and extrapolation alternatives are omitted from
the online callback, but remain available in the unchanged offline dictionary.
Shock detection, sampling formulas and thermalization physics are unchanged.

## Single-bubble interpretation

All statistics and histograms exclude surface samples at or inside the injection
radius. Histograms are **cell-count weighted**, not area weighted. Invalid,
underflow and overflow counts accompany the summaries; the rightmost histogram
edge is included in the final bin.

Forward/reverse labels are a radial validation layer, using the existing split
rule. Surface-area deduplication is performed separately per band. Totals sum
available measured bands, and do not imply complete shock coverage. These labels
and deduplication rules must not be applied globally to many interacting bubbles
without a separate instance definition. No temporal identity tracking or
cumulative energy integration is added here.

Density contrast is the **sampled** `rho2/rho1` at the pressure-selected positions,
not the theoretical Rankine-Hugoniot density ratio inferred from Mach. A valid
sample means finite positive densities, not necessarily a physically resolved
compressive jump. Values below one are kept in the histogram. In the 32³ example
they occur frequently; pressure-selected states do not recover a physically
compressive density jump at this coarse resolution. The online/offline check
establishes agreement of methods, not physical accuracy of those samples.

No detected surface gives missing (`NaN`) Mach, area and power statistics rather
than silently interpreting unresolved dissipation as zero. The initial uniform
state still records zero detected cells and zero histogram counts.

## Local demonstration

From the repository root, using a compatible environment:

```bash
python -m experiments.wind_bubble.run_online_bubble --verify-offline
```

The default is 32³, injection radius 0.0625, t_end=0.20, with five explicit
analysis times including both endpoints. The output directory must be new, to
avoid overwriting an earlier experiment. No MHD mode is included.

With `--verify-offline`, a second simulation intentionally retains a small
snapshot series at the identical times and timestep cap. The independent host
adapter analyzes those snapshots, and compact statistics, histogram counts and
the final gas state are compared. This mode is restricted to at most 64³ to
avoid accidentally allocating a large reference series. It is not the intended
production storage workflow. Explicit times differ from the historical
runner's approximate regular snapshot times, so old CSV files are not used as
pointwise references.

For an IWR validation after syncing these changes:

```bash
python -m experiments.wind_bubble.run_online_bubble --num-cells 32 --num-analysis 5 --t-end 0.20 --verify-offline --require-gpu --output-dir outputs/online_single_bubble_iwr_n032
```

The GPU must be checked and selected before this command using the existing IWR
workflow. Local CPU agreement does not establish GPU agreement.

## Tests and next milestone

```bash
python -m unittest -v tests.experiments.test_online_bubble tests.experiments.test_shared_shock_analyzer tests.time_stepping.test_callback_timing
```

The accounting tests check histogram sample conservation, empty detections and
separate area integration for two radial bands sharing projected coordinates.
The shared analyzer is also checked against the original pre-refactor Git
implementation. The simulation comparison above is the integration check.

Next: validate the small run on GPU, then test a resolved 64³ single bubble.
Measure runtime, peak memory and output size with and without analysis before
making efficiency claims or generalizing to multiple bubbles.
