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
- `online/sampling_quality.csv`: counts of known/unknown endpoint geometry and
  samples entering injection, for all surfaces and each radial band; flags are
  advisory and do not filter the primary measurements;
- `metadata.json`: physical setup, requested/actual times, source revision and
  source hashes, estimator, units, and run status;
- `online_histories.png`: basic scalar-history visualization;
- optionally, `online/surfaces_XXXX.npz` with `--save-surfaces`: detected
  point positions, normals, primary properties and validity flags outside injection.
- optionally, `online/tracks.csv` with `--track-shocks`: single-bubble candidate
  motion, continuity statuses and quality checks;
- optionally, `online/energy.csv` with `--integrate-energy`: finite-interval
  cumulative thermalization and time-coverage flags. See
  [ONLINE_TEMPORAL.md](ONLINE_TEMPORAL.md) for the implementation and validation scope.

The online run does not allocate or retain the full gas snapshot series. It
returns a final gas state. The host reducer retains only static grid geometry
and scalar rows; each event's full diagnostic grids are discarded after use.

## First correctness implementation, not a transfer optimization

The callback currently transfers **fixed-shape diagnostic grids**, including
masks, normals, offsets, Mach, flux, sampled densities and, in the current runner,
the two selected sampling-distance grids. These are compressed
on the CPU; no dense diagnostic fields or gas snapshots are written to disk.
Device-to-host transfer still scales with grid volume. No peak-memory saving,
runtime overhead or performance improvement is established by this initial
demonstration. Subsequent compact-only benchmarks are documented in
`outputs/online_benchmark_iwr_comparison/README.md`. Optional CPU-written sparse
catalogs and a standalone 3D plotter are now implemented; see
[SPARSE_SURFACES.md](SPARSE_SURFACES.md) for their schema and validation workflow.
Device-side packing remains future work. No dense grids are saved, even when
the sparse-output option is enabled.

Source-overlap flags were added after the original benchmarks. The current
runner's added transfers and quality table have not been benchmarked. Sparse
catalogs now use schema 2; older preserved schema-1 files remain unchanged.
See [the sampling-quality validation](../../outputs/sampling_quality_validation/README.md)
for the CPU end-to-end check and reclassification of saved GPU endpoints.

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
without a separate instance definition. Optional single-bubble temporal
diagnostics now extend these radial candidates using the established offline
continuity and energy rules; see `ONLINE_TEMPORAL.md`. They do not provide
general identity matching for interacting shocks.

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
workflow. GPU verification has now passed at both 32³ and 64³ on IWR, as recorded below.

## Tests and next milestone

```bash
python -m unittest -v tests.experiments.test_online_bubble tests.experiments.test_shared_shock_analyzer tests.time_stepping.test_callback_timing
```

The accounting tests check histogram sample conservation, empty detections and
separate area integration for two radial bands sharing projected coordinates.
The shared analyzer is also checked against the original pre-refactor Git
implementation. The simulation comparison above is the integration check.

## Completed IWR validation

Both 32³ and 64³ runs completed on the H100 selected as physical GPU 3 on
compgpu11, using JAX 0.6.2 and commit
`bbc48664b0aacfde9c1e798cffd3513a75713958`. Each used five analysis times
through t=0.20 and fixed injection radius 0.0625. All 41 scalar fields matched
exactly between online and offline runs, as did histogram counts. Final gas
states agreed within rtol=1e-6 and atol=1e-8. The six compact-product and shared
analyzer tests also passed remotely.

Copied results are in `outputs/online_single_bubble_iwr_n032/` and
`outputs/online_single_bubble_iwr_n064/`. Their histories, histogram arrays and
recorded source hashes were verified after transfer. This establishes
online/offline correctness for these experiments, not physical convergence.

These correctness checks do not establish efficiency or generalize the radial
classification to multiple bubbles. Compact-only cost measurements are
documented separately below; additional optional-product costs remain unmeasured.


The controlled three-workflow benchmarking harness is implemented; see
`ONLINE_BENCHMARK.md`. Completed compact-only GPU measurements at 64³ and 128³
are documented in `outputs/online_benchmark_iwr_comparison/README.md`.

## Sparse-output GPU validation

The optional `--save-surfaces` path passed the 32³, five-event comparison on
compgpu1, physical GPU 0, with JAX 0.6.2 at revision `49734c9`. All 15 focused
tests passed, and the standalone plotter displayed all 872 final surface samples.
Copied histories, histograms and sparse catalog arrays matched exactly on local
readback. Final gas-state agreement was checked remotely within rtol=1e-6 and
atol=1e-8; full gas arrays were not saved for local rechecking. Evidence is in
`outputs/online_sparse_iwr_compgpu1_n032/`. The optional catalog writes are not
included in the earlier compact-only performance results.

## Temporal-output GPU validation

The optional `--track-shocks --integrate-energy` path passed the 32³, five-event
comparison on compgpu3, physical GPU 0, using JAX 0.6.2 at revision `34642eb`.
All 43 focused/regression tests passed. Copied histories, tracking and energy
rows, histograms and catalog arrays matched exactly between online and offline
execution. Energy also matched the original batch integrator and a local
recalculation from saved powers. Final gas-state agreement was checked remotely
within the established tolerances. See `ONLINE_TEMPORAL.md` and
`outputs/online_temporal_iwr_compgpu3_n032/README.md` for evidence and limits.
The simulation detected only the forward candidate; the reverse candidate
remains unmeasured, not a zero-energy surface. These options do not extend the
radial method to general interacting-shock identities.

The extended sampling-quality callback now also has a 64³, 21-event live GPU
check on compgpu9, physical GPU 3. All online/offline products passed; local
readback confirmed exactly unchanged legacy history, histograms, tracking and
energy outputs relative to the earlier 21-event GPU run. Evidence is in
[`outputs/sampling_quality_iwr_compgpu9_n064/README.md`](../../outputs/sampling_quality_iwr_compgpu9_n064/README.md).
