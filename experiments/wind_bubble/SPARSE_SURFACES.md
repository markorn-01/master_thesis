# Sparse surface output: first implementation step

The existing callback computes dense diagnostic grids. With `--save-surfaces`,
the CPU writer additionally selects detected surface cells outside the injection
region and saves their geometry and properties in one compressed NumPy archive
per event. No gas snapshot sequence is stored. The catalog is a point cloud,
not a mesh, a list of identified shock objects, or a temporal track.

This step does not change the detector, Mach estimator, thermalization formula
or callback schedule. Schema version 2 adds source-endpoint quality information:
the online runner transfers two additional distance grids for CPU geometry checks.
It does not reduce device-to-host transfer.
Sparse catalog output is optional and off by default. Previous compact-only
storage/runtime measurements do not include this new optional output.

## Run and visualize

In a compatible simulation environment, from the repository root, use a new
output directory:

```bash
python -m experiments.wind_bubble.run_online_bubble \
  --num-cells 32 --num-analysis 5 --t-end 0.20 \
  --save-surfaces --verify-offline \
  --output-dir outputs/online_sparse_validation_n032
```

GPU runs should select an available GPU first and add `--require-gpu`.
`--verify-offline` intentionally retains reference snapshots and remains limited
to 64³. It compares the existing 41 scalar fields and histograms, then compares
all sparse fields: indices/flags/schema exactly, floating values with rtol=1e-6,
atol=1e-8 and matching NaNs. The final physical gas states are also compared.
Do not mistake this shared-method consistency test for physical validation.
The reference uses the established host analyzer for Mach/flux history fields
and compiles the primary payload for offsets and sampled densities, matching
the online evaluation mode. Eager versus compiled float32 interpolation can
differ slightly even for identical inputs. The pointwise tolerances are not
relaxed to accommodate that difference.

Plot a catalog without running a solver or initializing JAX:

```bash
python -m experiments.wind_bubble.plot_sparse_surfaces \
  outputs/online_sparse_validation_n032/online/surfaces_0004.npz
```

Use `--field thermalization_flux` or `--field density_jump` for other properties.
The default output is a PNG next to the input catalog. Points with invalid
properties appear gray, rather than disappearing from the geometry. Empty
events produce an empty view with an explanatory label. `--max-points` limits
the displayed points only (default 100000), selected deterministically in file
order; it never removes records from the archive. A subsampled view should not
be used to infer sampling-uniformity or quantitative distributions.

## Catalog schema, version 2

Files are `online/surfaces_0000.npz`, etc. The reference writer uses the same
names in `offline_reference/` when verification is enabled. Load with
`np.load(path, allow_pickle=False)`.

Let M be the number of detected cells with finite refined radius greater than
the injection radius. All M samples are retained, including invalid Mach,
density and flux values. There is no thinning in the stored output.

| Field | Shape | Meaning |
| --- | --- | --- |
| `grid_indices` | M × 3 | Physical-grid indices in `np.argwhere(mask)` order; not persistent IDs |
| `positions` | M × 3 | Cell center + grid spacing × refinement offset × local normal |
| `normals` | M × 3 | Finder's continuous local normal |
| `radii` | M | Refined distance from the bubble center (0.5, 0.5, 0.5) |
| `mach` | M | Profile-aware pressure-jump Mach |
| `thermalization_flux` | M | Local flux, energy per area per simulation time, in code units |
| `rho_upstream`, `rho_downstream` | M | Densities at the pressure-selected profile positions |
| `density_jump` | M | Sampled downstream/upstream ratio, NaN when sampling/densities invalid |
| `sampling_valid` | M | Profile-sampling validity flag |
| `density_jump_valid` | M | Sampling valid and both densities finite and positive |
| `mach_valid`, `thermalization_flux_valid` | M | Corresponding value finite and positive |
| `upstream_positions`, `downstream_positions` | M × 3 | Actual pressure-selected endpoints, from the refined center and local normal |
| `upstream_radii`, `downstream_radii` | M | Distance of each endpoint from the source center |
| `sample_positions_valid` | M | Endpoint geometry is known; independent of thermodynamic sampling validity |
| `upstream_inside_injection`, `downstream_inside_injection` | M | Known endpoint radius is less than or equal to the injection radius |

Scalar/static fields: `schema_version`, `time`, `grid_spacing`,
`injection_radius`, `grid_shape`, `domain_center`, `selection`, `units`, and
`mach_estimator`. Every event, including an empty initial detection, has a file
with the same schema; empty vector fields have shape (0, 3).

Older version-1 catalogs remain unchanged and lack endpoint geometry. The
low-level writer still accepts the legacy eight-grid payload: new geometry
fields are NaN and `sample_positions_valid` is false when distances are absent.
False overlap flags without valid positions do not establish clean sampling.
The current online runner supplies both distances by default, including in
the offline verification path. These flags do not remove samples or modify
Mach values, histograms, tracking, power or cumulative energy.

## Sampling-quality history

The current runner writes `sampling_quality.csv` even without `--save-surfaces`.
Each event records time and event index, then six integer counts for each of
`all`, `reverse` and `forward`: `surface_samples`, `sample_positions_known`,
`sample_positions_unknown`, `sampling_valid`, `upstream_inside_injection`, and
`downstream_inside_injection`. The selected population and radial split are
the same as the primary diagnostics. Empty populations have zero counts.
Overlap counts include every sample with known endpoint geometry; consult the
separate thermodynamic-validity count as well. Known plus unknown counts equal
the surface-sample count. Upstream/downstream overlap counts can refer to the
same sample and should not be added to obtain a unique-sample count.

The existing 41-column `history.csv` is unchanged. Online/offline checks now
also compare quality rows and the new sparse fields. The benchmark entrypoint
keeps its legacy distance-free payload by default; earlier published timings
do not measure the new runner's additional transfers and quality output.

Validity flags indicate numerical usability, not resolved physical accuracy.
Positive density ratios below one remain present, as in the existing histograms.
The stored geometry contains all selected surface cells, including samples that
surface-area deduplication would discard. No area weights or deduplication
selection are exported yet: do not sum local flux values to infer shock power.
Use the area-weighted power in the existing history instead.

## Checks and next steps

```bash
python -m unittest -v tests.experiments.test_sparse_surfaces \
  tests.experiments.test_online_bubble \
  tests.experiments.test_shared_shock_analyzer \
  tests.experiments.test_online_benchmark
```

The tests cover refined positions, injection exclusion, invalid-property
retention, empty catalogs, agreement with histograms, unchanged compact products
when enabling exports, and detection of mismatched sparse properties.

The first IWR GPU validation passed on compgpu1, physical GPU 0, with JAX 0.6.2
at revision `49734c9`. All 15 focused tests passed. The 32³, five-event run
passed online/offline comparison and the final catalog plotted all 872 samples.
After transfer, catalog arrays, histories and histograms were independently
rechecked locally and matched exactly. See
[`outputs/online_sparse_iwr_compgpu1_n032/README.md`](../../outputs/online_sparse_iwr_compgpu1_n032/README.md)
for evidence, source-hash checks and the scope of the final-state check.

After reviewing a real bubble catalog and its online/offline comparison, the
next separate steps are continuity-based temporal tracking and cumulative
energy integration. They can consume the online results directly and need not
read these catalogs back from disk. Neither capability is added by this change.
These subsequent optional temporal products are now implemented separately;
see [ONLINE_TEMPORAL.md](ONLINE_TEMPORAL.md) for their current validation status.

Thesis sections to revisit after validation: 5.4 output schema and storage scope,
5.5 new sparse-equivalence checks, and 5.7 point-cloud visualization. Keep the
existing benchmark figures labeled as compact-only results.

## GPU confirmation of schema-2 quality output

The extended callback and quality output passed a 64³, 21-event GPU
online/offline comparison on compgpu9, physical GPU 3. Independent local
readback reproduced the catalog geometry, flags and per-band counts. Existing
histories, histograms, tracking and energy CSV contents also matched the earlier
21-event GPU run exactly. The t=0.12 reverse population contains 48 upstream
endpoints inside injection, with no unknown positions. Evidence and scope:
[`outputs/sampling_quality_iwr_compgpu9_n064/README.md`](../../outputs/sampling_quality_iwr_compgpu9_n064/README.md).
This establishes the tested implementation's consistency, not accurate physical
state reconstruction or new performance measurements.
