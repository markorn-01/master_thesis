# Source-endpoint sampling-quality flags

The current online runner records whether the actual pressure-selected
upstream/downstream endpoints lie at or inside injection. These are advisory
flags: no samples are removed and no Mach, histogram, tracking, power or energy
values are modified. This is specific to the centered single-source validation
setup, not a general source mask for arbitrary geometries.

## Implementation

The device analyzer already computes the selected distances in cell units.
The online runner now transfers those two additional fixed-shape grids and
reconstructs endpoints on the CPU:

    x_up = x_surface + dx * pre_distance * normal
    x_down = x_surface - dx * post_distance * normal

`x_surface` includes subcell refinement. A known endpoint is flagged when
its radius is <= 0.0625. Both distances, positions and nonzero normals must
be finite, and distances nonnegative. Geometric validity is separate from
thermodynamic sampling validity. Missing distances remain explicitly unknown;
false overlap flags alone do not establish that a sample is outside injection.

Schema-2 sparse catalogs add endpoint coordinates, radii, geometry-validity
and overlap flags. `sampling_quality.csv` records counts for all selected
surfaces and for each radial band, even if sparse output is disabled. It
also separates known/unknown geometry and thermodynamic validity. The original
41-column history is unchanged. See
[`SPARSE_SURFACES.md`](../../experiments/wind_bubble/SPARSE_SURFACES.md).

The benchmark entrypoint still requests its legacy eight-grid payload, without
distances or a quality table. Published timings describe earlier code and do
not measure these additional online-runner transfers and writes. No new
performance claim is made.

## Checks completed locally

1. **28 focused tests passed**, covering sparse geometry/validity, online output,
   temporal products, profile plotting and benchmark helpers. New tests check
   endpoint signs, the inclusive source boundary, unknown/invalid distances,
   unchanged measurements and detection of altered quality rows.
2. **35 existing radial-tracking/energy regression tests passed.**
3. A new **32³ five-event CPU online/offline run passed**, including 41 primary
   fields, histograms, all sparse fields, five quality rows, ten tracking rows,
   five energy rows, the offline batch-energy reference and final-state agreement.
   See `online_cpu_n032/equivalence.json`.
4. Independent readback verifies the new catalog radii/flags and per-band quality
   counts. Histories, tracking CSVs, energy CSVs and histogram arrays are exactly
   unchanged relative to the earlier preserved CPU run. See `cpu_readback_checks.json`.
5. The shared overlap classifier reproduces every saved reverse endpoint flag at
   the three inspected times in both the 64³ and 128³ GPU result folders. At
   64³, 48/48 upstream endpoints at t=0.12 lie inside injection; all other
   inspected endpoints, including all 128³ endpoints, lie outside. See
   `saved_gpu_endpoint_recheck.json`.

These are CPU tests and reclassification of saved GPU endpoint radii. They do
not constitute a live GPU validation of the newly extended callback payload.
The 32³ CPU run has no reverse detections; reverse source overlap is exercised
by the saved 64³ endpoint check and synthetic geometry tests. No source flags
were retroactively inserted into old catalogs.

The implementation was tested as a working-tree change based on commit
`993935c`; run metadata source hashes identify the actual tested files. No
full gas-state arrays are saved. Independent readback cannot repeat the final
state comparison; that was performed by the live runner.

## Repeat checks

With NumPy only, from the repository root:

```bash
python outputs/sampling_quality_validation/verify_cpu_outputs.py
```

To reclassify preserved GPU inspection endpoints, use a new report filename:

```bash
python -m experiments.wind_bubble.check_sampling_quality \
  outputs/reverse_profiles_iwr_compgpu3_n064 \
  outputs/reverse_profiles_iwr_compgpu3_n128 \
  --output outputs/sampling_quality_validation/saved_gpu_endpoint_recheck_repeat.json
```

With the compatible simulation environment:

```bash
python -m experiments.wind_bubble.run_online_bubble \
  --num-cells 32 --num-analysis 5 --t-end 0.20 \
  --save-surfaces --track-shocks --integrate-energy --verify-offline \
  --output-dir outputs/sampling_quality_validation/online_cpu_n032_repeat
```

## Remaining GPU check

After syncing the new implementation and selecting an available GPU, run the
64³ case with 21 events, all optional outputs and `--verify-offline` in a fresh
directory. Confirm online/offline quality equality and inspect the t=0.12
reverse counts. This tests the full new GPU callback path in the case with
known source overlap. No physical estimator redesign or filtering is needed.

## Thesis text

An Overleaf-ready subsection is in
[`thesis_sampling_limits.tex`](../reverse_profiles_iwr_resolution_comparison/thesis_sampling_limits.tex).
It separates execution-path consistency from physical sampling accuracy and
records the nonmonotonic resolution comparison. It has not been inserted into
the user's live Overleaf project.

`SHA256SUMS` is a local preservation manifest, not a remote transfer checksum.
