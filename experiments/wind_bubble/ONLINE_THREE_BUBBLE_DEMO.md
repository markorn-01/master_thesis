# Final three-bubble demonstration from callback tracking

## Completed result

Step 1 is complete. The new three-bubble callback run passed GPU online/offline
and batch checks. Its copied products were verified locally and rendered as
`outputs/three_bubble_online_tracking_demo_n064/animation/three_bubble_group_evolution.gif`:
17 frames, 45 group observations, 15 segments, one additional conditional link
first visible at event 15. The checked meshes match the actual new catalogs and
are reused unchanged. The focused CPU suite passes 65 tests.

The completed input is `outputs/three_bubbles_online_tracking_iwr_compgpu8_n064`.
Its directory keeps the original launch label; the user reports execution on
**compgpu2, GPU 7** after changing hosts. `copy_checks.json` and the final demo's
`qa.json` record the distinction. No additional simulation is required to close
this demonstration. The commands below document the original launch recipe;
always choose a fresh directory for a new run.

## Purpose and completion boundary

Produce one movie with four synchronized panels: checked smooth shock surfaces,
geometric group relationships, per-group Mach distributions summarized by
median/p16/p84, and original detected-sample counts. All group colors,
statistics and relationship-availability times come from the same tracking
records written during the simulation.

The existing three-source geometry, 64-cubed grid, 17 events through t = 0.16,
source-budget reference and association settings are fixed. This step adds no
new algorithm, resolution study or performance benchmark.

`streaming_group_demo` converts `online/tracking/` records to the table format
used by the existing `animate_group_history` renderer. It recomputes statistics
and row/position fingerprints from each original catalog, reconstructs the
graph from the saved adjacent decisions, checks all four delayed-support
channels, and retains the last interval's pending status. The callback CSV
must agree with its own records. When offline verification was requested,
failed or missing equivalence results cannot become a successful demonstration.

Local reconstruction of floating statistic values uses the established relative
and absolute tolerances of 1e-6 and 1e-8. NumPy versions can round float32
percentile interpolation differently: two p84 values differed by about 1.2e-7
in this copied run. The displayed and exported table values remain the recorded
callback values exactly. Counts, categorical labels and original sample
fingerprints retain exact checks. Two regression tests cover accepted rounding
differences and rejected differences outside tolerance.

Accepted events appear at their child event. Added conditional support appears
only when C has arrived; it never changes segment labels or merges statistic
curves. All original measurements in a group are used, including points hidden
by the display cutaway.

## Reuse of the checked meshes

The renderer reuses `outputs/three_bubble_adaptive_sequence_final_n064` without
refitting. It verifies its manifest and checks the new catalogs' positions,
normals, Mach values, source flags and actual times against the mesh originals.
Group-coordinate fingerprints and continuation segment labels must agree at
every event. A different physical sequence cannot silently reuse those meshes.
If this check fails, inspect the mismatch before generating new display meshes;
do not relax the identity checks or rewrite the archived results.

## Current local checks

The saved three-bubble GPU catalogs and archived incremental replay exercise the
full adapter/rendering path using an explicit `--replay-tracking` option. That
output is labeled **saved_catalog_replay_preview**. It is not evidence of a new
three-bubble callback run.

The preview reproduces the prior verified JSON tables exactly and the group
history CSV byte for byte: 45 observations, 15 segments, and one additional
conditional relationship. All 17 GIF frames decode in the correct event order;
the conditional link is hidden at event 14 and becomes visible at event 15.
The mesh geometry and original physical samples are unchanged.

New tests cover callback table conversion, original-coordinate fingerprints,
group CSV consistency, early/duplicate support records, failed equivalence and
explicit separation of replay previews from callback results.

## Final GPU run and local rendering

The existing GPU runner already supports the required flag. On compgpu14 with
a free slot 2, in `.venv-iwr`:

```sh
export CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m experiments.wind_bubble.run_three_bubbles \
  --num-cells 64 --num-analysis 17 --t-end 0.16 \
  --track-shocks --verify-offline --require-gpu \
  --output-dir outputs/three_bubbles_online_tracking_iwr_compgpu14_n064

cat outputs/three_bubbles_online_tracking_iwr_compgpu14_n064/equivalence.json
```

After the checks pass, copy the result from a local terminal in the repository:

```sh
scp -r \
  mngo@compgpu14.iwr.uni-heidelberg.de:master_thesis/outputs/three_bubbles_online_tracking_iwr_compgpu14_n064 \
  outputs/
```

Run the renderer locally with the NumPy/SciPy/Matplotlib analysis environment:

```sh
python -m experiments.wind_bubble.streaming_group_demo \
  --directory outputs/three_bubbles_online_tracking_iwr_compgpu14_n064 \
  --output-dir outputs/three_bubble_online_tracking_demo_n064 --fps 1
```

Use fresh output directories. The final movie will be in
`outputs/three_bubble_online_tracking_demo_n064/animation/three_bubble_group_evolution.gif`.
Rendering remains CPU postprocessing; it need not happen during integration.

The step is complete once the three-bubble callback run passes equivalence,
its displayed surfaces/statistics/labels pass the sample joins, and the encoded
movie passes event-order and causal-visibility checks. This completes the
demonstration, not the separate cost measurement or validation summary.

## Limits

G labels identify geometric continuation segments. Combinations/separations and
conditional evidence do not establish unique physical ancestry or source
ownership. Sample counts are neither area nor energy. Smooth meshes are display
geometry. Existing source-overlap and missing-measurement flags remain visible.
This fixed three-bubble demonstration does not establish turbulent-box scaling.
