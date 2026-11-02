# One-bubble evolution visualization

The first animation uses existing GPU-validated sparse catalogs, with a
synchronized median/percentile Mach history. It is postprocessing only, without
new hydrodynamic integration or a full gas-state series. See
[`outputs/single_bubble_evolution_iwr_n064/README.md`](../../outputs/single_bubble_evolution_iwr_n064/README.md)
for the GIF, previews, verification and interpretation.

```bash
python -m experiments.wind_bubble.animate_sparse_evolution \
  outputs/sampling_quality_iwr_compgpu9_n064/online \
  --output-dir outputs/single_bubble_evolution_iwr_n064_repeat
```

Requires NumPy, Matplotlib and Pillow; no JAX or GPU. Fixed spatial bounds and
a fixed Mach color scale permit comparison between frames. One saved event
maps to one frame; --fps controls illustrative playback and --max-points only
subsamples the displayed point cloud. History statistics retain the full sample
population. Source files are never modified, and an existing output directory
is not overwritten.

The animation shows detected surface samples, not material particle trajectories
or identified connected shock objects. It is specific to the current centered
unit-box single-source setup. General object identities and interacting-bubble
labels need additional definitions; the GIF does not add those capabilities.
No source-overlap filtering, Mach redesign, energy change or new physical
validation is introduced. Power/energy panels and interactive controls can be
added separately after review of this first animation.

Three synchronization tests passed:

```bash
python -m unittest tests.experiments.test_sparse_evolution
```
