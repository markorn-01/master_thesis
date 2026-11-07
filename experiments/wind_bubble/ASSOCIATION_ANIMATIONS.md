# Two-bubble candidate-association animations at 128³

These animations review the existing geometric candidates through the complete
saved collision sequence, t=0–.16. They are postprocessing products, not a new
simulation or a validated physical tracker.

Outputs: `outputs/two_bubbles_association_animations_n128/`.

- `native_cells_associations.gif`: native spatial radius 1.5 cells, cutoff 20.
  Retains smaller fragments, including newly appearing components, competing
  matches at .13→.14 and redundant links at .14→.15.
- `native_cells_scaled_count_associations.gif`: same graph, cutoff 80 (the
  earlier 20-sample reference cutoff scaled by N²). The larger retained groups
  give simpler candidate links. This excludes fragments from matching; it does
  not establish that those fragments are physically irrelevant.

Both use spatial and association signed-normal limits of 60°, a common physical
association gate of 2.5/64 code units (five cells at 128³), parent and pooled-child
coverage thresholds .5, and parent dominance margin .25. These are frozen saved
settings, not parameters tuned by the animation. The wider gate was chosen to
make previously identified ambiguity and redundancy visible, not because it is
known to be optimal. Narrower-gate comparisons remain in the static results.

## Reading each frame

Each frame compares **two adjacent saved times**. There are 17 saved events and
16 intervals, hence 16 frames per GIF. At 2 fps each loop lasts eight seconds;
playback speed is independent of the physical simulation cadence (.01).
There is no interpolation between saved states or bridging of missing matches.

The two spatial columns show previous/next samples in fixed central xy/xz
slabs, each ±.04 wide in the perpendicular direction. These are projections of
3D points, not continuous meshes. Stars mark sources; gray crosses show
size-excluded samples. Graph decisions use all 3D samples.

The right column shows event-local group labels and sample counts. Green lines
are accepted candidates, orange dashed lines are accepted but redundant, and
red dotted/dashed lines show competing best/runner-up targets of ambiguous
parents. Red-bordered child nodes have no accepted pooled support. A new
unsupported child does not, by itself, establish the birth of a physical shock.
Colors and node layout are assigned independently at each event; they are not
persistent identities or physical trajectories. The footer records accepted,
redundant, ambiguous, insufficient-support and unsupported-child counts.

The same static inspection renderer is reused for every adjacent pair. It
reconstructs the candidate pair coverages, checks them against the saved
scores, and renders a temporary PNG. Pillow resizes each PNG, adds the interval
status strip, and assembles the GIF. Temporary full-resolution frames are
removed automatically; selected readable keyframes are retained as PNGs.
This implementation does not use FuncAnimation or modify the earlier animation
scripts. It reuses the established static view to keep its meaning consistent.

## Checks and scope

Both GIFs were decoded and checked for 16 frames, dimensions 1700×995, and
500 ms per frame. Each decoded status strip was matched against the rendered
sequence to check frame order despite GIF palette quantization. Saved audits
were checked against all 32 source configurations, including node labels/counts,
link categories, both ambiguous targets and displayed status totals. Inputs
retain their original hashes. See `verification.json` and `readback.json`.

These checks establish display consistency. They do not solve the known
near-parallel-sheet false-link control, establish physical shock ownership,
measure convergence, or make temporal association general to a turbulent box.
No production tracker, shock estimator, saved simulation, or grouping changed.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/astrolink-mpl \
/opt/anaconda3/envs/astrolink-env/bin/python \
  -m experiments.wind_bubble.animate_resolution_links \
  outputs/two_bubbles_temporal_comparison_n064_n128 \
  --fps 2 \
  --output-dir outputs/two_bubbles_association_animations_n128_new
```

Use a new output directory. NumPy, SciPy, Matplotlib and Pillow suffice; no GPU,
JAX or FuzzyCat is required. Provenance, full frame audits and checksums are
saved alongside the animations. Source-run and budget-matching limitations
remain those in `TWO_BUBBLE_RESOLUTION_RESULTS.md`.
