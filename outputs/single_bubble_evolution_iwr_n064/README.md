# First synchronized one-bubble evolution animation

`shock_evolution.gif` shows all 21 saved events from the verified 64³ online
GPU run on compgpu9 (physical GPU 3), spanning simulation time 0–0.20.
It is generated entirely from saved sparse catalogs and history.csv: no new
simulation, gas snapshots, detector evaluation or estimator changes are needed.

## What is shown

- Left: detected shock-surface points outside injection, colored by their
  profile-aware pressure-jump Mach. The star marks the central wind source.
  Camera, coordinate bounds and color scale remain fixed throughout the animation.
- Right: cell-count-weighted median Mach of all selected detected surface samples,
  with 16th–84th percentiles. Shading is sample spread, not uncertainty or an
  area-weighted distribution. Gray shows the complete saved history; teal and
  the orange marker show progression to the current event.
- Time, total detected sample count, displayed count and median update together.
  At t=0 there are no detections and the median is undefined, not zero.

The expanding outer population and later inner reverse candidate are visible.
This is a sequence of detected point clouds, not gas-density rendering, a
connected surface mesh, fluid particles or persistent identities of individual
points. The same radial tracking labels are not used to assign point identities.
Physical limitations of the reverse reconstruction remain; this animation
neither filters problematic samples nor establishes converged shock measurements.
Power and energy histories are not included in this first visualization.

All points are displayed for this run (maximum 5128). The color range is fixed
at 0–16, covering every finite positive Mach value across the run. Spatial bounds
are a fixed source-centered crop covering all saved surface points with margin,
not a changing zoom. Coordinates are baseline code units.

## Playback and previews

There is one frame per event, with no interpolation between saved states.
The requested playback is 2 frames/s; GIF timing is quantized to 500 ms/frame,
for 10.5 s per 21-frame loop. This is illustrative playback speed, not wall-clock
simulation execution time. These data have uniform requested physical cadence;
for irregularly spaced events the renderer still assigns one frame per event.
It does not claim playback duration proportional to arbitrary time intervals.

`preview_0000.png`, `preview_0010.png` and `preview_0020.png` show t=0, 0.10
and 0.20. They were visually inspected; an overlapping axis label was corrected
before this final render. All 21 GIF frames decode, and original source file
hashes and the rendering-script hash match render_metadata.json.

## Reproduction

With NumPy, Matplotlib and Pillow, from the repository root:

```bash
python -m experiments.wind_bubble.animate_sparse_evolution \
  outputs/sampling_quality_iwr_compgpu9_n064/online \
  --fps 2 \
  --output-dir outputs/single_bubble_evolution_iwr_n064_repeat
```

Use a fresh output directory. --fps controls playback only. --max-points limits
3D display points deterministically; history and percentiles always use the full
recorded sample population. Invalid Mach points, if present, display gray.
The renderer does not import JAX or require a GPU. Schema-1 and schema-2 catalogs
are supported when they contain the required geometry and primary properties.

The loader verifies contiguous catalog numbering, event times, point counts,
Mach-valid counts and percentile agreement with the history before rendering.
Three focused synchronization tests passed: empty initial measurement remains
missing; shifted event times are rejected; missing catalogs or inconsistent
statistics are rejected. No existing simulation code was modified.

render_metadata.json records all frame times, fixed scales, display counts,
input hashes and rendering source hash. verification.json records successful
frame decoding, frame durations and source checks. SHA256SUMS is a local
preservation manifest. No runtime or memory comparison is claimed.

This is the first offline visualization of the validated online outputs.
An interactive timeline, additional power/energy panels and multiple-bubble
visualization remain further steps, rather than features of this GIF.

Playback was slowed to 2 frames/s by changing GIF frame-delay metadata only.
All 21 decoded frames were verified pixel-identical to the previous animation;
simulation times, diagnostics, and static previews are unchanged.
