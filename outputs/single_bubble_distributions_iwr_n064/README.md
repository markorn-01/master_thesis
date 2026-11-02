# Shock-surface evolution with Mach and density-contrast histograms

shock_distributions.gif is a second visualization of the same 21-event, 64³
GPU-validated online run on compgpu9 (physical GPU 3). It covers simulation
times 0–0.20 and uses only saved catalogs, histograms and history. No new
simulation, shock detection, reconstruction or filtering was performed.
The earlier synchronized Mach-history animation is preserved separately.

## Reading the animation

- Left: detected shock-surface points outside injection, colored by profile-aware
  Mach. The star marks the source; the camera, coordinate crop and color scale
  remain fixed across all events. Invalid Mach points, if any, display gray.
- Top right: the original saved Mach histogram, 60 uniform bins over 0–30.
- Bottom right: the original saved density-jump histogram, 64 uniform bins over
  0–8. Density contrast is the directly sampled rho2/rho1 at pressure-selected
  upstream/downstream positions, not a value inferred from Mach.
- Orange vertical lines show medians of all valid positive values, including
  values outside histogram ranges. A gray dotted line marks density ratio one.
- Each histogram explicitly reports in-range, invalid, below-range and above-range
  sample counts. The final bin includes the rightmost edge. The initial empty
  event has zero counts and undefined medians, rather than measured zeros.

Histogram axes are fixed, with separate count limits for Mach and density.
The same vertical scale across events reveals growth of the detected sample
population, but early Mach bars can be small compared with later peaks. These
are raw cell counts, not normalized probabilities, area-weighted distributions,
or counts of connected shock objects. Surface display subsampling never affects
histograms. All surface points are displayed in this run, up to 5128 at t=0.20.

Positive sampled density ratios below one are retained, including a median of
approximately 0.97 at t=0.01. Numerical validity does not guarantee resolved
compressive states, and pressure-derived Mach and sampled density jumps need
not recover mutually consistent idealized jumps in a broadened/nonuniform flow.
This visualization does not change or independently validate those measurements.
See the preserved reverse-sampling inspection for the relevant physical limits.

The point clouds are detected geometry, not a mesh or material particles with
persistent identities. Spatial topology, radial tracking/identity labels, area,
power and cumulative-energy histories are not added by this animation. The
current scene remains the centered, unit-box single-source demonstration;
there is no generalization of radial diagnostics to many bubbles in this change.

## Reproduce and verify

Requires NumPy, Matplotlib and Pillow; no JAX or GPU. From the repository root,
use a new output directory:

```bash
python -m experiments.wind_bubble.animate_shock_distributions \
  outputs/sampling_quality_iwr_compgpu9_n064/online \
  --output-dir outputs/single_bubble_distributions_iwr_n064_repeat
```

Optional --fps changes illustrative playback; --max-points changes only the
number of displayed 3D points. No input files are modified. Original bin edges
and integer counts are preserved, without rebinning, smoothing or normalization.
The loader checks contiguous event numbering, synchronized times, compatible
fixed bin edges, nonnegative integer counts and complete sample accounting.
Both histograms are independently reconstructed from all catalog samples and
required to match exactly. Median/percentile and range counts must match history.

Six visualization synchronization/accounting tests passed locally, including
empty initial data, shifted times, missing events, changed counts, a positive
subunity density ratio, a sample exactly at the final bin edge, invalid values,
and overflow accounting. All 21 GIF frames decode. The initial, early and final
previews were visually inspected. Three frames per second are requested; GIF
duration is quantized to 330 ms per frame (6.93 s total). One saved event maps
to one frame, without temporal interpolation; playback is not wall-clock speed.

render_metadata.json records event times, fixed scales, bins, count axes,
source/input hashes and successful validation. preview_0000.png, preview_0001.png,
preview_0010.png and preview_0020.png provide static views. SHA256SUMS is a local
preservation manifest. No new runtime, memory or physical-accuracy claim is made.
