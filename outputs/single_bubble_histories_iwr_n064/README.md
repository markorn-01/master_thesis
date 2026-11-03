# Synchronized surface-area, power and energy histories

shock_histories.gif uses the existing 64³, 21-event GPU-validated online data
from compgpu9, physical GPU 3, over simulation time 0–0.20. It reads only
history.csv, energy.csv and the optional sampling_quality.csv. No simulation,
shock finding, integration-rule change or new spatial reconstruction is needed.
The 3D/Mach and histogram animations remain separate and unchanged.

## What the three panels mean

- **Measured shock-surface area:** existing orientation-corrected and
  projected-patch-deduplicated estimates, separately for the forward and reverse
  radial populations. These are measured surfaces, not proven complete coverage.
- **Estimated thermalization rate:** the existing area-weighted shock-heating
  estimate for each radial population, in code energy per code time. This is
  neither direct solver entropy production nor the bubble's total energy change.
- **Accumulated thermalized energy:** the stored finite-endpoint trapezoidal
  accumulation for each population, in code-energy units. This is not total
  stored thermal energy or a complete energy budget since injection began.

Teal is the forward radial candidate; purple is the reverse candidate. Faint
curves show the entire saved history for context. Solid curves and current
markers progress with the orange time cursor. Axis ranges remain fixed across
all 21 frames. Labels assume a single centered source and do not establish
identities for arbitrary interacting shocks.

The cards show current values, the first finite power time (the energy origin),
covered integration duration, interval status and source-overlap sample counts.
Unknown measurements remain undefined, not zero. Upstream source overlap is
advisory and does not remove samples or modify accumulated energy. In this
input all selected endpoint positions are known. The displayed ratio is the
number of source-overlapping upstream samples divided by the population size,
not an area-weighted fraction or a physical completeness measure.

## Energy origins and gaps

The forward accumulation begins at t approximately 0.01; the reverse accumulation
begins at t approximately 0.12. At those origins the energy is zero by definition,
not because earlier heating is known to vanish. Final values are approximately
0.0206260 (forward, coverage 0.19) and 0.00132977 (reverse, coverage 0.08).
These cover different measured intervals. No combined-total curve is introduced.

Only explicitly valid adjacent integration intervals are drawn as energy line
segments. At a missing power event, the stored accumulated value can carry
forward, but its point is not presented as a new measurement and no line bridges
the invalid interval. A later valid power point can appear without connecting
the gap. This input has no gaps after the bands become active; the no-gap drawing
behavior is verified using synthetic missing-data cases.

At t=0.12, the reverse card shows zero accumulated energy, zero covered duration,
an origin/no-valid-interval status and 48/48 upstream source overlaps. This
preserves the limitation found in the physical inspection. Sampling-quality flags
and physical reconstruction limitations remain relevant at later times too;
a clean source flag does not establish accurate shock-state reconstruction.

## Reproduction and checks

Requires NumPy, Matplotlib and Pillow, without JAX or a GPU. From the repository
root, use a fresh output directory:

```bash
python -m experiments.wind_bubble.animate_shock_histories \
  outputs/sampling_quality_iwr_compgpu9_n064/online \
  --fps 2 \
  --output-dir outputs/single_bubble_histories_iwr_n064_repeat
```

--fps controls illustrative playback only. There is one frame per saved event,
with no interpolation of states. The requested playback is 2 frames/s; actual
GIF timing is 500 ms/frame (10.5 s total), not simulation execution speed.

Before rendering, the loader checks event alignment, numbering, power values,
finite-endpoint energy accumulation, origins, covered durations and interval
flags. It also checks the stored combined energy for consistency but does not
plot it. Nine visualization tests passed, including three new history tests
covering missing measurements, carried energy, no gap-bridging, distinct origins,
changed cumulative values/flags and mismatched times.

All 21 frames decode. Initial, middle and final previews and the t=0.12 energy
origin frame were visually inspected. preview_0012.png is extracted from the
GIF; the other previews are direct high-resolution renders. Source and input
hashes match render_metadata.json. verification.json records readback success.
SHA256SUMS records local preservation. No previous source outputs were changed,
and no new runtime, memory or physical-convergence claim is made.

This provides the separate heating-history view requested for the first
single-bubble demonstration. Interactive controls, general many-bubble area
integration and cross-bubble identity tracking remain separate development.

Playback was slowed to 2 frames/s by changing GIF frame-delay metadata only.
All 21 decoded frames were verified pixel-identical to the previous animation;
simulation times, diagnostics, and static previews are unchanged.
