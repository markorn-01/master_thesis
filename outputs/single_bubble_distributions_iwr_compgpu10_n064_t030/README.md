# Extended single-bubble distributions animation

shock_distributions.gif: 31 saved analysis events over simulation time 0–0.30, played at
2 fps (500 ms per frame; 15.5 seconds per loop). One frame per recorded event;
no interpolated physical states. Inputs are the GPU-validated online products in
../online_temporal_iwr_compgpu10_n064_t030/online. No simulation is run by rendering.

Reproduce from the repository root, with a fresh output directory:

```bash
python -m experiments.wind_bubble.animate_shock_distributions \
  outputs/online_temporal_iwr_compgpu10_n064_t030/online \
  --fps 2 --output-dir outputs/single_bubble_distributions_iwr_compgpu10_n064_t030_repeat
```

The original renderer's render_metadata.json records input/source hashes and
rendering settings. verification.json records local full-frame decoding, playback
length and input-hash checks. Representative static previews were visually inspected.
Source physics and measurements are unchanged. Histograms use surface-cell counts,
not area weights; point clouds are not meshes or persistent material particles.
Heating histories cover measured finite intervals and retain the existing sampling
limitations. See the run README for equivalence, boundary clearance and scope.
Earlier t=0.20 animations and benchmark results are preserved separately.
