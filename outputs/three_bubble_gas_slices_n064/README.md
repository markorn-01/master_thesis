# Gas slices showing the colliding fronts

`three_bubble_gas_density_pressure.gif` shows all 17 saved events at one event
per second (17 seconds), through t=0.16. `density_contact_sequence.png` compares
t=0.04, 0.05, 0.06, 0.07, 0.08 and 0.10 around the initial interaction.

The panels are gas density / ambient density, gas pressure / ambient pressure,
and the same density map with detected shock samples near the slice overlaid.
All use the actual saved XY gas plane z=0.5078125, with fixed global logarithmic
color limits and native 64-by-64 cells. No field smoothing or temporal
interpolation is applied. Axes use code units.

Filled stars mark wind sources within half a cell of the slice; the hollow
star marks the third source's projected XY position (its center is off-plane).
Dotted circles show injection spheres intersecting the slice. The third bubble
can intersect the slice even though its source center is outside that plane.

Cyan open circles show original detected surface samples satisfying
|z_sample-z_slice| <= dx/2, where dx=0.015625. They are a thin-slab overlay,
not the exact mathematical intersection of the triangle mesh and slice.
Source-overlap flags remain reported. Some density boundaries may be contacts
or source-driven features; density alone does not establish shock identity.

These are volume-density and pressure **slices**, not surface density
Sigma(x,y)=integral rho(x,y,z) dz. The compact saved run has no full 3D gas
states from which to reconstruct that integral. A true column-density movie
would require an additional simulation/export that reduces density along
the chosen depth axis at the requested times. It cannot be recovered from
the sparse shock catalogs or one gas plane alone.

This is a new view of the same existing three-bubble simulation, not a new
hydrodynamic experiment. All 17 gas slices and relevant catalog arrays were
compared exactly to their saved offline reference. Input hashes, selected
catalog row indices, image orientation, fixed color limits, encoded frame
order and playback are checked in `verification.json`. No detection,
grouping, tracking, physical measurements or original arrays are changed.

Reproduce from the repository root with NumPy/SciPy/Matplotlib/Pillow:

```sh
python -m experiments.wind_bubble.animate_gas_slices \
  --output-dir outputs/three_bubble_gas_slices_n064_repeat
```

## MP4

`three_bubble_gas_density_pressure.mp4` contains the same 17 PNG frames at
one frame per second, encoded with macOS AVFoundation as H.264. All frames
were decoded and checked against the rendered PNG order and presentation
times. The movie is 17 seconds long. `mp4_verification.json` records duration,
dimensions, compression errors and the encoding source fingerprint.
