# Gas structure with detected shock samples

`gas_shock_context.png` shows one saved 128³ two-bubble event at t=0.08:

1. Gas density: dilute interiors and swept-up denser shells.
2. Temperature proxy P/rho: hot bubble interiors compared with ambient gas.
3. Exactly the same temperature view, overlaid with detected shock samples.

The background is the saved xy cell-center plane at z=0.50390625, rather
than an interpolated slice at z=0.5. The temperature proxy is calculated
cell by cell from saved pressure and density; it is in code units, not
Kelvin. Both temperature panels have identical logarithmic color limits.
Arrays retain x,y order and are transposed only for imshow with origin
lower and extent [0,1]×[0,1]. Nearest-pixel display adds no field smoothing.

Cyan circles mark all 248 original detected samples within half a cell
(0.00390625) of the saved plane. This is a thin-slab projection, not exact
surface-plane intersections. There are 11,424 catalog samples in the full
3D event. Samples outside the slab are not shown in this view. Source
stars are projected source centers; dotted circles are intersections of
the prescribed injection spheres with the plane. Colors describe gas
fields; the cyan overlay describes shock-finder detections, not Mach.

The volume-rendering reference motivated putting gas structure beside
shock measurements. This first step uses existing 2D outputs. It does not
reproduce that reference's SPH smoothing, transfer function or 3D rendering.
The sparse shock catalogs and saved slices do not contain a full gas volume.
The visual correspondence is explanatory, not independent physical validation
of shock finding or evidence for a confirmed physical merger.

`display_data.npz` preserves density, pressure, derived proxy and exact
overlay positions/indices. `verification.json` records source/input hashes,
slice and catalog time agreement, limits and selected counts. Saved fields
and positions are read back; the proxy and slab selection are recomputed.
The image is decoded and visually inspected. Four tests check grid order,
proxy calculation, slab boundaries, invalid fields and shape mismatch.
No gas evolution, shock detection, grouping or tracking decision changed.

Reproduce with NumPy and Matplotlib (Pillow for image readback), using a new
output directory:

```sh
python -m experiments.wind_bubble.plot_gas_shock_context --event-index 8 --output-dir outputs/two_bubbles_gas_shock_context_n128_event0008
python -m unittest tests.experiments.test_gas_shock_context
```

No GPU, new simulation or full snapshots are required for this figure.
