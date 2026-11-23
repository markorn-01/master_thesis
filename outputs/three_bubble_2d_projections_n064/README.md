# Two-dimensional projections of the final three-bubble demonstration

Share `three_bubble_xy_xz_projections.mp4` (H.264), or open
`three_bubble_xy_xz_projections.gif`: 17 events over t = 0–0.16,
1 frame per second, 17 seconds per loop.

Left: full XY projection, looking along z. Right: full XZ projection, looking
along y. The source is the same actual-callback three-bubble run used in
`outputs/three_bubble_online_tracking_demo_n064/animation/three_bubble_group_evolution.gif`.
The existing final checked triangle meshes are reused without refitting.

Both views collapse the entire depth onto the plane. There is no central-slice,
thin-slab selection or z cutaway. The original 3D movie uses a z <= 0.5 cutaway;
this companion instead shows the full saved surfaces transparently. It is not
a gas-density slice, column-density integral, physical emission map, or exact
intersection of the shock mesh with a plane. Overlap in a projection does not
establish physical contact, connectivity or ancestry.

Colors use the same tab20 continuation-segment palette as the 3D demonstration;
combinations start new segments. Black stars are projected wind-source locations.
Gray points, if present, preserve original samples outside approved mesh groups.
Each group's projected triangles are rasterized into one footprint at 1200
pixels across the fixed x extent. Opacity is applied once per footprint,
avoiding wire patterns from overlapping front/back triangle rasterization.
The largest mesh groups are drawn first and smaller groups afterward with
fixed alpha=89/255. This display order improves visibility and is not depth sorting.
Opacity has no physical meaning. Axes and bounds are fixed over all events.
Legend counts refer to entire 3D groups, not projected areas or triangle counts.
Source-overlap sampling flags remain displayed.

The 17 original analysis times, 45 group observations and existing segment
assignments are checked using the original demonstration's table/mesh joins.
All input/source manifests are verified on load and again after rendering.
Every saved triangle is projected by selecting its x/y or x/z coordinates;
no vertices, faces, physical measurements or tracking decisions are changed.
The GIF is decoded and every frame header is matched against the expected
event order; frame count, size and duration are checked. See `verification.json`.
`frames/` preserves each PNG, including the empty initial state, for readback
and future movie encoding. These checks concern display consistency, not
independent physical validation.

Reproduce from the repository root with NumPy/SciPy/Matplotlib/Pillow:

```sh
python -m experiments.wind_bubble.animate_surface_projections \
  --output-dir outputs/three_bubble_2d_projections_n064_repeat --fps 1
```

No new hydrodynamic run or GPU is required. No temporal interpolation or
smoothing is applied between the saved observations.

## MP4 companion

The MP4 uses the same 17 rendered PNGs at one frame per second, for 17 seconds.
It was encoded locally with macOS AVFoundation; no Python environment or system
packages were changed. The saved `mp4_encoder.m` is the encoding/readback source
for this one-frame-per-second export. All 17 frames were decoded again and
matched to the expected PNG header and presentation time. `mp4_verification.json`
records frame order, duration, dimensions, source checksums and RGB compression
errors. The maximum full-frame mean absolute RGB error is below 0.52 on the
0–255 channel scale. No tracking or physical quantities are derived from video
pixels. The GIF and MP4 are display products of the same existing data.

On macOS with command-line developer tools, compile the included source with
Foundation, AVFoundation, AppKit, ImageIO, CoreVideo, CoreMedia and CoreGraphics
frameworks. Its arguments are the PNG folder, a fresh MP4 output path, a fresh
readback folder and the number of frames (17). The GIF renderer itself remains
platform-independent.
