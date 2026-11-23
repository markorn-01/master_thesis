"""XY/XZ movies of saved three-bubble surfaces; no new fitting or simulation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.wind_bubble.animate_group_history import load_inputs
from experiments.wind_bubble.animate_adaptive_sequence import verify_encoded_frames


def run(tables, meshes, output, fps=1.):
    if not np.isfinite(fps) or fps <= 0 or output.exists():
        raise ValueError('Require a positive finite FPS and a fresh output directory.')
    frames, rows, accounting, _, _, _, hashes, report = load_inputs(tables, meshes)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from PIL import Image, ImageDraw

    palette = list(plt.get_cmap('tab20').colors)
    if report['segments'] > len(palette):
        raise ValueError('Require a distinct color for each existing segment.')
    points = np.concatenate([f['mesh']['original_positions'] for f in frames]
                            + [frames[0]['mesh']['sources']])
    low, high = points.min(axis=0)-.04, points.max(axis=0)+.04
    output.mkdir(parents=True)
    pngs = output/'frames'
    pngs.mkdir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.5), dpi=120)
    fig.subplots_adjust(left=.065, right=.97, top=.83, bottom=.31, wspace=.18)
    title = fig.suptitle('', y=.975, fontsize=18)
    fig.text(.5, .918, 'XY and XZ orthographic projections of the full saved 3D surfaces',
             ha='center', fontsize=12)
    note = fig.text(.055, .14, '', fontsize=11)
    fig.text(.055, .025,
             'Full depth is collapsed: these are projections, not slices or column-density maps. Transparency shows overlapping groups.\n'
             'Same groups and times as the 3D demonstration; this view uses full surfaces, whereas the 3D movie uses a z ≤ 0.5 cutaway.\n'
             'Colors identify geometric continuation segments. Black stars are projected wind sources. Meshes are display geometry.',
             fontsize=10)
    headers, audit = [], []
    legend = None
    for index, frame in enumerate(frames):
        mesh = frame['mesh']
        current = [r for r in rows if r['event'] == frame['event']]
        mapping = {r['group']: r['segment_id'] for r in current}
        vertices, faces, groups = mesh['vertices'], mesh['faces'], mesh['vertex_groups']
        if len(faces) and not np.all(groups[faces] == groups[faces[:, :1]]):
            raise ValueError('A display triangle crosses group labels.')
        face_groups = groups[faces[:, 0]]
        face_order = sorted(np.unique(face_groups),
                            key=lambda g: -int((face_groups == g).sum()))
        fallback = ~np.isin(mesh['original_labels'], mesh['approved_groups'])
        event_audit = dict(event=frame['event'], time=float(mesh['time']),
                           original_samples=len(mesh['original_positions']),
                           faces=len(faces), fallback_samples=int(fallback.sum()),
                           groups=[dict(group=r['group'], segment_id=r['segment_id'],
                                        samples=r['surface_samples'], color=list(palette[r['segment_id']]))
                                   for r in current], views=[])
        for ax, coord, label, depth in zip(axes, ((0, 1), (0, 2)), ('XY', 'XZ'), ('z', 'y')):
            ax.clear()
            # Orthographic coordinate selection only: no rotation, fit, slice or cutaway.
            projected = vertices[:, coord]
            x0, y0 = low[list(coord)]
            x1, y1 = high[list(coord)]
            width = 1200
            height = round(width*(y1-y0)/(x1-x0))
            for group in face_order:
                selected = faces[face_groups == group]
                # Rasterize the union of projected triangles, then apply opacity
                # once per group. Repeated front/back triangles must not create
                # artificial wire patterns or imply depth-integrated intensity.
                mask = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(mask)
                triangles = projected[selected].copy()
                triangles[:, :, 0] = (triangles[:, :, 0]-x0)/(x1-x0)*(width-1)
                triangles[:, :, 1] = (y1-triangles[:, :, 1])/(y1-y0)*(height-1)
                for triangle in triangles:
                    draw.polygon([tuple(point) for point in triangle], fill=255)
                rgba = np.empty((height, width, 4), dtype=np.uint8)
                rgba[:, :, :3] = np.round(np.asarray(palette[mapping[int(group)]])*255).astype(np.uint8)
                rgba[:, :, 3] = np.where(np.asarray(mask) != 0, 89, 0)
                ax.imshow(rgba, origin='upper', extent=(x0, x1, y0, y1),
                          interpolation='nearest', zorder=1)
            p = mesh['original_positions']
            ax.scatter(p[fallback, coord[0]], p[fallback, coord[1]], s=5,
                       c='#858585', linewidths=0, zorder=3)
            sources = mesh['sources'][:, coord]
            ax.scatter(*sources.T, marker='*', c='black', s=85, zorder=5)
            ax.set(xlim=(low[coord[0]], high[coord[0]]),
                   ylim=(low[coord[1]], high[coord[1]]),
                   xlabel='xyz'[coord[0]], ylabel='xyz'[coord[1]],
                   title=f'{label} projection | collapse {depth} depth')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=.15)
            if not current:
                ax.text(.5, .5, 'No detected surface groups', transform=ax.transAxes,
                        ha='center', va='center', color='#666666')
            event_audit['views'].append(dict(plane=label, axes=list(coord),
                projected_faces=len(faces), source_coordinates=sources.tolist(),
                bounds=[low[list(coord)].tolist(), high[list(coord)].tolist()]))
        if legend is not None:
            legend.remove()
        handles = [Patch(facecolor=palette[r['segment_id']],
                         label=f'G{r["segment_id"]:02d}: {r["surface_samples"]:,} samples')
                   for r in sorted(current, key=lambda r: r['segment_id'])]
        if fallback.any():
            handles.append(Line2D([], [], marker='.', color='#858585', linestyle='none', label='Raw-point fallback'))
        legend = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .19),
                            ncol=4, frameon=False, fontsize=11) if handles else None
        title.set_text(f'Three-bubble surface evolution | t = {float(mesh["time"]):.2f} | event {frame["event"]:02d}')
        a = accounting[index]
        note.set_text(f'{len(mesh["original_positions"]):,} original 3D samples | {len(current)} retained groups | '
                      f'{a["unretained_samples"]:,} samples outside groups\n'
                      f'Source-overlap flags (all samples): upstream {a["all_upstream_source_overlaps"]:,}, '
                      f'downstream {a["all_downstream_source_overlaps"]:,}. Counts describe full 3D groups.')
        fig.canvas.draw()
        headers.append(np.asarray(fig.canvas.buffer_rgba())[:90, :, :3].copy())
        fig.savefig(pngs/f'event_{frame["event"]:04d}.png', dpi=120)
        audit.append(event_audit)
    plt.close(fig)

    # Save once, preserving exactly one frame for each analysis event.
    images = []
    for frame in frames:
        with Image.open(pngs/f'event_{frame["event"]:04d}.png') as im:
            images.append(im.convert('RGB'))
    movie = output/'three_bubble_xy_xz_projections.gif'
    duration = round(1000/fps)
    images[0].save(movie, save_all=True, append_images=images[1:], duration=duration,
                   loop=0, optimize=False, disposal=2)
    encoded = verify_encoded_frames(movie, headers)
    if not all(d == duration for d in encoded['frame_durations_ms']):
        raise ValueError('Encoded GIF playback differs from the requested cadence.')
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Input changed during projection rendering: {path}')
    verification = dict(passed=True, events=len(frames), fps=fps,
        group_observations=len(rows), segment_count=report['segments'], **encoded,
        projection='orthographic coordinate selection: (x,y) and (x,z)',
        rendering='Raster union of projected triangles per group, fixed alpha=89/255 applied once per group',
        full_geometry=True, cutaway=False, slab_selection=False,
        coordinates_changed=False, normals_changed=False, physical_measurements_changed=False,
        mesh_refitted=False, triangle_connectivity_changed=False, tracking_decisions_changed=False,
        palette_matches_3d_demo=True, playback_matches_3d_demo=(fps == 1.),
        overlap_interpretation='Collapsed depth can overlap unrelated structures; opacity is display-only.',
        per_event=audit, input_sha256=hashes,
        source_sha256={str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        output_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(output.rglob('*')) if p.is_file()})
    (output/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    (output/'README.md').write_text(f'''# Two-dimensional projections of the final three-bubble demonstration

Open `three_bubble_xy_xz_projections.gif`: {len(frames)} events over t = 0–0.16,
{fps:g} frame per second, {encoded['playback_seconds']:g} seconds per loop.

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
python -m experiments.wind_bubble.animate_surface_projections \\
  --output-dir outputs/three_bubble_2d_projections_n064_repeat --fps 1
```

No new hydrodynamic run or GPU is required. No temporal interpolation or
smoothing is applied between the saved observations.
''')
    print(json.dumps({key:verification[key] for key in
        ('passed','events','decoded_frames','frame_size','playback_seconds','group_observations')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', type=Path,
                        default=Path('outputs/three_bubble_online_tracking_demo_n064/tables'))
    parser.add_argument('--meshes', type=Path,
                        default=Path('outputs/three_bubble_adaptive_sequence_final_n064'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=1.)
    args = parser.parse_args()
    run(args.tables, args.meshes, args.output_dir, args.fps)
