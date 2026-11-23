"""Show saved XY gas-density/pressure slices beside near-plane shock samples."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.wind_bubble.animate_adaptive_sequence import verify_encoded_frames
from experiments.wind_bubble.plot_gas_shock_context import prepare


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frames(root):
    meta = json.loads((root/'metadata.json').read_text())
    equivalence = json.loads((root/'equivalence.json').read_text())
    if meta['status'] != 'complete' or not equivalence['passed']:
        raise ValueError('Require a complete run with passing recorded equivalence.')
    if meta['domain'] != [0., 1.]:
        raise ValueError('This viewer uses the saved unit Cartesian domain.')
    hashes = {str(p): sha(p) for p in (root/'metadata.json', root/'equivalence.json')}
    frames = []
    for index, time in enumerate(meta['actual_times']):
        paths = [root/'online'/f'{prefix}_{index:04d}.npz' for prefix in ('slice', 'surfaces')]
        for p in paths:
            hashes[str(p)] = sha(p)
        with np.load(paths[0]) as data:
            gas = {k: data[k].copy() for k in data.files}
        with np.load(paths[1]) as data:
            catalog = {k: data[k].copy() for k in ('positions', 'source_positions',
                       'injection_radius', 'grid_spacing', 'grid_shape', 'time',
                       'upstream_inside_any_source', 'downstream_inside_any_source')}
        density, pressure = gas['density'], gas['pressure']
        z, dx = float(gas['z_coordinate']), float(gas['grid_spacing'])
        _, rows = prepare(density, pressure, catalog['positions'], z, dx)
        independent = np.array([i for i, point in enumerate(catalog['positions'])
                                if z-dx/2 <= float(point[2]) <= z+dx/2], dtype=int)
        np.testing.assert_array_equal(rows, independent)
        np.testing.assert_array_equal(density.shape, catalog['grid_shape'][:2])
        np.testing.assert_allclose([float(gas['time']), float(catalog['time'])], time, rtol=0, atol=1e-7)
        if dx != float(catalog['grid_spacing']) or not np.isclose(len(density)*dx, 1.):
            raise ValueError('Gas and catalog grid spacing differ.')
        if not np.isclose(z, (density.shape[0]//2+.5)*dx, rtol=0, atol=1e-12):
            raise ValueError('Unexpected saved slice plane.')
        # Recheck the copied raw arrays against the stored offline reference.
        for prefix, original in [('slice', gas), ('surfaces', catalog)]:
            reference = root/'offline_reference'/f'{prefix}_{index:04d}.npz'
            hashes[str(reference)] = sha(reference)
            with np.load(reference) as data:
                for key, value in original.items():
                    np.testing.assert_array_equal(value, data[key])
        frames.append(dict(event=index, time=time, density=density, pressure=pressure,
            z=z, dx=dx, positions=catalog['positions'], sources=catalog['source_positions'],
            injection_radius=float(catalog['injection_radius']), rows=rows,
            upstream_overlap=int(catalog['upstream_inside_any_source'][rows].sum()),
            downstream_overlap=int(catalog['downstream_inside_any_source'][rows].sum())))
    if not frames or not np.all(np.diff([f['time'] for f in frames]) > 0):
        raise ValueError('Require increasing analysis times.')
    return frames, meta, hashes


def run(root, output):
    if output.exists():
        raise ValueError('Use a fresh output directory.')
    frames, meta, hashes = load_frames(root)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle
    from PIL import Image

    rho0, p0 = float(meta['ambient_density']), float(meta['ambient_pressure'])
    if rho0 <= 0 or p0 <= 0:
        raise ValueError('Require positive ambient normalization.')
    limits = {name: [min(float(f[field].min()/scale) for f in frames),
                     max(float(f[field].max()/scale) for f in frames)]
              for name, field, scale in [('density', 'density', rho0), ('pressure', 'pressure', p0)]}
    norms = {key: LogNorm(*values) for key, values in limits.items()}
    output.mkdir(parents=True)
    pngs = output/'frames'
    pngs.mkdir()

    def sources(ax, frame):
        for index, source in enumerate(frame['sources']):
            on = abs(source[2]-frame['z']) <= frame['dx']/2+1e-12
            ax.scatter(*source[:2], marker='*', s=90, facecolors='white' if on else 'none',
                       edgecolors='black' if on else 'white', linewidths=1., zorder=5)
            dz = source[2]-frame['z']
            if abs(dz) < frame['injection_radius']:
                ax.add_patch(Circle(source[:2], np.sqrt(frame['injection_radius']**2-dz**2),
                    fill=False, edgecolor='white', linewidth=.8, linestyle=':', alpha=.8))

    fig, axes = plt.subplots(1, 3, figsize=(17, 7.5), dpi=120)
    fig.subplots_adjust(left=.04, right=.935, top=.80, bottom=.26, wspace=.27)
    title = fig.suptitle('', y=.97, fontsize=18)
    fig.text(.5, .91, f'Actual gas slice at z = {frames[0]["z"]:.7f} | fixed logarithmic colors at all times',
             ha='center', fontsize=12)
    images = []
    for ax, name, heading in zip(axes, ('density', 'pressure', 'density'),
            ('Gas density: ρ / ρambient', 'Gas pressure: P / Pambient', 'Density + detected shock samples')):
        im = ax.imshow(np.ones_like(frames[0]['density']).T, origin='lower', extent=(0, 1, 0, 1),
                       interpolation='nearest', norm=norms[name], cmap='magma' if name == 'density' else 'inferno')
        images.append(im)
        ax.set(xlim=(.1, .9), ylim=(.1, .9), aspect='equal', xlabel='x', ylabel='y', title=heading)
        cb = fig.colorbar(im, ax=ax, pad=.025, fraction=.045)
        cb.ax.tick_params(labelsize=9)
        cb.set_label('ρ / ρambient' if name == 'density' else 'P / Pambient', fontsize=10)
    note = fig.text(.035, .16, '', fontsize=11)
    fig.text(.035, .035,
        'Cyan circles: detected shock samples within half a cell of the slice; this is not an exact mesh–plane intersection.\n'
        'Filled stars: near-slice sources. Hollow star: off-plane source. Dotted circles: injection-region intersections.\n'
        'Native 64² gas cells, no field smoothing. These are volume-density/pressure slices, not integrated surface-density Σ maps.', fontsize=10)
    headers, audit = [], []
    overlays = []
    for frame in frames:
        for artist in overlays:
            artist.remove()
        overlays = []
        for ax, im, field, scale in zip(axes, images, ('density', 'pressure', 'density'), (rho0, p0, rho0)):
            im.set_data(frame[field].astype(float).T/scale)
            np.testing.assert_array_equal(np.asarray(im.get_array()), frame[field].astype(float).T/scale)
            before = set(ax.get_children())
            sources(ax, frame)
            overlays.extend(a for a in ax.get_children() if a not in before)
        selected = frame['positions'][frame['rows']]
        overlays.append(axes[2].scatter(selected[:, 0], selected[:, 1], s=17,
            facecolors='none', edgecolors='#4be7ed', linewidths=.75, zorder=4))
        title.set_text(f'Three-bubble gas structure and shock fronts | t = {frame["time"]:.2f} | event {frame["event"]:02d}')
        note.set_text(f'{len(selected)} near-plane detected samples shown; {len(frame["positions"]):,} samples in the complete 3D catalog.\n'
                      f'Near-plane endpoint source-overlap flags: upstream {frame["upstream_overlap"]}, downstream {frame["downstream_overlap"]}. '
                      'Density boundaries alone do not establish a shock.')
        fig.canvas.draw()
        headers.append(np.asarray(fig.canvas.buffer_rgba())[:90, :, :3].copy())
        fig.savefig(pngs/f'event_{frame["event"]:04d}.png', dpi=120)
        audit.append(dict(event=frame['event'], time=frame['time'], z=frame['z'], dx=frame['dx'],
            selected_catalog_rows=frame['rows'].tolist(), selected_samples=len(selected),
            total_samples=len(frame['positions']), density_range=[float(frame['density'].min()), float(frame['density'].max())],
            pressure_range=[float(frame['pressure'].min()), float(frame['pressure'].max())]))
    plt.close(fig)
    movie = output/'three_bubble_gas_density_pressure.gif'
    pictures = []
    for frame in frames:
        with Image.open(pngs/f'event_{frame["event"]:04d}.png') as im:
            pictures.append(im.convert('RGB'))
    pictures[0].save(movie, save_all=True, append_images=pictures[1:], duration=1000,
                     disposal=2, loop=0, optimize=False)
    encoded = verify_encoded_frames(movie, headers)
    if encoded['frame_durations_ms'] != [1000]*len(frames):
        raise ValueError('Encoded cadence differs from one analysis event per second.')

    # A density-only sequence for inspecting the initial interaction without overlay clutter.
    contact_events = [4, 5, 6, 7, 8, 10]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 9), dpi=150)
    fig.subplots_adjust(left=.06, right=.88, bottom=.13, top=.88, hspace=.29, wspace=.20)
    for ax, index in zip(axes.flat, contact_events):
        frame = frames[index]
        im = ax.imshow(frame['density'].T/rho0, origin='lower', extent=(0, 1, 0, 1),
                       interpolation='nearest', cmap='magma', norm=norms['density'])
        sources(ax, frame)
        ax.set(xlim=(.15, .85), ylim=(.15, .85), aspect='equal', xlabel='x', ylabel='y', title=f't = {frame["time"]:.2f}')
    cax = fig.add_axes([.91, .22, .018, .57])
    fig.colorbar(im, cax=cax, label='Gas density / ambient density (log colors)')
    fig.suptitle('Three-bubble density evolution around contact', y=.97, fontsize=17)
    fig.text(.06, .035,
        f'Same XY slice, z = {frames[0]["z"]:.7f}, and same color scale in every panel. Native cells; no field smoothing.\n'
        'Filled stars: near-slice wind sources. Hollow star: off-plane source. White dotted circles: source-region intersections.\n'
        'Density alone does not classify a shock; use the companion shock-sample overlay for comparison. This is not column density.', fontsize=10)
    still = output/'density_contact_sequence.png'
    fig.savefig(still, dpi=150)
    plt.close(fig)
    for name, expected in hashes.items():
        if sha(Path(name)) != expected:
            raise ValueError(f'Input changed during rendering: {name}')
    verification = dict(passed=True, events=len(frames), **encoded, per_event=audit,
        fields='Native rho and P in saved XY plane; normalized only by fixed ambient values.',
        column_density_computed=False, full_3d_gas_states_available=False,
        interpolation='nearest; no spatial or temporal smoothing',
        overlay_definition='abs(z_sample-z_slice) <= dx/2; original detected positions',
        fixed_color_limits=limits, xy_array_transposed_only_for_display=True,
        offline_reference_fields_and_selected_catalog_arrays_exact=True,
        physics_changed=False, shock_detection_changed=False, tracking_changed=False,
        input_sha256=hashes, source_sha256={str(Path(__file__)):sha(Path(__file__))},
        output_sha256={str(p):sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    (output/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    (output/'README.md').write_text('''# Gas slices showing the colliding fronts

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
python -m experiments.wind_bubble.animate_gas_slices \\
  --output-dir outputs/three_bubble_gas_slices_n064_repeat
```
''')
    print(json.dumps({k:verification[k] for k in ('passed', 'events', 'decoded_frames', 'playback_seconds', 'frame_size')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/three_bubbles_online_tracking_iwr_compgpu8_n064'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output_dir)
