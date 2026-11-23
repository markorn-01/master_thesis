"""Bounded saved-slice check of the lower-source collision; not a 3D gate audit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def run(root, output, event=8):
    if output.exists():
        raise ValueError('Use a fresh output directory.')
    meta = json.loads((root/'metadata.json').read_text())
    expected = np.array([[.375, .375, .5], [.625, .375, .5], [.5, .625, .375]])
    np.testing.assert_array_equal(meta['source_positions'], expected)
    if meta['num_cells'] != 64 or meta['status'] != 'complete':
        raise ValueError('This check uses the completed three-source 64-cubed case.')
    hashes = {str(root/'metadata.json'): hashlib.sha256((root/'metadata.json').read_bytes()).hexdigest()}
    rows, frames = [], []
    for index, time in enumerate(meta['actual_times']):
        paths = [root/'online'/f'{prefix}_{index:04d}.npz' for prefix in ('slice', 'surfaces')]
        for path in paths:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(paths[0]) as data:
            rho, pressure = data['density'].astype(float), data['pressure'].astype(float)
            z, dx = float(data['z_coordinate']), float(data['grid_spacing'])
            np.testing.assert_allclose(float(data['time']), time, atol=1e-7, rtol=0)
        with np.load(paths[1]) as data:
            points = data['positions'].copy()
            np.testing.assert_allclose(float(data['time']), time, atol=1e-7, rtol=0)
        x = (np.arange(len(rho))+.5)*dx
        region = (abs(x[:, None]-.5) <= .05) & (abs(x[None, :]-.375) <= .04)
        xy = (abs(points[:, 0]-.5) <= .05) & (abs(points[:, 1]-.375) <= .04)
        near = abs(points[:, 2]-z) <= dx/2
        local = abs(points[:, 2]-.5) <= .08
        temperature = pressure/rho
        gr = np.stack(np.gradient(rho, dx, edge_order=2))
        gt = np.stack(np.gradient(temperature, dx, edge_order=2))
        planar_dot = (gr*gt).sum(axis=0)
        rows.append(dict(event=index, time=time, central_slice_samples=int((xy & near).sum()),
            central_local_3d_samples=int((xy & local).sum()), central_xy_all_depth_samples=int(xy.sum()),
            planar_roi_cells=int(region.sum()), planar_alignment_positive=int((region & (planar_dot > 0)).sum()),
            planar_alignment_negative=int((region & (planar_dot < 0)).sum()),
            planar_dot_min=float(planar_dot[region].min()), planar_dot_max=float(planar_dot[region].max())))
        frames.append(dict(time=time, rho=rho, pressure=pressure, temperature=temperature,
                           x=x, z=z, dx=dx, points=points, near=near))
    output.mkdir(parents=True)
    with (output/'slice_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    frame = frames[event]
    yrow = int(np.argmin(abs(frame['x']-.375)))
    profile = [dict(x=float(x), y=float(frame['x'][yrow]), density=float(frame['rho'][i, yrow]),
                    pressure=float(frame['pressure'][i, yrow]), temperature_proxy=float(frame['temperature'][i, yrow]))
               for i, x in enumerate(frame['x'])]
    with (output/'cross_collision_profile.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(profile[0])); writer.writeheader(); writer.writerows(profile)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    fig.subplots_adjust(left=.08, right=.95, bottom=.23, top=.86, wspace=.29, hspace=.39)
    ax = axes[0, 0]
    image = ax.imshow(frame['rho'].T, origin='lower', extent=(0, 1, 0, 1),
        interpolation='nearest', cmap='magma', norm=LogNorm(frame['rho'].min(), frame['rho'].max()))
    points = frame['points'][frame['near']]
    ax.scatter(*points[:, :2].T, facecolors='none', edgecolors='#4be7ed', s=13, linewidths=.65)
    ax.add_patch(Rectangle((.45, .335), .10, .08, fill=False, edgecolor='white', linewidth=1.5))
    ax.axhline(frame['x'][yrow], color='white', linewidth=1, linestyle='--')
    ax.set(xlim=(.30, .70), ylim=(.25, .50), xlabel='x', ylabel='y',
           title='Density + near-plane detections; box = checked region')
    fig.colorbar(image, ax=ax, pad=.02, label='ρ (code units)')
    for ax, field, heading in zip((axes[0, 1], axes[1, 0], axes[1, 1]),
                                  ('rho', 'pressure', 'temperature'),
                                  ('Density', 'Pressure', 'Temperature proxy P/ρ')):
        ax.plot(frame['x'], frame[field][:, yrow], 'o-', markersize=3, color='#087f8c')
        ax.axvspan(.45, .55, color='#aabbbf', alpha=.2)
        ax.axvline(.5, color='gray', linestyle=':', linewidth=1)
        ax.set(xlim=(.43, .57), xlabel='x', ylabel=heading+' (code units)', title=heading+' across the interaction')
        selected = (frame['x'] >= .43) & (frame['x'] <= .57)
        values = frame[field][selected, yrow]
        ax.set_ylim(0, float(values.max())*1.15)
        ax.grid(alpha=.2)
    fig.suptitle(f'Lower-source collision: preliminary slice check | t = {frame["time"]:.2f}', fontsize=16)
    fig.text(.08, .065,
        f'Saved plane z = {frame["z"]:.7f}; profiles at y = {frame["x"][yrow]:.7f}. Cyan: original detections within half a cell of the plane.\n'
        'Density rises toward the central layer while P/ρ falls. The planar gradient term is negative in the checked region.\n'
        'This is not the full 3D alignment criterion: out-of-plane gradients and velocity/compression were not saved.\n'
        'No detection settings changed. This figure does not establish whether the layer contains an unresolved shock.', fontsize=9)
    fig.savefig(output/'collision_slice_check.png', dpi=150)
    plt.close(fig)
    sources = [Path(__file__), Path('astronomix/_physics_modules/_shock_finder/_shock_zones.py'),
               Path('astronomix/_physics_modules/_shock_finder/_gradients.py')]
    for path, expected_hash in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Input changed during slice inspection.')
    result = dict(saved_data_check_passed=True, exact_3d_rejection_reason_established=False,
        scope='Slice thermodynamics and exported-catalog counts only; no production 3D mask/velocity replay.',
        checked_xy_bounds=[[.45, .55], [.335, .415]], local_z_bounds=[.42, .58],
        slice_z=frame['z'], slice_half_width=frame['dx']/2,
        planar_alignment_definition='dT/dx*drho/dx + dT/dy*drho/dy; excludes the unknown z contribution',
        no_local_central_detections_from_event=7,
        all_planar_roi_cells_negative_from_event=6,
        physical_layer_classification='unresolved: do not label as contact-only or confirmed missed shock from this check',
        missing_for_full_audit=['3D density and pressure gradients', 'velocity divergence',
                               'production jump masks', 'zone and surface masks before export'],
        physics_or_thresholds_changed=False, input_sha256=hashes,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    assert all(r['central_local_3d_samples'] == 0 for r in rows[7:])
    assert all(r['planar_alignment_negative'] == r['planar_roi_cells'] for r in rows[6:])
    (output/'findings.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'README.md').write_text('''# Preliminary check of the lower two bubbles' interaction

This reads only the existing three-bubble 64-cubed slice fields and exported
shock catalogs. It changes no solver, detector, thresholds, grouping or display
inputs. `collision_slice_check.png` shows the actual slice and a cross-layer
profile at t=0.08. Counts and all-time planar diagnostics are in `slice_summary.csv`.

## Established from saved data

The central ROI is x=0.45–0.55, y=0.335–0.415, z=0.42–0.58. It contains no
exported detected samples from t=0.07 through 0.16. Earlier, at t=0.05 and
0.06, some detections exist outside the displayed half-cell slab. Thus the
late central gap is already present before grouping or mesh fitting, and is
not solely caused by the thin-slice overlay. This ROI is outside the source
exclusion spheres; it is not a count of every possible interacting front.

Across the saved central layer density rises while the temperature proxy
P/rho falls. At t>=0.06, the in-plane term
dT/dx*drho/dx + dT/dy*drho/dy is negative in all 36 ROI cells.
The production finder requires the full 3D gradient dot product to be positive,
so this is a specific reason to investigate its thermodynamic filtering.

## Not established

The saved plane omits dT/dz*drho/dz and the velocity field. We therefore cannot
reconstruct the production 3D alignment, compression, jump or surface masks.
We cannot yet attribute the gap to one exact production criterion, declare
the layer only a contact, or prove a physical shock was missed. Shock/contact
overlap and resolved gradients may matter; these remain hypotheses.

The next diagnostic is a matching-setup three-bubble replay that records the
unchanged production masks and velocity/thermodynamic profiles in this ROI,
checking the replay products against the existing run. Do not relax the
criteria merely to populate the displayed gap.

Method reference for the gradient/contact filter:
Schaal & Springel, Shock finding on a moving-mesh I, Section 2.3.2,
https://arxiv.org/html/1407.4117 . This reference explains the criterion's
purpose; it does not classify this particular simulated layer.
''')
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('sha256')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/three_bubbles_online_tracking_iwr_compgpu8_n064'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output_dir)
