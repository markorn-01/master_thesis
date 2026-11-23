"""Replay the saved three-bubble experiment and inspect the lower collision layer.

The detector, source prescription, analysis schedule and solver cap are unchanged.
Five full states are validation output; ordinary runs still save compact products.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from astronomix._physics_modules._shock_finder._gradients import (
    _calculate_density_gradient, _calculate_temperature_gradient,
)
from experiments.wind_bubble.audit_detection_stages import (
    StateCapture, compare_reference, selected_indices, stage_fields,
)
from experiments.wind_bubble.run_single_bubble import build_problem
from experiments.wind_bubble.run_two_bubbles import run as run_simulation
from experiments.wind_bubble.three_bubble_setup import (
    INJECTION_RADIUS, SOURCES, source_budget, weighted_budget,
)

REFERENCE = Path('outputs/three_bubbles_online_tracking_iwr_compgpu8_n064')
GATES = ('compression', 'gradient_alignment', 'pressure_jump_pass', 'temperature_jump_pass')
BOUNDS = np.array([[.45, .55], [.335, .415], [.42, .58]])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_reference(root):
    """Fail before integration if this is not the experiment under discussion."""
    meta = json.loads((root/'metadata.json').read_text())
    if meta.get('status') != 'complete' or meta.get('num_cells') != 64:
        raise ValueError('Require the completed 64-cubed three-bubble reference.')
    if not np.array_equal(meta.get('source_positions'), SOURCES):
        raise ValueError('Reference must use the original three-source geometry.')
    if not np.array_equal(meta.get('source_strengths'), [1., 1., 1.]):
        raise ValueError('Reference must use three equal source strengths.')
    times = np.asarray(meta['requested_times'], np.float32)
    if (times.shape != (17,) or not np.all(np.diff(times) > 0)
            or not np.isclose(meta['t_end'], .16, rtol=0, atol=1e-9)):
        raise ValueError('Require the original 17-event schedule ending at 0.16.')
    if meta['injection_radius'] != INJECTION_RADIUS:
        raise ValueError('Reference injection radius differs.')
    # Remote absolute paths are mapped to this checkout, without changing files.
    fingerprints = {str(root/'metadata.json'): digest(root/'metadata.json')}
    for name, expected in meta['source_sha256'].items():
        path = Path(name.split('master_thesis/')[-1])
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Reference source changed or missing: {path}')
        fingerprints[str(path)] = expected
    budget = Path(meta['source_budget_reference']['path'])
    if digest(budget) != meta['source_budget_reference']['sha256']:
        raise ValueError('Source-budget reference has changed.')
    fingerprints[str(budget)] = digest(budget)
    for index in range(len(times)):
        for prefix in ('surfaces', 'histograms', 'slice'):
            path = root/'online'/f'{prefix}_{index:04d}.npz'
            fingerprints[str(path)] = digest(path)
    return meta, fingerprints


def collision_fields(state, config, registry, helper):
    fields = stage_fields(state, config, registry, helper)
    rho, pressure = state[registry.density_index], state[registry.pressure_index]
    gt = _calculate_temperature_gradient(pressure, rho, config)
    gr = _calculate_density_gradient(rho, config)
    components = gt*gr
    fields.update(gradient_dot=jnp.sum(components, axis=0),
        gradient_dot_x=components[0], gradient_dot_y=components[1], gradient_dot_z=components[2])
    return fields


def validate_fields(fields):
    """Cross-check diagnostic decomposition against the actual production masks."""
    np.testing.assert_array_equal(fields['gradient_alignment'], fields['gradient_dot'] > 0)
    np.testing.assert_array_equal(fields['minimum_jump'],
                                  fields['pressure_jump_pass'] & fields['temperature_jump_pass'])
    np.testing.assert_array_equal(fields['shock_zone'],
                                  np.logical_and.reduce([fields[k] for k in GATES]))
    if (fields['shock_surface'] & ~fields['shock_zone']).any():
        raise AssertionError('Production surface must be a subset of the zone.')


def region_counts(fields, roi, exported):
    """Independent pass/fail counts and mutually exclusive joint patterns."""
    row = dict(cells=int(roi.sum()))
    for key in (*GATES, 'minimum_jump', 'shock_zone', 'shock_surface'):
        name = key.removesuffix('_pass')
        row[name+'_pass'] = int((roi & fields[key]).sum())
    for key in GATES:
        name = key.removesuffix('_pass')
        row[name+'_fail'] = int((roi & ~fields[key]).sum())
        others = np.logical_and.reduce([fields[k] for k in GATES if k != key])
        row[name+'_only_failure'] = int((roi & ~fields[key] & others).sum())
    row['zone_not_surface'] = int((roi & fields['shock_zone'] & ~fields['shock_surface']).sum())
    row['surface_removed_by_export'] = int((roi & fields['shock_surface'] & ~exported).sum())
    row['exported_surface'] = int((roi & exported).sum())
    pattern = sum(fields[k].astype(np.uint8)*(1 << i) for i, k in enumerate(GATES))
    for code in range(16):
        row[f'pass_pattern_{code:04b}'] = int((roi & (pattern == code)).sum())
    if sum(row[f'pass_pattern_{c:04b}'] for c in range(16)) != row['cells']:
        raise AssertionError('Joint patterns do not account for every ROI cell.')
    return row


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_saved_states(root, indices, meta):
    setup = argparse.Namespace(num_cells=meta['num_cells'], num_snapshots=17,
        t_end=meta['t_end'], num_injection_cells=meta['num_cells']//16, plasma_beta=None)
    _, config, _, registry, helper = build_problem(setup)
    compiled = jax.jit(lambda state: collision_fields(state, config, registry, helper))
    centers = np.asarray(helper.geometric_centers)
    dx = float(config.grid_spacing)
    x = centers[:, 0, 0, 0]
    iy, iz = int(np.argmin(abs(x-.375))), meta['num_cells']//2
    roi = ((centers >= BOUNDS[:, 0]) & (centers <= BOUNDS[:, 1])).all(axis=-1)
    plane = np.zeros(roi.shape, bool)
    plane[:, :, iz] = True
    # A wider crop retains both flanks rather than just the highest-density center.
    crop_bounds = np.array([[.40, .60], [.28, .47], [.42, .58]])
    crop = []
    for axis in range(3):
        cells = np.flatnonzero((x >= crop_bounds[axis, 0]) & (x <= crop_bounds[axis, 1]))
        crop.append(slice(int(cells[0]), int(cells[-1])+1))
    crop = tuple(crop)
    rows, profiles, snapshots, state_hashes = [], [], [], {}
    for index in indices:
        path = root/'audit'/f'state_{index:04d}.npy'
        state = np.load(path, allow_pickle=False)
        if state.shape != (registry.num_vars,)+roi.shape or not np.isfinite(state).all():
            raise ValueError(f'Unexpected physical state: {path}')
        fields = {k:np.asarray(v) for k, v in compiled(jnp.asarray(state)).items()}
        validate_fields(fields)
        positions = centers+dx*fields['surface_offsets'][..., None]*fields['direction']
        distances = np.linalg.norm(positions[..., None, :]-np.asarray(SOURCES), axis=-1)
        exported = (fields['shock_surface'] & np.isfinite(distances).all(axis=-1)
                    & (distances > INJECTION_RADIUS).all(axis=-1))
        with np.load(root/'online'/f'surfaces_{index:04d}.npz') as catalog:
            np.testing.assert_array_equal(np.argwhere(exported), catalog['grid_indices'])
            np.testing.assert_array_equal(positions[exported], catalog['positions'])
            time = float(catalog['time'])
        np.testing.assert_allclose(time, meta['requested_times'][index], rtol=0, atol=1e-6)
        refined_roi = ((positions >= BOUNDS[:, 0]) & (positions <= BOUNDS[:, 1])).all(axis=-1)
        for name, mask in (('central_volume_grid_centers', roi), ('central_slice_grid_centers', roi & plane)):
            row = dict(event=index, time=time, region=name, **region_counts(fields, mask, exported))
            row['exported_refined_positions_in_volume'] = int((exported & refined_roi).sum())
            rows.append(row)
        rho, pressure = state[registry.density_index], state[registry.pressure_index]
        velocity = state[[registry.velocity_index.x, registry.velocity_index.y, registry.velocity_index.z]]
        temperature = pressure/rho
        cs = np.sqrt(meta['gamma']*pressure/rho)
        entropy = pressure/rho**meta['gamma']
        for ix in range(len(x)):
            cell = (ix, iy, iz)
            row = dict(event=index, time=time, x=float(x[ix]), y=float(x[iy]), z=float(x[iz]),
                density=float(rho[cell]), pressure=float(pressure[cell]), temperature_proxy=float(temperature[cell]),
                velocity_x=float(velocity[(0,)+cell]), velocity_y=float(velocity[(1,)+cell]),
                velocity_z=float(velocity[(2,)+cell]), sound_speed=float(cs[cell]), entropy_proxy=float(entropy[cell]))
            for key in (*GATES, 'shock_zone', 'shock_surface', 'divergence', 'gradient_dot',
                        'gradient_dot_x', 'gradient_dot_y', 'gradient_dot_z',
                        'log_pressure_jump', 'log_temperature_jump'):
                row[key] = fields[key][cell].item()
            profiles.append(row)
        np.savez_compressed(root/'audit'/f'collision_crop_{index:04d}.npz',
            **{k:v[crop] for k,v in fields.items()}, state=state[(slice(None),)+crop],
            centers=centers[crop], exported_surface=exported[crop], central_roi=roi[crop],
            time=time, grid_spacing=dx, density_index=registry.density_index,
            pressure_index=registry.pressure_index, velocity_indices=np.array([
                registry.velocity_index.x, registry.velocity_index.y, registry.velocity_index.z]))
        snapshots.append(dict(index=index, time=time, fields={k:v[:, :, iz] for k,v in fields.items()},
                              rho=rho[:, :, iz], pressure=pressure[:, :, iz], velocity=velocity[:, :, :, iz]))
        state_hashes[str(path)] = digest(path)
        central = rows[-1]
        print(f't={time:.2f}: slice ROI {central["cells"]} cells; '
              f'compression={central["compression_pass"]}, alignment={central["gradient_alignment_pass"]}, '
              f'P jump={central["pressure_jump_pass"]}, T jump={central["temperature_jump_pass"]}, '
              f'zone={central["shock_zone_pass"]}, surface={central["shock_surface_pass"]}', flush=True)
    write_csv(root/'audit'/'collision_stage_counts.csv', rows)
    write_csv(root/'audit'/'cross_collision_profiles.csv', profiles)
    make_figures(root/'audit', snapshots, profiles, x, float(x[iz]), float(x[iy]), dx)
    return rows, state_hashes


def make_figures(output, snapshots, profiles, x, z, y, dx):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    for snap in snapshots:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=140)
        fig.subplots_adjust(left=.07, right=.98, top=.85, bottom=.16, hspace=.42, wspace=.24)
        keys = (*GATES, 'shock_zone', 'shock_surface')
        titles = ('Compression: ∇·v < 0', 'Alignment: ∇T·∇ρ > 0',
                  'Local pressure jump ≥ threshold', 'Local temperature jump ≥ threshold',
                  'Shock zone: all four tests', 'Selected surface before export')
        for ax, key, title in zip(axes.flat, keys, titles):
            ax.imshow(snap['fields'][key].T, origin='lower', extent=(0,1,0,1),
                      cmap=ListedColormap(['#efefef', '#087f8c']), vmin=0, vmax=1, interpolation='nearest')
            ax.contour(x, x, snap['rho'].T, levels=[.1, .5, 1., 2.], colors='#a44b27', linewidths=.5)
            ax.add_patch(Rectangle((.45,.335),.10,.08, fill=False, edgecolor='black', linewidth=1))
            ax.set(xlim=(.40,.60), ylim=(.28,.47), xlabel='x', ylabel='y', title=title)
        fig.suptitle(f'Unchanged production detector | lower collision | t = {snap["time"]:.2f}', fontsize=15)
        fig.text(.07,.035, f'z = {z:.7f}; teal = passes / selected, gray = fails. Orange contours: density.\n'
                 'Box: counted central region. Tests can fail simultaneously; surface selection follows shock-zone detection.\n'
                 'A failed test identifies a detector decision, not by itself the physical type of the layer.', fontsize=9)
        fig.savefig(output/f'production_masks_{snap["index"]:04d}.png')
        plt.close(fig)
    # One profile figure per captured state; retain signs and units without a shock-frame Mach claim.
    for snap in snapshots:
        data = [r for r in profiles if r['event'] == snap['index']]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=140)
        fig.subplots_adjust(left=.07, right=.97, top=.85, bottom=.19, hspace=.45, wspace=.36)
        for ax, key, title in zip(axes.flat,
                ('density','pressure','temperature_proxy','velocity_x','divergence','gradient_dot'),
                ('Density ρ','Pressure P','Temperature proxy P/ρ','x velocity (lab frame)','3D velocity divergence','3D ∇T·∇ρ')):
            ax.plot(x, [r[key] for r in data], 'o-', color='#087f8c', markersize=3, linewidth=1)
            ax.axvspan(.45,.55,color='#b4bfc1',alpha=.2)
            ax.axvline(.5,color='gray',linestyle=':',linewidth=1)
            ax.set(xlim=(.40,.60), xlabel='x', ylabel=title, title=title)
            ax.grid(alpha=.2)
            if key in ('velocity_x','divergence','gradient_dot'):ax.axhline(0,color='gray',linewidth=.7)
        fig.suptitle(f'Cross-layer thermodynamics and flow | t = {snap["time"]:.2f}', fontsize=15)
        fig.text(.07,.045, f'y = {y:.7f}, z = {z:.7f}; grid spacing = {dx:.7f}; code units.\n'
                 'Positive vx on the left and negative vx on the right indicate approach toward the center.\n'
                 'The profiles show resolved grid values. They do not assume a stationary shock or infer shock Mach from lab velocity.',fontsize=9)
        fig.savefig(output/f'collision_profiles_{snap["index"]:04d}.png')
        plt.close(fig)


def run(args):
    meta, inputs = check_reference(args.reference_run)
    times = np.asarray(meta['requested_times'], np.float32)
    indices = selected_indices(args.capture_events, len(times))
    if not args.audit_only:
        if args.output_dir.exists():
            raise ValueError('Use a fresh output directory; existing data are never overwritten.')
        capture = StateCapture(args.output_dir/'audit', times)
        simulation_args = argparse.Namespace(num_cells=meta['num_cells'], num_analysis=len(times),
            t_end=meta['t_end'], source_strengths=meta['source_strengths'],
            match_source_budget=Path(meta['source_budget_reference']['path']),
            verify_offline=False, require_gpu=args.require_gpu, track_shocks=False, output_dir=args.output_dir)
        run_simulation(simulation_args, state_observer=capture.observer(indices), analysis_times=times,
            solver_dt_max=meta['solver_dt_max'], source_positions=SOURCES,
            budget_builder=source_budget, strength_builder=weighted_budget,
            extra_source_files=[Path(__file__), Path('experiments/wind_bubble/three_bubble_setup.py')])
        if sorted(capture.saved) != indices:
            raise AssertionError('Selected physical states were not all captured.')
    elif args.require_gpu and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU requested but unavailable.')
    rows, hashes = audit_saved_states(args.output_dir, indices, meta)
    try:
        reference = compare_reference(args.output_dir, args.reference_run)
    except AssertionError as error:
        reference = dict(passed=False, error=str(error), scope='Strict replay-product comparison failed; '
                         'do not attribute these mask counts to the original run.')
    for path, expected in inputs.items():
        if digest(path) != expected:
            raise AssertionError(f'Input changed during replay: {path}')
    gamma, m2 = 5/3, 1.3**2
    pratio = (2*gamma*m2-(gamma-1))/(gamma+1)
    tratio = pratio*((gamma-1)*m2+2)/((gamma+1)*m2)
    sources = [Path(__file__), Path('experiments/wind_bubble/audit_detection_stages.py')]
    sources += sorted(Path('astronomix/_physics_modules/_shock_finder').glob('*.py'))
    result = dict(passed=reference['passed'], captured_events=indices, capture_times=times[indices].tolist(),
        reference_replay=reference, production_mask_decomposition_matches=True,
        exported_indices_and_positions_match=True, physics_or_thresholds_changed=False,
        schedule_changed=False, grouping_and_meshing_used=False, backend=jax.default_backend(),
        roi_bounds=BOUNDS.tolist(), roi_definition='grid-cell centers; refined-position counts recorded separately',
        pattern_bits_low_to_high=list(GATES), thresholds=dict(mach_min=1.3, pressure_ratio=pratio, temperature_ratio=tratio),
        full_state_bytes=sum(Path(p).stat().st_size for p in hashes), state_sha256=hashes,
        input_sha256=inputs, source_sha256={str(p):digest(p) for p in sources},
        central_slice_counts=[r for r in rows if r['region']=='central_slice_grid_centers'],
        physical_layer_classification='Requires interpretation of resolved profiles; masks alone do not prove shock/contact identity.',
        scope='Selected-state production-mask audit with replay-product comparison, not independent physical validation or full-state equivalence.')
    (args.output_dir/'audit'/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','captured_events','reference_replay','physics_or_thresholds_changed')},indent=2))
    if not reference['passed']:
        raise RuntimeError('Replay products differ: see audit/verification.json; do not use as the original-run explanation.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, default=REFERENCE)
    parser.add_argument('--capture-events', nargs='+', type=int, default=[4,6,8,12,16])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--audit-only', action='store_true', help='Analyze previously captured states without another simulation.')
    run(parser.parse_args())
