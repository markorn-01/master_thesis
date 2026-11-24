"""Read-only original/transition-aware finder comparison on captured bubble gas.

The target is the lower pair's interaction volume, including both flanks.
Pressure features are location cues, not independently certified shock fronts.
No time integration, grouping, meshing, or physical threshold is changed.
"""
import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import find_peaks

from astronomix import (CARTESIAN, SimulationConfig, construct_primitive_state,
    finalize_config, get_helper_data, get_registered_variables)
from astronomix.option_classes.simulation_config import StaticFloatVector
from astronomix._physics_modules._shock_finder._gradients import (
    _calculate_shock_direction, _calculate_density_gradient,
    _calculate_temperature_gradient, _calculate_velocity_divergence)
from astronomix._physics_modules._shock_finder._shock_zones import (
    _shock_zone_criterion_minimum_mach, _make_interior_mask,
    _transition_jump_candidates, get_profile_aware_post_pre_shock_values,
    _sample_field_at_distances)
from astronomix._physics_modules._shock_finder._shock_surface import (
    identify_shock_surface, calculate_shock_surface_offsets)
from astronomix._physics_modules._shock_finder._shock_mach import _calculate_mach_at_surface
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import find_shocks_pfrommer
from experiments.shock.run_evolved_collision import sha, write_csv
from experiments.wind_bubble.three_bubble_setup import SOURCES, INJECTION_RADIUS


ROOTS = (Path('outputs/three_bubble_collision_detection_audit_iwr_compgpu8_n064'),
         Path('outputs/three_bubble_collision_resolution_iwr_compgpu8_n128'))
WINDOW = np.array([[.40, .60], [.28, .47], [.42, .58]])
CENTRAL = np.array([[.45, .55], [.335, .415], [.42, .58]])
EVENTS = (4, 6, 8, 12, 16)
PLOT_EVENTS = (4, 8, 16)
HALO = 24
HALO_FLOAT_RTOL = 1e-5
HALO_FLOAT_ATOL = 2e-5


def validate_archive(root, events=EVENTS):
    """Validate frozen gas, not equality of historical and corrected sources."""
    root = Path(root)
    meta_path, report_path = root/'metadata.json', root/'audit/verification.json'
    meta, report = json.loads(meta_path.read_text()), json.loads(report_path.read_text())
    if (meta.get('status') != 'complete' or not report.get('passed')
            or meta.get('backend') != 'gpu' or meta['num_cells'] not in (64, 128)):
        raise ValueError('Require the completed, audited 64/128 cubed GPU captures.')
    np.testing.assert_array_equal(meta['source_positions'], SOURCES)
    np.testing.assert_array_equal(meta['source_strengths'], [1., 1., 1.])
    if meta['gamma'] != 5/3 or meta['injection_radius'] != INJECTION_RADIUS:
        raise ValueError('Source radius or ideal-gas prescription differs.')
    inputs = {str(meta_path):sha(meta_path), str(report_path):sha(report_path)}
    historical = {Path(k).name:v for k, v in report['state_sha256'].items()}
    source_changes = {}
    for name, old in report.get('source_sha256', {}).items():
        path = Path(name.split('master_thesis/')[-1])
        if path.is_file() and sha(path) != old:
            source_changes[str(path)] = dict(historical=old, current=sha(path))
    for event in events:
        path = root/'audit'/f'state_{event:04d}.npy'
        if event not in report['captured_events'] or sha(path) != historical.get(path.name):
            raise ValueError(f'Captured state is absent or changed: {path}')
        state = np.load(path, mmap_mode='r', allow_pickle=False)
        n = meta['num_cells']
        if (state.shape != (5, n, n, n) or not np.isfinite(state).all()
                or (state[[0, 4]] <= 0).any()):
            raise ValueError(f'Invalid captured primitive gas: {path}')
        inputs[str(path)] = historical[path.name]
        for relative in (f'audit/collision_crop_{event:04d}.npz', f'online/surfaces_{event:04d}.npz'):
            p = root/relative; inputs[str(p)] = sha(p)
    return meta, inputs, source_changes


def crop_layout(n, halo, bounds=WINDOW):
    """Keep an untouched target window plus a declared dependency halo."""
    if halo < 0 or int(halo) != halo:
        raise ValueError('Halo must be a nonnegative integer.')
    axis = (np.arange(n)+.5)/n
    target, slices = [], []
    for low, high in bounds:
        selected = np.flatnonzero((axis >= low) & (axis <= high))
        if not len(selected):
            raise ValueError('Target window contains no cells.')
        target.append(selected)
        slices.append(slice(max(0, int(selected[0])-halo), min(n, int(selected[-1])+halo+1)))
    start = np.array([s.start for s in slices])
    target_slices = tuple(slice(int(indices[0]-begin), int(indices[-1]-begin+1))
                          for indices, begin in zip(target, start))
    centers = np.stack(np.meshgrid(*[axis[indices] for indices in target], indexing='ij'), axis=-1)
    return tuple(slices), target_slices, centers


def make_analyzer(n, halo=HALO):
    crop, target, centers = crop_layout(n, halo)
    shape = tuple(s.stop-s.start for s in crop)
    config = SimulationConfig(geometry=CARTESIAN, dimensionality=3, num_cells=shape[0],
        box_size=StaticFloatVector(*[size/n for size in shape]), mhd=False)
    registry = get_registered_variables(config)
    template = construct_primitive_state(config=config, registered_variables=registry,
        density=jnp.ones(shape), gas_pressure=jnp.ones(shape), velocity_x=jnp.zeros(shape),
        velocity_y=jnp.zeros(shape), velocity_z=jnp.zeros(shape))
    config = finalize_config(config, template.shape); helper = get_helper_data(config)
    if config.grid_spacing != 1/n:
        raise AssertionError('Cropping must preserve physical grid spacing.')

    @jax.jit
    def evaluate(state):
        p, rho = state[registry.pressure_index], state[registry.density_index]
        direction = _calculate_shock_direction(p, rho, config)
        divergence = _calculate_velocity_divergence(state, config, registry)
        grad_t = _calculate_temperature_gradient(p, rho, config)
        grad_rho = _calculate_density_gradient(rho, config)
        dot = jnp.sum(grad_t*grad_rho, axis=0)
        comp, aligned = divergence < 0, dot > 0
        old_jump = _shock_zone_criterion_minimum_mach(state, config, registry, helper, direction)
        extra = _transition_jump_candidates(p, rho, direction,
            comp & aligned & _make_interior_mask(shape))
        output = dict(compression=comp, gradient_alignment=aligned, gradient_dot=dot,
            divergence=divergence, old_jump=old_jump, extra_jump=extra,
            temperature_gradient_x=grad_t[0], density_gradient_x=grad_rho[0],
            normal_x=direction[0], normal_y=direction[1], normal_z=direction[2])
        results = {}
        for label, enabled in (('baseline', False), ('transition', True)):
            result = find_shocks_pfrommer(state, config, registry, helper,
                mach_min=1.3, profile_aware_mach=True, transition_aware_detection=enabled)
            results[label] = result
            output.update({label+'_zone':result.shock_zones,
                label+'_surface':result.shock_surface_cells, label+'_mach':result.mach_numbers,
                label+'_offset':result.shock_surface_offsets})
        zone = results['transition'].shock_zones
        raw_surface = identify_shock_surface(state, zone, direction, config, registry)
        offsets = calculate_shock_surface_offsets(state, raw_surface, direction, config, registry)
        raw_mach = _calculate_mach_at_surface(state, raw_surface, zone, direction,
            offsets, config, registry, profile_aware=True)
        displacement = direction*offsets[jnp.newaxis, ...]
        _, _, _, _, valid, post, pre = get_profile_aware_post_pre_shock_values(
            direction, zone, p, rho, max_steps=8, center_offsets=displacement)
        log_entropy = jnp.log(p)-(5/3)*jnp.log(rho)
        s2, s1 = _sample_field_at_distances(log_entropy, direction, post, pre, displacement)
        final = raw_surface & valid & jnp.isfinite(raw_mach) & (raw_mach >= 1.3) & (s2-s1 > 1e-5)
        output.update(transition_raw_surface=raw_surface, transition_raw_mach=raw_mach,
            transition_log_entropy_jump=s2-s1, transition_sampling_valid=valid,
            transition_mach_pass=jnp.isfinite(raw_mach) & (raw_mach >= 1.3),
            transition_entropy_pass=s2-s1 > 1e-5,
            transition_post_distance=post, transition_pre_distance=pre)
        output['zone_mismatch'] = (results['baseline'].shock_zones != (comp & aligned & old_jump)) | (
            zone != (comp & aligned & (old_jump | extra)))
        output['surface_mismatch'] = results['transition'].shock_surface_cells != final
        return {k:v[target] for k, v in output.items()}

    def analyze(state):
        patch = np.asarray(state[(slice(None),)+crop])
        fields = {k:np.asarray(v) for k, v in evaluate(jnp.asarray(patch)).items()}
        if fields.pop('zone_mismatch').any() or fields.pop('surface_mismatch').any():
            raise AssertionError('Instrumented stages do not match the production API.')
        normal = np.stack([fields['normal_'+axis] for axis in 'xyz'], axis=-1)
        for label in ('baseline', 'transition'):
            points = centers+fields[label+'_offset'][..., None]*normal/n
            distances = np.linalg.norm(points[..., None, :]-np.asarray(SOURCES), axis=-1)
            fields[label+'_exported'] = fields[label+'_surface'] & (distances > INJECTION_RADIUS).all(axis=-1)
        # Endpoint diagnostics use the original pre-acceptance refinement.
        # Offsets are zeroed on final rejections by the API; compute distances
        # for accepted surfaces, where API and raw offsets coincide exactly.
        refined = centers+normal*fields['transition_offset'][..., None]/n
        post = refined-normal*fields['transition_post_distance'][..., None]/n
        pre = refined+normal*fields['transition_pre_distance'][..., None]/n
        fields['transition_upstream_inside_source'] = fields['transition_surface'] & (
            np.linalg.norm(pre[..., None, :]-np.asarray(SOURCES), axis=-1) <= INJECTION_RADIUS).any(axis=-1)
        fields['transition_downstream_inside_source'] = fields['transition_surface'] & (
            np.linalg.norm(post[..., None, :]-np.asarray(SOURCES), axis=-1) <= INJECTION_RADIUS).any(axis=-1)
        fields['centers'] = centers
        fields['state'] = np.asarray(state[(slice(None),)+tuple(slice(c.start+t.start, c.start+t.stop)
            for c, t in zip(crop, target))])
        return fields
    return analyze


def region_summary(fields, roi):
    base, new = fields['baseline_exported'] & roi, fields['transition_exported'] & roi
    raw = fields['transition_raw_surface'] & roi
    added = new & ~fields['baseline_exported']
    result = dict(cells=int(roi.sum()), compression=int((roi & fields['compression']).sum()),
        aligned=int((roi & fields['gradient_alignment']).sum()),
        compressive_and_aligned=int((roi & fields['compression'] & fields['gradient_alignment']).sum()),
        baseline_zone=int((roi & fields['baseline_zone']).sum()),
        transition_zone=int((roi & fields['transition_zone']).sum()),
        baseline_exported_surface_cells=int(base.sum()), transition_exported_surface_cells=int(new.sum()),
        added_surface_cells=int(added.sum()), removed_surface_cells=int((base & ~new).sum()),
        transition_candidates_before_final_acceptance=int(raw.sum()),
        raw_candidates_fail_mach=int((raw & ~fields['transition_mach_pass']).sum()),
        raw_candidates_fail_entropy=int((raw & ~fields['transition_entropy_pass']).sum()),
        raw_candidates_invalid_sampling=int((raw & ~fields['transition_sampling_valid']).sum()),
        added_with_upstream_inside_source=int((added & fields['transition_upstream_inside_source']).sum()),
        added_with_downstream_inside_source=int((added & fields['transition_downstream_inside_source']).sum()))
    return result


def native_axis_indices(centers, n):
    """Same native upper-z slice and nearest source-row as the original audit."""
    return (int(np.argmin(abs(centers[0, :, 0, 1]-.375))),
            int(np.argmin(abs(centers[0, 0, :, 2]-(.5+.5/n)))))


def pressure_features(fields, n):
    """Retain all positive signed pressure-gradient crests on each half-axis.

    These are geometric inspection locations. No motion, entropy/conservation
    certification, unique ancestry, or physical shock label is inferred.
    """
    centers = fields['centers']
    j, k = native_axis_indices(centers, n)
    x = centers[:, j, k, 0]; q = fields['state'][:, :, j, k]
    dp = np.gradient(q[4], 1/n); rows = []
    for side, sign in (('left', 1), ('right', -1)):
        peaks = find_peaks(sign*dp)[0]
        peaks = [i for i in peaks if sign*dp[i] > 0 and (x[i] < .5 if side == 'left' else x[i] > .5)]
        for rank, i in enumerate(sorted(peaks, key=lambda i:abs(dp[i]), reverse=True), 1):
            near = abs(x-x[i]) <= 2/n
            row = dict(side=side, strength_rank_on_half_axis=rank, x=float(x[i]),
                y=float(centers[i, j, k, 1]), z=float(centers[i, j, k, 2]),
                pressure_gradient=float(dp[i]), in_central_x_window=bool(.45 <= x[i] <= .55),
                density=float(q[0, i]), pressure=float(q[4, i]), temperature=float(q[4, i]/q[0, i]),
                lab_velocity_x=float(q[1, i]), full_gradient_dot=float(fields['gradient_dot'][i,j,k]),
                temperature_gradient_x=float(fields['temperature_gradient_x'][i,j,k]),
                density_gradient_x=float(fields['density_gradient_x'][i,j,k]),
                temperature_normal_x=float(fields['normal_x'][i,j,k]))
            for name in ('compression', 'gradient_alignment', 'old_jump', 'extra_jump',
                         'baseline_zone', 'transition_zone', 'baseline_exported', 'transition_exported'):
                row[name+'_at_feature'] = bool(fields[name][i, j, k])
                row[name+'_within_2cells'] = int(fields[name][:, j, k][near].sum())
            rows.append(row)
    return rows


def run(args):
    if args.output_dir.exists():
        raise ValueError('Use a fresh output directory.')
    inputs, historical_changes, metadata = {}, {}, []
    for root in args.roots:
        meta, hashes, changes = validate_archive(root)
        metadata.append(meta); inputs.update(hashes); historical_changes[str(root)] = changes
    if sorted(meta['num_cells'] for meta in metadata) != [64, 128]:
        raise ValueError('Require one 64 cubed and one 128 cubed archive.')
    for key in ('source_positions', 'source_strengths', 'injection_radius', 'gamma',
                'discrete_mass_rates', 'discrete_energy_rates', 'requested_times'):
        np.testing.assert_allclose(metadata[0][key], metadata[1][key], rtol=1e-12, atol=1e-12)
    sources = [Path(__file__), Path('experiments/wind_bubble/three_bubble_setup.py')]
    sources += sorted(Path('astronomix').rglob('*.py'))
    source_hashes = {str(p):sha(p) for p in sources}
    args.output_dir.mkdir(parents=True)
    rows, feature_rows, historical_rows, halo_checks, snapshots = [], [], [], [], {}
    for root, meta in zip(args.roots, metadata):
        n = meta['num_cells']; evaluate = make_analyzer(n)
        # Expand the halo for early and late captures. No crop-boundary feature
        # can be interpreted unless target masks/measurements remain invariant.
        larger = make_analyzer(n, HALO+4)
        for event in EVENTS:
            path = root/'audit'/f'state_{event:04d}.npy'
            state = np.load(path, allow_pickle=False)
            fields = evaluate(state)
            if event in (EVENTS[0], EVENTS[-1]):
                other = larger(state); maximum = 0.; differences = {}
                # Keep ALL Boolean decisions exact. Float32 interpolation at
                # coordinates shifted by a crop origin has small roundoff;
                # measure it separately, without altering physical thresholds.
                for key, value in fields.items():
                    if value.dtype == bool:
                        np.testing.assert_array_equal(value, other[key], err_msg=key)
                for key, value in fields.items():
                    if value.dtype == bool:
                        continue
                    elif key not in ('transition_post_distance', 'transition_pre_distance',
                                     'transition_log_entropy_jump'):
                        # Diagnostic endpoints outside actual candidate cells
                        # can depend on unavailable exits and are not reported.
                        if key.startswith('transition_raw_mach'):
                            mask = fields['transition_raw_surface']
                            np.testing.assert_allclose(value[mask], other[key][mask],
                                rtol=HALO_FLOAT_RTOL, atol=HALO_FLOAT_ATOL)
                        else:
                            np.testing.assert_allclose(value, other[key],
                                rtol=HALO_FLOAT_RTOL, atol=HALO_FLOAT_ATOL, err_msg=key)
                        delta = float(np.max(abs(value-other[key])))
                        maximum = max(maximum, delta); differences[key] = delta
                halo_checks.append(dict(resolution=n, event=event, halo_cells=HALO,
                    larger_halo_cells=HALO+4, passed=True, maximum_field_difference=maximum,
                    maximum_baseline_mach_difference=differences['baseline_mach'],
                    maximum_transition_mach_difference=differences['transition_mach']))
            centers = fields['centers']; time = float(meta['actual_times'][event])
            central = ((centers >= CENTRAL[:, 0]) & (centers <= CENTRAL[:, 1])).all(axis=-1)
            _, iz = native_axis_indices(centers, n)
            plane = np.zeros(central.shape, bool); plane[:, :, iz] = True
            for name, roi in (('collision_window_volume', np.ones(central.shape, bool)),
                              ('central_volume', central), ('collision_window_xy_slice', plane),
                              ('central_xy_slice', central & plane)):
                rows.append(dict(resolution=n, event=event, time=time, region=name, **region_summary(fields, roi)))
            for row in pressure_features(fields, n):
                feature_rows.append(dict(resolution=n, event=event, time=time, **row))
            old = np.load(root/'audit'/f'collision_crop_{event:04d}.npz', allow_pickle=False)
            np.testing.assert_allclose(old['centers'], centers, atol=0, rtol=0)
            for old_key, key in (('compression', 'compression'), ('gradient_alignment', 'gradient_alignment'),
                ('minimum_jump', 'old_jump'), ('shock_zone', 'baseline_zone'), ('shock_surface', 'baseline_surface')):
                historical_rows.append(dict(resolution=n, event=event, field=old_key,
                    differing_cells=int((old[old_key] != fields[key]).sum())))
            old.close()
            np.savez_compressed(args.output_dir/f'comparison_n{n:03d}_{event:04d}.npz', time=time,
                grid_spacing=1/n, **fields)
            snapshots[n, event] = fields
            central_row = rows[-1]
            print(f'{n} cubed t={time:.2f}: central slice surfaces '
                f'{central_row["baseline_exported_surface_cells"]} -> '
                f'{central_row["transition_exported_surface_cells"]}; '
                f'compression AND alignment={central_row["compressive_and_aligned"]}', flush=True)
    write_csv(args.output_dir/'region_comparison.csv', rows)
    write_csv(args.output_dir/'pressure_features.csv', feature_rows)
    write_csv(args.output_dir/'historical_stage_comparison.csv', historical_rows)
    write_csv(args.output_dir/'halo_checks.csv', halo_checks)
    profiles = save_profiles(args.output_dir, snapshots, metadata)
    make_figures(args.output_dir, snapshots, profiles, metadata)
    for path, digest in {**inputs, **source_hashes}.items():
        if sha(path) != digest:
            raise AssertionError(f'Input or source changed during saved-state comparison: {path}')
    central_late = [r for r in rows if r['region'] == 'central_xy_slice' and r['event'] >= 6]
    report = dict(comparison_completed=True, states_compared=len(snapshots), resolutions=[m['num_cells'] for m in metadata],
        captured_events=list(EVENTS), window_bounds=WINDOW.tolist(), central_bounds=CENTRAL.tolist(),
        halo_cells=HALO, halo_invariance_checks=halo_checks,
        halo_boolean_comparison='exact', halo_float_rtol=HALO_FLOAT_RTOL, halo_float_atol=HALO_FLOAT_ATOL,
        historical_stage_differing_cells=sum(r['differing_cells'] for r in historical_rows),
        late_central_slice_added_surface_cells=sum(r['added_surface_cells'] for r in central_late),
        late_central_slice_count_increased=any(r['transition_exported_surface_cells'] > r['baseline_exported_surface_cells'] for r in central_late),
        final_mach_minimum=1.3, hydrodynamics_rerun=False, grouping_or_smoothing_changed=False,
        production_default_changed=False, physical_fronts_independently_certified=False,
        physical_shock_absence_proven=False, temporal_tracking_performed=False,
        feature_scope='Signed pressure-gradient crests on a native grid axis locate inspection sites only. They are not certified shocks or linked physical fronts.',
        historical_source_changes=historical_changes, input_sha256=inputs, source_sha256=source_hashes,
        scope='Same-gas detector comparison inside a fixed 3D collision window. No independent shock-motion oracle or physical completeness verdict.')
    (args.output_dir/'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k, v in report.items() if k not in ('input_sha256','source_sha256','historical_source_changes')}, indent=2))
    return report


def save_profiles(output, snapshots, metadata):
    rows = []
    for meta in metadata:
        n = meta['num_cells']
        for event in EVENTS:
            f = snapshots[n, event]; centers = f['centers']; q = f['state']
            j, k = native_axis_indices(centers, n)
            for i in range(len(centers)):
                row = dict(resolution=n, event=event, time=float(meta['actual_times'][event]),
                    x=float(centers[i, j, k, 0]), y=float(centers[i, j, k, 1]), z=float(centers[i, j, k, 2]),
                    density=float(q[0, i, j, k]), pressure=float(q[4, i, j, k]),
                    temperature=float(q[4, i, j, k]/q[0, i, j, k]), velocity_x=float(q[1, i, j, k]))
                for name in ('compression', 'gradient_alignment', 'gradient_dot', 'divergence', 'old_jump',
                    'extra_jump', 'baseline_zone', 'transition_zone', 'baseline_exported', 'transition_exported',
                    'baseline_mach', 'transition_mach', 'transition_raw_surface', 'transition_entropy_pass'):
                    row[name] = f[name][i, j, k].item()
                rows.append(row)
    write_csv(output/'native_axis_profiles.csv', rows)
    return rows


def make_figures(output, snapshots, profiles, metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, ListedColormap
    from matplotlib.patches import Rectangle
    for meta in metadata:
        n = meta['num_cells']
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=150)
        fig.subplots_adjust(left=.07, right=.96, bottom=.19, top=.86, wspace=.27, hspace=.36)
        for column, event in enumerate(PLOT_EVENTS):
            f = snapshots[n, event]; centers = f['centers']; q = f['state']
            _, k = native_axis_indices(centers, n)
            xx, yy = centers[:, 0, 0, 0], centers[0, :, 0, 1]
            extent = (xx[0]-.5/n, xx[-1]+.5/n, yy[0]-.5/n, yy[-1]+.5/n)
            ax = axes[0, column]
            image = ax.imshow(q[0, :, :, k].T, origin='lower', extent=extent,
                cmap='magma', norm=LogNorm(.02, 4.), interpolation='nearest')
            base, new = f['baseline_exported'][:, :, k], f['transition_exported'][:, :, k]
            for mask, color, marker, label in ((base & new, '#f8ba54', '.', 'Retained baseline'),
                    (new & ~base, '#39e0d0', '^', 'Added'), (base & ~new, '#4cc3ff', 'x', 'Removed')):
                i, j = np.nonzero(mask)
                ax.scatter(xx[i], yy[j], s=13, color=color, marker=marker, linewidths=.7, label=label)
            ax.set_title(f'Density and accepted surfaces | t={meta["actual_times"][event]:.2f}', fontsize=10)
            if column == 0:
                ax.legend(fontsize=7, loc='lower left')
            ax = axes[1, column]
            mask = f['compression'][:, :, k] & f['gradient_alignment'][:, :, k]
            ax.imshow(mask.T, origin='lower', extent=extent,
                cmap=ListedColormap(['#eeeeee','#6e9699']), vmin=0, vmax=1, interpolation='nearest')
            for key, color, label in (('baseline_zone', '#d98b21', 'Baseline zone'), ('transition_zone', '#006b70', 'Extended zone')):
                values = f[key][:, :, k]
                if values.any() and not values.all():
                    ax.contour(xx, yy, values.T, levels=[.5], colors=[color], linewidths=.8)
            ax.set_title('Compression AND alignment; zone contours', fontsize=10)
            for ax in axes[:, column]:
                ax.add_patch(Rectangle((.45, .335), .10, .08, fill=False, edgecolor='#3e8ec7', lw=1))
                ax.axvline(.5, color='gray', ls=':', lw=.7)
                ax.set(xlabel='x', ylabel='y', xlim=(.40,.60), ylim=(.28,.47))
        fig.colorbar(image, ax=list(axes[0]), fraction=.02, pad=.02, label='Density')
        fig.suptitle(f'Original versus transition-aware detection | saved {n} cubed gas', fontsize=14)
        fig.text(.07, .045, 'Top: identical density field, source-filtered surface cells; added/removed refer to cells, not physical shock identities.\n'
            'Bottom: teal cells pass both early criteria. Orange/dark teal contours show baseline/extended zones. Blue box: original central audit region.\n'
            f'Native XY slice near z=0.5; z={centers[0,0,k,2]:.7f}. No projection, grouping, smoothing, or new gas evolution. A gap is not proof of shock absence.', fontsize=8)
        fig.savefig(output/f'detector_comparison_n{n:03d}.png'); plt.close(fig)

        fig, axes = plt.subplots(5, 3, figsize=(13, 12), dpi=150)
        fig.subplots_adjust(left=.08, right=.98, bottom=.14, top=.9, wspace=.25, hspace=.35)
        for column, event in enumerate(PLOT_EVENTS):
            data = [r for r in profiles if r['resolution'] == n and r['event'] == event]
            x = [r['x'] for r in data]
            for row, name in enumerate(('density','pressure','temperature','velocity_x')):
                ax = axes[row, column]; ax.plot(x, [r[name] for r in data], 'o-', ms=3, color='#354a58')
                for label, color, marker in (('baseline_exported', '#d98b21', 'x'), ('transition_exported', '#008e91', 'v')):
                    selected = [r for r in data if r[label]]
                    ax.scatter([r['x'] for r in selected], [r[name] for r in selected], s=26, color=color, marker=marker)
                ax.set(ylabel=name.replace('_',' '), xlim=(.4,.6)); ax.grid(alpha=.2)
                ax.axvline(.5, color='gray', ls=':', lw=.7)
                if row == 0:
                    ax.set_title(f't={meta["actual_times"][event]:.2f}')
                if name in ('density','pressure','temperature'):
                    ax.set_yscale('log')
                else:
                    ax.axhline(0, color='gray', lw=.6)
            names = ('compression','gradient_alignment','old_jump','extra_jump','baseline_exported','transition_exported')
            matrix = np.array([[r[name] for r in data] for name in names], dtype=int)
            ax = axes[4, column]
            ax.imshow(matrix, aspect='auto', extent=(x[0]-.5/n, x[-1]+.5/n, 5.5, -.5),
                vmin=0, vmax=1, cmap=ListedColormap(['white','#087f8c']), interpolation='nearest')
            ax.set_yticks(range(6), ['compression','T/rho alignment','one-cell jump','additional jump','baseline surface','new surface'], fontsize=7)
            ax.set(xlabel='x', xlim=(.4,.6))
        fig.suptitle(f'Native cross-layer profiles and detection stages | {n} cubed', fontsize=14)
        fig.text(.08, .035, f'Native axis y={data[0]["y"]:.7f}, z={data[0]["z"]:.7f}; all profiles and Boolean tests refer to the same grid cells.\n'
            'Orange crosses: baseline surfaces; teal triangles: corrected-mode surfaces (source-filtered). Colored stage cells mean the test passes.\n'
            'Velocity is in the lab frame, not shock-frame Mach. Pressure features and snapshots alone do not certify shock motion or physical absence.', fontsize=9)
        fig.savefig(output/f'cross_layer_profiles_n{n:03d}.png'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs=2, type=Path, default=ROOTS)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/bubble_transition_jump_comparison_n064_n128'))
    run(parser.parse_args())
