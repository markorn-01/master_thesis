"""Apply the frozen planar reconstruction to guarded, native saved bubble rays.

This is one bounded transfer check, not a detector change. A window cannot
cross an injection sphere or combine the two sides of the collision. Inputs
are native cell values, without transverse interpolation or artificial padding.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from experiments.shock.joint_shock_contact_fit import FitSettings, fit_profile
from experiments.wind_bubble.collision_front_motion import pressure_ridges


TIMES = (.04, .06, .08, .12, .16)
WINDOW_CELLS = 16  # Existing fitter's minimum; not a physical information bound.
REGIONS = ('outer_left', 'outer_right', 'central_left', 'central_right')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def allowed_cells(n, y, z, sources, radius, region):
    """Conservative full-cell source exclusion and a fixed collision-side guard."""
    sources = np.asarray(sources, float)
    if region not in REGIONS or sources.ndim != 2 or sources.shape[1] != 3 or len(sources) < 2:
        raise ValueError('Require a named region and at least the lower source pair.')
    dx = 1/n; x = (np.arange(n)+.5)*dx
    centers = np.column_stack([x, np.full(n, y), np.full(n, z)])
    distance = np.maximum(abs(centers[:, None, :]-sources[None, :, :])-dx/2, 0.)
    mask = np.all(np.sum(distance**2, axis=2) >= radius**2, axis=1)
    left, right = sources[0, 0], sources[1, 0]; middle = (left+right)/2
    if not left < right:
        raise ValueError('The first two sources must be ordered left to right.')
    if region == 'outer_left':
        mask &= x+dx/2 <= left
    elif region == 'outer_right':
        mask &= x-dx/2 >= right
    elif region == 'central_left':
        mask &= (x-dx/2 >= left) & (x+dx/2 <= middle)
    else:
        mask &= (x-dx/2 >= middle) & (x+dx/2 <= right)
    mask[[0, -1]] = False
    return mask


def guarded_window(candidate_index, mask, count=WINDOW_CELLS):
    """Return one centered/clamped window, never search for the best fit window."""
    if candidate_index is None or not 0 <= candidate_index < len(mask) or not mask[candidate_index]:
        return None, 0
    start = stop = candidate_index
    while start > 0 and mask[start-1]:
        start -= 1
    while stop+1 < len(mask) and mask[stop+1]:
        stop += 1
    available = stop-start+1
    if available < count:
        return None, available
    first = int(np.clip(candidate_index-count//2, start, stop-count+1))
    return (first, first+count), available


def choose_candidate(ridges, mask, region):
    candidates = [r for r in ridges if mask[r['index']]]
    if not candidates:
        return None
    # Outermost front for the positive control; closest-to-midplane feature
    # for the central question. Selection uses geometry/pressure, never fits.
    return (min if region in ('outer_left', 'central_right') else max)(candidates, key=lambda r:r['x'])


def interpretation(rows):
    outer = [r for r in rows if r['region'].startswith('outer')]
    center = [r for r in rows if r['region'].startswith('central')]
    outer_supported = sum(r['joint_reconstruction_supported'] for r in outer)
    central_supported = sum(r['joint_reconstruction_supported'] for r in center)
    outer_shocks = sum(r['shock_model_supported'] for r in outer)
    # A good algebraic fit is still only conditional. Never certify physical
    # fronts from single-snapshot reconstruction in this transfer experiment.
    return dict(outer_cases=len(outer), central_cases=len(center),
        outer_fitted=sum(r['fit_attempted'] for r in outer),
        central_fitted=sum(r['fit_attempted'] for r in center),
        outer_supported_joint_pairs=outer_supported,
        outer_supported_shock_models=outer_shocks,
        central_supported_joint_pairs=central_supported,
        central_short_windows=sum(r['status'] == 'insufficient_guarded_cells' for r in center),
        outer_positive_control_passes=outer_shocks > 0,
        decision='restricted_model_does_not_transfer' if outer_shocks == 0
            else 'conditional_fits_only_no_physical_certification',
        missing_bubble_fronts_recovered=False, physical_shock_absence_established=False,
        production_promotion_allowed=False)


def run(args):
    if args.output_dir.exists():
        raise ValueError('Use a fresh output directory.')
    settings = FitSettings()
    calibration_path = args.calibration/'verification.json'
    calibration = json.loads(calibration_path.read_text())
    fitter_path = Path('experiments/shock/joint_shock_contact_fit.py')
    if (not calibration['passed'] or calibration['fixed_settings'] != json.loads(json.dumps(asdict(settings)))
            or sha(fitter_path) != calibration['source_sha256'][str(fitter_path)]):
        raise ValueError('Require the unchanged, successfully tested planar fitter and settings.')
    meta_path = args.root/'metadata.json'; meta = json.loads(meta_path.read_text())
    if meta['status'] != 'complete' or meta['num_cells'] != 128 or meta['gamma'] != settings.gamma:
        raise ValueError('Require the completed 128-cubed strip capture with matching gamma.')
    source_paths = [Path(__file__), fitter_path,
        Path('experiments/shock/check_joint_shock_contact_fit.py'),
        Path('experiments/wind_bubble/collision_front_motion.py')]
    sources = {str(p):sha(p) for p in source_paths}
    production = {str(p):sha(p) for p in Path('astronomix/_physics_modules/_shock_finder').glob('*.py')}
    inputs = {str(p):sha(p) for p in (meta_path, calibration_path)}
    times = np.asarray(meta['actual_times']); ys = np.asarray(meta['transverse_y']); zs = np.asarray(meta['transverse_z'])
    js = sorted(np.argsort(abs(ys-meta['target_y']))[:2])
    ks = sorted(np.argsort(abs(zs-meta['target_z']))[:2])
    n = meta['num_cells']; dx = 1/n; x = (np.arange(n)+.5)*dx
    args.output_dir.mkdir(parents=True)
    rows, fits, profiles = [], [], {}
    for requested_time in TIMES:
        event = int(np.argmin(abs(times-requested_time)))
        if abs(times[event]-requested_time) > 1.e-6:
            raise ValueError('Required analysis time is unavailable.')
        path = args.root/'strips'/f'strip_{event:04d}.npz'
        inputs[str(path)] = sha(path)
        if inputs[str(path)] != meta['strip_sha256'][path.name]:
            raise ValueError(f'Saved strip differs from its original fingerprint: {path}')
        with np.load(path, allow_pickle=False) as data:
            band, stages = data['primitive'], data['production_stages']
            if not np.isclose(float(data['time']), times[event], rtol=0, atol=1.e-8):
                raise ValueError('Strip time does not match metadata.')
        for j in js:
            for k in ks:
                line = band[:, :, j, k].astype(float)
                ridges = pressure_ridges(line, x)
                for region in REGIONS:
                    mask = allowed_cells(n, ys[j], zs[k], meta['source_positions'], meta['injection_radius'], region)
                    candidate = choose_candidate(ridges, mask, region)
                    window, available = guarded_window(candidate['index'] if candidate else None, mask)
                    row = dict(case=f'e{event:04d}_y{j}_z{k}_{region}', event=event, time=float(times[event]),
                        region=region, y=float(ys[j]), z=float(zs[k]), candidate_found=candidate is not None,
                        candidate_x=candidate['x'] if candidate else None, available_guarded_cells=available,
                        window_first=None, window_stop=None, fit_attempted=False,
                        status='no_pressure_feature' if candidate is None else 'insufficient_guarded_cells',
                        selected_model=None, joint_reconstruction_supported=False,
                        shock_model_supported=False,
                        joint_maximum_residual=None, best_any_maximum_residual=None,
                        fitted_mach_if_supported=None, fitted_shock_x_if_supported=None,
                        transverse_speed_over_sound_max=None, saved_production_surface_nearby=False)
                    if candidate:
                        i = candidate['index']; nearby = slice(max(i-2, 0), min(i+3, n))
                        stage_index = meta['stage_names'].index('shock_surface')
                        row['saved_production_surface_nearby'] = bool(stages[stage_index, nearby, j, k].any())
                    if window is not None:
                        first, stop = window
                        primitive = line[[0, 1, 4], first:stop].T.copy()
                        before = primitive.copy()
                        fit = fit_profile(primitive, dx, settings)
                        np.testing.assert_array_equal(primitive, before)
                        joint = next(r for r in fit['models'] if r['model'] == 'joint')
                        shock = next(r for r in fit['models'] if r['model'] == 'shock')
                        supports_shock = bool(fit['joint_reconstruction_supported'] or
                            (fit['selected_model'] == 'shock' and shock['parameters'][3] >= 1.3))
                        cs = np.sqrt(settings.gamma*line[4, first:stop]/line[0, first:stop])
                        row.update(window_first=first, window_stop=stop, fit_attempted=True,
                            status=fit['selected_model'], selected_model=fit['selected_model'],
                            joint_reconstruction_supported=fit['joint_reconstruction_supported'],
                            shock_model_supported=supports_shock,
                            joint_maximum_residual=joint['maximum_residual'],
                            best_any_maximum_residual=min(r['maximum_residual'] for r in fit['models']),
                            transverse_speed_over_sound_max=float((np.hypot(line[2, first:stop], line[3, first:stop])/cs).max()))
                        if supports_shock:
                            if fit['joint_reconstruction_supported']:
                                mach, shock_x = fit['recovered_mach'], fit['recovered_shock_x']
                            else:
                                mach, position = shock['parameters'][3], shock['parameters'][4]
                                shock_x = (position if shock['orientation'] == 1 else stop-first-position)*dx
                            row.update(fitted_mach_if_supported=mach,
                                fitted_shock_x_if_supported=first*dx+shock_x)
                        fits.append(dict(case=row['case'], fit=fit))
                        profiles[row['case']] = line[:, first:stop]
                    rows.append(row)
        latest = [r for r in rows if r['event'] == event]
        print(f't={times[event]:.2f}: fitted={sum(r["fit_attempted"] for r in latest)}; '
            f'joint-supported={sum(r["joint_reconstruction_supported"] for r in latest)}; '
            f'short windows={sum(r["status"] == "insufficient_guarded_cells" for r in latest)}', flush=True)
    with (args.output_dir/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (args.output_dir/'fits.jsonl').open('w') as stream:
        for fit in fits:
            stream.write(json.dumps(fit, allow_nan=False)+'\n')
    np.savez_compressed(args.output_dir/'fitted_profiles.npz', **profiles)
    make_figure(args.output_dir, rows, fits, profiles, n, meta)
    for path, digest in {**sources, **production, **inputs}.items():
        if sha(path) != digest:
            raise AssertionError(f'Read-only input or source changed: {path}')
    verdict = interpretation(rows)
    result = dict(completed=True, **verdict, fixed_fit_settings=asdict(settings),
        fixed_times=TIMES, window_cells=WINDOW_CELLS, native_rays=len(js)*len(ks),
        source_free_guard='Entire cell cubes excluded if they intersect any injection sphere.',
        midplane_guard='Each central window stays on its own side of the collision midplane.',
        interpolation_or_padding_used=False, fitting_windows_tuned_to_results=False,
        simulation_rerun=False, production_changed=False, thresholds_changed=False,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        source_sha256=sources, production_sha256=production, input_sha256=inputs,
        saved_stage_scope='Historical masks from the original strip capture, not recomputed current masks.',
        minimum_window_scope='16 cells is the current prototype input requirement, not a fundamental impossibility for shorter profiles.',
        scope='Transfer of one fixed planar, constant-state fan and blur family to selected native bubble rays. Rejecting a window or fit cannot establish physical shock absence.')
    (args.output_dir/'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(verdict, indent=2))
    return result


def make_figure(output, rows, fits, profiles, n, meta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from experiments.shock.check_joint_shock_contact_fit import predicted_joint
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=160)
    fig.subplots_adjust(top=.88, bottom=.17, wspace=.3, hspace=.35)
    for col, region in enumerate(('outer_left', 'outer_right')):
        selected = next(r for r in rows if r['region'] == region and abs(r['time']-.12) < 1.e-6 and r['fit_attempted'])
        fit = next(r['fit'] for r in fits if r['case'] == selected['case'])
        q = profiles[selected['case']]; predicted = predicted_joint(fit, q.shape[1])
        x = (np.arange(q.shape[1])+selected['window_first']+.5)/n
        for ax, observed, fitted, label in [(axes[0, col], q[0], predicted[:, 0], 'Density'),
                                          (axes[1, col], q[4], predicted[:, 2], 'Pressure')]:
            ax.plot(x, observed, 'o-', ms=3, label='Saved native cells', color='#555555')
            label_fit = 'Conditional joint-model fit' if selected['joint_reconstruction_supported'] else 'Rejected joint-model fit'
            ax.plot(x, fitted, '--', color='#c76824', label=label_fit)
            ax.set(xlabel='x', ylabel=label); ax.grid(alpha=.2)
        axes[0, col].set_title(f'{region.replace("_", " ").title()} control, t=0.12\n'
            f'max scaled joint residual = {selected["joint_maximum_residual"]:.3f}')
        axes[0, col].legend(fontsize=8)
    verdict = interpretation(rows)
    fig.suptitle(f'Saved bubble transfer: {verdict["outer_supported_shock_models"]}/{verdict["outer_fitted"]} outer shock models accepted', fontsize=14)
    fig.text(.07, .035, 'Dashed curves show joint-model predictions; acceptance requires maximum scaled residual at most 0.001.\n'
        'Central candidates have only 8 source-free cells on each side; the unchanged prototype requires 16.\n'
        'No source-crossing, opposing-front pairing, interpolation, padding, or threshold relaxation is used. This does not prove shock absence.', fontsize=9)
    fig.savefig(output/'bubble_transfer.png'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/collision_front_motion_iwr_n128'))
    parser.add_argument('--calibration', type=Path, default=Path('outputs/joint_shock_contact_feasibility'))
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())
