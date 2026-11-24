"""Compare verified 128/256-cubed flow strips without reintegrating the gas.

The outermost pressure fronts are controls; an inner noncentral feature cannot
stand in for them. This comparison does not infer shock absence from rejection.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from experiments.wind_bubble.check_outer_control_calibration import (
    density_example, evaluate_window,
)
from experiments.wind_bubble.collision_front_motion import MotionSettings, strip_fields
from experiments.wind_bubble.review_collision_front_motion import run as review_run, sha, write_csv


ROOTS = (Path('outputs/collision_front_motion_iwr_n128'),
         Path('outputs/collision_front_motion_iwr_compgpu7_n256_flow_only'))
OUTPUT = Path('outputs/collision_front_resolution_comparison_n128_n256/comparison')
TARGET_TIMES = (.08, .12, .13)


def load_checked_run(root):
    root = Path(root); path = root/'metadata.json'
    meta = json.loads(path.read_text())
    if meta.get('status') != 'complete':
        raise ValueError('Require a completed integration.')
    hashes = {str(path): sha(path)}
    names = sorted(meta['strip_sha256'])
    if set(names) != {p.name for p in (root/'strips').glob('*.npz')}:
        raise ValueError('Saved strip file set differs from metadata.')
    if names != [f'strip_{i:04d}.npz' for i in range(len(names))]:
        raise ValueError('Require consecutive strip events.')
    n = meta['num_cells']; bands, actual = [], []
    for name in names:
        path = root/'strips'/name; hashes[str(path)] = sha(path)
        if hashes[str(path)] != meta['strip_sha256'][name]:
            raise ValueError(f'Strip hash mismatch: {path}')
        with np.load(path, allow_pickle=False) as data:
            q = data['primitive']
            if q.shape != (5, n, 4, 4) or not np.isfinite(q).all() or (q[[0, 4]] <= 0).any():
                raise ValueError(f'Invalid primitive strip: {path}')
            if bool(meta['stage_names']) != ('production_stages' in data):
                raise ValueError('Saved masks differ from declared availability.')
            bands.append(q); actual.append(float(data['time']))
    np.testing.assert_array_equal(actual, meta['actual_times'])
    np.testing.assert_allclose(actual, meta['requested_times'], rtol=0, atol=1.e-6)
    if not np.all(np.diff(actual) > 0):
        raise ValueError('Event times must increase.')
    lines, dy, dz = strip_fields(np.array(bands), np.array(meta['transverse_y']),
        np.array(meta['transverse_z']), 1/n, meta['target_y'], meta['target_z'])
    return dict(root=root, meta=meta, times=np.array(actual), lines=lines, dy=dy, dz=dz,
                x=(np.arange(n)+.5)/n, hashes=hashes)


def validate_matching_setup(a, b):
    for key in ('source_positions', 'source_strengths', 'injection_radius', 'gamma',
                'solver_dt_max', 'requested_times', 'target_y', 'target_z'):
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    for key in ('discrete_mass_rates', 'discrete_energy_rates'):
        np.testing.assert_allclose(a[key], b[key], rtol=1.e-12, atol=1.e-12, err_msg=key)
    def normalized(meta):
        return {k.split('master_thesis/')[-1]: v for k, v in meta['source_sha256'].items()}
    sa, sb = normalized(a), normalized(b)
    changed = sorted(k for k in sa.keys() & sb.keys() if sa[k] != sb[k])
    integration_changes = [k for k in changed if k.startswith('astronomix/')
                           and '/_shock_finder/' not in k]
    if integration_changes:
        raise ValueError(f'Hydrodynamic source fingerprints differ: {integration_changes}')
    return changed


def physical_distances(n):
    """Keep the original 128-cubed endpoint separations fixed in physical units."""
    return np.array(MotionSettings().distances_cells)*n/128


def run(roots=ROOTS, output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a fresh comparison directory.')
    datasets = [load_checked_run(root) for root in roots]
    if [d['meta']['num_cells'] for d in datasets] != [128, 256]:
        raise ValueError('Require the 128 and 256 cubed runs in that order.')
    changed = validate_matching_setup(*(d['meta'] for d in datasets))
    hashes = {p: h for d in datasets for p, h in d['hashes'].items()}
    rows, pairs, density, summaries, inner_examples, central = [], [], [], [], [], []
    review_source = 'experiments/wind_bubble/review_collision_front_motion.py'
    for data in datasets:
        meta = data['meta']; n = meta['num_cells']
        review_dir = output.parent/f'review_n{n}'
        if not review_dir.exists():
            review_run(data['root'], review_dir)
        path = review_dir/'verification.json'
        report = json.loads(path.read_text()); hashes[str(path)] = sha(path)
        if sha(review_source) not in report['analysis_source_sha256'].values():
            raise ValueError('The saved review predates the corrected control selection.')
        for path, expected in report['input_sha256'].items():
            if sha(path) != expected:
                raise ValueError(f'Review input changed: {path}')
        summaries.append(dict(num_cells=n, verdict=report['verdict'],
            persistent_outer_controls=report['persistent_outer_controls'],
            noncentral_compatible_features=report['persistent_noncentral_compatible_features'],
            persistent_late_central_fronts=report['persistent_late_central_fronts'],
            events_with_both_outer_controls_passing=report['events_with_both_outer_controls_passing'],
            production_masks_recorded=bool(meta['stage_names'])))
        for target in TARGET_TIMES:
            event = int(np.argmin(abs(data['times']-target)))
            if abs(data['times'][event]-target) > 1.e-6:
                raise ValueError('Target time unavailable.')
            settings = [(half, None, 'fixed_cell_distances') for half in (4, 8, 16)]
            if n == 256:
                settings.append((16, physical_distances(n), 'matched_physical_distances'))
            center = None
            for half, distances, label in settings:
                row, selected = evaluate_window(data['lines'], data['times'], data['x'], event, half,
                    data['dy'], data['dz'], sources=meta['source_positions'], distances=distances)
                if row is None:
                    raise ValueError('Outermost left front cannot be temporally bracketed.')
                row.update(num_cells=n, sampling_grid=label)
                rows.append(row); center = row['x']
                for pair in selected:
                    pair.update(num_cells=n, sampling_grid=label)
                pairs.extend(selected)
            density.append(dict(num_cells=n, event=event, time=float(data['times'][event]),
                front_x=center, **density_example(data['lines'][event], data['x'], center)))
            q = np.array([np.interp(.5, data['x'], field) for field in data['lines'][event]])
            central.append(dict(num_cells=n, time=float(data['times'][event]), x=.5,
                density=float(q[0]), pressure=float(q[4]), temperature_proxy=float(q[4]/q[0]),
                velocity_x=float(q[1])))
        # These passing noncentral samples caused the old broad-region label
        # to be mistaken for an outermost control. Preserve their actual positions.
        if n == 256:
            import csv
            path = review_dir/'front_observations.csv'; hashes[str(path)] = sha(path)
            with path.open() as stream:
                inner_examples = [r for r in csv.DictReader(stream)
                    if r['compatible_10pct'] == 'True' and r['region'] == 'outer_control'
                    and r['outermost_positive_control_candidate'] == 'False']
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise AssertionError(f'Read-only input changed: {path}')
    output.mkdir(parents=True)
    for name, values in (('outer_windows.csv', rows), ('sampled_pairs.csv', pairs),
                         ('density_examples.csv', density), ('passing_inner_features.csv', inner_examples),
                         ('central_midplane.csv', central)):
        write_csv(output/name, values)
    make_figures(output, datasets, density, inner_examples)
    result = dict(comparison_completed=True, copied_strips_verified=sum(len(d['times']) for d in datasets),
        matching_schedule_and_source_budgets=True, matched_physical_ray=[.375, .5],
        source_fingerprint_differences=changed, hydrodynamic_source_fingerprint_differences=[],
        per_run=summaries, selected_time_outer_comparisons=len(rows),
        selected_time_outer_passes_10pct=sum(r['compatible_10pct'] for r in rows),
        selected_time_outer_passes_20pct=sum(r['compatible_20pct'] for r in rows),
        passing_inner_observations=len(inner_examples),
        production_changed=False, thresholds_changed=False, simulation_rerun=False,
        physical_shock_absence_established=False, original_gap_resolved=False,
        input_sha256=hashes, source_sha256={str(Path(__file__)): sha(__file__),
            review_source: sha(review_source),
            'experiments/wind_bubble/check_outer_control_calibration.py': sha('experiments/wind_bubble/check_outer_control_calibration.py')},
        settings=asdict(MotionSettings()),
        scope='Two-resolution sampled-profile comparison and corrected control identity. Not continuum convergence, full-domain completeness, or production-mask comparison at 256 cubed.')
    (output/'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('sha256')}, indent=2))
    return result


def make_figures(output, datasets, density, inner):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {128: '#d47724', 256: '#087f8c'}
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True, sharey='row', constrained_layout=True)
    for col, time in enumerate(TARGET_TIMES):
        axes[0, col].set_title(f't={time:.2f}')
        for data in datasets:
            n = data['meta']['num_cells']; e = int(np.argmin(abs(data['times']-time)))
            example = next(r for r in density if r['num_cells'] == n and r['event'] == e)
            distance = data['x']-example['front_x']; q = data['lines'][e]
            visible = (distance >= -.04-1/n) & (distance <= .06+1/n)
            for row, variable in enumerate((0, 4, 1)):
                axes[row, col].plot(distance[visible], q[variable, visible], '.-', ms=3, color=colors[n], label=f'{n} cubed')
        for ax in axes[:, col]:
            ax.set_xlim(-.04, .06); ax.axvline(0, color='.6', ls=':', lw=.8)
        axes[-1, col].set_xlabel('Distance from own outer pressure crest')
    for ax, label in zip(axes[:, 0], ('Density', 'Pressure', 'x velocity')):
        ax.set_ylabel(label+' (code units)')
    axes[0, 0].set_ylim(0, 4.2); axes[0, 0].legend()
    fig.suptitle('Outermost left front: sharper density peak at higher resolution\n'
        'Same physical ray and time; each profile shifted to its own pressure crest', fontsize=12)
    fig.savefig(output/'outer_profiles.png', dpi=155); plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True, sharey='row', constrained_layout=True)
    for col, time in enumerate(TARGET_TIMES):
        axes[0, col].set_title(f't={time:.2f}')
        for data in datasets:
            n = data['meta']['num_cells']; e = int(np.argmin(abs(data['times']-time)))
            q = data['lines'][e]
            visible = (data['x'] >= .43-1/n) & (data['x'] <= .57+1/n)
            for row, field in enumerate((q[0], q[4], q[4]/q[0], q[1])):
                axes[row, col].plot(data['x'][visible], field[visible], '.-', ms=2, color=colors[n], label=f'{n} cubed')
        for ax in axes[:, col]:
            ax.set_xlim(.43, .57); ax.axvline(.5, color='.6', ls=':', lw=.8)
        axes[-1, col].set_xlabel('Physical x coordinate')
    for row, label in enumerate(('Density', 'Pressure', 'Temperature proxy P/rho', 'x velocity')):
        axes[row, 0].set_ylabel(label)
        if row < 3:
            for ax in axes[row]:
                ax.set_yscale('log')
    axes[0, 0].legend()
    fig.suptitle('Central layer: physical profiles, not shock classifications\n'
        'Same ray y=0.375, z=0.5; dotted line is the collision midplane', fontsize=12)
    fig.savefig(output/'central_profiles.png', dpi=155); plt.close(fig)

    data = datasets[1]; e = int(np.argmin(abs(data['times']-.13)))
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.plot(data['x'], data['lines'][e, 4], color=colors[256], label='256-cubed pressure at t=0.13')
    outer = next(r['front_x'] for r in density if r['num_cells'] == 256 and r['event'] == e)
    for i, position in enumerate((outer, 1-outer)):
        ax.axvline(position, color='.2', ls='--', label='Outermost front (control fails)' if i == 0 else None)
    positions = [float(r['x']) for r in inner if abs(float(r['time'])-.13) < 1.e-6]
    ax.scatter(positions, np.interp(positions, data['x'], data['lines'][e, 4]),
        marker='x', s=75, lw=2, color='#a043a6', label='Passing inner feature (not an outer control)', zorder=5)
    for i, center in enumerate((.375, .625)):
        ax.axvspan(center-.0625, center+.0625, color='.8', alpha=.35,
            label='Prescribed injection interval on this ray' if i == 0 else None)
    ax.set(xlim=(.1, .9), yscale='log', xlabel='Physical x coordinate', ylabel='Pressure (code units)',
        title='The two passing noncentral tracks are not the outermost fronts')
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2)
    fig.savefig(output/'control_identity.png', dpi=155); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs=2, type=Path, default=ROOTS)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args(); run(args.runs, args.output_dir)
