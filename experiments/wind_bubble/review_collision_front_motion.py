"""Review saved flow strips without reintegrating the hydrodynamic solution."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.wind_bubble.collision_front_motion import MotionSettings, analyze_motion, strip_fields


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def persistent_runs(observations, key, minimum=5, late_only=False):
    """Five accepted consecutive events on one short pressure-feature path."""
    records = {}
    for row in observations:
        if row.get(key, False) and (not late_only or row['time'] >= .06):
            records.setdefault((row['track'], row['region']), []).append(row)
    result = []
    for (track, region), rows in records.items():
        events = np.array([r['event'] for r in rows])
        breaks = np.flatnonzero(np.diff(events) != 1)+1
        for group in np.split(np.arange(len(rows)), breaks):
            if len(group) >= minimum:
                result.append(dict(track=track, region=region, count=len(group),
                    start_time=rows[group[0]]['time'], end_time=rows[group[-1]]['time']))
    return result


def production_comparison(observations, stages, x, y, z, names):
    """Cell-center mask support within two cells; not refined point equivalence."""
    if 'shock_surface' not in names:
        for row in observations:
            row['production_masks_available'] = False
            row['production_shock_surface_nearby'] = None
            row['compatible_without_nearby_surface_10pct'] = None
            row['above_threshold_without_nearby_surface_10pct'] = None
        return
    dx = x[1]-x[0]
    for row in observations:
        row['production_masks_available'] = True
        mask = stages[row['event']]
        ids = np.flatnonzero(abs(x-row['x']) <= 2*dx)
        dist2 = ((x[ids, None, None]-row['x'])**2 + (y[None, :, None]-.375)**2
                 + (z[None, None, :]-.5)**2)
        nearby = dist2 <= (2*dx)**2
        for index, name in enumerate(names):
            row[f'production_{name}_nearby'] = bool(np.any(mask[index, ids] & nearby))
        row['compatible_without_nearby_surface_10pct'] = bool(
            row['compatible_10pct'] and not row['production_shock_surface_nearby'])
        row['above_threshold_without_nearby_surface_10pct'] = bool(
            row.get('compatible_above_detection_threshold_10pct', False)
            and not row['production_shock_surface_nearby'])


def run(root, output=None):
    root = Path(root); output = root/'assessment' if output is None else Path(output)
    meta_path = root/'metadata.json'
    meta = json.loads(meta_path.read_text())
    if meta.get('status') != 'complete':
        raise ValueError('Require a completed strip simulation; --analyze-only never reintegrates.')
    if output.exists():
        raise ValueError('Use a fresh assessment directory with --analysis-dir.')
    hashes = {str(meta_path): sha(meta_path)}
    masks_recorded = bool(meta['stage_names'])
    if meta.get('production_masks_retained', masks_recorded) != masks_recorded:
        raise ValueError('Inconsistent production-mask metadata.')
    bands, stages, actual = [], [], []
    for name, expected in sorted(meta['strip_sha256'].items()):
        path = root/'strips'/name
        hashes[str(path)] = sha(path)
        if hashes[str(path)] != expected:
            raise ValueError(f'Saved strip changed: {path}')
        with np.load(path, allow_pickle=False) as d:
            bands.append(d['primitive']); actual.append(float(d['time']))
            if masks_recorded:
                stages.append(d['production_stages'].astype(bool))
            elif 'production_stages' in d:
                raise ValueError('Flow-only metadata conflicts with saved masks.')
    np.testing.assert_array_equal(actual, meta['actual_times'])
    np.testing.assert_allclose(actual, meta['requested_times'], rtol=0, atol=1.e-6)
    n = meta['num_cells']; x = (np.arange(n)+.5)/n
    y, z = np.array(meta['transverse_y']), np.array(meta['transverse_z'])
    lines, dy, dz = strip_fields(np.asarray(bands), y, z, 1/n)
    settings = MotionSettings(**{key: value for key, value in meta['motion_settings'].items()})
    observations, pairs, tracks = analyze_motion(lines, actual, x, dy, dz, settings,
        source_positions=meta['source_positions'], source_radius=meta['injection_radius'])
    production_comparison(observations, np.asarray(stages), x, y, z, meta['stage_names'])
    output.mkdir(parents=True)
    write_csv(output/'front_observations.csv', observations)
    write_csv(output/'sampled_jump_checks.csv', pairs)
    runs = persistent_runs(observations, 'compatible_10pct')
    missing = persistent_runs(observations, 'above_threshold_without_nearby_surface_10pct', late_only=True)
    weak = [r for r in persistent_runs(observations, 'compatible_below_detection_threshold_10pct', late_only=True)
            if r['region'].startswith('central')]
    outer = [r for r in runs if r['region'] == 'outer_control']
    central_late = [r for r in persistent_runs(observations, 'compatible_10pct', late_only=True)
                    if r['region'].startswith('central')]
    missing = [r for r in missing if r['region'].startswith('central')]
    verdict = ('inconclusive_outer_control_failed' if not outer else
               'flow_only_physical_check_no_production_mask_comparison' if not masks_recorded else
               'persistent_compatible_central_front_without_production_surface' if missing else
               'persistent_weak_central_front_below_configured_threshold' if weak else
               'persistent_central_front_requires_mask_and_threshold_review' if central_late else
               'no_certified_late_central_front_in_sampled_strip')
    report = dict(analysis_completed=True, verdict=verdict, simulation_resolution=n,
        cpu_smoke_only=n < 64, saved_events=len(actual), primitive_strip_shape=list(np.shape(bands)),
        measured_speed_from='Nine-event local fit of pressure-ridge position versus time; not inferred from jump states.',
        pressure_feature_tracks=len(tracks), observations=len(observations), sampled_pairs=len(pairs),
        persistent_outer_controls=outer, persistent_late_central_fronts=central_late,
        persistent_late_central_fronts_without_nearby_production_surface=missing if masks_recorded else None,
        production_masks_recorded=masks_recorded, production_comparison_performed=masks_recorded,
        persistent_late_weak_central_fronts=weak,
        full_shock_absence_proven=False, full_domain_search=False, production_changed=False,
        success_scope='Persistent entropy-admissible, shock-frame-consistent fronts using independently measured motion, on one physical ray.',
        limitations=['A pressure ridge is not itself a shock.',
            'Finite-window state sampling and local speed fits have finite-resolution errors.',
            'The fitted-speed scatter is not a complete physical uncertainty estimate.',
            'The x-directed strip cannot certify strongly tilted fronts or other parts of the interface.',
            'A failed outer positive control makes central non-detection inconclusive.',
            'No accepted central front does not establish absence of physical shocks.'],
        settings=asdict(settings), input_sha256=hashes,
        analysis_source_sha256={str(Path(__file__)): sha(__file__),
            'experiments/wind_bubble/collision_front_motion.py': sha('experiments/wind_bubble/collision_front_motion.py')})
    if not masks_recorded:
        report['limitations'].append('Production masks were not recorded; this run cannot establish production detections or misses.')
    make_figure(output/'front_motion.png', lines, actual, x, observations)
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise AssertionError(f'Read-only input changed: {path}')
    (output/'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ('analysis_completed', 'verdict', 'saved_events',
        'pressure_feature_tracks', 'persistent_outer_controls', 'persistent_late_central_fronts',
        'persistent_late_central_fronts_without_nearby_production_surface')}, indent=2))
    return report


def make_figure(path, lines, times, x, observations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), dpi=150)
    fig.subplots_adjust(left=.06, right=.97, bottom=.23, top=.83, wspace=.32)
    for ax, values, label in zip(axes, (lines[:, 0], lines[:, 4], lines[:, 4]/lines[:, 0]**(5/3)),
                                 ('Density', 'Pressure', 'Entropy proxy P/ρᵞ')):
        picture = ax.pcolormesh(x, times, np.log10(np.maximum(values, 1.e-30)), shading='nearest', cmap='magma')
        ax.set(xlim=(.4375, .5625), xlabel='x', ylabel='Time', title=label+' (log₁₀)')
        fig.colorbar(picture, ax=ax, pad=.02)
        candidates = [r for r in observations if r['region'].startswith('central') and r['motion_fit_available']]
        if candidates:
            ax.scatter([r['x'] for r in candidates], [r['time'] for r in candidates],
                       color='white', s=3, alpha=.4, label='Pressure feature with motion fit')
        accepted = [r for r in candidates if r['compatible_10pct']]
        if accepted:
            ax.scatter([r['x'] for r in accepted], [r['time'] for r in accepted],
                       color='#00d5b5', s=8, label='Motion + jump checks pass')
    fig.suptitle('Actual flow evolution through the lower collision | independently measured front speed', fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .09), ncol=2)
    fig.text(.06, .025, 'Ray: y=0.375, z=0.5. White marks are pressure features, not shock labels. '
             'Teal requires 10% flux/state/endpoint checks and entropy admissibility.\n'
             'Interpret central rejection only with the outer-shock control reported in verification.json.', fontsize=9)
    fig.savefig(path); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    run(args.root, args.output_dir)
