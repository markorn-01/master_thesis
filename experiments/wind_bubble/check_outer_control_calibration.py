"""Bounded speed-versus-state calibration on saved outer-front controls.

This does not change the detector or turn a failed control into shock absence.
All gas and fitted motion come from the same saved high-cadence evolution.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.shock.colliding_shocks_reference import (
    SymmetricShockCollision, conserved, primitive,
)
from experiments.wind_bubble.collision_front_motion import (
    MotionSettings, analyze_motion, strip_fields,
)
from experiments.wind_bubble.review_collision_front_motion import production_comparison, write_csv


ROOT = Path('outputs/collision_front_motion_iwr_n128')
OUTPUT = Path('outputs/outer_control_calibration_n128')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_window(lines, times, x, event, half, dy=None, dz=None, sources=None, distances=None):
    """Evaluate the outermost left pressure ridge, with unchanged sample gates."""
    if event < half or event+half >= len(times):
        raise ValueError('A complete two-sided time window is required.')
    window = slice(event-half, event+half+1)
    settings = replace(MotionSettings(), fit_half_events=half)
    if distances is not None:
        settings = replace(settings, distances_cells=tuple(distances))
    obs, pairs, _ = analyze_motion(lines[window], times[window], x,
        None if dy is None else dy[window], None if dz is None else dz[window],
        settings=settings, source_positions=sources)
    choices = [r for r in obs if r['event'] == half and r['x'] < .3125
               and r['region'] == 'outer_control' and r['motion_fit_available']]
    if not choices:
        return None, []
    row = min(choices, key=lambda r: r['x'])
    selected = [r for r in pairs if r['event'] == half and r['track'] == row['track']]
    necessary = [r for r in selected if r['necessary_shock_conditions']]
    best = min(necessary, key=lambda r: r['largest_relative_error']) if necessary else None
    result = dict(event=event, time=float(times[event]), fit_events=2*half+1,
        x=row['x'], measured_x_speed=row['measured_x_speed'],
        speed_fit_scatter=row['speed_fit_scatter'], sampled_pairs=len(selected),
        sampling_grid='original' if distances is None else 'quarter_cell',
        necessary_conditions_pairs=len(necessary), compatible_10pct=row['compatible_10pct'],
        compatible_20pct=row['compatible_20pct'])
    result['compatible_above_threshold_10pct'] = row['compatible_above_detection_threshold_10pct']
    if best:
        result.update({'best_'+key: best[key] for key in (
            'largest_relative_error', 'mass_relative_error', 'momentum_relative_error',
            'enthalpy_relative_error', 'density_ratio_error', 'endpoint_relative_change',
            'entropy_signal_error', 'upstream_x', 'downstream_x')})
    for pair in selected:
        pair.update(event=event, time=float(times[event]), fit_events=2*half+1,
            sampling_grid=result['sampling_grid'])
    return result, selected


def density_example(line, x, center, gamma=5/3):
    """Pressure/density compatibility at the recorded downstream density maximum.

    The prediction is a comparison, never a replacement for the measured gas.
    This example alone is not an exhaustive test of all possible endpoints.
    """
    dx = x[1]-x[0]
    up = np.array([np.interp(center-6*dx, x, q) for q in line])
    ids = np.flatnonzero((x > center) & (x <= center+8*dx))
    if not len(ids):
        raise ValueError('No downstream cells in the declared window.')
    i = ids[np.argmax(line[0, ids])]
    pressure_ratio = line[4, i]/up[4]
    predicted = ((gamma+1)*pressure_ratio+gamma-1)/((gamma-1)*pressure_ratio+gamma+1)
    return dict(upstream_x=float(center-6*dx), downstream_x=float(x[i]),
        upstream_density=float(up[0]),
        sampled_density_ratio=float(line[0, i]/up[0]), pressure_ratio=float(pressure_ratio),
        pressure_predicted_density_ratio=float(predicted),
        relative_density_shortfall=float(1-line[0, i]/up[0]/predicted))


def moving_control(left, right, speed, times, n=128):
    edges = np.arange(n+1)/n
    states = []
    for time in times-times[len(times)//2]:
        fraction = np.clip((.2+speed*time-edges[:-1])*n, 0, 1)
        q = primitive(fraction[:, None]*conserved(left)+(1-fraction[:, None])*conserved(right))
        state = np.zeros((5, n)); state[[0, 1, 4]] = q.T
        states.append(state)
    return np.array(states), (edges[:-1]+edges[1:])/2


def known_controls():
    """Same measured-speed procedure on independently specified shocks/contacts."""
    times = np.arange(65)*.0005
    reference = SymmetricShockCollision()
    cases = [
        ('incident_M3', reference.incoming_left, reference.ambient, reference.incoming_speed, True),
        ('reflected', reference.incoming_left, reference.central, -reference.outgoing_speed, True),
        ('contact', np.array([4., 2., 1.]), np.array([1., 2., 1.]), 2., False),
    ]
    weak = SymmetricShockCollision(incoming_mach=1.2)
    cases.append(('weak_M1p2', weak.incoming_left, weak.ambient, weak.incoming_speed, True))
    rows = []
    for name, left, right, speed, expected in cases:
        lines, x = moving_control(left, right, speed, times)
        for half in (4, 8, 16):
            row, _ = evaluate_window(lines, times, x, 32, half)
            accepted = bool(row and row['compatible_10pct'])
            rows.append(dict(case=name, fit_events=2*half+1, expected_compatible=expected,
                compatible_10pct=accepted, passed=accepted == expected,
                measured_x_speed=None if row is None else row['measured_x_speed'],
                reference_speed=speed))
    x = (np.arange(128)+.5)/128
    smooth = []
    for time in times-times[32]:
        p = 1+5*(1-np.tanh((x-.2-time)*128/2))
        rho = p**(3/5); c = np.sqrt((5/3)*p/rho)
        q = np.zeros((5, len(x))); q[0], q[1], q[4] = rho, 3*(c-np.sqrt(5/3)), p
        smooth.append(q)
    for half in (4, 8, 16):
        row, _ = evaluate_window(np.array(smooth), times, x, 32, half)
        accepted = bool(row and row['compatible_10pct'])
        rows.append(dict(case='smooth_isentropic_profile', fit_events=2*half+1,
            expected_compatible=False, compatible_10pct=accepted, passed=not accepted,
            measured_x_speed=None if row is None else row['measured_x_speed'], reference_speed=None))
    return rows


def run(root=ROOT, output=OUTPUT):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('Use a fresh output directory.')
    meta_path = root/'metadata.json'; meta = json.loads(meta_path.read_text())
    if meta.get('status') != 'complete' or meta['num_cells'] != 128:
        raise ValueError('Require the completed 128-cubed strip experiment.')
    hashes = {str(meta_path): sha(meta_path)}
    bands, stages, times = [], [], []
    for name, expected in sorted(meta['strip_sha256'].items()):
        path = root/'strips'/name; actual = sha(path)
        if actual != expected:
            raise ValueError(f'Strip hash mismatch: {path}')
        hashes[str(path)] = actual
        with np.load(path, allow_pickle=False) as data:
            bands.append(data['primitive']); stages.append(data['production_stages'])
            times.append(float(data['time']))
    times = np.array(times)
    np.testing.assert_array_equal(times, meta['actual_times'])
    x = (np.arange(128)+.5)/128
    y, z = np.array(meta['transverse_y']), np.array(meta['transverse_z'])
    lines, dy, dz = strip_fields(np.array(bands), y, z, 1/128)
    rows, pairs, examples = [], [], []
    for target in (.08, .12):
        event = int(np.argmin(abs(times-target)))
        if abs(times[event]-target) > 1.e-6:
            raise ValueError('Required time unavailable.')
        for half, distances in ((4, None), (8, None), (16, None),
                                (16, np.arange(1, 8.001, .25))):
            row, selected = evaluate_window(lines, times, x, event, half, dy, dz,
                sources=meta['source_positions'], distances=distances)
            if row is None:
                raise ValueError('Outermost pressure crest unavailable.')
            support = dict(event=event, x=row['x'], compatible_10pct=False)
            production_comparison([support], np.array(stages), x, y, z, meta['stage_names'])
            row['historical_production_surface_nearby'] = support['production_shock_surface_nearby']
            rows.append(row); pairs.extend(selected)
        examples.append(dict(event=event, time=float(times[event]),
            **density_example(lines[event], x, rows[-1]['x'])))
    controls = known_controls()
    if not all(r['passed'] for r in controls):
        raise AssertionError('Known-state calibration failed; inspect controls before interpreting bubbles.')
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise AssertionError(f'Input changed: {path}')
    output.mkdir(parents=True)
    for name, data in (('outer_windows.csv', rows), ('sampled_pairs.csv', pairs),
                       ('density_examples.csv', examples), ('known_controls.csv', controls)):
        write_csv(output/name, data)
    make_figure(output/'speed_and_state_calibration.png', rows, examples, lines, x)
    result = dict(calibration_completed=True, known_controls_passed=True,
        known_control_checks=len(controls), outer_comparisons=len(rows),
        outer_control_passes_10pct=sum(r['compatible_10pct'] for r in rows),
        outer_control_passes_20pct=sum(r['compatible_20pct'] for r in rows),
        strict_outer_control_repaired=all(r['compatible_10pct'] for r in rows if r['fit_events'] == 33),
        production_changed=False, thresholds_changed=False, simulation_rerun=False,
        original_gap_physically_resolved=False, input_sha256=hashes,
        source_sha256={name: sha(name) for name in (str(Path(__file__)),
            'experiments/wind_bubble/collision_front_motion.py')},
        scope='Bounded calibration of a failed independent control. No shock-absence claim; no speed inferred from Rankine-Hugoniot balance.')
    (output/'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('sha256')}, indent=2))
    return result


def make_figure(path, rows, examples, lines, x):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for col, example in enumerate(examples):
        chosen = [r for r in rows if r['event'] == example['event'] and r['sampling_grid'] == 'original']
        axes[0, col].plot([r['fit_events'] for r in chosen],
            [r['measured_x_speed'] for r in chosen], 'o-', color='#087f8c')
        axes[0, col].set(xlabel='Events in two-sided speed fit', ylabel='Measured x-speed (code units)',
            title=f"Outer front: t={example['time']:.2f}", xticks=[9, 17, 33], ylim=(-1.28, -1.0))
        center = chosen[0]['x']; q = lines[example['event']]
        ax = axes[1, col]
        ax.plot((x-center)*128, q[0]/example['upstream_density'], 'o-', ms=3, color='#087f8c', label='Recorded density ratio')
        ax.axhline(example['pressure_predicted_density_ratio'], ls='--', color='#d47724',
            label='RH ratio at density-maximum sample')
        ax.set(xlim=(-5, 8), ylim=(0, 4.5), xlabel='Cell widths from pressure crest',
            ylabel='Density / sampled upstream density')
        ax.legend(fontsize=8)
        ax.text(.03, .06, f"Density shortfall: {100*example['relative_density_shortfall']:.1f}%",
            transform=ax.transAxes, fontsize=9)
    fig.suptitle('Longer speed fits do not repair the sampled jump control\n'
        'Same saved gas and motion; original jump gates unchanged', fontsize=12)
    fig.savefig(path, dpi=160); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args(); run(args.run_dir, args.output_dir)
