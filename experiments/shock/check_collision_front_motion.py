"""Known space-time references for the independently measured-speed diagnostic."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.shock.colliding_shocks_reference import SymmetricShockCollision, conserved, primitive
from experiments.wind_bubble.collision_front_motion import MotionSettings, analyze_motion
from experiments.wind_bubble.review_collision_front_motion import persistent_runs, write_csv


def moving_jump(left, right, speed, times, n, origin=.1):
    edges = np.arange(n+1)/n
    states = []
    for time in times:
        f = np.clip((origin+speed*time-edges[:-1])*n, 0, 1)
        q = primitive(f[:, None]*conserved(left)+(1-f[:, None])*conserved(right))
        state = np.zeros((5, n)); state[[0, 1, 4]] = q.T
        states.append(state)
    return np.asarray(states)


def controls(n):
    times = np.arange(41)*.0005
    for mach in (3., 1.2):
        reference = SymmetricShockCollision(incoming_mach=mach)
        yield f'moving_shock_mach{mach:g}', times, moving_jump(reference.incoming_left,
            reference.ambient, reference.incoming_speed, times, n), True
    reference = SymmetricShockCollision()
    collision_times = .075+times
    states = []
    for time in collision_times:
        q, _ = reference.cell_averages(n, time)
        state = np.zeros((5, n)); state[[0, 1, 4]] = q.T
        states.append(state)
    yield 'two_outgoing_shocks', collision_times, np.asarray(states), True
    yield 'moving_contact', times, moving_jump([4., 2., 1.], [1., 2., 1.], 2., times, n), False
    x = (np.arange(n)+.5)/n
    states = []
    for time in times:
        p = 1+5*(1-np.tanh((x-.1-time)*n/2))
        rho = p**(3/5); c = np.sqrt((5/3)*p/rho)
        q = np.zeros((5, n)); q[0], q[1], q[4] = rho, 3*(c-np.sqrt(5/3)), p
        states.append(q)
    yield 'translated_smooth_isentropic_profile', times, np.asarray(states), False


def run(args):
    if args.output_dir.exists():
        raise ValueError('Use a fresh output directory.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    settings = MotionSettings()
    paths = [Path(__file__), Path('experiments/wind_bubble/collision_front_motion.py'),
             Path('experiments/shock/colliding_shocks_reference.py')]
    fingerprints = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output_dir.mkdir(parents=True)
    fig, axes = plt.subplots(1, 5, figsize=(15, 5), dpi=150)
    fig.subplots_adjust(left=.06, right=.98, bottom=.24, top=.73, wspace=.35)
    titles = {'moving_shock_mach3': 'Moving shock, Mach 3',
              'moving_shock_mach1.2': 'Moving shock, Mach 1.2',
              'two_outgoing_shocks': 'Two outgoing shocks',
              'moving_contact': 'Moving contact',
              'translated_smooth_isentropic_profile': 'Smooth compression'}
    rows = []
    x = (np.arange(args.num_cells)+.5)/args.num_cells
    for ax, (name, times, lines, expected) in zip(axes, controls(args.num_cells)):
        observations, pairs, _ = analyze_motion(lines, times, x, settings=settings)
        accepted = [r for r in observations if r['compatible_10pct']]
        persistent = persistent_runs(observations, 'compatible_10pct')
        row = dict(case=name, shock_expected=expected, pressure_observations=len(observations),
            compatible_observations_10pct=len(accepted), persistent_runs_10pct=len(persistent),
            expected_classification_matches=bool(persistent) == expected)
        if name == 'two_outgoing_shocks':
            row['expected_classification_matches'] &= all(any(r['region'] == side for r in persistent)
                                                        for side in ('central_left', 'central_right'))
        rows.append(row)
        ax.pcolormesh(x, times, lines[:, 4], shading='nearest', cmap='magma')
        for subset, color, label in ((observations, 'white', 'Pressure feature'),
                                      (accepted, '#00d5b5', 'Motion + jump checks pass')):
            if subset:
                ax.scatter([r['x'] for r in subset], [r['time'] for r in subset], s=7, c=color, label=label)
        ax.set(xlim=((.4375, .5625) if name == 'two_outgoing_shocks' else (.07, .23)),
               xlabel='x', ylabel='Time')
        ax.set_title(titles[name]+f'\nAccepted events: {len(accepted)}', fontsize=11)
        print(row, flush=True)
    fig.suptitle(f'Measured-speed controls | {args.num_cells} cells | fixed settings before the bubble run', fontsize=14)
    fig.text(.06, .055, 'Background: pressure. White: pressure features. Teal: measured motion and full jump checks pass.\n'
        'Shock/contact sequences use exact Euler cell averages. The smooth negative sequence is a translated isentropic profile, not an Euler evolution.', fontsize=9)
    fig.savefig(args.output_dir/'motion_controls.png'); plt.close(fig)
    write_csv(args.output_dir/'summary.csv', rows)
    for path, expected in fingerprints.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise AssertionError('Calibration source changed.')
    report = dict(passed=all(r['expected_classification_matches'] for r in rows),
        num_cells=args.num_cells, source_sha256=fingerprints, settings=asdict(settings),
        hydrodynamic_simulation_evolved=False, production_changed=False, controls=rows,
        scope='Known space-time reference and negative-control validation, not bubble validation.')
    (args.output_dir/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['passed']:
        raise AssertionError('Measured-speed control classification failed; do not promote bubble results.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-cells', type=int, default=128)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())
