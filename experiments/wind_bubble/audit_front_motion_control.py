"""Explain a failed strict jump control using a completed saved-strip run.

This is a failure audit, not an alternative shock finder. Density examples
expose finite-grid state mismatch; sonic checks are necessary conditions only.
No acceptance thresholds, production masks, or historical verdicts are changed.
"""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.wind_bubble.assess_collision_physics import source_free_segments
from experiments.wind_bubble.collision_front_motion import MotionSettings, fit_motion, strip_fields
from experiments.wind_bubble.review_collision_front_motion import write_csv


ERRORS = ('mass_relative_error', 'momentum_relative_error', 'enthalpy_relative_error',
          'tangential_error', 'density_ratio_error', 'endpoint_relative_change')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def one_side_fit(track, index, half):
    """Do not fit one velocity across two sides of the collision midplane."""
    if index < half or index+half >= len(track):
        return None, 'insufficient_events'
    window = track[index-half:index+half+1]
    if min(r['x'] for r in window) < .5 < max(r['x'] for r in window):
        return None, 'crosses_collision_midplane'
    fit = fit_motion(track, index, replace(MotionSettings(), fit_half_events=half))
    return fit, 'available' if fit is not None else 'nonconsecutive_events'


def slope_resolution_allowance(times, cell_width):
    """OLS slope sensitivity to an assumed +/- half-cell error at each point.

    This is a declared position-error sensitivity, not a confidence interval
    for shock speed. A pressure crest need not coincide with the physical shock.
    """
    t = np.asarray(times, float); t = t-t.mean()
    if len(t) < 3 or t @ t <= 0:
        raise ValueError('Require at least three distinct times.')
    return float(.5*cell_width*abs(t).sum()/(t @ t))


def upstream_sonic_samples(line, x, center, normal, speed, sources, dx, region):
    """Broad source-free upstream sample set; no density-jump or plateau gate."""
    positions = center-np.sign(normal[0])*np.arange(.25, 8.001, .25)*dx
    valid = (positions >= x[0]) & (positions <= x[-1])
    if region.startswith('central'):
        valid &= (positions-.5)*(center-.5) > 0
    a = np.column_stack([positions, np.full(len(positions), .375), np.full(len(positions), .5)])
    b = np.tile([center, .375, .5], (len(positions), 1))
    valid &= source_free_segments(a, b, np.asarray(sources), .0625)
    positions = positions[valid]
    if not len(positions):
        return positions, np.zeros(0), np.zeros(0)
    q = np.stack([np.interp(positions, x, field) for field in line])
    sound = np.sqrt((5/3)*q[4]/q[0])
    mach = ((normal @ q[1:4])-normal[0]*speed)/sound
    return positions, mach, sound


def rejection_summary(rows):
    result = []
    for region in ('outer_control', 'central_left', 'central_right'):
        selected = [r for r in rows if r['region'] == region]
        entry = dict(region=region, sampled_pairs=len(selected),
            necessary_conditions_pass=sum(r['necessary_shock_conditions'] == 'True' for r in selected))
        for field in ERRORS:
            entry[field+'_fails_10pct'] = sum(float(r[field]) > .1 for r in selected)
        entry['entropy_signal_fails'] = sum(float(r['entropy_signal_error']) > .5 for r in selected)
        for label in ('05', '10', '20'):
            entry['compatible_pairs_'+label+'pct'] = sum(r['compatible_'+label+'pct'] == 'True' for r in selected)
        result.append(entry)
    return result


def run(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('Use a fresh audit output directory.')
    meta_path = root/'metadata.json'; meta = json.loads(meta_path.read_text())
    if meta['status'] != 'complete' or meta['num_cells'] != 128:
        raise ValueError('Require the completed 128-cubed strip experiment.')
    inputs = {str(meta_path): sha(meta_path)}
    bands = []
    for name, expected in sorted(meta['strip_sha256'].items()):
        path = root/'strips'/name; inputs[str(path)] = sha(path)
        if inputs[str(path)] != expected:
            raise ValueError(f'Strip hash mismatch: {path}')
        with np.load(path, allow_pickle=False) as data:
            bands.append(data['primitive'])
    paths = [root/'assessment'/name for name in ('front_observations.csv', 'sampled_jump_checks.csv', 'verification.json')]
    for path in paths:
        inputs[str(path)] = sha(path)
    with paths[0].open() as stream:
        observations = list(csv.DictReader(stream))
    with paths[1].open() as stream:
        pairs = list(csv.DictReader(stream))
    dx = 1/meta['num_cells']; x = (np.arange(meta['num_cells'])+.5)*dx
    lines, dy, dz = strip_fields(np.stack(bands), np.array(meta['transverse_y']), np.array(meta['transverse_z']), dx)
    tracks = {}
    for row in observations:
        tracks.setdefault(int(row['track']), []).append(dict(event=int(row['event']),
            time=float(row['time']), x=float(row['x']), original=row))
    sonic, fit_counts = [], {}
    for track in tracks.values():
        for index, point in enumerate(track):
            original = point['original']
            if not original.get('normal_x'):
                continue
            n = np.array([float(original['normal_'+axis]) for axis in 'xyz'])
            if abs(n[0]) < .8:
                continue
            for half in (4, 8, 16):
                fit, reason = one_side_fit(track, index, half)
                fit_counts[reason] = fit_counts.get(reason, 0)+1
                if fit is None:
                    continue
                speed, scatter, duration = fit
                positions, mach, sound = upstream_sonic_samples(lines[point['event']], x, point['x'], n,
                    speed, meta['source_positions'], dx, original['region'])
                if not len(mach):
                    continue
                times = [p['time'] for p in track[index-half:index+half+1]]
                allowance = slope_resolution_allowance(times, dx)
                sensitivity = abs(n[0])*(allowance+3*scatter)/sound
                sonic.append(dict(track=int(original['track']), event=point['event'], time=point['time'],
                    x=point['x'], region=original['region'], fit_events=2*half+1,
                    fit_duration=duration, measured_x_speed=speed, speed_fit_scatter=scatter,
                    half_cell_slope_allowance=allowance, upstream_samples=len(mach),
                    maximum_sampled_upstream_mach=float(mach.max()),
                    maximum_with_position_sensitivity=float((mach+sensitivity).max()),
                    maximum_lower_sensitivity_value=float((mach-sensitivity).max()),
                    all_sampled_upstream_values_subsonic=bool(np.all(mach < 1)),
                    subsonic_with_position_sensitivity=bool(np.all(mach+sensitivity < 1))))

    examples = []
    for event in (80, 160, 240, 300):
        candidates = [r for r in observations if int(r['event']) == event and r['region'] == 'outer_control' and float(r['x']) < .3125]
        original = min(candidates, key=lambda r: float(r['x']))
        center = float(original['x']); n = np.array([float(original['normal_'+axis]) for axis in 'xyz'])
        q = lines[event]
        up_x = center-np.sign(n[0])*6*dx
        up = np.array([np.interp(up_x, x, field) for field in q])
        downstream = ((x-center)*np.sign(n[0]) > 0) & (abs(x-center) <= 8*dx)
        ids = np.flatnonzero(downstream)
        a = np.tile([center, .375, .5], (len(ids), 1))
        b = np.column_stack([x[ids], np.full(len(ids), .375), np.full(len(ids), .5)])
        ids = ids[source_free_segments(a, b, np.array(meta['source_positions']), .0625)]
        densest = ids[np.argmax(q[0, ids])]
        p = q[4, densest]/up[4]; g = 5/3
        predicted = ((g+1)*p+g-1)/((g-1)*p+g+1)
        actual = q[0, densest]/up[0]
        examples.append(dict(event=event, time=float(meta['actual_times'][event]),
            front_x=center, upstream_x=up_x, densest_downstream_x=float(x[densest]),
            upstream_density=float(up[0]), largest_recorded_downstream_density=float(q[0, densest]),
            pressure_at_densest_downstream=float(q[4, densest]), pressure_ratio=float(p),
            sampled_density_ratio=float(actual), pressure_predicted_density_ratio=float(predicted),
            relative_density_shortfall=float(1-actual/predicted),
            production_surface_nearby=original['production_shock_surface_nearby'] == 'True'))
    counts = rejection_summary(pairs)
    output.mkdir(parents=True)
    write_csv(output/'strict_gate_failures.csv', counts)
    write_csv(output/'outer_density_examples.csv', examples)
    write_csv(output/'sonic_sensitivity.csv', sonic)
    make_figure(output/'control_failure_and_kinematics.png', lines, x, examples, sonic)
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise AssertionError(f'Read-only input changed: {path}')
    matching_sources, changed_sources = 0, []
    for name, expected in meta['source_sha256'].items():
        local = Path(name.split('master_thesis/')[-1])
        if local.is_file() and sha(local) == expected:
            matching_sources += 1
        else:
            changed_sources.append(str(local))
    selected_times = []
    for time in (.04, .06, .07, .08, .10, .12, .15):
        selected = [r for r in sonic if abs(r['time']-time) < 1.e-6
                    and .45 <= r['x'] < .5 and r['fit_events'] in (17, 33)]
        selected_times.extend(selected)
    write_csv(output/'central_time_examples.csv', selected_times)
    result = dict(audit_completed=True, verified_strips=len(bands), simulation_rerun=False,
        production_changed=False, thresholds_changed=False, original_verdict_preserved=True,
        fit_window_counts=fit_counts, outer_density_examples=examples, strict_gate_failures=counts,
        integration_sources_matching_current_checkout=matching_sources,
        integration_sources_different_from_current_checkout=changed_sources,
        source_sha256={str(Path(__file__)): sha(__file__),
            'experiments/wind_bubble/collision_front_motion.py': sha('experiments/wind_bubble/collision_front_motion.py')},
        input_sha256=inputs, original_question_fully_resolved=False,
        scope='Explains failure of a strict sampled-state control and records sonic necessary-condition sensitivity. No new shock certification or physical-absence proof.',
        interpretation=['The production finder detects the outer control; its sampled finite-grid states fail the additional strict jump gate.',
            'The largest recorded outer-shell density is below the pressure-jump prediction in the recorded examples.',
            'Short peak-position fits are sensitive to grid phase; longer fits crossing the collision midplane are excluded.',
            'A subsonic upstream flow relative to the measured feature conflicts with interpreting that feature as a resolved compressive shock.',
            'Sampled sonic incompatibility does not exclude an unresolved shock inside a mixed cell, a different feature, or other interface locations.',
            'Half-cell position sensitivity plus three fit scatters is a declared sensitivity model, not a rigorous total error bound.'])
    (output/'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('audit_completed', 'verified_strips', 'fit_window_counts', 'outer_density_examples')}, indent=2))
    return result


def make_figure(path, lines, x, examples, sonic):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), dpi=150)
    fig.subplots_adjust(left=.07, right=.98, bottom=.25, top=.83, wspace=.33)
    example = examples[2]; event = example['event']; center = example['front_x']
    ax = axes[0]
    ax.plot((x-center)*128, lines[event, 0], 'o-', ms=3, color='#087f8c', label='Recorded density')
    ax.axhline(example['upstream_density']*example['pressure_predicted_density_ratio'], color='#d98a27', ls='--',
               label='Jump prediction at densest sample')
    ax.set(xlim=(-5, 8), ylim=(0, 4.4), xlabel='Cell widths from outer pressure crest', ylabel='Density',
           title='Outer shock control at t=0.12')
    ax.legend(fontsize=8)
    for ax, outer, title in ((axes[1], True, 'Outer front: sampled upstream Mach'),
                              (axes[2], False, 'Central features: sampled upstream Mach')):
        shown = []
        for count, color in ((17, '#d98a27'), (33, '#087f8c')):
            rows = [r for r in sonic if r['fit_events'] == count and
                ((r['region'] == 'outer_control' and r['x'] < .28) if outer else
                 (r['region'].startswith('central') and .45 <= r['x'] < .5 and r['time'] >= .06))]
            # The outermost tracked front is selected per event for this plot.
            if outer:
                by_event = {}
                for r in rows:
                    if r['event'] not in by_event or r['x'] < by_event[r['event']]['x']:
                        by_event[r['event']] = r
                rows = list(by_event.values())
            if rows:
                shown.extend(rows)
                ax.scatter([r['time'] for r in rows], [r['maximum_sampled_upstream_mach'] for r in rows],
                           s=8, color=color, label=f'{count}-event fit')
                if not outer and count == 33:
                    ax.scatter([r['time'] for r in rows], [r['maximum_with_position_sensitivity'] for r in rows],
                               s=5, color=color, alpha=.35, marker='^', label='33-event position sensitivity')
        ax.axhline(1, color='#444444', ls='--', lw=1)
        ax.set(xlim=(.035, .16), xlabel='Time', ylabel='Maximum over sampled upstream positions', title=title)
        if not outer:
            ax.set_xlim(.06, .16)
            lower = min((r['maximum_sampled_upstream_mach'] for r in shown), default=0)
            upper = max((r['maximum_with_position_sensitivity'] for r in shown if r['fit_events'] == 33), default=1)
            ax.set_ylim(min(-.2, lower-.1), max(1.2, upper+.1))
            ax.set_title('Central features after t=0.06 (left half)')
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('Why the strict outer control failed | original GPU strips, unchanged production finder', fontsize=14)
    fig.text(.07, .045, 'Density panel: no clean downstream state satisfying the ideal jump is represented by these local samples.\n'
        'Mach panels: kinematic necessary conditions, not shock certifications. Missing points lack a usable fit.\n'
        'CSV retains position-error sensitivity; plots show nominal values. Unresolved shocks and untested locations remain possible.', fontsize=9)
    fig.savefig(path); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output_dir)
