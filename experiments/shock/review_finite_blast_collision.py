"""Read-only review of finite-blast detector misses and their local stages."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.shock.run_evolved_collision import sha, write_csv


STAGES = ('compression', 'gradient_alignment', 'pressure_jump_pass',
          'temperature_jump_pass', 'minimum_jump', 'shock_zone', 'shock_surface')


def local_stage_counts(data, event, position, n):
    x = (np.arange(n)+.5)/n; roi = abs(x-position) <= 2/n
    masks = {name:data[name][event].astype(bool) for name in STAGES}
    before_jump = roi & masks['compression'] & masks['gradient_alignment']
    return dict(roi_cells=int(roi.sum()), **{k:int((v & roi).sum()) for k, v in masks.items()},
        compression_and_alignment=int(before_jump.sum()),
        before_jump_pressure_pass=int((before_jump & masks['pressure_jump_pass']).sum()),
        before_jump_temperature_pass=int((before_jump & masks['temperature_jump_pass']).sum()),
        before_jump_both_jumps_pass=int((before_jump & masks['minimum_jump']).sum()))


def run(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('Use a fresh review output directory.')
    verification = root/'verification.json'; report = json.loads(verification.read_text())
    if not report['completed'] or not report['independent_controls_passed']:
        raise ValueError('Require completed evolution and passing independent evidence controls.')
    inputs = {str(verification):sha(verification)}
    for path in (root/'observations.csv', root/'tracks.csv', root/'controls.json', root/'resolution_comparison.csv'):
        inputs[str(path)] = sha(path)
    for path, digest in {**report['source_sha256'], **report['input_sha256']}.items():
        if sha(path) != digest:
            raise AssertionError(f'Run source or control input changed: {path}')
    with (root/'observations.csv').open() as stream:
        observations = list(csv.DictReader(stream))
    misses, saved = [], {}
    for n in report['resolutions']:
        path = root/f'finite_blast_n{n:04d}.npz'; inputs[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as archive:
            data = {k:archive[k] for k in archive.files}
        np.testing.assert_array_equal(data['shock_zone'],
            data['compression'] & data['gradient_alignment'] & data['minimum_jump'])
        np.testing.assert_array_equal(data['minimum_jump'],
            data['pressure_jump_pass'] & data['temperature_jump_pass'])
        saved[n] = data
        for row in observations:
            if (int(row['resolution']) != n or row['reflection_candidate_track'] != 'True'
                    or row['evidence'] != 'jump_supported_above_threshold'
                    or row['production_surface_nearby'] == 'True'):
                continue
            event = int(row['event']); position = float(row['x'])
            misses.append(dict(resolution=n, event=event, time=float(row['time']), side=row['side'],
                candidate_x=position, nominal_mach_min=float(row['nominal_mach_min']),
                nominal_mach_max=float(row['nominal_mach_max']),
                mach_lower_sensitivity=float(row['mach_low']),
                mach_upper_sensitivity=float(row['mach_high']),
                **local_stage_counts(data, event, position, n)))
    output.mkdir(parents=True)
    plot_supported_intervals(output, observations)
    if misses:
        write_csv(output/'above_supported_misses.csv', misses)
        # A reproducible selection rule, not the largest discrepancy: earliest
        # left-side miss at the highest resolution where one is present.
        highest = max(r['resolution'] for r in misses)
        example = min((r for r in misses if r['resolution'] == highest and r['side'] == 'left'),
                      key=lambda r:r['time'])
        data = saved[highest]; event = example['event']; n = highest
        x = (np.arange(n)+.5)/n; mask = abs(x-example['candidate_x']) <= 10/n
        local = dict(x=x[mask], profile=data['profiles'][event, mask],
            **{key:data[key][event, mask] for key in STAGES},
            log_pressure_jump=data['log_pressure_jump'][event, mask],
            log_temperature_jump=data['log_temperature_jump'][event, mask])
        np.savez_compressed(output/'example.npz', **local)
        (output/'example.json').write_text(json.dumps(example, indent=2)+'\n')
        plot(output, example, local)
    else:
        example = None
    results = dict(review_completed=True, numerical_integral_controls_passed=report['numerical_integral_controls_passed'],
        exact_stage_intersections_verified=True, independent_above_supported_misses=len(misses),
        misses_without_zone=sum(r['shock_zone'] == 0 for r in misses),
        misses_with_compression_and_alignment_but_no_joint_jump=sum(
            r['compression_and_alignment'] > 0 and r['before_jump_both_jumps_pass'] == 0 for r in misses),
        misses_with_no_compression_alignment_intersection=sum(r['compression_and_alignment'] == 0 for r in misses),
        robust_threshold_crossing_observed=report['observed_robust_threshold_crossing'], example=example,
        input_sha256=inputs, source_sha256={str(Path(__file__)):sha(__file__)},
        production_changed=False, bubble_question_resolved=False,
        scope='Stages of missing detections beside independently motion/jump-supported finite-blast candidates. Sampled compatibility is conditional, not an exact physical oracle or a proof about the original bubble.')
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise AssertionError(f'Read-only review input changed: {path}')
    (output/'verification.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k, v in results.items() if not k.endswith('sha256')}, indent=2))
    return results


def plot(output, example, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), dpi=150)
    fig.subplots_adjust(top=.85, bottom=.15, left=.08, right=.98, hspace=.4, wspace=.3)
    x = (data['x']-example['candidate_x'])*example['resolution']
    q = data['profile']
    for ax, field, label in [(axes[0, 0], q[:, 0], 'Density'), (axes[1, 0], q[:, 2], 'Pressure'),
        (axes[2, 0], q[:, 1], 'Velocity'), (axes[0, 1], q[:, 2]/q[:, 0], 'Temperature proxy P/rho')]:
        ax.plot(x, field, 'o-', ms=3, color='#087f8c')
        ax.axvline(0, color='#c76824', ls='--'); ax.grid(alpha=.2)
        ax.set(xlabel='Cell widths from pressure feature', ylabel=label)
    ax = axes[1, 1]
    matrix = np.array([data[k] for k in STAGES], int)
    ax.imshow(matrix, extent=(x[0]-.5, x[-1]+.5, len(STAGES)-.5, -.5),
        aspect='auto', cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    ax.set_yticks(np.arange(len(STAGES)), [k.replace('_', ' ') for k in STAGES], fontsize=8)
    ax.axvline(0, color='#c76824', ls='--'); ax.set_xlabel('Cell widths from pressure feature')
    ax.set_title('Black = local detector test passes', fontsize=10)
    ax = axes[2, 1]
    gamma = 5/3; mach2 = 1.3**2
    pmin = (2*gamma*mach2-gamma+1)/(gamma+1)
    rmin = (gamma+1)*mach2/((gamma-1)*mach2+2)
    ax.plot(x, data['log_pressure_jump'], label='Measured local log P jump')
    ax.plot(x, data['log_temperature_jump'], label='Measured local log T jump')
    ax.axhline(np.log(pmin), color='#1f77b4', ls=':', label='Required log P jump')
    ax.axhline(np.log(pmin/rmin), color='#ff7f0e', ls=':', label='Required log T jump')
    ax.axvline(0, color='#c76824', ls='--'); ax.grid(alpha=.2)
    ax.set(xlabel='Cell widths from pressure feature', ylabel='Local screening jumps')
    ax.legend(fontsize=7)
    fig.suptitle(f'Finite blast: earliest left miss at {example["resolution"]} cells | t={example["time"]:.4f}\n'
        f'Independent sampled Mach {example["nominal_mach_min"]:.3f}–{example["nominal_mach_max"]:.3f}; '
        f'lower sensitivity value {example["mach_lower_sensitivity"]:.3f} > 1.3', fontsize=13)
    fig.text(.08, .035, 'Orange dashed: independently tracked pressure feature, not an exact physical shock position.\n'
        'A local jump criterion samples only part of the numerical transition. Passing motion/jump evidence is conditional on sampled states.\n'
        'No production thresholds or sampling distances were changed.', fontsize=9)
    fig.savefig(output/'missed_front_stages.png'); plt.close(fig)


def plot_supported_intervals(output, observations):
    """Break lines at every unverified interval; do not interpolate evidence."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=160)
    fig.subplots_adjust(top=.84, bottom=.24, left=.08, right=.97, wspace=.3)
    resolutions = sorted({int(r['resolution']) for r in observations}); highest = max(resolutions)
    for number, n in enumerate(resolutions):
        rows = sorted((r for r in observations if int(r['resolution']) == n
            and r['reflection_candidate_track'] == 'True' and r['side'] == 'left'),
            key=lambda r:(float(r['time']), int(r['track'])))
        x, y, lo, hi = [], [], [], []
        for row in rows:
            verified = row['evidence'].startswith('jump_supported')
            x.append(float(row['time'])); y.append(float(row['nominal_mach_min']) if verified else np.nan)
            lo.append(float(row['mach_low']) if verified else np.nan)
            hi.append(float(row['mach_high']) if verified else np.nan)
        axes[0].plot(x, y, '.-', ms=3, lw=.8, label=f'{n} cells')
        if n == highest:
            axes[1].plot(x, y, color='#087f8c', lw=1, label='Independent sampled Mach')
            axes[1].fill_between(x, lo, hi, color='#087f8c', alpha=.15, label='Declared position/speed sensitivity')
            found = [r for r in rows if r['evidence'].startswith('jump_supported')
                     and r['production_mach_median']]
            axes[1].scatter([float(r['time']) for r in found],
                [float(r['production_mach_median']) for r in found], s=12,
                color='#e67722', label='Production Mach at matched surface')
    for ax in axes:
        ax.axhline(1.3, color='#555555', ls='--', label='Threshold 1.3')
        ax.set(xlabel='Simulation time', ylabel='Mach relative to sampled upstream gas')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_title('Left outgoing candidate: supported intervals')
    axes[1].set_title(f'Independent evidence versus detections | {highest} cells')
    fig.suptitle('Finite blasts: supported weakening without an observed threshold crossing', fontsize=13)
    fig.text(.08, .045, 'Blank intervals are unverified; lines and envelopes do not bridge them. Curves end at the last supported measurement.\n'
        'No supported M < 1.3 phase was observed. An ended track is not evidence that the physical front faded or disappeared.', fontsize=9)
    fig.savefig(output/'mach_supported_intervals.png'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/finite_blast_collision_n0512_n1024_n2048'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); run(args.root, args.output_dir)
