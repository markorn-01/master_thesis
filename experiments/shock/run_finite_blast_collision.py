"""Finite blast collisions: evolving gas, independent evidence, finder comparison.

The test may find weakening without observing a threshold crossing. A missing
pressure track, failed jump check, or missing detector sample is never relabeled
as physical fading. This is a symmetric planar experiment, not the bubble run.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform

import jax
import numpy as np

from experiments.shock.finite_blast_collision import BlastSettings, evolve_blasts
from experiments.shock.blast_front_evidence import EvidenceSettings, analyze, summaries
from experiments.shock.colliding_shocks_reference import SymmetricShockCollision, conserved, primitive
from experiments.shock.run_evolved_collision import make_slab_analyzer, sha, write_csv


EVOLVED_CONTROL = Path('outputs/evolved_planar_collision_n0256_n0512_n1024/M3_n1024.npz')


def reference_controls(output, settings):
    """The evidence rules must distinguish known strong/weak shocks from smooth flow."""
    n = 2048; reports = []
    for label, incoming, wanted in [('above_threshold', 3., 'jump_supported_above_threshold'),
                                    ('below_threshold', 1.2, 'jump_supported_below_threshold')]:
        reference = SymmetricShockCollision(incoming)
        times = reference.collision_time+np.arange(81)*.001
        profiles = np.array([reference.cell_averages(n, float(t))[0] for t in times])
        rows, _, tracks = analyze(profiles, times, reference.collision_time, settings)
        supported = [r for r in rows if r['evidence'].startswith('jump_supported')]
        valid = (len(supported) >= 20 and {r['side'] for r in supported} == {'left', 'right'}
            and all(r['evidence'] == wanted for r in supported)
            and all(r['nominal_mach_min'] <= reference.outgoing_mach+.01
                    and r['nominal_mach_max'] >= reference.outgoing_mach-.01 for r in supported))
        reports.append(dict(case=label, passed=bool(valid), supported_events=len(supported),
            expected_mach=reference.outgoing_mach, evidence_labels=sorted({r['evidence'] for r in rows})))
        np.savez_compressed(output/f'control_{label}.npz', profiles=profiles, times=times)
    times = np.arange(81)*.001; x = (np.arange(n)+.5)/n; gamma = settings.gamma
    smooth = []
    for t in times:
        left, right = .5-2*t, .5+2*t
        pressure = 1+1.5*(np.tanh((x-left)*n/2)-np.tanh((x-right)*n/2))
        density = pressure**(1/gamma); sound = np.sqrt(gamma*pressure/density)
        center_sound = np.sqrt(gamma*4/(4**(1/gamma)))
        velocity = np.where(x < .5, 1., -1.)*2/(gamma-1)*(center_sound-sound)
        smooth.append(np.column_stack([density, velocity, pressure]))
    smooth = np.array(smooth)
    rows, _, _ = analyze(smooth, times, 0., settings)
    false_claims = sum(r['evidence'].startswith('jump_supported') for r in rows)
    reports.append(dict(case='translated_smooth_isentropic_control', passed=false_claims == 0,
        supported_events=false_claims, expected_mach=None,
        evidence_labels=sorted({r['evidence'] for r in rows})))
    np.savez_compressed(output/'control_smooth.npz', profiles=smooth, times=times)
    # An exact-profile success is insufficient: require the already validated
    # evolved numerical control, which contains small pressure oscillations.
    with np.load(EVOLVED_CONTROL, allow_pickle=False) as data:
        profiles, times = data['profiles'], data['times']
    reference = SymmetricShockCollision(3.)
    start = float(times[np.flatnonzero(times > reference.collision_time)[0]])
    rows, _, _ = analyze(profiles, times, start, settings)
    supported = [r for r in rows if r['evidence'].startswith('jump_supported')]
    passed = (len(supported) >= 20 and {r['side'] for r in supported} == {'left', 'right'}
        and all(r['evidence'] == 'jump_supported_above_threshold' for r in supported)
        and all(abs(.5*(r['nominal_mach_min']+r['nominal_mach_max'])/reference.outgoing_mach-1) < .05
                for r in supported))
    reports.append(dict(case='evolved_numerical_control', passed=bool(passed),
        supported_events=len(supported), expected_mach=reference.outgoing_mach,
        evidence_labels=sorted({r['evidence'] for r in rows})))
    return reports


def compare_detector(row, field, n):
    dx = 1/n; x = (np.arange(n)+.5)*dx
    position = x+dx*field['surface_offsets']*field['normal_x']
    sign = row['normal_upstream_to_downstream']
    near = (abs(position-row['x']) <= 2*dx) & (field['normal_x']*sign < -.9)
    selected = field['shock_surface'] & near
    valid = selected & np.isfinite(field['mach']) & (field['mach'] > 0)
    cells = abs(x-row['x']) <= 2*dx
    row.update(production_surface_nearby=bool(selected.any()),
        production_mach_median=float(np.median(field['mach'][valid])) if valid.any() else None,
        production_compression_nearby=bool((field['compression'] & cells).any()),
        production_alignment_nearby=bool((field['gradient_alignment'] & cells).any()),
        production_jump_nearby=bool((field['minimum_jump'] & cells).any()),
        production_zone_nearby=bool((field['shock_zone'] & cells).any()))


def conservation(profiles, settings):
    values = conserved(profiles, settings.gamma)
    total = values.mean(axis=1); scale = np.maximum(abs(total[0]), 1.)
    boundary = profiles[:, [0, 1, -2, -1], :]
    ambient = np.array([settings.density, 0., settings.ambient_pressure])
    boundary_error = float(np.max(abs(boundary-ambient)/np.maximum(abs(ambient), 1.)))
    return dict(maximum_normalized_integral_change=float((abs(total-total[0])/scale).max()),
        maximum_boundary_state_change=boundary_error,
        boundaries_undisturbed=boundary_error < 1.e-5,
        initial_total_mass=float(total[0, 0]), initial_total_energy=float(total[0, 2]),
        maximum_symmetry_error=float((abs(profiles-profiles[:, ::-1]*[1., -1., 1.])/
            np.maximum(abs(profiles).max(axis=(0, 1)), 1.)).max()))


def at_time(profiles, times, time):
    if time < times[0]-2.e-6 or time > times[-1]+2.e-6:
        raise ValueError('Comparison time lies outside saved evolution.')
    i = int(np.clip(np.searchsorted(times, time), 1, len(times)-1))
    weight = np.clip((time-times[i-1])/(times[i]-times[i-1]), 0., 1.)
    # Only for resolution comparisons; never modify detector input profiles.
    return (1-weight)*conserved(profiles[i-1])+weight*conserved(profiles[i])


def run(args):
    if args.output_dir.exists():
        raise ValueError('Use a fresh output directory.')
    resolutions = sorted(set(args.resolutions))
    if len(resolutions) < 2 or any(n < 256 or n % 4 for n in resolutions):
        raise ValueError('Require at least two resolutions, multiples of four >=256.')
    blast, evidence = BlastSettings(), EvidenceSettings()
    files = [Path(__file__), Path('experiments/shock/finite_blast_collision.py'),
        Path('experiments/shock/blast_front_evidence.py'), Path('experiments/shock/run_evolved_collision.py'),
        Path('experiments/shock/colliding_shocks_reference.py'),
        Path('experiments/wind_bubble/audit_detection_stages.py'),
        Path('experiments/wind_bubble/collision_front_motion.py'),
        Path('experiments/wind_bubble/audit_front_motion_control.py')]
    files += sorted(Path('astronomix').rglob('*.py'))
    hashes = {str(p):sha(p) for p in files}
    inputs = {str(EVOLVED_CONTROL):sha(EVOLVED_CONTROL)}
    args.output_dir.mkdir(parents=True)
    controls = reference_controls(args.output_dir, evidence)
    (args.output_dir/'controls.json').write_text(json.dumps(controls, indent=2)+'\n')
    if not all(c['passed'] for c in controls):
        raise AssertionError('Independent evidence controls failed; do not interpret finite-blast candidates.')
    all_rows, all_pairs, track_rows, simulations, cases = [], [], [], {}, []
    for n in resolutions:
        profiles, times, solver = evolve_blasts(n, blast)
        center_pressure = profiles[:, n//2-1:n//2+1, 2].mean(axis=1)
        indices = np.flatnonzero(center_pressure > 2*blast.ambient_pressure)
        if not len(indices):
            raise AssertionError('No central interaction onset during the prescribed run.')
        onset = float(times[indices[0]])  # Explicit pressure-rise proxy, not exact collision time.
        rows, pairs, tracks = analyze(profiles, times, onset, evidence)
        track_summary = summaries(rows, tracks, 1/n)
        retained = {r['track'] for r in track_summary if r['reflection_candidate']}
        for row in rows:
            row['reflection_candidate_track'] = row['track'] in retained
        evaluate = make_slab_analyzer(n)
        fields = [evaluate(q) for q in profiles]
        for row in rows:
            compare_detector(row, fields[row['event']], n)
        reflection = [r for r in rows if r['track'] in retained]
        above = [r for r in reflection if r['evidence'] == 'jump_supported_above_threshold']
        below = [r for r in reflection if r['evidence'] == 'jump_supported_below_threshold']
        fluid = conservation(profiles, blast)
        case = dict(resolution=n, events=len(times), solver=solver, fluid_control=fluid,
            interaction_pressure_rise_proxy_time=onset, pressure_feature_tracks=len(tracks),
            central_born_outward_tracks=len(retained),
            jump_supported_above_threshold_events=len(above),
            above_threshold_without_detection=sum(not r['production_surface_nearby'] for r in above),
            jump_supported_below_threshold_events=len(below),
            above_then_below_candidate_tracks=sum(r['ordered_above_then_below'] for r in track_summary
                if r['reflection_candidate']),
            nominal_supported_mach_min=min((r['nominal_mach_min'] for r in above+below), default=None),
            nominal_supported_mach_max=max((r['nominal_mach_max'] for r in above+below), default=None))
        cases.append(case)
        all_rows.extend(dict(resolution=n, **r) for r in rows)
        all_pairs.extend(dict(resolution=n, **r) for r in pairs)
        track_rows.extend(dict(resolution=n, **r) for r in track_summary)
        arrays = {key:np.stack([f[key] for f in fields]) for key in fields[0]}
        arrays.update(profiles=profiles, times=times, center_pressure=center_pressure)
        np.savez_compressed(args.output_dir/f'finite_blast_n{n:04d}.npz', **arrays)
        simulations[n] = (profiles, times, fields, rows)
        print(f'{n} cells: outward candidates={len(retained)}; above-supported={len(above)}; '
            f'above without detection={case["above_threshold_without_detection"]}; below-supported={len(below)}', flush=True)
    convergence = []
    finest = resolutions[-1]; fine_profiles, fine_times, _, _ = simulations[finest]
    for n in resolutions[:-1]:
        if finest % n:
            raise ValueError('Finest grid must be an integer refinement of each compared grid.')
        q, times, _, _ = simulations[n]
        for time in (.015, .02, .025, .03, .04, .06):
            coarse = at_time(q, times, time)
            fine = at_time(fine_profiles, fine_times, time).reshape(n, finest//n, 3).mean(axis=1)
            scales = np.maximum(abs(fine).mean(axis=0), 1.)
            convergence.append(dict(resolution=n, comparison_resolution=finest, time=time,
                normalized_conserved_l1=float((abs(coarse-fine)/scales).mean())))
    write_csv(args.output_dir/'observations.csv', all_rows)
    write_csv(args.output_dir/'sampled_jump_checks.csv', all_pairs)
    write_csv(args.output_dir/'tracks.csv', track_rows)
    write_csv(args.output_dir/'resolution_comparison.csv', convergence)
    plot(args.output_dir, simulations, cases)
    for name, digest in {**hashes, **inputs}.items():
        if sha(name) != digest:
            raise AssertionError(f'Experiment source changed during execution: {name}')
    fluid_pass = all(c['fluid_control']['boundaries_undisturbed']
        and c['fluid_control']['maximum_normalized_integral_change'] < 5.e-5 for c in cases)
    result = dict(completed=True, finite_energy=True, simulation_evolved=True,
        blast_settings=asdict(blast), evidence_settings=asdict(evidence),
        independent_controls_passed=all(c['passed'] for c in controls), numerical_integral_controls_passed=fluid_pass,
        observed_robust_threshold_crossing=any(c['above_then_below_candidate_tracks'] for c in cases),
        production_changed=False, thresholds_changed=False, bubble_question_resolved=False,
        resolutions=resolutions, cases=cases, source_sha256=hashes, input_sha256=inputs,
        python=platform.python_version(), jax=jax.__version__, numpy=np.__version__, backend=jax.default_backend(),
        motion_scope='Pressure-feature tracks and sampled jump compatibility, independent of production masks. Necessary physical evidence, not exact trajectories or universal shock certification.',
        sensitivity_scope='Half-cell position sensitivity, three speed-fit scatters, and short/long-fit difference; not a rigorous total error bound.',
        fading_scope='Only an ordered, supported above/below transition on a retained outward track counts as observed threshold crossing. An unverified or lost track is not fading.',
        spatial_scope='Fixed diagnostic band 0.3<x<0.7. Exiting that band is censoring, not disappearance.',
        reference_scope='Finest numerical comparison is not an exact solution. Smooth control is a kinematic isentropic counterexample, not a claimed Euler evolution.',
        reference='https://www.clawpack.org/gallery/_static/amrclaw/examples/euler_1d_wcblast/README.html')
    (args.output_dir/'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k, v in result.items() if k not in ('source_sha256', 'cases')}, indent=2))
    return result


def plot(output, simulations, cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = max(simulations); profiles, times, fields, rows = simulations[n]
    x = (np.arange(n)+.5)/n
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), dpi=160, sharey=True)
    fig.subplots_adjust(top=.85, bottom=.22, left=.06, right=.97, wspace=.25)
    for ax, index, title in zip(axes, (0, 2, 1), ('Density', 'Pressure', 'Velocity')):
        data = np.log10(profiles[:, :, index]) if index in (0, 2) else profiles[:, :, index]
        image = ax.pcolormesh(x, times, data, shading='auto', cmap='coolwarm' if index == 1 else 'viridis')
        for event, field in enumerate(fields):
            selected = field['shock_surface']; position = x+field['surface_offsets']*field['normal_x']/n
            ax.scatter(position[selected], np.full(selected.sum(), times[event]), s=3, color='#ff7733', edgecolors='none')
        candidate = [r for r in rows if r['reflection_candidate_track']]
        ax.scatter([r['x'] for r in candidate], [r['time'] for r in candidate], s=3,
            color='white', marker='.', alpha=.8)
        ax.set(xlim=(.25, .75), ylim=(0, times[-1]), xlabel='x', title=('log10 '+title) if index != 1 else title)
        fig.colorbar(image, ax=ax, fraction=.05, pad=.03)
    axes[0].set_ylabel('Simulation time')
    fig.suptitle(f'Finite blast collision | {n} cells | no continuing source', fontsize=14)
    fig.text(.06, .055, 'Orange: production detections. White: independently tracked central-born outward pressure features (not automatic shock labels).\n'
        'Motion and sampled jump evidence are assessed separately. A lost track or central detection gap does not establish physical fading.\n'
        'Diagnostic track band: 0.3 < x < 0.7. Its boundary is a measurement limit, not a physical termination.', fontsize=9)
    fig.savefig(output/'finite_blast_spacetime.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=160)
    fig.subplots_adjust(top=.84, bottom=.23, left=.08, right=.97, wspace=.3)
    colors = {resolution:plt.get_cmap('tab10')(i) for i, resolution in enumerate(sorted(simulations))}
    for resolution, (_, _, _, observations) in sorted(simulations.items()):
        selected = [r for r in observations if r['reflection_candidate_track'] and r['side'] == 'left'
                    and r['evidence'].startswith('jump_supported')]
        if selected:
            axes[0].plot([r['time'] for r in selected], [r['nominal_mach_min'] for r in selected], '.-',
                ms=3, lw=.8, color=colors[resolution], label=f'{resolution} cells')
        if resolution == n and selected:
            axes[1].plot([r['time'] for r in selected], [r['nominal_mach_min'] for r in selected],
                color='#087f8c', label='Independent sampled jump/motion evidence')
            axes[1].fill_between([r['time'] for r in selected], [r['mach_low'] for r in selected],
                [r['mach_high'] for r in selected], color='#087f8c', alpha=.15, label='Declared sensitivity envelope')
            detected = [r for r in selected if r['production_mach_median'] is not None]
            axes[1].scatter([r['time'] for r in detected], [r['production_mach_median'] for r in detected],
                s=12, color='#e67722', label='Production Mach at matched surface')
    for ax in axes:
        ax.axhline(1.3, color='#555555', ls='--', label='Detection threshold 1.3')
        ax.set(xlabel='Simulation time', ylabel='Mach relative to sampled upstream gas')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_title('Left outgoing candidate: resolution comparison')
    axes[1].set_title(f'Independent evidence and detections | {n} cells')
    fig.suptitle('Weakening must be measured; missing detection alone is not fading', fontsize=13)
    fig.text(.08, .055, 'Only events passing the independent sampled-jump and motion checks are shown. Unverified intervals are retained in CSV.\n'
        'Curves do not establish shock absence elsewhere, exact intermediate states, or a rigorous uncertainty interval.', fontsize=9)
    fig.savefig(output/'finite_blast_mach.png'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolutions', type=int, nargs='+', default=[512, 1024, 2048])
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())
