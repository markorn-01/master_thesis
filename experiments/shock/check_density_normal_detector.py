"""One fixed detector comparison, gated before any saved-bubble application.

Compare frozen production results with a density-normal, averaged-state hydro
analogue of SHOCKFIND. Failure ends this comparison; no per-case tuning or
automatic promotion to the production shock finder is performed.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform

import jax
import numpy as np

from experiments.shock.check_colliding_shocks import fingerprint, make_analyzer, write_csv
from experiments.shock.check_pressure_velocity_diagnostic import negative_controls
from experiments.shock.density_normal_detector import DensityNormalSettings, analyze_line


def location_match(candidate, original):
    if original['kind'] not in ('pair', 'shock'):
        return False
    shock = float(original['shock_x']); n = int(original['resolution'])
    # Frozen reference cases are right-going. Direction conventions differ:
    # production hot-to-cold points right; the alternative up-to-down points left.
    match = abs(candidate['x']-shock)*n <= 2 and candidate['normal_upstream_to_downstream'] == -1
    if original['kind'] == 'pair':
        match &= candidate['x'] > .5*(shock+float(original['contact_x']))
    return bool(match)


def evaluate_profile(profile, old, settings):
    candidates = analyze_line(profile, 1/int(old['resolution']), settings)
    located = [r for r in candidates if location_match(r, old)]
    accepted = [r for r in located if r['accepted']]
    kinematic = [r for r in located if r['kinematic_candidate']]
    mach = float(np.median([r['mass_inferred_mach'] for r in accepted])) if accepted else None
    expected = float(old['reference_mach']) if old['kind'] in ('shock', 'pair') else None
    row = dict(case=old['case'], kind=old['kind'], resolution=int(old['resolution']),
        baseline_outcome=old['outcome'], filter_cells=old.get('filter_cells', ''),
        baseline_surface_count=int(old.get('all_interior_surface_samples', 0)),
        alternative_candidates=len(candidates),
        alternative_kinematic_count=sum(r['kinematic_candidate'] for r in candidates),
        alternative_accepted_count=sum(r['accepted'] for r in candidates),
        alternative_shock_located_kinematically=bool(kinematic),
        alternative_shock_accepted=bool(accepted), alternative_mach=mach,
        alternative_mach_relative_error=mach/expected-1 if mach is not None else None,
        reference_mach=expected,
        entropy_rejects_located_kinematic_candidate=bool(kinematic and not accepted))
    return row, candidates


def acceptance_decision(rows):
    missed = [r for r in rows if r['baseline_outcome'] == 'no_zone_candidates']
    isolated = [r for r in rows if r['kind'] == 'shock' and int(r['filter_cells']) == 1]
    negatives = [r for r in rows if r['kind'] in ('contact', 'negative_control')]
    prior = [r for r in rows if r['kind'] == 'pair' and r['baseline_outcome'] == 'recovered']
    gates = dict(
        every_known_zone_rejected_shock_recovered=bool(missed) and all(r['alternative_shock_accepted'] for r in missed),
        no_contact_or_smooth_control_accepted=not any(r['alternative_accepted_count'] for r in negatives),
        isolated_exact_shocks_preserved=bool(isolated) and all(r['alternative_shock_accepted']
            and abs(r['alternative_mach_relative_error']) <= .05 for r in isolated),
        no_previously_recovered_pair_detection_lost=all(r['alternative_shock_accepted'] for r in prior))
    return dict(passed=all(gates.values()), gates=gates,
        decision='eligible_for_separate_bubble_comparison' if all(gates.values()) else 'reject_alternative_do_not_apply_to_bubble',
        known_zone_rejected_cases=len(missed),
        known_zone_rejected_located_kinematically=sum(r['alternative_shock_located_kinematically'] for r in missed),
        known_zone_rejected_accepted=sum(r['alternative_shock_accepted'] for r in missed),
        negative_accepted_samples=sum(r['alternative_accepted_count'] for r in negatives),
        negative_kinematic_samples=sum(r['alternative_kinematic_count'] for r in negatives),
        previously_recovered_pair_detections_lost=sum(not r['alternative_shock_accepted'] for r in prior))


def run(args):
    if args.output_dir.exists():
        raise ValueError('Use a fresh output directory.')
    settings = DensityNormalSettings()
    source_paths = [Path(__file__), Path('experiments/shock/density_normal_detector.py'),
        Path('experiments/shock/check_pressure_velocity_diagnostic.py'),
        Path('experiments/shock/check_colliding_shocks.py')]
    source_hashes = {str(p): fingerprint(p) for p in source_paths}
    baseline_path = args.reference/'verification.json'
    baseline = json.loads(baseline_path.read_text())
    for name, expected in baseline['source_sha256'].items():
        path = Path(name.split('master_thesis/')[-1])
        if fingerprint(path) != expected:
            raise ValueError(f'Production baseline source changed; cannot use frozen result: {path}')
        source_hashes[str(path)] = expected
    input_hashes = {str(baseline_path): fingerprint(baseline_path)}
    import csv
    summary_path = args.reference/'summary.csv'
    input_hashes[str(summary_path)] = fingerprint(summary_path)
    with summary_path.open() as stream:
        old_rows = list(csv.DictReader(stream))
    args.output_dir.mkdir(parents=True)
    rows, records = [], []
    for old in old_rows:
        path = args.reference/'profiles'/f'{old["case"]}.npz'
        input_hashes[str(path)] = fingerprint(path)
        with np.load(path, allow_pickle=False) as d:
            profile = np.stack([d['density'], d['velocity_x'], d['pressure']], axis=1)
        row, candidates = evaluate_profile(profile, old, settings)
        rows.append(row)
        records.extend(dict(case=old['case'], **candidate) for candidate in candidates)
    # The original frozen matrix has contacts but no smooth-compression rows.
    # Evaluate the unchanged production finder on those exact additional inputs.
    production = make_analyzer(64)
    for name, state in negative_controls(64):
        fields = production(state.T)
        count = int(fields['shock_surface'][2:-2, 2:-2, 2:-2].sum())
        old = dict(case=name, kind='negative_control', resolution=64, filter_cells='',
            outcome='smooth_control_false_surface' if count else 'smooth_control_rejected',
            all_interior_surface_samples=count)
        row, candidates = evaluate_profile(state.T, old, settings)
        rows.append(row)
        records.extend(dict(case=name, **candidate) for candidate in candidates)
    write_csv(args.output_dir/'summary.csv', rows)
    write_csv(args.output_dir/'candidates.csv', records)
    decision = acceptance_decision(rows)
    make_figure(args.output_dir, args.reference, rows, records)
    for path, expected in {**source_hashes, **input_hashes}.items():
        if fingerprint(path) != expected:
            raise AssertionError(f'Comparison source/input changed: {path}')
    result = dict(comparison_completed=True, fixed_settings=asdict(settings),
        backend=jax.default_backend(), python=platform.python_version(),
        jax=jax.__version__, numpy=np.__version__,
        production_changed=False, simulation_evolved=False, bubble_application_performed=False,
        reference_cases=len(old_rows), additional_negative_controls=len(rows)-len(old_rows),
        sampling_tuned_per_case=False, **decision,
        method_scope='Planar hydro analogue inspired by SHOCKFIND candidate geometry/averaging; not the published MHD classifier.',
        temperature_gradient_used=False, tight_jump_residual_gate_used=False,
        speed_scope='Mass-inferred speed from averaged states, not independent propagation evidence.',
        entropy_scope='Entropy sign of sampled side averages; a mixed contact can invalidate these as physical shock states.',
        source_sha256=source_hashes, input_sha256=input_hashes,
        references=['https://arxiv.org/pdf/1608.02050', 'https://www.clawpack.org/riemann_book/html/Euler.html'])
    (args.output_dir/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('sha256')}, indent=2))
    return result


def make_figure(output, reference, rows, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    examples = [('Isolated shock', 'n064_shock_c0.25_s0_f1_p0'),
        ('Known shock + dense contact', 'n064_pair_c8_s0.5_f1_p0'),
        ('Moving contact only', 'n064_contact_c8_s0_f1_p0'),
        ('Smooth isentropic compression', 'isentropic_compression_w2')]
    negatives = dict(negative_controls(64))
    fig, axes = plt.subplots(3, 4, figsize=(13, 9), dpi=150)
    fig.subplots_adjust(left=.07, right=.98, top=.83, bottom=.18, wspace=.34, hspace=.32)
    for col, (title, name) in enumerate(examples):
        if name in negatives:
            state = negatives[name]; x = (np.arange(64)+.5)/64
        else:
            with np.load(reference/'profiles'/f'{name}.npz') as d:
                state = np.stack([d['density'], d['velocity_x'], d['pressure']]); x = d['x']
        summary = next(r for r in rows if r['case'] == name)
        for ax, field, label in zip(axes[:, col], (state[2], state[1], state[0]), ('Pressure', 'Velocity', 'Density')):
            ax.plot(x, field, 'o-', color='#555555', lw=1, ms=3)
            for r in records:
                if r['case'] != name or not r['kinematic_candidate']:
                    continue
                ax.axvline(r['x'], color='#087f8c' if r['accepted'] else '#d98a27',
                           ls='-' if r['accepted'] else '--', lw=1.3)
            ax.set(xlim=(.37, .63), ylabel=label); ax.grid(alpha=.2)
        axes[0, col].set_title(f'{title}\nKinematic={summary["alternative_kinematic_count"]}; accepted={summary["alternative_accepted_count"]}', fontsize=10)
        axes[2, col].set_xlabel('x')
        if col == 1:
            axes[2, col].axhline(3., color='#087f8c', ls=':', lw=1)
            axes[2, col].text(.04, .78, 'True immediate shocked density: 3\nSampled contact density: 24',
                             transform=axes[2, col].transAxes, fontsize=8)
    fig.suptitle('Bounded density-normal detector comparison | fixed sampling, unchanged production baseline', fontsize=13)
    fig.text(.07, .045, 'Teal: accepted hydro jump. Orange dashed: kinematic candidate rejected by the sampled entropy sign.\n'
        'Extra candidates are not automatically recovered physical shocks. This is a planar hydro analogue, not a test of the full SHOCKFIND MHD code.\n'
        'The acceptance gates in verification.json decide whether a separate bubble comparison is justified.', fontsize=9)
    fig.savefig(output/'detector_comparison.png'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('outputs/shock_contact_reference_n032_n064'))
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())
