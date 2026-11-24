"""Bounded, blind feasibility test of joint planar shock/contact reconstruction.

Truth is used to generate inputs and score fits, never passed to fit_profile.
This test does not alter the production finder or classify the bubble layer.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

from experiments.shock.check_pressure_velocity_diagnostic import negative_controls
from experiments.shock.joint_shock_contact_fit import (
    FitSettings, fit_profile, forward_fan, profile_parameter, to_conserved, to_primitive,
)
from experiments.shock.shock_contact_reference import ShockContactReference


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases():
    reference = ShockContactReference()
    for n in (32, 64):
        for width in (1, 3, 5):
            q, truth = reference.profile(n, .5, width)
            yield f'n{n:03d}_hidden_f{width}', q, truth, 'joint', True
    for label, orientation, boost in (('reflected', -1, 0.), ('boosted', 1, 17.)):
        q, truth = reference.profile(64, .5, orientation=orientation)
        q[:, 1] += boost
        truth['velocity_boost'] = boost
        yield label, q, truth, 'joint', True
    # New strength, density contrast, phase and separation: none are fit inputs.
    heldout = ShockContactReference(mach=2.4, contact_density_factor=5.)
    q, truth = heldout.profile(64, .35, phase=.23)
    yield 'heldout_M2p4', q, truth, 'joint', True
    q, truth = reference.profile(64, .5)
    conserved = to_conserved(q, reference.gamma)
    rng = np.random.default_rng(20261124)
    conserved += 1.e-4*np.ptp(conserved, axis=0)*rng.uniform(-1, 1, conserved.shape)
    yield 'perturbed_1e-4', to_primitive(conserved, reference.gamma), truth, 'joint', True
    for kind, expected in (('shock', 'shock'), ('contact', 'contact')):
        q, truth = reference.control(64, kind, phase=.37)
        yield kind+'_only', q, truth, expected, False
    for name, q in negative_controls(64):
        expected = 'isentropic' if name.startswith('isentropic') else 'model_mismatch'
        yield name, q.T, {}, expected, False
    # Deliberately outside the fitted top-hat blur family. Failure is a scope
    # control, not evidence that this known shock has disappeared physically.
    q, truth = reference.profile(64, .5)
    values = gaussian_filter1d(to_conserved(q, reference.gamma), .8, axis=0, mode='nearest')
    yield 'unmodeled_gaussian_blur', to_primitive(values, reference.gamma), truth, 'model_mismatch', False


def score_fit(name, q, truth, expected, expect_joint, result):
    n = len(q)
    row = dict(case=name, cells=n, expected_model=expected, selected_model=result['selected_model'],
        joint_reconstruction_supported=result['joint_reconstruction_supported'],
        local_full_rank=result['local_full_rank'], recovered_mach=result['recovered_mach'],
        recovered_downstream_density=result['recovered_downstream'][0],
        recovered_gap_cells=result['recovered_gap_cells'],
        joint_max_residual=next(r['maximum_residual'] for r in result['models'] if r['model'] == 'joint'),
        mach_relative_error=None, downstream_density_relative_error=None,
        shock_position_error_cells=None, contact_position_error_cells=None)
    accurate = True
    if expect_joint:
        density = ShockContactReference(mach=truth['mach'],
            contact_density_factor=truth['contact_density_factor']).downstream[0]
        row.update(mach_relative_error=result['recovered_mach']/truth['mach']-1,
            downstream_density_relative_error=result['recovered_downstream'][0]/density-1,
            shock_position_error_cells=(result['recovered_shock_x']-truth['shock_x'])*n,
            contact_position_error_cells=(result['recovered_contact_x']-truth['contact_x'])*n)
        accurate = (abs(row['mach_relative_error']) <= .05
            and abs(row['downstream_density_relative_error']) <= .05
            and abs(row['shock_position_error_cells']) <= .1
            and abs(row['contact_position_error_cells']) <= .1)
    row['passed'] = bool(result['selected_model'] == expected
        and result['joint_reconstruction_supported'] == expect_joint and accurate)
    return row


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def predicted_joint(fit, n):
    best = next(r for r in fit['models'] if r['model'] == 'joint')
    values = forward_fan(best['parameters'], n, best['blur_cells'], velocity_scale=fit['velocity_scale'])
    q = to_primitive(values, 5/3)
    if best['orientation'] == -1:
        q = q[::-1].copy(); q[:, 1] *= -1
    q[:, 1] += fit['galilean_shift']
    return q


def make_figure(output, inputs, fits, rows, scans):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), dpi=160)
    fig.subplots_adjust(top=.88, bottom=.15, hspace=.38, wspace=.3)
    q = inputs['n064_hidden_f1']; fit = fits['n064_hidden_f1']; x = np.arange(len(q))+.5
    predicted = predicted_joint(fit, len(q))
    ax = axes[0, 0]
    ax.plot(x, q[:, 0], 'o', ms=4, color='#555555', label='Observed cell density')
    ax.plot(x, predicted[:, 0], color='#087f8c', lw=1.5, label='Fitted cell averages')
    xs = fit['recovered_shock_x']*64; xc = fit['recovered_contact_x']*64
    ax.plot([29., xc, xc, xs, xs, 35.], [24, 24, 3, 3, 1, 1], '--', color='#c76824',
            label='Inferred states (truth agrees)')
    ax.set(xlim=(29, 35), ylim=(0, 26), xlabel='Position / cell width', ylabel='Density',
           title='Hidden shocked slab: no pure cell at density 3')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for name, color in [('mach', '#087f8c'), ('gap', '#c76824')]:
        data = [r for r in scans if r['parameter'] == name]
        optimum = fit['recovered_mach'] if name == 'mach' else fit['recovered_gap_cells']
        ax.plot([r['fixed_value']/optimum for r in data],
            [max(r['maximum_residual'], 1.e-12) for r in data], 'o-', ms=3, color=color,
            label='Mach' if name == 'mach' else 'Shock–contact gap')
    ax.axhline(FitSettings().fit_tolerance, color='#777777', ls=':', label='Fit tolerance')
    ax.set(yscale='log', xlabel='Fixed value / best-fit value', ylabel='Largest normalized residual',
           title='Refit other parameters while varying Mach or gap')
    ax.legend(fontsize=8)
    for ax, name, title in [(axes[1, 0], 'isentropic_compression_w2', 'Smooth compression: isentropic model selected'),
                            (axes[1, 1], 'unmodeled_gaussian_blur', 'Different blur: all tested models rejected')]:
        q = inputs[name]; x = (np.arange(len(q))+.5)/len(q)
        ax.plot(x, q[:, 0]/q[:, 0].max(), label='Density / maximum', color='#555555')
        ax.plot(x, q[:, 2]/q[:, 2].max(), label='Pressure / maximum', color='#087f8c')
        ax.set(xlim=(.38, .62), xlabel='x', ylabel='Normalized field', title=title)
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('Joint shock–contact reconstruction: conditional feasibility, not a bubble detector', fontsize=13)
    fig.text(.07, .035, 'Fit inputs: density, velocity, pressure and grid spacing. True Mach, states, locations and blur labels are withheld.\n'
        'Top: the assumed planar Euler fan can recover a hidden state. Bottom: controls expose the limits of that assumption.\n'
        'Local rank, restarts and parameter scans test only the declared model family; unrestricted smooth alternatives are not excluded.', fontsize=9)
    fig.savefig(output/'joint_reconstruction.png'); plt.close(fig)


def run(args):
    if args.output_dir.exists():
        raise ValueError('Use a fresh output directory.')
    settings = FitSettings()
    source_paths = [Path(__file__), Path('experiments/shock/joint_shock_contact_fit.py'),
        Path('experiments/shock/shock_contact_reference.py'),
        Path('experiments/shock/colliding_shocks_reference.py'),
        Path('experiments/shock/check_pressure_velocity_diagnostic.py')]
    sources = {str(p):fingerprint(p) for p in source_paths}
    production = {str(p):fingerprint(p) for p in Path('astronomix/_physics_modules/_shock_finder').glob('*.py')}
    args.output_dir.mkdir(parents=True); (args.output_dir/'profiles').mkdir()
    inputs, fits, rows, hashes = {}, {}, [], {}
    for name, q, truth, expected, expect_joint in cases():
        path = args.output_dir/'profiles'/f'{name}.npz'
        np.savez_compressed(path, density=q[:, 0], velocity_x=q[:, 1], pressure=q[:, 2], dx=1/len(q))
        hashes[str(path)] = fingerprint(path)
        start = time.monotonic()
        result = fit_profile(q, 1/len(q), settings)
        result['elapsed_seconds'] = time.monotonic()-start
        row = score_fit(name, q, truth, expected, expect_joint, result)
        rows.append(row); fits[name] = result; inputs[name] = q
        (args.output_dir/f'{name}.json').write_text(json.dumps(dict(truth_for_scoring_only=truth,
            fit=result, evaluation=row), indent=2, allow_nan=False)+'\n')
        print(f'{name}: {result["selected_model"]}; supported={result["joint_reconstruction_supported"]}; check={row["passed"]}', flush=True)
    main = fits['n064_hidden_f1']; scans = []
    for parameter, values in [('mach', np.linspace(2., 4., 17)), ('gap', np.linspace(0., 1.25, 21))]:
        optimum = main['recovered_mach'] if parameter == 'mach' else main['recovered_gap_cells']
        scans.extend(profile_parameter(inputs['n064_hidden_f1'], 1/64, main, parameter,
            sorted(set(values.tolist()+[optimum])), settings))
    write_csv(args.output_dir/'summary.csv', rows); write_csv(args.output_dir/'parameter_scans.csv', scans)
    make_figure(args.output_dir, inputs, fits, rows, scans)
    for path, digest in {**sources, **production, **hashes}.items():
        if fingerprint(path) != digest:
            raise AssertionError(f'Source or input changed during comparison: {path}')
    passed = all(row['passed'] for row in rows)
    result = dict(completed=True, passed=passed, cases=len(rows),
        positive_reconstructions=sum(r['joint_reconstruction_supported'] for r in rows),
        false_joint_claims=sum(r['joint_reconstruction_supported'] for r in rows if r['expected_model'] != 'joint'),
        fixed_settings=asdict(settings), python=platform.python_version(), numpy=np.__version__,
        fitter_receives_truth=False, production_changed=False, bubble_application_performed=False,
        source_sha256=sources, production_sha256=production, input_sha256=hashes,
        decision='conditional_planar_feasibility_only' if passed else 'feasibility_gates_failed',
        scope='Joint one-shock/one-contact fan with observable far states and a finite top-hat blur family. Not general detection, global physical uniqueness, or validation on the bubble layer.',
        unmodeled_blur_is_a_known_shock=True,
        model_mismatch_does_not_prove_no_shock=True,
        score_scope='BIC-style residual/complexity ranking with a declared numerical floor; not calibrated evidence or physical probabilities.',
        reference='https://www.clawpack.org/riemann_book/html/Euler.html')
    (args.output_dir/'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k, v in result.items() if not k.endswith('sha256')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())
