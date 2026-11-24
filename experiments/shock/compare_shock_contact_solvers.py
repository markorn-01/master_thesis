"""Bounded HLL/HLLC experiment with an exact shock/contact Riemann solution.

The gas is evolved in 1D on CPU. The unchanged 3D finder is evaluated on a
thin extruded slab. Reference states score results, never seed detections or
replace measured endpoints. No bubble run or production default is changed.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import cumulative_trapezoid

from astronomix import (CARTESIAN, SimulationConfig, SimulationParams,
    construct_primitive_state, finalize_config, get_registered_variables, time_integration)
from astronomix.option_classes.simulation_config import HLL, HLLC, SnapshotSettings
from experiments.shock.colliding_shocks_reference import conserved, flux
from experiments.shock.run_evolved_collision import make_slab_analyzer, sha
from experiments.shock.shock_contact_solver_reference import ShockContactReference
from experiments.wind_bubble.collision_front_motion import MotionSettings, analyze_motion
from experiments.wind_bubble.review_collision_front_motion import write_csv


SOLVERS = {'HLL': HLL, 'HLLC': HLLC}
OUTPUT = Path('outputs/shock_contact_hll_hllc_n128_n256')
TIMES_TO_COMPARE = (.04, .08, .16)


def evolve_case(reference, n, scheme, contact_only=False, events=201, t_end=.2):
    if n < 32 or n % 8 or scheme not in SOLVERS or events < 33 or not 0 < t_end <= .2:
        raise ValueError('Require n>=32 divisible by 8, HLL/HLLC, >=33 events, and 0<t_end<=0.2.')
    reference.positions(t_end)
    config = SimulationConfig(geometry=CARTESIAN, dimensionality=1, num_cells=n,
        box_size=1., riemann_solver=SOLVERS[scheme], return_snapshots=True,
        num_snapshots=events, use_specific_snapshot_timepoints=True,
        snapshot_settings=SnapshotSettings(return_states=True, return_final_state=True),
        print_elapsed_time=False, progress_bar=False)
    variables = get_registered_variables(config)
    initial = reference.cell_averages(n, 0., contact_only)
    state = construct_primitive_state(config=config, registered_variables=variables,
        density=jnp.asarray(initial[:, 0], jnp.float32),
        velocity_x=jnp.asarray(initial[:, 1], jnp.float32),
        gas_pressure=jnp.asarray(initial[:, 2], jnp.float32))
    config = finalize_config(config, state.shape)
    times = np.linspace(0., t_end, events, dtype=np.float32)
    params = SimulationParams(gamma=reference.gamma, t_end=t_end,
        snapshot_timepoints=jnp.asarray(times), C_cfl=.4)
    output = time_integration(state, config, params, variables)
    jax.block_until_ready(output.states)
    profiles = np.moveaxis(np.asarray(output.states)[:,
        [variables.density_index, variables.velocity_index, variables.pressure_index]], 1, -1)
    actual = np.asarray(output.time_points)
    np.testing.assert_allclose(actual, times, rtol=0, atol=2.e-7)
    if not np.isfinite(profiles).all() or (profiles[..., [0, 2]] <= 0).any():
        raise AssertionError('The evolved gas became nonfinite or nonpositive.')
    return profiles, actual, dict(scheme=scheme, riemann_solver=config.riemann_solver,
        num_iterations=int(output.num_iterations), solver_mode=config.solver_mode,
        limiter=config.limiter, reconstruction_order=config.reconstruction_order,
        time_integrator=config.time_integrator, C_cfl=.4, dtype=str(profiles.dtype),
        boundary='open boundaries; compare total conserved content with the exact finite-domain solution')


def motion_checks(profiles, times, reference):
    n = profiles.shape[1]; x = (np.arange(n)+.5)/n
    canonical = np.zeros((len(times), 5, n))
    canonical[:, [0, 1, 4]] = np.moveaxis(profiles, 2, 1)
    rows = []
    for half in (4, 16):
        settings = replace(MotionSettings(), fit_half_events=half, ridge_windows=((.02, .98),))
        observations, _, _ = analyze_motion(canonical, times, x, settings=settings)
        for event, time in enumerate(times):
            if time == 0:
                continue
            target = reference.positions(float(time))[1]
            choices = [r for r in observations if r['event'] == event
                       and abs(r['x']-target)*n <= 2 and r['motion_fit_available']]
            # Multiple nearby features are ambiguous, not a choice of whichever
            # happens to fit the exact speed or pass the jump criteria.
            selected = choices[0] if len(choices) == 1 else None
            rows.append(dict(event=event, time=float(time), fit_events=2*half+1,
                unique_position_matched_feature=selected is not None,
                measured_speed=None if selected is None else selected['measured_x_speed'],
                relative_speed_error=None if selected is None else abs(selected['measured_x_speed']/reference.shock_speed-1),
                compatible_10pct=bool(selected and selected['compatible_10pct']),
                compatible_20pct=bool(selected and selected['compatible_20pct'])))
    return rows


def snapshot_score(profile, exact, fields, reference, time, contact_only=False):
    n = len(profile); x = (np.arange(n)+.5)/n
    result = dict(time=float(time), contact_only=contact_only,
        density_relative_l1=float(abs(profile[:, 0]-exact[:, 0]).sum()/abs(exact[:, 0]).sum()),
        pressure_relative_l1=float(abs(profile[:, 2]-exact[:, 2]).sum()/abs(exact[:, 2]).sum()),
        detected_surface_cells=int(np.count_nonzero(fields['shock_surface'])))
    contact, shock = reference.positions(float(time))
    if contact_only:
        low, high = reference.contact_density, reference.shocked[0]
        result['contact_10_90_width_cells'] = int(np.count_nonzero(
            (profile[:, 0] > low+.1*(high-low)) & (profile[:, 0] < low+.9*(high-low))))
        return result
    refined = x+fields['surface_offsets']*fields['normal_x']/n
    selected = fields['shock_surface'].astype(bool) & (fields['normal_x'] > .9)
    matched = selected & (abs(refined-shock)*n <= 2)
    valid = matched & np.isfinite(fields['mach']) & (fields['mach'] > 0)
    ids = (x >= contact-2/n) & (x <= shock+2/n)
    peak = float(profile[ids, 0].max()); exact_peak = float(exact[ids, 0].max())
    result.update(shock_exists=time > 0, separation_cells=(shock-contact)*n,
        numerical_peak_density=peak, exact_cell_average_peak_density=exact_peak,
        exact_plateau_density=float(reference.shocked[0]),
        peak_density_relative_error=abs(peak/exact_peak-1),
        plateau_density_shortfall=1-peak/reference.shocked[0],
        shock_located=bool(time > 0 and valid.any()),
        measured_mach=float(np.median(fields['mach'][valid])) if time > 0 and valid.any() else None)
    return result


def conservation_metrics(profiles, times, exact):
    """Account for measured open-boundary outflow, not fixed reference fluxes.

    Small numerical waves may reach the open boundary even though no wave in
    the exact fan does. Exact-solution integral error is then an accuracy error,
    not a measure of whether the numerical flux update conserves. The sampled
    boundary budget uses trapezoidal quadrature and is not an exact discrete
    solver-flux audit.
    """
    total = conserved(profiles).mean(axis=1)
    exact_total = conserved(exact).mean(axis=1)
    net_flux = np.array([flux(q[0])-flux(q[-1]) for q in profiles])
    budget = total[0]+cumulative_trapezoid(net_flux, times, axis=0, initial=0)
    error = abs(total-budget)/np.maximum(abs(budget), 1.e-12)
    exact_error = abs(total-exact_total)/np.maximum(abs(exact_total), 1.e-12)
    return dict(maximum_relative_conserved_integral_error=float(error.max()),
        maximum_relative_exact_integral_error=float(exact_error.max()),
        maximum_left_boundary_pressure_departure=float(abs(profiles[:, 0, 2]-profiles[0, 0, 2]).max()))


def decision(rows, motion, stability, resolutions):
    """Predeclared practical gate for considering one later bubble trial.

    Require >=20% reduction of median cell-average peak-density error at the
    selected times at both grids, passing late 33-event strict checks at both
    grids, no pure-contact false detections, and finite positive conservative
    evolutions. This is a trial-selection rule, not universal solver validation.
    """
    density = []
    for n in resolutions:
        values = {}
        for solver in SOLVERS:
            chosen = [r for r in rows if r['num_cells'] == n and r['scheme'] == solver
                      and not r['contact_only'] and any(abs(r['time']-t) < 1.e-6 for t in TIMES_TO_COMPARE)]
            values[solver] = float(np.median([r['peak_density_relative_error'] for r in chosen]))
        density.append(dict(num_cells=n, hll_median_peak_error=values['HLL'],
            hllc_median_peak_error=values['HLLC'],
            at_least_20pct_reduction=bool(values['HLLC'] <= .8*values['HLL'])))
    negatives = [r for r in rows if r['contact_only']]
    strict = {n: sum(r['compatible_10pct'] for r in motion if r['num_cells'] == n
        and r['scheme'] == 'HLLC' and r['fit_events'] == 33 and r['time'] >= .12)
        for n in resolutions}
    negative_ok = bool(negatives) and all(r['detected_surface_cells'] == 0 for r in negatives)
    conservation_ok = all(r['maximum_relative_conserved_integral_error'] <= .001 for r in stability)
    return dict(density_comparison=density, pure_contact_checks=len(negatives),
        pure_contacts_rejected=negative_ok, conserved_integrals_within_0p1pct=conservation_ok,
        hllc_late_strict_pass_counts=strict,
        supports_one_later_bubble_trial=bool(all(r['at_least_20pct_reduction'] for r in density)
            and all(strict.values()) and negative_ok and conservation_ok))


def run(resolutions=(128, 256), output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a fresh output directory.')
    if tuple(resolutions) != (128, 256):
        raise ValueError('This bounded comparison uses the declared 128/256 pair.')
    reference = ShockContactReference()
    if np.max(abs(reference.flux_residuals())) > 1.e-12:
        raise AssertionError('Exact reference jump conditions failed.')
    files = sorted(Path('astronomix').rglob('*.py')) + [Path(__file__),
        Path('experiments/shock/shock_contact_reference.py'),
        Path('experiments/shock/shock_contact_solver_reference.py'),
        Path('experiments/shock/colliding_shocks_reference.py'),
        Path('experiments/shock/run_evolved_collision.py'),
        Path('experiments/wind_bubble/collision_front_motion.py'),
        Path('experiments/wind_bubble/review_collision_front_motion.py')]
    hashes = {str(p): sha(p) for p in files}
    output.mkdir(parents=True)
    rows, motion, stability, configs, datasets = [], [], [], [], {}
    for n in resolutions:
        evaluate = make_slab_analyzer(n)
        for contact_only in (False, True):
            case = 'contact_only' if contact_only else 'shock_contact'
            for scheme in SOLVERS:
                print(f'Evolving {case}, {scheme}, n={n} on {jax.default_backend()}...', flush=True)
                profiles, times, config = evolve_case(reference, n, scheme, contact_only)
                exact = np.array([reference.cell_averages(n, float(t), contact_only) for t in times])
                np.savez_compressed(output/f'{case}_{scheme}_n{n:04d}.npz',
                    profiles=profiles, exact_profiles=exact, times=times)
                datasets[case, scheme, n] = (profiles, times, exact)
                configs.append(dict(case=case, num_cells=n, **config))
                stability.append(dict(case=case, scheme=scheme, num_cells=n,
                    **conservation_metrics(profiles, times, exact), finite_positive=True))
                if not contact_only:
                    motion.extend(dict(num_cells=n, scheme=scheme, **r) for r in motion_checks(profiles, times, reference))
                for event in range(0, len(times), 5):
                    fields = evaluate(profiles[event])
                    rows.append(dict(num_cells=n, scheme=scheme, event=event,
                        **snapshot_score(profiles[event], exact[event], fields, reference, times[event], contact_only)))
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise AssertionError(f'Experiment source changed: {path}')
    for name, values in (('snapshots.csv', rows), ('motion_checks.csv', motion),
                         ('stability.csv', stability), ('solver_configs.csv', configs)):
        write_csv(output/name, values)
    result = dict(experiment_completed=True, backend=jax.default_backend(), jax=jax.__version__,
        python=platform.python_version(), gamma=reference.gamma, exact_mach=reference.mach,
        exact_shock_speed=float(reference.shock_speed), exact_contact_speed=float(reference.shocked[1]),
        exact_downstream_state=reference.shocked.tolist(), reference=asdict(reference),
        reference_flux_residual_max=float(np.max(abs(reference.flux_residuals()))),
        resolutions=list(resolutions), evolved_cases=len(configs), snapshot_checks=len(rows),
        **decision(rows, motion, stability, resolutions),
        production_changed=False, thresholds_changed=False, bubble_simulation_performed=False,
        conservation_method='Sampled actual open-boundary Euler fluxes with trapezoidal time integration; exact-reference integral accuracy is recorded separately.',
        saved_case_sha256={str(path): sha(path) for path in sorted(output.glob('*.npz'))},
        source_sha256=hashes,
        scope='Exact two-wave planar Riemann test of shock/contact broadening and existing solver options; not physical certification of the bubble collision layer.')
    make_figures(output, datasets, rows, reference)
    (output/'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, indent=2))
    return result


def make_figures(output, datasets, rows, reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'HLL': '#d47724', 'HLLC': '#087f8c'}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True, constrained_layout=True)
    for row, n in enumerate((128, 256)):
        for col, time in enumerate(TIMES_TO_COMPARE):
            ax = axes[row, col]
            for scheme in SOLVERS:
                q, ts, exact = datasets['shock_contact', scheme, n]
                e = int(np.argmin(abs(ts-time))); x = (np.arange(n)+.5)/n
                contact, shock = reference.positions(float(ts[e]))
                visible = (x >= contact-.04) & (x <= shock+.04)
                ax.plot(x[visible], q[e, visible, 0], '.-', ms=3, color=colors[scheme], label=scheme)
                if scheme == 'HLL':
                    ax.plot(x[visible], exact[e, visible, 0], 'k--', lw=1, label='Exact conserved-cell averages')
                    ax.axvline(contact, color='.55', ls=':', lw=1)
                    ax.axvline(shock, color='.25', ls=':', lw=1)
                    ax.set_xlim(contact-.035, shock+.035)
            ax.set_title(f'{n} cells, t={time:.2f}; shell width={(shock-contact)*n:.1f} cells')
            ax.set_xlabel('x'); ax.set_ylim(0, 4.2)
        axes[row, 0].set_ylabel('Density')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Known thin layer between a shock and a density contact\n'
        'Same initial states, grid, limiter, time integrator, and output schedule; only HLL/HLLC differs', fontsize=12)
    fig.savefig(output/'shock_contact_density.png', dpi=155); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, n in zip(axes, (128, 256)):
        for scheme in SOLVERS:
            q, ts, exact = datasets['contact_only', scheme, n]
            x = (np.arange(n)+.5)/n; center = reference.positions(float(ts[-1]))[0]
            visible = abs(x-center) <= .13
            ax.plot(x[visible], q[-1, visible, 0], '.-', ms=3, color=colors[scheme], label=scheme)
            if scheme == 'HLL':
                ax.plot(x[visible], exact[-1, visible, 0], 'k--', lw=1, label='Exact conserved-cell averages')
        ax.set(title=f'Pure contact, {n} cells, t=0.20', xlabel='x', ylabel='Density')
    axes[0].legend(fontsize=8)
    fig.suptitle('Negative control: a sharper density boundary must remain undetected as a shock', fontsize=12)
    fig.savefig(output/'pure_contact_density.png', dpi=155); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for scheme in SOLVERS:
        selected = [r for r in rows if not r['contact_only'] and r['scheme'] == scheme
                    and r['time'] > 0 and r['measured_mach'] is not None]
        ax.scatter([100*r['peak_density_relative_error'] for r in selected],
            [100*abs(r['measured_mach']/reference.mach-1) for r in selected],
            s=16, alpha=.7, color=colors[scheme], label=scheme)
    ax.axhline(5, color='.4', ls='--', lw=1, label='5% Mach-error guide')
    ax.axvline(10, color='.6', ls=':', lw=1, label='10% density-error guide')
    ax.set(xlabel='Peak density error relative to exact cell averages (%)',
        ylabel='Pressure-based Mach-number error (%)',
        title='Known shocks can have accurate Mach estimates despite a diluted density peak')
    ax.legend(fontsize=8)
    fig.savefig(output/'density_and_mach_errors.png', dpi=155); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args(); run(output=args.output_dir)
