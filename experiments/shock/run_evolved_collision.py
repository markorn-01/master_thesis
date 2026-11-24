"""Evolve known colliding planar shocks, then test the production 3D finder.

The gas is genuinely advanced by Astronomix in 1D. Each saved profile is
extruded onto a thin Cartesian slab solely for the unchanged 3D detection
path. The independent exact fan is used for scoring, never to seed detections.
Constant incoming reservoirs do not model later fading inside bubble cavities.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import jax
import jax.numpy as jnp
import numpy as np

from astronomix import (
    CARTESIAN, SimulationConfig, SimulationParams, construct_primitive_state,
    finalize_config, get_helper_data, get_registered_variables, time_integration,
)
from astronomix.option_classes.simulation_config import SnapshotSettings, StaticFloatVector
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import find_shocks_pfrommer
from experiments.wind_bubble.audit_detection_stages import stage_fields
from experiments.shock.colliding_shocks_reference import SymmetricShockCollision, conserved, flux


SLAB_WIDTH = 8
MIN_SEPARATION_CELLS = 16.
LOCATION_TOLERANCE_CELLS = 2.
MACH_RELATIVE_TOLERANCE = .05
SPEED_RELATIVE_TOLERANCE = .05


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evolve(reference, resolution, events=81, final_collision_time_ratio=1.6):
    """Default HLL/minmod/RK2 solver, as in the hydrodynamic bubble setup."""
    if resolution < 32 or resolution % 4 or events < 5 or not 1.2 <= final_collision_time_ratio <= 2.:
        raise ValueError('Require resolution divisible by four, >=32; >=5 events; final t/t_collision in [1.2,2].')
    profile, _ = reference.cell_averages(resolution, 0.)
    config = SimulationConfig(geometry=CARTESIAN, dimensionality=1, num_cells=resolution,
        box_size=1., return_snapshots=True, num_snapshots=events,
        use_specific_snapshot_timepoints=True,
        snapshot_settings=SnapshotSettings(return_states=True, return_final_state=True),
        print_elapsed_time=False, progress_bar=False)
    variables = get_registered_variables(config)
    initial = construct_primitive_state(config=config, registered_variables=variables,
        density=jnp.asarray(profile[:, 0], jnp.float32),
        velocity_x=jnp.asarray(profile[:, 1], jnp.float32),
        gas_pressure=jnp.asarray(profile[:, 2], jnp.float32))
    config = finalize_config(config, initial.shape)
    end = final_collision_time_ratio*reference.collision_time
    times = np.linspace(0., end, events, dtype=np.float32)
    params = SimulationParams(gamma=reference.gamma, t_end=end,
        snapshot_timepoints=jnp.asarray(times), C_cfl=.4)
    output = time_integration(initial, config, params, variables)
    jax.block_until_ready(output.states)
    actual = np.asarray(output.time_points)
    values = np.asarray(output.states)[:, [variables.density_index, variables.velocity_index, variables.pressure_index], :]
    profiles = np.moveaxis(values, 1, -1)
    if not np.isfinite(profiles).all() or (profiles[:, :, [0, 2]] <= 0).any():
        raise AssertionError('Nonfinite or nonpositive evolved hydrodynamic state.')
    np.testing.assert_allclose(actual, times, rtol=0, atol=2.e-7*reference.collision_time)
    return profiles, actual, dict(num_iterations=int(output.num_iterations),
        solver_mode=config.solver_mode, riemann_solver=config.riemann_solver,
        limiter=config.limiter, reconstruction_order=config.reconstruction_order,
        time_integrator=config.time_integrator, grid_spacing=float(config.grid_spacing),
        boundary='open constant reservoirs; exact boundary flux is accounted for',
        dtype=str(profiles.dtype))


def make_slab_analyzer(resolution, width=SLAB_WIDTH):
    """O(N*width^2) planar 3D evaluation; no N-cubed simulation or smoothing."""
    config = SimulationConfig(geometry=CARTESIAN, dimensionality=3, num_cells=resolution,
        box_size=StaticFloatVector(1., width/resolution, width/resolution), mhd=False)
    variables = get_registered_variables(config)
    shape = (resolution, width, width)
    ones, zeros = jnp.ones(shape), jnp.zeros(shape)
    template = construct_primitive_state(config=config, registered_variables=variables,
        density=ones, velocity_x=zeros, velocity_y=zeros, velocity_z=zeros, gas_pressure=ones)
    config = finalize_config(config, template.shape)
    helper = get_helper_data(config)

    @jax.jit
    def analyze(profile):
        state = template.at[variables.density_index].set(jnp.broadcast_to(profile[:, 0, None, None], shape))
        state = state.at[variables.velocity_index.x].set(jnp.broadcast_to(profile[:, 1, None, None], shape))
        state = state.at[variables.pressure_index].set(jnp.broadcast_to(profile[:, 2, None, None], shape))
        result = find_shocks_pfrommer(state, config, variables, helper, profile_aware_mach=True)
        stages = stage_fields(state, config, variables, helper)
        # All interior transverse rays must agree: extrusion cannot supply new
        # physical evidence or hide a failure on the selected central ray.
        plane = (slice(None), slice(2, -2), slice(2, -2))
        zone = result.shock_zones[plane]; surface = result.shock_surface_cells[plane]
        mismatch = jnp.sum(stages['shock_zone'] != result.shock_zones)
        mismatch += jnp.sum(stages['shock_surface'] != result.shock_surface_cells)
        mismatch += jnp.sum(zone != zone[:, :1, :1])+jnp.sum(surface != surface[:, :1, :1])
        center = width//2
        fields = {k:v[:, center, center] for k, v in stages.items() if v.ndim == 3}
        fields.update(normal_x=stages['direction'][:, center, center, 0],
            mach=result.mach_numbers[:, center, center], mask_mismatch=mismatch)
        return fields

    def evaluate(profile):
        fields = {k:np.asarray(v) for k, v in analyze(jnp.asarray(profile, jnp.float32)).items()}
        if int(fields.pop('mask_mismatch')):
            raise AssertionError('Stage instrumentation or transverse extrusion differs from production.')
        return fields
    return evaluate


def known_structure(reference, time):
    if abs(time-reference.collision_time) <= 2.e-7*reference.collision_time:
        return reference.structure(reference.collision_time)
    return reference.structure(time)


def front_metrics(fields, structure, n):
    """Separate sides and normal signs; one midpoint detection cannot match both."""
    x = (np.arange(n)+.5)/n
    refined = x+fields['surface_offsets']*fields['normal_x']/n
    selected = fields['shock_surface'].astype(bool)
    rows = []
    if structure['phase'] == 'meeting':
        return rows
    for side, (target, normal) in enumerate(zip(structure['positions'], structure['hot_to_cold_directions'])):
        half = refined < .5 if side == 0 else refined > .5
        candidate = selected & half & (fields['normal_x']*normal > .9)
        matched = candidate & (abs(refined-target)*n <= LOCATION_TOLERANCE_CELLS)
        valid = matched & np.isfinite(fields['mach']) & (fields['mach'] > 0)
        mach = float(np.median(fields['mach'][valid])) if valid.any() else None
        neighborhood = abs(x-target)*n <= 3
        row = dict(side='left' if side == 0 else 'right', expected_x=float(target),
            expected_mach=float(structure['mach']), matched_surface_cells=int(matched.sum()),
            located=bool(matched.any()), measured_mach=mach,
            mach_relative_error=mach/structure['mach']-1 if mach is not None else None,
            location_error_cells=float(np.median(abs(refined[matched]-target)*n)) if matched.any() else None,
            compression_nearby=bool((fields['compression'] & neighborhood).any()),
            aligned_gradients_nearby=bool((fields['gradient_alignment'] & neighborhood).any()),
            jump_nearby=bool((fields['minimum_jump'] & neighborhood).any()),
            zone_nearby=bool((fields['shock_zone'] & neighborhood).any()))
        row['located_and_mach_accurate'] = bool(row['located'] and mach is not None
            and abs(row['mach_relative_error']) <= MACH_RELATIVE_TOLERANCE)
        rows.append(row)
    return rows


def pressure_fronts(profile, dx):
    """Independent numerical pressure-gradient crests, without finder masks."""
    strength = abs(np.gradient(profile[:, 2], dx)); n = len(profile)
    positions = []
    for start, stop in ((2, n//2-1), (n//2+1, n-2)):
        i = start+int(np.argmax(strength[start:stop]))
        lower, center, upper = strength[i-1:i+2]
        denominator = lower-2*center+upper
        offset = np.clip(.5*(lower-upper)/denominator, -.5, .5) if denominator < 0 else 0.
        positions.append((i+.5+offset)*dx)
    return np.asarray(positions)


def hydrodynamic_metrics(profile, reference, time):
    n = len(profile); x = (np.arange(n)+.5)/n
    exact, structure = reference.cell_averages(n, time)
    got, expected = conserved(profile), conserved(exact)
    scale = np.maximum(np.mean(abs(expected), axis=0), 1.e-12)
    initial, _ = reference.cell_averages(n, 0.)
    total_expected = conserved(initial).mean(axis=0)+time*(flux(reference.incoming_left)-flux(reference.incoming_right))
    total_scale = np.maximum(np.mean(abs(expected), axis=0), 1.)
    integral_errors = abs(got.mean(axis=0)-total_expected)/total_scale
    row = dict(conserved_l1_normalized=float(np.mean(abs(got-expected)/scale)),
        maximum_boundary_flux_balance_error=float(integral_errors.max()),
        symmetry_error=float(np.max(abs(profile-profile[::-1]*[1., -1., 1.])/
            np.maximum(np.max(abs(profile), axis=0), 1.))),
        slab_density_median=None, slab_pressure_median=None, slab_velocity_rms_over_c2=None,
        midpoint_density=None, midpoint_temperature=None)
    if time > reference.collision_time:
        # Central half of the exact slab avoids the fronts but includes any
        # midpoint artifact. Full profiles and midpoint values are also saved.
        half = .5*(structure['positions'][1]-.5)
        slab = abs(x-.5) <= half
        midpoint = np.argsort(abs(x-.5))[:2]
        if slab.any():
            c2 = np.sqrt(reference.gamma*reference.incoming_left[2]/reference.incoming_left[0])
            row.update(slab_density_median=float(np.median(profile[slab, 0])),
                slab_pressure_median=float(np.median(profile[slab, 2])),
                slab_velocity_rms_over_c2=float(np.sqrt(np.mean(profile[slab, 1]**2))/c2))
        row.update(midpoint_density=float(np.mean(profile[midpoint, 0])),
            midpoint_temperature=float(np.mean(profile[midpoint, 2]/profile[midpoint, 0])))
    return row


def summarize_case(rows, fronts, profiles, times, reference, n):
    required = [r for r in rows if r['separated_outgoing_control']]
    required_fronts = [r for r in fronts if r['separated_outgoing_control']]
    speeds = []
    keep = times/reference.collision_time >= 1.2-1.e-6
    crests = np.array([pressure_fronts(q, 1/n) for q in profiles])
    for side, sign in enumerate((-1, 1)):
        t = times[keep]-np.mean(times[keep]); positions = crests[keep, side]
        speed = float(t @ (positions-positions.mean())/(t @ t))
        speeds.append(dict(side='left' if side == 0 else 'right', measured_speed=speed,
            expected_speed=sign*reference.outgoing_speed,
            relative_error=speed/(sign*reference.outgoing_speed)-1))
    return dict(resolution=n, incoming_mach=reference.incoming_mach,
        expected_reflected_mach=reference.outgoing_mach,
        expected_reflected_pressure_ratio=float(reference.central[2]/reference.incoming_left[2]),
        expected_reflected_density_ratio=float(reference.central[0]/reference.incoming_left[0]),
        required_separated_events=len(required),
        every_separated_front_located=bool(required_fronts) and all(r['located'] for r in required_fronts),
        every_separated_mach_within_5pct=bool(required_fronts) and all(r['located_and_mach_accurate'] for r in required_fronts),
        missing_required_fronts=sum(not r['located'] for r in required_fronts),
        inaccurate_or_missing_required_fronts=sum(not r['located_and_mach_accurate'] for r in required_fronts),
        midpoint_surface_events=sum(r['midpoint_surface_samples'] > 0 for r in required),
        measured_front_motion=speeds,
        pressure_front_speeds_within_5pct=all(abs(r['relative_error']) <= SPEED_RELATIVE_TOLERANCE for r in speeds),
        maximum_flux_balance_error=max(r['maximum_boundary_flux_balance_error'] for r in rows),
        final_conserved_l1=rows[-1]['conserved_l1_normalized'],
        final_slab_density_relative_error=rows[-1]['slab_density_median']/reference.central[0]-1,
        final_slab_pressure_relative_error=rows[-1]['slab_pressure_median']/reference.central[2]-1,
        final_slab_velocity_rms_over_c2=rows[-1]['slab_velocity_rms_over_c2'])


def write_csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run(args):
    if args.output_dir.exists():
        raise ValueError('Use a fresh output directory.')
    resolutions = sorted(set(args.resolutions)); machs = sorted(set(args.incoming_machs))
    if len(resolutions) < 2:
        raise ValueError('Use at least two resolutions to expose discretization dependence.')
    files = [Path(__file__), Path('experiments/shock/colliding_shocks_reference.py'),
        Path('experiments/wind_bubble/audit_detection_stages.py')]
    files += sorted(Path('astronomix').rglob('*.py'))
    hashes = {str(p):sha(p) for p in files}
    args.output_dir.mkdir(parents=True)
    event_rows, front_rows, summaries, simulations = [], [], [], {}
    analyzers = {n:make_slab_analyzer(n) for n in resolutions}
    for mach in machs:
        reference = SymmetricShockCollision(mach)
        np.testing.assert_allclose(reference.flux_residuals(), 0., atol=1.e-12, rtol=0.)
        for n in resolutions:
            case = f'M{mach:g}_n{n:04d}'
            profiles, times, solver = evolve(reference, n, args.events)
            fields = [analyzers[n](profile) for profile in profiles]
            rows, fronts = [], []
            for event, (time, profile, field) in enumerate(zip(times, profiles, fields)):
                structure = known_structure(reference, float(time))
                separation = float(np.ptp(structure['positions'])*n)
                required = bool(structure['phase'] == 'outgoing' and
                    separation >= MIN_SEPARATION_CELLS and time/reference.collision_time >= 1.1)
                data = dict(case=case, resolution=n, incoming_mach=mach, event=event, time=float(time),
                    time_over_collision=float(time/reference.collision_time), phase=structure['phase'],
                    exact_separation_cells=separation, separated_outgoing_control=required)
                measures = front_metrics(field, structure, n)
                fronts.extend(dict(**data, **r) for r in measures)
                x = (np.arange(n)+.5)/n
                midpoint = abs(x-.5) <= 2/n
                row = dict(**data, surface_cells=int(field['shock_surface'].sum()),
                    midpoint_surface_samples=int((field['shock_surface'] & midpoint).sum()),
                    both_fronts_located=bool(measures) and all(r['located'] for r in measures),
                    both_fronts_mach_accurate=bool(measures) and all(r['located_and_mach_accurate'] for r in measures),
                    **hydrodynamic_metrics(profile, reference, float(time)))
                rows.append(row)
            summary = summarize_case(rows, fronts, profiles, times, reference, n)
            summary.update(case=case, events=len(times), solver=solver)
            summaries.append(summary); event_rows.extend(rows); front_rows.extend(fronts)
            arrays = {key:np.stack([f[key] for f in fields]) for key in fields[0]}
            arrays.update(profiles=profiles, times=times,
                reference_positions=np.array([known_structure(reference, float(t))['positions'] for t in times]))
            np.savez_compressed(args.output_dir/f'{case}.npz', **arrays)
            simulations[case] = (profiles, times, fields, reference)
            print(f'{case}: evolved {solver["num_iterations"]} steps; separated events={summary["required_separated_events"]}; '
                f'located={summary["every_separated_front_located"]}; Mach={summary["every_separated_mach_within_5pct"]}; '
                f'pressure-motion={summary["pressure_front_speeds_within_5pct"]}', flush=True)
    write_csv(args.output_dir/'events.csv', event_rows)
    write_csv(args.output_dir/'fronts.csv', front_rows)
    make_figures(args.output_dir, simulations, summaries)
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise AssertionError(f'Source changed during evolved experiment: {path}')
    numerical = all(s['pressure_front_speeds_within_5pct'] and s['maximum_flux_balance_error'] < 5.e-5
        and abs(s['final_slab_density_relative_error']) < .05
        and abs(s['final_slab_pressure_relative_error']) < .05
        and s['final_slab_velocity_rms_over_c2'] < .01 for s in summaries)
    detection = all(s['every_separated_mach_within_5pct'] and s['midpoint_surface_events'] == 0 for s in summaries)
    report = dict(completed=True, simulation_evolved=True, backend=jax.default_backend(),
        python=platform.python_version(), jax=jax.__version__, numpy=np.__version__,
        incoming_machs=machs, resolutions=resolutions, events_per_run=args.events,
        numerical_control_passed=numerical, separated_detection_gate_passed=detection,
        decision='separated_reflected_shocks_recovered' if numerical and detection else
            'detector_failures_on_validated_flow' if numerical else 'numerical_control_needs_review',
        detector_path='Production 3D finder on (N,8,8) extrusion of genuine 1D hydro states',
        minimum_separation_cells=MIN_SEPARATION_CELLS,
        location_tolerance_cells=LOCATION_TOLERANCE_CELLS, mach_relative_tolerance=MACH_RELATIVE_TOLERANCE,
        native_1d_surface_loop_corrected=True, three_dimensional_detector_changed=False,
        extra_smoothing=False, wind_sources=False, cases=summaries, source_sha256=hashes,
        fading_modeled=False, bubble_question_resolved=False,
        scope='Evolved constant-reservoir planar collision. Tests reflected fronts and smooth slab, not finite-blast fading, curved bubble flow, contacts, or detection completeness in the original bubble.')
    (args.output_dir/'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k, v in report.items() if k not in ('source_sha256', 'cases')}, indent=2))
    return report


def make_figures(output, simulations, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mach = max(s['incoming_mach'] for s in summaries)
    n = max(s['resolution'] for s in summaries)
    profiles, times, fields, reference = simulations[f'M{mach:g}_n{n:04d}']
    x = (np.arange(n)+.5)/n; tau = times/reference.collision_time
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), dpi=160, sharey=True)
    fig.subplots_adjust(top=.82, bottom=.22, left=.06, right=.96, wspace=.25)
    for ax, index, title in zip(axes, (0, 2, 1), ('Density', 'Pressure', 'Velocity')):
        mesh = ax.pcolormesh(x, tau, profiles[:, :, index], shading='auto',
            cmap='coolwarm' if index == 1 else 'viridis', rasterized=True)
        expected = np.array([known_structure(reference, float(t))['positions'] for t in times])
        ax.plot(expected[:, 0], tau, 'w--', lw=1.2); ax.plot(expected[:, 1], tau, 'w--', lw=1.2)
        ax.axhline(1., color='white', ls=':', lw=1)
        for time_ratio, field in zip(tau, fields):
            surface = field['shock_surface'].astype(bool)
            positions = x+field['surface_offsets']*field['normal_x']/n
            ax.scatter(positions[surface], np.full(surface.sum(), time_ratio), s=6, c='#ff7733', edgecolors='none')
        ax.set(xlim=(.22, .78), ylim=(0, 1.6), xlabel='x', title=title)
        fig.colorbar(mesh, ax=ax, fraction=.05, pad=.03)
    axes[0].set_ylabel('Time / exact collision time')
    fig.suptitle(f'Evolved collision | incident Mach {mach:g} | {n} cells | expected reflected Mach {reference.outgoing_mach:.3f}', fontsize=13)
    fig.text(.06, .055, 'White dashed: independent exact front trajectories. Orange: production 3D finder on thin-extruded evolved profiles.\n'
        'At t/t_collision = 1 the incident fronts meet; two reflected fronts then move apart around the central slab.\n'
        'This constant-reservoir problem does not model late weakening inside hot bubble interiors.', fontsize=9)
    fig.savefig(output/'evolved_collision_spacetime.png'); plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), dpi=150, sharex=True)
    fig.subplots_adjust(top=.86, bottom=.13, left=.07, right=.98, hspace=.3, wspace=.22)
    for col, target in enumerate((.8, 1.04, 1.6)):
        event = int(np.argmin(abs(tau-target)))
        exact, structure = reference.cell_averages(n, float(times[event]))
        field = fields[event]; selected = field['shock_surface'].astype(bool)
        for ax, index, label in zip(axes[:, col], (0, 2, 1), ('Density', 'Pressure', 'Velocity')):
            ax.plot(x, exact[:, index], '--', color='#777777', label='Exact cell averages')
            ax.plot(x, profiles[event, :, index], color='#087f8c', lw=1.2, label='Evolved gas')
            ax.scatter(x[selected], profiles[event, selected, index], color='#e67722', s=25,
                label='Detected surface cell', zorder=3)
            for position in structure['positions']:
                ax.axvline(position, color='#777777', ls=':', lw=.8)
            ax.set(xlim=(.36, .64), ylabel=label); ax.grid(alpha=.2)
        axes[0, col].set_title(f't/t_collision = {tau[event]:.2f}')
        axes[-1, col].set_xlabel('x')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Before collision, close reflected fronts, and separated reflected fronts', fontsize=14)
    fig.text(.07, .035, 'No physical shock is prescribed at the post-collision midplane. Full profiles retain any numerical center artifact.\n'
        'Exact positions are used for scoring/overlays only; they are not supplied to the detector.', fontsize=9)
    fig.savefig(output/'evolved_collision_profiles.png'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolutions', type=int, nargs='+', default=[256, 512, 1024])
    parser.add_argument('--incoming-machs', type=float, nargs='+', default=[3., 10.])
    parser.add_argument('--events', type=int, default=81)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())
