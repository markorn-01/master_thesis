"""Replay frozen finite-blast gas with baseline and transition-aware screening.

No gas evolution, tracking changes, threshold fitting, or bubble interpretation.
Independent motion/jump labels are read unchanged from the preceding experiment.
"""
import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from astronomix import (CARTESIAN, SimulationConfig, construct_primitive_state,
    finalize_config, get_helper_data, get_registered_variables)
from astronomix.option_classes.simulation_config import StaticFloatVector
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix._physics_modules._shock_finder._shock_zones import (
    _shock_zone_criterion_converging_flow, _shock_zone_criterion_aligned_gradients,
    _shock_zone_criterion_minimum_mach, _make_interior_mask, _transition_jump_candidates)
from astronomix._physics_modules._shock_finder._gradients import _calculate_shock_direction
from experiments.shock.run_evolved_collision import sha, write_csv, front_metrics, known_structure
from experiments.shock.run_finite_blast_collision import compare_detector
from experiments.shock.colliding_shocks_reference import SymmetricShockCollision, isolated_control
from experiments.shock.shock_contact_reference import ShockContactReference


ARCHIVE = Path('outputs/finite_blast_collision_n0512_n1024_n2048')
EVOLVED = Path('outputs/evolved_planar_collision_n0256_n0512_n1024')


def make_comparison_analyzer(n):
    """Both production API paths, same unmodified 3D slab input and estimator."""
    config = SimulationConfig(geometry=CARTESIAN, dimensionality=3, num_cells=n,
        box_size=StaticFloatVector(1., 8/n, 8/n), mhd=False)
    variables = get_registered_variables(config)
    shape = (n, 8, 8); ones = jnp.ones(shape); zeros = jnp.zeros(shape)
    template = construct_primitive_state(config=config, registered_variables=variables,
        density=ones, gas_pressure=ones, velocity_x=zeros, velocity_y=zeros, velocity_z=zeros)
    config = finalize_config(config, template.shape); helper = get_helper_data(config)

    @jax.jit
    def evaluate(profile):
        state = template.at[variables.density_index].set(jnp.broadcast_to(profile[:, 0, None, None], shape))
        state = state.at[variables.velocity_index.x].set(jnp.broadcast_to(profile[:, 1, None, None], shape))
        state = state.at[variables.pressure_index].set(jnp.broadcast_to(profile[:, 2, None, None], shape))
        pressure, density = state[variables.pressure_index], state[variables.density_index]
        direction = _calculate_shock_direction(pressure, density, config)
        comp = _shock_zone_criterion_converging_flow(state, config, variables)
        aligned = _shock_zone_criterion_aligned_gradients(pressure, density, config)
        provisional = comp & aligned & _make_interior_mask(shape)
        old_jump = _shock_zone_criterion_minimum_mach(state, config, variables, helper, direction)
        extra = _transition_jump_candidates(pressure, density, direction, provisional)
        outputs = {}
        for label, enabled in (('baseline', False), ('transition', True)):
            result = find_shocks_pfrommer(state, config, variables, helper,
                profile_aware_mach=True, transition_aware_detection=enabled)
            jump = old_jump | extra if enabled else old_jump
            mismatch = jnp.sum(result.shock_zones != (comp & aligned & jump))
            plane = (slice(None), slice(2, -2), slice(2, -2))
            for array in (result.shock_zones, result.shock_surface_cells):
                rays = array[plane]
                mismatch += jnp.sum(rays != rays[:, :1, :1])
            fields = dict(shock_surface=result.shock_surface_cells,
                shock_zone=result.shock_zones, surface_offsets=result.shock_surface_offsets,
                normal_x=result.shock_direction[0], mach=result.mach_numbers,
                compression=comp, gradient_alignment=aligned, minimum_jump=jump,
                thermal_flux=result.thermal_energy_flux)
            outputs[label] = {k:v[:, 4, 4] for k, v in fields.items()}
            outputs[label]['mask_mismatch'] = mismatch
        outputs['extra_jump'] = extra[:, 4, 4]
        return outputs

    def analyze(q):
        raw = evaluate(jnp.asarray(q, jnp.float32)); output = {}
        for label in ('baseline', 'transition'):
            fields = {k:np.asarray(v) for k, v in raw[label].items()}
            if int(fields.pop('mask_mismatch')):
                raise AssertionError('API zone mask or interior transverse ray disagrees.')
            selected = fields['shock_surface']
            if label == 'transition':
                assert np.all(np.isfinite(fields['mach'][selected]) & (fields['mach'][selected] >= 1.3))
            for key in ('mach', 'surface_offsets', 'thermal_flux'):
                assert np.all(fields[key][~selected] == 0), (label, key)
            output[label] = fields
        output['extra_jump'] = np.asarray(raw['extra_jump'])
        return output
    return analyze


def smooth_profile(n, width, pressure_ratio=11., phase=.37):
    """Isentropic compressive snapshot, not an asserted exact Euler history."""
    x = np.arange(n)+.5
    pressure = 1+(pressure_ratio-1)*.5*(1-np.tanh((x-n/2-phase)/width))
    rho = pressure**(3/5)
    sound = np.sqrt((5/3)*pressure/rho)
    u = 3*(sound-np.sqrt(5/3))
    return np.column_stack([rho, u, pressure])


def control_suite(evaluate, n):
    rows = []

    def score(name, q, target=None, expected=None, negative=False):
        fields = evaluate(q)
        values = {}
        for label in ('baseline', 'transition'):
            f = fields[label]; x = (np.arange(n)+.5+f['surface_offsets']*f['normal_x'])/n
            selected = f['shock_surface']
            near = selected if target is None else selected & (abs(x-target)*n <= 2)
            mach = float(np.median(f['mach'][near])) if near.any() else None
            values[label] = dict(surface_cells=int(selected.sum()), located=bool(near.any()), mach=mach)
        passed = (not values['transition']['surface_cells']) if negative else (
            values['transition']['located'] and
            abs(values['transition']['mach']/expected-1) <= .05)
        rows.append(dict(case=name, resolution=n, negative=negative, passed=bool(passed),
            baseline_surfaces=values['baseline']['surface_cells'],
            transition_surfaces=values['transition']['surface_cells'],
            transition_mach=values['transition']['mach']))

    for orientation in (-1, 1):
        for width in (1, 3, 5):
            for mach in (1.2, 1.5, 3., 10.):
                reference = ShockContactReference(mach=mach)
                q, layout = reference.control(n, 'shock', width, phase=.37, orientation=orientation)
                score(f'isolated_M{mach:g}_w{width}_o{orientation}', q,
                      layout['shock_x'], mach, negative=mach < 1.3)
            for factor in (.25, 4., 8.):
                q, _ = ShockContactReference(contact_density_factor=factor).control(
                    n, 'contact', width, phase=.37, orientation=orientation)
                score(f'contact_r{factor:g}_w{width}_o{orientation}', q, negative=True)
        for width in (1., 2., 4., 8., 16.):
            q = smooth_profile(n, width)
            if orientation == -1:
                q = q[::-1].copy(); q[:, 1] *= -1
            score(f'smooth_w{width:g}_o{orientation}', q, negative=True)
        for factor in (.25, 8.):
            reference = ShockContactReference(contact_density_factor=factor)
            q, layout = reference.profile(n, 8., orientation=orientation)
            score(f'separated_shock_contact_r{factor:g}_o{orientation}', q, layout['shock_x'], reference.mach)
    return rows


def isolated_front_is_accurate(fields, structure, n):
    """One midpoint shock has no left/right half-domain ownership condition."""
    position = (np.arange(n)+.5+fields['surface_offsets']*fields['normal_x'])/n
    selected = (fields['shock_surface']
                & (fields['normal_x']*structure['hot_to_cold_directions'][0] > .9)
                & (abs(position-structure['positions'][0])*n <= 2))
    return bool(selected.any() and np.isfinite(fields['mach'][selected]).all()
                and abs(float(np.median(fields['mach'][selected]))/structure['mach']-1) <= .05)


def opposing_controls(evaluate, n):
    reference = SymmetricShockCollision(); rows = []
    for width in (1, 3, 5):
        for phase in ('incoming', 'outgoing'):
            for separation in (.5, 1., 2., 4., 8., 16.):
                time = reference.time_at_separation(phase, separation, n)
                q, structure = reference.cell_averages(n, time, width)
                fields = evaluate(q)
                before = front_metrics(fields['baseline'], structure, n)
                after = front_metrics(fields['transition'], structure, n)
                rows.append(dict(resolution=n, phase=phase, filter_cells=width,
                    separation_cells=separation,
                    baseline_accurate=all(r['located_and_mach_accurate'] for r in before),
                    transition_accurate=all(r['located_and_mach_accurate'] for r in after)))
            q, structure = isolated_control(reference, n, phase, width)
            fields = evaluate(q)
            rows.append(dict(resolution=n, phase=phase, filter_cells=width,
                separation_cells=None,
                baseline_accurate=isolated_front_is_accurate(fields['baseline'], structure, n),
                transition_accurate=isolated_front_is_accurate(fields['transition'], structure, n)))
    return rows


def run(args):
    if args.output_dir.exists():
        raise ValueError('Use a fresh output directory.')
    archive = args.archive
    review_path = archive/'review/verification.json'
    review = json.loads(review_path.read_text())
    if not review['review_completed'] or not review['numerical_integral_controls_passed']:
        raise ValueError('Require reviewed finite-blast inputs with passing integral controls.')
    inputs = {str(review_path):sha(review_path)}
    # The old source hashes are historical; changing the two detector modules
    # is intentional. Frozen gas and independently formed labels must match.
    for path, digest in review['input_sha256'].items():
        if sha(path) != digest:
            raise AssertionError(f'Frozen finite-blast input changed: {path}')
        inputs[path] = digest
    report = json.loads((archive/'verification.json').read_text())
    sources = [Path(__file__)] + sorted(Path('astronomix').rglob('*.py'))
    sources += [Path('experiments/shock/run_finite_blast_collision.py'),
        Path('experiments/shock/colliding_shocks_reference.py'),
        Path('experiments/shock/shock_contact_reference.py'),
        Path('experiments/shock/run_evolved_collision.py')]
    hashes = {str(p):sha(p) for p in sources}
    args.output_dir.mkdir(parents=True)
    observations = list(csv.DictReader((archive/'observations.csv').open()))
    results, case_rows, controls, opposing, evolved_rows = [], [], [], [], []
    maximum_default_difference = 0.; profiles_saved = {}
    for n in report['resolutions']:
        with np.load(archive/f'finite_blast_n{n:04d}.npz', allow_pickle=False) as data:
            frozen = {key:data[key] for key in data.files}
        supported = [r for r in observations if int(r['resolution']) == n
            and r['reflection_candidate_track'] == 'True' and r['evidence'] == 'jump_supported_above_threshold']
        events = sorted({int(r['event']) for r in supported})
        evaluate = make_comparison_analyzer(n)
        # Replay every independently supported front-event, not another gas
        # simulation or arbitrary unsupported times. Keep the evidence frozen.
        fields = {event:evaluate(frozen['profiles'][event]) for event in events}
        for label, key in (('shock_surface', 'shock_surface'), ('shock_zone', 'shock_zone'),
                           ('surface_offsets', 'surface_offsets'), ('normal_x', 'normal_x'), ('mach', 'mach')):
            values = np.array([fields[event]['baseline'][key] for event in events])
            if values.dtype == bool:
                np.testing.assert_array_equal(values, frozen[label][events])
            else:
                np.testing.assert_allclose(values, frozen[label][events], rtol=1e-6, atol=1e-6)
                maximum_default_difference = max(maximum_default_difference, float(abs(values-frozen[label][events]).max()))
        for row in supported:
            event = int(row['event']); item = dict(resolution=n, event=event, time=float(row['time']),
                side=row['side'], x=float(row['x']), independent_mach_min=float(row['nominal_mach_min']),
                independent_mach_max=float(row['nominal_mach_max']))
            for label in ('baseline', 'transition'):
                target = dict(x=item['x'], normal_upstream_to_downstream=float(row['normal_upstream_to_downstream']))
                compare_detector(target, fields[event][label], n)
                item[f'{label}_located'] = target['production_surface_nearby']
                item[f'{label}_mach'] = target['production_mach_median']
            item['recovered_miss'] = not item['baseline_located'] and item['transition_located']
            item['lost_supported_front'] = item['baseline_located'] and not item['transition_located']
            results.append(item)
        local = [r for r in results if r['resolution'] == n]
        case_rows.append(dict(resolution=n, saved_events=len(frozen['times']), replayed_events=len(fields), supported_front_events=len(local),
            baseline_misses=sum(not r['baseline_located'] for r in local),
            transition_misses=sum(not r['transition_located'] for r in local),
            recovered_misses=sum(r['recovered_miss'] for r in local),
            lost_supported_fronts=sum(r['lost_supported_front'] for r in local)))
        print(case_rows[-1], flush=True)
        profiles_saved[n] = (frozen, fields)
    for n in (64, 128):
        evaluate = make_comparison_analyzer(n)
        controls += control_suite(evaluate, n)
        opposing += opposing_controls(evaluate, n)
    # Retest the already evolved separated controls, not just constructed steps.
    for path in sorted(EVOLVED.glob('M*_n*.npz')):
        inputs[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as data:
            q, times = data['profiles'], data['times']
        mach = float(path.stem.split('_')[0][1:]); n = q.shape[1]
        reference = SymmetricShockCollision(mach); evaluate = make_comparison_analyzer(n)
        for event, (profile, time) in enumerate(zip(q, times)):
            structure = known_structure(reference, float(time))
            if (structure['phase'] != 'outgoing' or time/reference.collision_time < 1.1
                    or np.ptp(structure['positions'])*n < 16):
                continue
            fields = evaluate(profile)
            for row in front_metrics(fields['transition'], structure, n):
                evolved_rows.append(dict(case=path.stem, event=event, **row))
    write_csv(args.output_dir/'front_comparison.csv', results)
    write_csv(args.output_dir/'controls.csv', controls)
    write_csv(args.output_dir/'opposing_controls.csv', opposing)
    write_csv(args.output_dir/'evolved_controls.csv', evolved_rows)
    gates = dict(frozen_baseline_masks_exact=True, baseline_float_fields_within_tolerance=True,
        all_archived_supported_misses_recovered=all(r['transition_misses'] == 0 for r in case_rows),
        no_supported_front_lost=all(not r['lost_supported_front'] for r in results),
        isolated_weak_contact_and_smooth_controls_pass=all(r['passed'] for r in controls),
        all_isolated_collision_branch_controls_pass=all(r['transition_accurate'] for r in opposing
            if r['separation_cells'] is None),
        no_accurate_opposing_control_regressed=all(not r['baseline_accurate'] or r['transition_accurate'] for r in opposing),
        all_separated_evolved_controls_pass=bool(evolved_rows) and all(r['located_and_mach_accurate'] for r in evolved_rows))
    plot(args.output_dir, results, profiles_saved)
    for path, digest in {**inputs, **hashes}.items():
        if sha(path) != digest:
            raise AssertionError(f'Input or source changed during replay: {path}')
    output = dict(completed=True, passed=all(gates.values()), gates=gates, cases=case_rows,
        recovered_front_events=sum(r['recovered_miss'] for r in results), controls=len(controls),
        opposing_control_cases=len(opposing), separated_evolved_front_checks=len(evolved_rows),
        maximum_default_float_difference=maximum_default_difference,
        final_mach_threshold=1.3, detection_search_limit_cells=8, added_candidate_log_entropy_floor=1e-5,
        final_surface_log_entropy_floor=1e-5,
        replay_scope='All independently above-supported finite-blast events; baseline archive comparison covers these replayed events, not every saved time.',
        production_default_changed=False, opt_in_api='find_shocks_pfrommer(..., transition_aware_detection=True)',
        hydrodynamics_rerun=False, independent_labels_changed=False, original_bubble_question_resolved=False,
        source_sha256=hashes, input_sha256=inputs,
        scope='Bounded screening correction on frozen planar gas. Supported front-events are repeated conditional observations, not distinct shocks. No universal completeness, bubble-gap repair, or curved-flow validation.')
    (args.output_dir/'verification.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k, v in output.items() if not k.endswith('sha256')}, indent=2))
    return output


def plot(output, results, saved):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150)
    fig.subplots_adjust(left=.08, right=.97, bottom=.22, top=.84, wspace=.3)
    n = max(saved); data, fields = saved[n]
    # Same reproducible example as the preceding review.
    example = min((r for r in results if r['resolution'] == n and r['side'] == 'left'
                   and not r['baseline_located']), key=lambda r:r['event'])
    event = example['event']; x = (np.arange(n)+.5)/n
    axes[0].plot((x-example['x'])*n, data['profiles'][event, :, 2], 'o-', ms=3, color='#354a58')
    for label, color, marker in (('baseline', '#c66a25', 'x'), ('transition', '#087f8c', 'v')):
        f = fields[event][label]; mask = f['shock_surface']
        axes[0].scatter((x[mask]+f['surface_offsets'][mask]*f['normal_x'][mask]/n-example['x'])*n,
            data['profiles'][event, mask, 2], color=color, marker=marker, s=80, label=label.capitalize(), zorder=4)
    axes[0].axvline(0, color='gray', ls=':')
    axes[0].set(xlim=(-10, 10), xlabel='Cell widths from independently tracked front', ylabel='Pressure',
        title=f'Previously missed front | {n} cells | t={example["time"]:.4f}')
    axes[0].legend()
    rows = sorted((r for r in results if r['resolution'] == n and r['side'] == 'left'), key=lambda r:r['event'])
    for label, color, marker in (('baseline', '#c66a25', 'x'), ('transition', '#087f8c', '.')):
        usable = [r for r in rows if r[f'{label}_mach'] is not None]
        axes[1].scatter([r['time'] for r in usable], [r[f'{label}_mach'] for r in usable],
            color=color, marker=marker, s=22, label=label.capitalize())
    axes[1].scatter([r['time'] for r in rows], [.5*(r['independent_mach_min']+r['independent_mach_max']) for r in rows],
        facecolors='none', edgecolors='#354a58', s=28, label='Independent sampled evidence')
    axes[1].axhline(1.3, color='gray', ls=':'); axes[1].legend(fontsize=8)
    axes[1].set(xlabel='Simulation time', ylabel='Mach number', title='Only independently supported intervals shown')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Bounded transition-jump screening | identical saved gas, unchanged Mach estimator', fontsize=13)
    fig.text(.08, .055, 'New candidates measure the jump across a compressive, aligned transition; maximum search 8 cell widths per side.\n'
        'Opposing-front/domain guards and positive sampled entropy constrain the extension. Final reconstructed Mach must be at least 1.3.\n'
        'Symbols are separate observations; unverified intervals are not joined. This planar replay does not settle the original 3D bubble gap.', fontsize=8)
    fig.savefig(output/'screening_correction.png'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/transition_jump_screening_validation'))
    args = parser.parse_args()
    result = run(args)
    if not result['passed']:
        raise SystemExit('One or more correction gates failed; do not promote the option.')
