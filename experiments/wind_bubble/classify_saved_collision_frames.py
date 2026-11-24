"""Control-gated, frame-aware classification of the ten compared gas states.

Supplemental high-cadence strips are from a separate 128^3 evolution. They do
not silently supply exact front velocities for the 64^3 or original 128^3 gas.
Failed outer controls force ambiguity, never a claim of physical shock absence.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.shock.run_evolved_collision import sha, write_csv
from experiments.wind_bubble.compare_collision_jump_screening import ROOTS, EVENTS
from experiments.wind_bubble.collision_front_motion import (
    MotionSettings, analyze_motion, strip_fields)
from experiments.wind_bubble.audit_front_motion_control import one_side_fit, slope_resolution_allowance


STRIPS = Path('outputs/collision_front_motion_iwr_n128')


def classification(control_passed, matching_evolution, bracketed, shocks_supported,
                   noncompressive_subsonic_supported):
    """A failed control cannot become a negative physical classification."""
    if not control_passed or not matching_evolution or not bracketed:
        return 'ambiguous'
    if shocks_supported:
        return 'resolved shock'
    if noncompressive_subsonic_supported:
        return 'rarefaction/contact-compatible'
    return 'ambiguous'


def choose_observation(observations, tracks, event, side, outer=False):
    choices = [r for r in observations if r['event'] == event and
        ((r['region'] == 'outer_control' and (r['x'] < .3125 if side == 'left' else r['x'] > .6875))
         if outer else (.45 <= r['x'] < .5 if side == 'left' else .5 < r['x'] <= .55))]
    if not choices:
        return None
    if outer:
        return min(choices, key=lambda r:r['x']) if side == 'left' else max(choices, key=lambda r:r['x'])
    # Locate one inspection feature deterministically. This is not a shock
    # identity assignment or selection for a favorable jump fit.
    strengths = {(p['track'], p['event']):abs(p['pressure_gradient']) for track in tracks for p in track}
    return max(choices, key=lambda r:strengths[r['track'], event])


def feature_evidence(observation, pairs, tracks, times, dx):
    result = dict(position=None, measured_x_speed=None, normal_x=None, normal_y=None, normal_z=None,
        speed_sensitivity=None, bracketed=False, sampled_pairs=0, compatible_pairs=0,
        shock_frame_mach_lower=None, shock_frame_mach_upper=None, apparent_pressure_mach=None,
        best_mass_residual=None, best_momentum_residual=None, best_enthalpy_residual=None,
        best_density_ratio_error=None, best_entropy_jump=None, best_characteristic_ordering=None,
        best_upstream_mach=None, best_downstream_mach=None, best_necessary_shock_conditions=None,
        best_endpoint_change=None, status='no_feature')
    if observation is None:
        return result
    event, track_id = observation['event'], observation['track']
    result['position'] = observation['x']
    track = tracks[track_id]
    index = next(i for i,p in enumerate(track) if p['event'] == event)
    short, reason_short = one_side_fit(track, index, 4)
    long, reason_long = one_side_fit(track, index, 8)
    if short is None or long is None or not observation.get('normal_x'):
        result['status'] = f'no_bracketed_motion:{reason_short}/{reason_long}'
        return result
    speed, scatter, _ = short
    allowance = slope_resolution_allowance([p['time'] for p in track[index-4:index+5]], dx)
    sensitivity = max(allowance+3*scatter, abs(speed-long[0]))
    selected = [r for r in pairs if r['event'] == event and r['track'] == track_id]
    result.update(measured_x_speed=speed, speed_sensitivity=sensitivity, bracketed=True,
        normal_x=observation['normal_x'], normal_y=observation['normal_y'], normal_z=observation['normal_z'],
        sampled_pairs=len(selected), compatible_pairs=sum(r['compatible_10pct'] for r in selected),
        status='sampled_jump_checks_failed' if selected else 'no_guarded_state_pairs')
    if not selected:
        return result
    # Keep all allowable sampling choices in the sensitivity envelope. The
    # best residual pair is diagnostic context, never enough for acceptance.
    lower, upper = [], []
    for pair in selected:
        c1 = np.sqrt((5/3)*pair['upstream_pressure']/pair['upstream_density'])
        delta = abs(observation['normal_x'])*sensitivity/c1
        lower.append(pair['shock_frame_mach']-delta); upper.append(pair['shock_frame_mach']+delta)
    best = min(selected, key=lambda r:r['largest_relative_error'])
    result.update(shock_frame_mach_lower=float(min(lower)), shock_frame_mach_upper=float(max(upper)),
        apparent_pressure_mach=float(np.sqrt((4*best['pressure_ratio']+1)/5)),
        best_mass_residual=best['mass_relative_error'], best_momentum_residual=best['momentum_relative_error'],
        best_enthalpy_residual=best['enthalpy_relative_error'], best_density_ratio_error=best['density_ratio_error'],
        best_entropy_jump=best['entropy_change_over_cv'],
        best_upstream_mach=best['shock_frame_mach'], best_downstream_mach=best['downstream_shock_frame_mach'],
        best_characteristic_ordering=bool(best['shock_frame_mach'] > 1 and 0 < best['downstream_shock_frame_mach'] < 1),
        best_necessary_shock_conditions=best['necessary_shock_conditions'],
        best_endpoint_change=best['endpoint_relative_change'])
    if result['compatible_pairs'] >= 3 and result['shock_frame_mach_lower'] > 1:
        result['status'] = 'conditionally_shock_supported'
    return result


def run(directory, strip_root=STRIPS):
    directory, strip_root = Path(directory), Path(strip_root)
    primary_path = directory/'verification.json'; primary = json.loads(primary_path.read_text())
    inputs = {str(primary_path):sha(primary_path)}
    for path,digest in {**primary['input_sha256'], **primary['source_sha256']}.items():
        if sha(path) != digest:
            raise AssertionError(f'Primary comparison input/source changed: {path}')
    meta_path = strip_root/'metadata.json'; meta = json.loads(meta_path.read_text())
    if meta.get('status') != 'complete' or meta['num_cells'] != 128:
        raise ValueError('Require the completed high-cadence 128 cubed strip history.')
    inputs[str(meta_path)] = sha(meta_path)
    bands, times = [], []
    for name,digest in sorted(meta['strip_sha256'].items()):
        path = strip_root/'strips'/name
        if sha(path) != digest:
            raise AssertionError(f'Saved high-cadence strip changed: {path}')
        inputs[str(path)] = digest
        with np.load(path, allow_pickle=False) as data:
            bands.append(data['primitive']); times.append(float(data['time']))
    times, bands = np.asarray(times), np.asarray(bands)
    np.testing.assert_array_equal(times, meta['actual_times'])
    x = (np.arange(128)+.5)/128
    lines, dy, dz = strip_fields(bands, np.array(meta['transverse_y']), np.array(meta['transverse_z']), 1/128)
    settings = MotionSettings(**meta['motion_settings'])
    observations, pairs, tracks = analyze_motion(lines, times, x, dy, dz, settings,
        source_positions=meta['source_positions'], source_radius=meta['injection_radius'])
    by_region = list(csv.DictReader((directory/'region_comparison.csv').open()))
    inputs[str(directory/'region_comparison.csv')] = sha(directory/'region_comparison.csv')
    frame_rows, raw_rows, state_matches = [], [], []
    for root in ROOTS:
        state_meta = json.loads((root/'metadata.json').read_text()); n = state_meta['num_cells']
        for key in ('source_positions','source_strengths','gamma','injection_radius','discrete_mass_rates','discrete_energy_rates'):
            np.testing.assert_allclose(state_meta[key], meta[key], rtol=1e-12, atol=1e-12)
        for event in EVENTS:
            time = float(state_meta['actual_times'][event]); strip_event = int(np.argmin(abs(times-time)))
            row = dict(resolution=n, event=event, time=time, classification='ambiguous',
                high_cadence_resolution_available=n == 128, matching_motion_evolution=False,
                outer_control_status='unavailable_at_this_resolution', temporal_bracket_available=False,
                central_slice_compression=None, central_slice_alignment=None,
                central_slice_baseline_surface_cells=None, central_slice_transition_surface_cells=None,
                supplemental_strip_state_max_difference=None, supplemental_strip_density_relative_l1=None,
                reason='No matching 64 cubed high-cadence history; do not transfer 128 cubed speeds.')
            region = next(r for r in by_region if int(r['resolution']) == n and int(r['event']) == event
                          and r['region'] == 'central_xy_slice')
            for key, source in (('central_slice_compression','compression'), ('central_slice_alignment','aligned'),
                ('central_slice_baseline_surface_cells','baseline_exported_surface_cells'),
                ('central_slice_transition_surface_cells','transition_exported_surface_cells')):
                row[key] = int(region[source])
            evidence = {}
            for location in ('left_outer','right_outer','left_layer','right_layer'):
                side, role = location.split('_')
                obs = choose_observation(observations, tracks, strip_event, side, outer=role == 'outer') if n == 128 else None
                item = feature_evidence(obs, pairs, tracks, times, 1/128)
                evidence[location] = item
                row.update({location+'_'+k:v for k,v in item.items()})
                raw_rows.append(dict(resolution=n,event=event,time=time,location=location,**item))
            if n == 128:
                # Both runs share physical parameters, but fine event cadence
                # perturbs solver step boundaries. Quantify the discrepancy;
                # never describe their states as pointwise equivalent.
                state_path = root/'audit'/f'state_{event:04d}.npy'
                state = np.load(state_path, allow_pickle=False)
                iy = np.array([np.argmin(abs(x-y)) for y in meta['transverse_y']])
                iz = np.array([np.argmin(abs(x-z)) for z in meta['transverse_z']])
                subset = state[:,:,iy,:][:,:,:,iz]
                delta = float(abs(subset-bands[strip_event]).max())
                l1 = float(abs(subset[0]-bands[strip_event,0]).sum()/abs(subset[0]).sum())
                same = bool(np.array_equal(subset, bands[strip_event]) and abs(times[strip_event]-time) <= 1e-6)
                row.update(matching_motion_evolution=same, supplemental_strip_state_max_difference=delta,
                    supplemental_strip_density_relative_l1=l1)
                state_matches.append(dict(event=event, exact_match=same, max_difference=delta, density_relative_l1=l1))
                controls = [evidence[k] for k in ('left_outer','right_outer')]
                bracketed = all(r['bracketed'] for r in controls)
                passed = all(r['status'] == 'conditionally_shock_supported' for r in controls)
                row['outer_control_status'] = ('passed' if passed else 'failed_flux_entropy_or_endpoint_checks'
                    if bracketed else 'unavailable_without_bracketed_motion')
                row['temporal_bracket_available'] = bracketed and all(evidence[k]['bracketed'] for k in ('left_layer','right_layer'))
                row['classification'] = classification(passed, same, row['temporal_bracket_available'],
                    all(evidence[k]['status'] == 'conditionally_shock_supported' for k in ('left_layer','right_layer')),
                    False)  # No positive contact/rarefaction certification is supplied here.
                row['reason'] = ('No two-sided temporal fit at the final saved time.' if not bracketed else
                    'Outer controls fail the same sampled Euler/entropy/endpoint procedure; central physical classification is ambiguous.')
                if not same:
                    row['reason'] += ' Supplemental strips are a separate-cadence evolution, not the exact captured state.'
            frame_rows.append(row)
    for path,digest in inputs.items():
        if sha(path) != digest:
            raise AssertionError(f'Frame-aware input changed: {path}')
    paths = [directory/name for name in ('frame_aware_classification.csv','frame_aware_feature_evidence.csv','frame_aware_verification.json')]
    if any(path.exists() for path in paths):
        raise ValueError('Do not overwrite an existing frame-aware assessment.')
    write_csv(paths[0], frame_rows); write_csv(paths[1], raw_rows)
    sources = [Path(__file__), Path('experiments/wind_bubble/collision_front_motion.py'),
        Path('experiments/wind_bubble/audit_front_motion_control.py'),
        Path('experiments/wind_bubble/compare_collision_jump_screening.py')]
    result = dict(analysis_completed=True, compared_states=len(frame_rows), high_cadence_events=len(times),
        controls_passed=sum(r['outer_control_status']=='passed' for r in frame_rows),
        classifications={name:sum(r['classification']==name for r in frame_rows)
            for name in ('resolved shock','rarefaction/contact-compatible','ambiguous')},
        bracketed_fit_events=[9,17], match_state_checks=state_matches,
        normal_from='3D pressure gradient from the saved strip and transverse derivatives; not the temperature normal.',
        speed_from='Bracketed pressure-feature motion; not solved from mass flux.',
        mach_bounds_scope='All guarded sampled pairs plus half-cell position/scatter/fit-window sensitivity; not rigorous confidence bounds.',
        thresholds_changed=False, new_simulation=False, physical_shock_absence_proven=False,
        conclusion='Wider sampling did not repair the original gap. Physical classifications remain ambiguous under failed/unavailable outer controls and nonidentical supplemental evolution.',
        input_sha256=inputs, source_sha256={str(p):sha(p) for p in sources})
    paths[2].write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','source_sha256')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dir',type=Path,default=Path('outputs/bubble_transition_jump_comparison_n064_n128'))
    run(parser.parse_args().comparison_dir)
