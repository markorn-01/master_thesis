"""Independent pressure-feature motion and sampled Euler evidence in 1D.

No production masks, normals, Mach estimates, or temperature gradients are used
to form tracks or assess jumps. A pressure feature alone is not a shock.
All labels are conditional on sampled states and a finite position sensitivity.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from experiments.wind_bubble.collision_front_motion import (
    MotionSettings, fit_motion, measured_speed_metrics,
)
from experiments.wind_bubble.audit_front_motion_control import slope_resolution_allowance


@dataclass(frozen=True)
class EvidenceSettings:
    gamma: float = 5/3
    bounds: tuple = (.3, .7)
    prominence_fraction: float = 1.e-5
    minimum_peak_dominance: float = 2.
    distances: tuple = (3., 4., 6., 8.)
    fit_half_events: tuple = (4, 8)
    jump_tolerance: float = .1
    entropy_signal_tolerance: float = .5
    minimum_compatible_pairs: int = 3
    detection_threshold: float = 1.3


def candidates(profile, dx, settings):
    pressure_gradient = np.gradient(profile[:, 2], dx)
    threshold = settings.prominence_fraction*max(float(abs(pressure_gradient).max()), 1.e-12)
    rows = []
    for sign, side in ((1, 'left'), (-1, 'right')):
        strength = sign*pressure_gradient
        peaks, _ = find_peaks(strength, prominence=threshold)
        for index in peaks:
            x = (index+.5)*dx
            inside = settings.bounds[0] < x < .5 if side == 'left' else .5 < x < settings.bounds[1]
            if not inside or not 1 < index < len(profile)-2 or strength[index] <= 0:
                continue
            lower, center, upper = strength[index-1:index+2]
            denominator = lower-2*center+upper
            delta = np.clip(.5*(lower-upper)/denominator, -.5, .5) if denominator < 0 else 0.
            rows.append(dict(index=int(index), x=float((index+.5+delta)*dx), side=side,
                normal_upstream_to_downstream=sign, pressure_gradient=float(pressure_gradient[index])))
    # A low-amplitude numerical peak must not steal the continuation of the
    # dominant outgoing front merely by lying closer to its previous position.
    # Competing peaks of comparable strength end the observation, not force it.
    selected = []
    for side in ('left', 'right'):
        choices = sorted((r for r in rows if r['side'] == side),
                         key=lambda r:abs(r['pressure_gradient']), reverse=True)
        if choices and (len(choices) == 1 or abs(choices[0]['pressure_gradient']) >=
                settings.minimum_peak_dominance*abs(choices[1]['pressure_gradient'])):
            selected.append(choices[0])
    return selected


def link_candidates(profiles, times, start_time, settings=EvidenceSettings()):
    dx = 1/profiles.shape[1]; tracks = []; previous = []
    for event, (q, time) in enumerate(zip(profiles, times)):
        if time < start_time:
            continue
        current = candidates(q, dx, settings)
        edges = []
        if previous:
            for j, item in enumerate(current):
                local = q[item['index']]
                gate = 1.5*dx+(time-times[event-1])*(abs(local[1])+np.sqrt(settings.gamma*local[2]/local[0]))
                for i, old in enumerate(previous):
                    if old['side'] == item['side'] and abs(item['x']-old['x']) <= gate:
                        edges.append((i, j, abs(item['x']-old['x'])))
        inherited = {}
        for i, j, distance in edges:
            old_choices = sorted((d, jj) for ii, jj, d in edges if ii == i)
            new_choices = sorted((d, ii) for ii, jj, d in edges if jj == j)
            if (old_choices[0][1] == j and new_choices[0][1] == i
                    and (len(old_choices) == 1 or old_choices[1][0]-distance > .25*dx)
                    and (len(new_choices) == 1 or new_choices[1][0]-distance > .25*dx)):
                inherited[j] = previous[i]['track']
        for j, item in enumerate(current):
            track = inherited.get(j)
            if track is None:
                track = len(tracks); tracks.append([])
            item.update(event=event, time=float(time), track=track)
            tracks[track].append(item)
        previous = current
    return tracks


def embed_state(state):
    rho, velocity, pressure = state
    return np.array([rho, velocity, 0., 0., pressure])


def assess_track_point(track, index, profiles, times, settings=EvidenceSettings()):
    point = track[index]; event = point['event']; dx = 1/profiles.shape[1]
    x = (np.arange(profiles.shape[1])+.5)*dx; q = profiles[event]
    sign = point['normal_upstream_to_downstream']
    row = dict(**point, motion_available=False, measured_x_speed=None,
        speed_sensitivity=None, short_long_speed_difference=None, outward_motion=False,
        compatible_pairs=0, mach_low=None, mach_high=None, nominal_mach_min=None,
        nominal_mach_max=None, evidence='motion_unavailable')
    motions = [fit_motion(track, index, MotionSettings(fit_half_events=half))
               for half in settings.fit_half_events]
    if any(fit is None for fit in motions):
        return row, []
    speed, scatter, _ = motions[0]
    half = settings.fit_half_events[0]
    time_window = [r['time'] for r in track[index-half:index+half+1]]
    sensitivity = max(slope_resolution_allowance(time_window, dx)+3*scatter,
                      abs(speed-motions[1][0]))
    outward = sign*speed < 0
    row.update(motion_available=True, measured_x_speed=float(speed),
        speed_sensitivity=float(sensitivity), short_long_speed_difference=float(abs(speed-motions[1][0])),
        outward_motion=bool(outward), evidence='not_outward' if not outward else 'jump_unverified')
    def sample(position):
        return embed_state(np.array([np.interp(position, x, field) for field in q.T]))
    pairs = []
    normal = np.array([sign, 0., 0.])
    for upstream in settings.distances:
        for downstream in settings.distances:
            a, b = point['x']-sign*upstream*dx, point['x']+sign*downstream*dx
            if min(a, b) < x[0] or max(a, b) > x[-1] or (a-.5)*(b-.5) <= 0:
                continue
            up, down = sample(a), sample(b)
            metrics = measured_speed_metrics(up, down, normal, sign*speed, settings.gamma)
            quality = max(metrics[k] for k in ('mass_relative_error', 'momentum_relative_error',
                'enthalpy_relative_error', 'density_ratio_error'))
            compatible = bool(outward and metrics['necessary_shock_conditions']
                and quality <= settings.jump_tolerance
                and metrics['entropy_signal_error'] <= settings.entropy_signal_tolerance)
            c1 = np.sqrt(settings.gamma*up[4]/up[0]); uncertainty = sensitivity/c1
            pairs.append(dict(track=point['track'], event=event, time=point['time'], side=point['side'],
                upstream_distance_cells=upstream, downstream_distance_cells=downstream,
                largest_jump_error=float(quality), compatible=compatible,
                mach_lower_sensitivity=float(metrics['shock_frame_mach']-uncertainty),
                mach_upper_sensitivity=float(metrics['shock_frame_mach']+uncertainty), **metrics))
    accepted = [p for p in pairs if p['compatible']]
    row['compatible_pairs'] = len(accepted)
    if len(accepted) >= settings.minimum_compatible_pairs:
        lower = min(p['mach_lower_sensitivity'] for p in accepted)
        upper = max(p['mach_upper_sensitivity'] for p in accepted)
        row.update(mach_low=float(lower), mach_high=float(upper),
            nominal_mach_min=float(min(p['shock_frame_mach'] for p in accepted)),
            nominal_mach_max=float(max(p['shock_frame_mach'] for p in accepted)),
            evidence='jump_supported_above_threshold' if lower > settings.detection_threshold else
                'jump_supported_below_threshold' if upper < settings.detection_threshold else 'threshold_uncertain')
    return row, pairs


def analyze(profiles, times, start_time, settings=EvidenceSettings()):
    profiles, times = np.asarray(profiles, float), np.asarray(times, float)
    if (profiles.ndim != 3 or profiles.shape[0] != len(times) or profiles.shape[2] != 3
            or profiles.shape[1] < 16 or not np.isfinite(profiles).all()
            or not np.isfinite(times).all() or not np.all(np.diff(times) > 0)
            or (profiles[:, :, [0, 2]] <= 0).any()):
        raise ValueError('Require positive finite rho/u/P profiles and strictly increasing times.')
    tracks = link_candidates(profiles, times, start_time, settings)
    rows, pairs = [], []
    for track in tracks:
        for index in range(len(track)):
            row, sampled = assess_track_point(track, index, profiles, times, settings)
            rows.append(row); pairs.extend(sampled)
    return rows, pairs, tracks


def summaries(rows, tracks, dx):
    def has_three_consecutive(values):
        events = sorted(r['event'] for r in values)
        return any(events[i+2] == events[i]+2 for i in range(len(events)-2))
    result = []
    for number, track in enumerate(tracks):
        selected = [r for r in rows if r['track'] == number]
        sign = track[0]['normal_upstream_to_downstream']
        displacement = -sign*(track[-1]['x']-track[0]['x'])/dx
        central_birth = abs(track[0]['x']-.5) <= 8*dx
        above = [r for r in selected if r['evidence'] == 'jump_supported_above_threshold']
        below = [r for r in selected if r['evidence'] == 'jump_supported_below_threshold']
        later_below = [r for r in below if above and r['time'] > max(a['time'] for a in above)]
        result.append(dict(track=number, side=track[0]['side'], observations=len(track),
            first_time=track[0]['time'], last_time=track[-1]['time'],
            first_x=track[0]['x'], last_x=track[-1]['x'], central_birth=central_birth,
            net_outward_displacement_cells=float(displacement),
            reflection_candidate=bool(central_birth and displacement >= 8 and len(track) >= 17),
            above_supported_events=len(above), below_supported_events=len(below),
            ordered_above_then_below=bool(has_three_consecutive(above) and has_three_consecutive(later_below))))
    return result
