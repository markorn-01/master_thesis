"""Pressure-front motion and Euler checks using independently measured speeds.

This diagnostic follows pressure ridges, not production shock objects. A ridge
can be a smooth wave. Full jump compatibility and persistence are reported
separately; neither a rejected ridge nor an empty search proves shock absence.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class MotionSettings:
    gamma: float = 5 / 3
    fit_half_events: int = 4
    distances_cells: tuple = (1., 2., 3., 4., 6., 8.)
    tolerances: tuple = (.05, .10, .20)
    pressure_prominence_fraction: float = 1.e-5
    minimum_normal_x: float = .8
    entropy_signal_tolerance: float = .5
    production_mach_threshold: float = 1.3


def measured_speed_metrics(upstream, downstream, normal, normal_speed, gamma=5/3):
    """Test jumps at a supplied speed; never solve mass balance for that speed."""
    a, b, n = map(lambda value: np.asarray(value, float), (upstream, downstream, normal))
    if (a.shape != (5,) or b.shape != (5,) or n.shape != (3,)
            or not np.isfinite(np.r_[a, b, n, normal_speed, gamma]).all()
            or min(a[0], b[0], a[4], b[4]) <= 0 or gamma <= 1
            or not np.isclose(n @ n, 1)):
        raise ValueError('Require positive finite primitive states, a unit normal, and a finite measured speed.')
    u1, u2 = a[1:4] @ n, b[1:4] @ n
    w1, w2 = u1 - normal_speed, u2 - normal_speed
    c1, c2 = np.sqrt(gamma * a[4] / a[0]), np.sqrt(gamma * b[4] / b[0])
    j1, j2 = a[0] * w1, b[0] * w2
    m1, m2 = a[4] + a[0] * w1**2, b[4] + b[0] * w2**2
    vt1, vt2 = a[1:4] - u1*n, b[1:4] - u2*n
    h1 = gamma/(gamma-1)*a[4]/a[0] + .5*(w1**2 + vt1 @ vt1)
    h2 = gamma/(gamma-1)*b[4]/b[0] + .5*(w2**2 + vt2 @ vt2)
    rho_ratio, pressure_ratio = b[0]/a[0], b[4]/a[4]
    predicted_rho_ratio = ((gamma+1)*pressure_ratio+gamma-1)/((gamma-1)*pressure_ratio+gamma+1)
    entropy = np.log(pressure_ratio) - gamma*np.log(rho_ratio)
    predicted_entropy = np.log(pressure_ratio) - gamma*np.log(predicted_rho_ratio)
    admissible = bool(rho_ratio > 1.0001 and pressure_ratio > 1.0001 and entropy > 1.e-10
                      and u1 > u2 and w1 > c1 and 0 < w2 < c2)
    return dict(
        measured_normal_speed=float(normal_speed), density_ratio=float(rho_ratio),
        pressure_ratio=float(pressure_ratio), entropy_change_over_cv=float(entropy),
        shock_frame_mach=float(w1/c1), downstream_shock_frame_mach=float(w2/c2),
        mass_relative_error=float(abs(j2-j1)/max(abs(j1), abs(j2), 1.e-12*a[0]*c1)),
        momentum_relative_error=float(abs(m2-m1)/max(m1, m2)),
        enthalpy_relative_error=float(abs(h2-h1)/max(h1, h2)),
        tangential_error=float(np.linalg.norm(vt2-vt1)/c1),
        density_ratio_error=float(abs(rho_ratio/predicted_rho_ratio-1)),
        entropy_signal_error=float(abs(entropy-predicted_entropy)/max(abs(predicted_entropy), 1.e-12)),
        necessary_shock_conditions=admissible,
    )


def bilinear_band(band, y, z, target_y=.375, target_z=.5):
    """Interpolate a small primitive-state strip to one specified physical ray."""
    q = np.asarray(band, float)
    y, z = np.asarray(y), np.asarray(z)
    if q.shape[-2:] != (len(y), len(z)) or not (y[0] <= target_y < y[-1] and z[0] <= target_z < z[-1]):
        raise ValueError('The transverse target must be bracketed by the saved strip.')
    j, k = np.searchsorted(y, target_y, side='right')-1, np.searchsorted(z, target_z, side='right')-1
    fy, fz = (target_y-y[j])/(y[j+1]-y[j]), (target_z-z[k])/(z[k+1]-z[k])
    return ((1-fy)*(1-fz)*q[..., j, k] + fy*(1-fz)*q[..., j+1, k]
            + (1-fy)*fz*q[..., j, k+1] + fy*fz*q[..., j+1, k+1])


def strip_fields(bands, y, z, dx, target_y=.375, target_z=.5):
    """Center ray plus transverse derivatives; retain all three velocities."""
    line = bilinear_band(bands, y, z, target_y, target_z)
    dy = (bilinear_band(bands, y, z, target_y+dx, target_z)
          - bilinear_band(bands, y, z, target_y-dx, target_z))/(2*dx)
    dz = (bilinear_band(bands, y, z, target_y, target_z+dx)
          - bilinear_band(bands, y, z, target_y, target_z-dx))/(2*dx)
    return line, dy, dz


def pressure_ridges(line, x, settings=MotionSettings()):
    """All significant pressure-gradient peaks; no temperature or Mach filter."""
    derivative = np.gradient(line[4], x)
    strength = abs(derivative)
    prominence = settings.pressure_prominence_fraction * max(float(line[4].max()), 1.e-12)/(x[1]-x[0])
    peaks, _ = find_peaks(strength, prominence=prominence)
    rows = []
    for index in peaks:
        if not 1 < index < len(x)-2:
            continue
        # Three-point quadratic peak refinement. A plateau is not forced to
        # have a preferred subcell direction.
        lower, center, upper = strength[index-1:index+2]
        denominator = lower-2*center+upper
        offset = np.clip(.5*(lower-upper)/denominator, -.5, .5) if denominator < 0 else 0.
        position = x[index] + offset*(x[1]-x[0])
        if (.02 <= position <= .3125 or .4375 <= position <= .5625 or .6875 <= position <= .98):
            rows.append(dict(index=int(index), x=float(position), sign=int(np.sign(derivative[index])),
                             pressure_gradient=float(derivative[index])))
    return rows


def link_ridges(lines, times, x, settings=MotionSettings()):
    """Conservative mutual-nearest adjacent links, ending paths at ambiguity.

    The gate uses local |u|+c and a 1.5-cell positional allowance. Multiple
    nearby candidates end the path instead of forcing a merge or ancestry.
    These are short pressure-feature paths, not the thesis object tracker.
    """
    tracks = []
    previous = []
    dx = x[1]-x[0]
    for event, (time, line) in enumerate(zip(times, lines)):
        current = pressure_ridges(line, x, settings)
        eligible = []
        if previous:
            dt = time-times[event-1]
            for j, item in enumerate(current):
                local = line[:, item['index']]
                gate = 1.5*dx + dt*(abs(local[1])+np.sqrt(settings.gamma*local[4]/local[0]))
                for i, old in enumerate(previous):
                    if item['sign'] == old['sign'] and abs(item['x']-old['x']) <= gate:
                        eligible.append((i, j, abs(item['x']-old['x'])))
        inherited = {}
        for i, j, distance in eligible:
            before = sorted((d, jj) for ii, jj, d in eligible if ii == i)
            after = sorted((d, ii) for ii, jj, d in eligible if jj == j)
            # Reject effectively tied neighbors (including flat/split crests).
            if (before[0][1] == j and after[0][1] == i
                    and (len(before) == 1 or before[1][0]-distance > .25*dx)
                    and (len(after) == 1 or after[1][0]-distance > .25*dx)):
                inherited[j] = previous[i]['track']
        for j, item in enumerate(current):
            track = inherited.get(j)
            if track is None:
                track = len(tracks)
                tracks.append([])
            item.update(event=event, time=float(time), track=track)
            tracks[track].append(item)
        previous = current
    return tracks


def fit_motion(track, index, settings=MotionSettings()):
    half = settings.fit_half_events
    if index < half or index+half >= len(track):
        return None
    window = track[index-half:index+half+1]
    if np.any(np.diff([r['event'] for r in window]) != 1):
        return None
    t = np.array([r['time'] for r in window]); t -= t.mean()
    position = np.array([r['x'] for r in window])
    speed = float(t @ (position-position.mean())/(t @ t))
    residual = position-position.mean()-speed*t
    scatter = float(np.sqrt(residual @ residual/(len(t)-2)/(t @ t)))
    return speed, scatter, float(np.ptp(t))


def analyze_motion(lines, times, x, transverse_y=None, transverse_z=None,
                   settings=MotionSettings(), source_positions=None, source_radius=.0625):
    """Save all finite sampling choices and each ridge's best declared checks."""
    lines, times, x = np.asarray(lines, float), np.asarray(times, float), np.asarray(x, float)
    if (lines.shape != (len(times), 5, len(x)) or len(times) < 2 or len(x) < 8
            or not np.isfinite(lines).all() or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0) or not np.all(np.diff(x) > 0)
            or (lines[:, [0, 4]] <= 0).any()):
        raise ValueError('Require finite positive primitive profiles on increasing space and time grids.')
    if not np.allclose(np.diff(x), x[1]-x[0]):
        raise ValueError('Uniform spatial grid required.')
    if transverse_y is None:
        transverse_y = np.zeros_like(lines)
    if transverse_z is None:
        transverse_z = np.zeros_like(lines)
    if np.shape(transverse_y) != lines.shape or np.shape(transverse_z) != lines.shape:
        raise ValueError('Transverse derivatives must match the primitive profiles.')
    dx = x[1]-x[0]
    gradient_x = np.gradient(lines[:, 4], dx, axis=-1)
    tracks = link_ridges(lines, times, x, settings)
    pairs, observations = [], []
    def sample(field, position):
        return np.array([np.interp(position, x, component) for component in field])
    for track in tracks:
        for number, point in enumerate(track):
            event, center = point['event'], point['x']
            region = ('central_left' if center < .5 else 'central_right') if .4375 <= center <= .5625 else 'outer_control'
            fit = fit_motion(track, number, settings)
            observation = dict(track=point['track'], event=event, time=point['time'], x=center, region=region,
                               motion_fit_available=fit is not None, sampled_pairs=0,
                               minimum_compatible_mach_10pct=None, maximum_compatible_mach_10pct=None,
                               compatible_above_detection_threshold_10pct=False,
                               compatible_below_detection_threshold_10pct=False)
            for tol in settings.tolerances:
                observation[f'compatible_{int(round(100*tol)):02d}pct'] = False
            if fit is None:
                observations.append(observation)
                continue
            speed, scatter, duration = fit
            gradient = np.array([np.interp(center, x, gradient_x[event]),
                np.interp(center, x, transverse_y[event, 4]), np.interp(center, x, transverse_z[event, 4])])
            normal = gradient / max(np.linalg.norm(gradient), 1.e-30)
            sign = np.sign(normal[0])
            observation.update(measured_x_speed=speed, speed_fit_scatter=scatter, fit_duration=duration,
                               normal_x=float(normal[0]), normal_y=float(normal[1]), normal_z=float(normal[2]))
            if abs(normal[0]) < settings.minimum_normal_x:
                observations.append(observation)
                continue
            q = lines[event]
            for upstream_distance in settings.distances_cells:
                for downstream_distance in settings.distances_cells:
                    u = center-sign*upstream_distance*dx
                    d = center+sign*downstream_distance*dx
                    ou, od = u-sign*.5*dx, d+sign*.5*dx
                    if min(ou, od) < x[0] or max(ou, od) > x[-1]:
                        continue
                    # Never pair states across the midplane of opposing flows.
                    if region.startswith('central') and (ou-.5)*(od-.5) <= 0:
                        continue
                    if source_positions is not None:
                        from experiments.wind_bubble.assess_collision_physics import source_free_segments
                        if not source_free_segments(np.array([[ou, .375, .5]]),
                            np.array([[od, .375, .5]]), np.asarray(source_positions), source_radius)[0]:
                            continue
                    # Opposing pressure directions indicate a second wave.
                    ids = (x >= min(ou, od)) & (x <= max(ou, od))
                    gradients = gradient_x[event, ids]
                    limit = 1.e-5*max(float(abs(gradient_x[event]).max()), 1.e-30)
                    if np.any(gradients*sign < -limit):
                        continue
                    a, b = sample(q, u), sample(q, d)
                    ao, bo = sample(q, ou), sample(q, od)
                    metrics = measured_speed_metrics(a, b, normal, normal[0]*speed, settings.gamma)
                    c1 = np.sqrt(settings.gamma*a[4]/a[0])
                    scales = np.r_[max(abs(b[0]-a[0]), 1.e-12*max(a[0], b[0])),
                        np.repeat(max(np.linalg.norm(b[1:4]-a[1:4]), 1.e-12*c1), 3),
                        max(abs(b[4]-a[4]), 1.e-12*max(a[4], b[4]))]
                    plateau = float(max(np.max(abs(ao-a)/scales), np.max(abs(bo-b)/scales)))
                    errors = [metrics[k] for k in ('mass_relative_error','momentum_relative_error',
                        'enthalpy_relative_error','tangential_error','density_ratio_error')]
                    maximum = max(*errors, plateau, scatter/c1)
                    row = dict(track=point['track'], event=event, time=point['time'], x=center, region=region,
                        measured_x_speed=speed, speed_fit_scatter=scatter, fit_duration=duration,
                        upstream_x=u, downstream_x=d, normal_x=float(normal[0]),
                        normal_y=float(normal[1]), normal_z=float(normal[2]),
                        upstream_density=float(a[0]), downstream_density=float(b[0]),
                        upstream_pressure=float(a[4]), downstream_pressure=float(b[4]),
                        endpoint_relative_change=plateau, largest_relative_error=maximum, **metrics)
                    for tol in settings.tolerances:
                        key = f'compatible_{int(round(100*tol)):02d}pct'
                        row[key] = bool(metrics['necessary_shock_conditions'] and maximum <= tol
                            and metrics['entropy_signal_error'] <= settings.entropy_signal_tolerance)
                        observation[key] |= row[key]
                        if row[key] and np.isclose(tol, .10):
                            mach = metrics['shock_frame_mach']
                            current_min = observation['minimum_compatible_mach_10pct']
                            current_max = observation['maximum_compatible_mach_10pct']
                            observation['minimum_compatible_mach_10pct'] = min(current_min, mach) if current_min is not None else mach
                            observation['maximum_compatible_mach_10pct'] = max(current_max, mach) if current_max is not None else mach
                    pairs.append(row)
                    observation['sampled_pairs'] += 1
            if observation['compatible_10pct']:
                observation['compatible_above_detection_threshold_10pct'] = bool(
                    observation['minimum_compatible_mach_10pct'] >= settings.production_mach_threshold)
                observation['compatible_below_detection_threshold_10pct'] = bool(
                    observation['maximum_compatible_mach_10pct'] < settings.production_mach_threshold)
            observations.append(observation)
    return observations, pairs, tracks
