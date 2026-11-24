"""Bounded planar hydro analogue of SHOCKFIND's density-normal strategy.

This is not the SHOCKFIND MHD implementation. Compression/density candidates,
density-gradient directions, averaged side states and compression maxima follow
its general design. Gas-dynamic sonic and entropy conditions replace magnetic
classification. No production shock-finder code is changed or called here.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DensityNormalSettings:
    gamma: float = 5 / 3
    minimum_mach: float = 1.3
    shock_thickness_cells: int = 3
    relative_contrast_floor: float = 1.e-4
    log_entropy_floor: float = 1.e-10


def classify_states(upstream, downstream, normal, settings=DensityNormalSettings()):
    """Separate kinematic candidates from entropy-admissible sampled jumps.

    Normal points toward higher density. Mass balance supplies a speed estimate,
    so that estimate is not independent propagation evidence. There is no tight
    numerical Rankine--Hugoniot residual gate or temperature-gradient criterion.
    """
    a, b = np.asarray(upstream, float), np.asarray(downstream, float)
    if (a.shape != (3,) or b.shape != (3,) or not np.isfinite(np.r_[a, b]).all()
            or min(a[0], b[0], a[2], b[2]) <= 0 or normal not in (-1, 1)):
        raise ValueError('Require finite rho/u/P states, positive rho/P and a signed unit normal.')
    if not np.isfinite([settings.gamma, settings.minimum_mach]).all() or settings.gamma <= 1 or settings.minimum_mach <= 1:
        raise ValueError('Require gamma > 1 and a supersonic detection threshold.')
    rho_ratio, pressure_ratio = b[0]/a[0], b[2]/a[2]
    c1, c2 = np.sqrt(settings.gamma*a[2]/a[0]), np.sqrt(settings.gamma*b[2]/b[0])
    u1, u2 = normal*a[1], normal*b[1]
    contrast = bool(rho_ratio > 1+settings.relative_contrast_floor
                    and pressure_ratio > 1+settings.relative_contrast_floor)
    drop = u1-u2
    finite_speed = abs(b[0]-a[0]) > 1.e-12*max(a[0], b[0])
    speed = (b[0]*u2-a[0]*u1)/(b[0]-a[0]) if finite_speed else np.nan
    w1, w2 = u1-speed, u2-speed
    sonic = bool(np.isfinite(speed) and w1 > c1 and 0 < w2 < c2)
    mach = w1/c1
    entropy = np.log(pressure_ratio)-settings.gamma*np.log(rho_ratio)
    kinematic = bool(contrast and drop > 1.e-10*c1 and sonic and mach >= settings.minimum_mach)
    entropy_ok = bool(entropy > settings.log_entropy_floor)
    pressure_mach = np.sqrt(((settings.gamma+1)*pressure_ratio+settings.gamma-1)/(2*settings.gamma))
    return dict(density_ratio=float(rho_ratio), pressure_ratio=float(pressure_ratio),
        normal_velocity_drop=float(drop), mass_inferred_normal_speed=float(speed),
        mass_inferred_mach=float(mach), pressure_mach=float(pressure_mach),
        downstream_shock_frame_mach=float(w2/c2),
        compressive_contrast=contrast, sonic_ordering=sonic,
        sampled_log_entropy_change=float(entropy), entropy_increase=entropy_ok,
        kinematic_candidate=kinematic, accepted=bool(kinematic and entropy_ok))


def analyze_line(profile, grid_spacing, settings=DensityNormalSettings()):
    """Analyze a planar rho/u/P profile with one fixed averaging distance.

    For thickness N=3, side averages use offsets 2, 3, 4 cells from the refined
    candidate center (three-cell blocks with edges at 1.5 and 4.5 cells).
    Transverse cylinder averaging is trivial for these planar inputs. This
    function makes no claim to be a general 3D surface reconstruction.
    """
    q = np.asarray(profile, float)
    if (q.ndim != 2 or q.shape[1] != 3 or q.shape[0] < 16
            or not np.isfinite(q).all() or (q[:, (0, 2)] <= 0).any()
            or not np.isfinite(grid_spacing) or grid_spacing <= 0):
        raise ValueError('Require at least 16 finite rho/u/P cells and positive spacing, density and pressure.')
    width = settings.shock_thickness_cells
    if not isinstance(width, (int, np.integer)) or width < 1 or width % 2 != 1:
        raise ValueError('Shock thickness must be a positive odd number of cells.')
    density_gradient = np.gradient(q[:, 0], grid_spacing)
    compression = -np.gradient(q[:, 1], grid_spacing)
    eps = 64*np.finfo(float).eps
    c = np.sqrt(settings.gamma*q[:, 2]/q[:, 0])
    compression_floor = eps*max(float(c.max()), float(np.ptp(q[:, 1])))/grid_spacing
    gradient_floor = eps*float(q[:, 0].max())/grid_spacing
    eligible = (compression > compression_floor) | (abs(density_gradient) > gradient_floor)
    eligible &= abs(density_gradient) > gradient_floor  # A density direction must exist.
    eligible[0:2] = eligible[-2:] = False
    eligible[1:-1] &= ((compression[1:-1] >= compression[:-2]-compression_floor)
                        & (compression[1:-1] >= compression[2:]-compression_floor))
    offsets = np.arange((width+1)//2, (width+1)//2+width, dtype=float)
    coordinates = np.arange(len(q), dtype=float)
    candidates = []
    for index in np.flatnonzero(eligible):
        lower, center, upper = compression[index-1:index+2]
        denominator = lower-2*center+upper
        delta = np.clip(.5*(lower-upper)/denominator, -.5, .5) if denominator < -compression_floor else 0.
        location = index+delta
        normal = int(np.sign(density_gradient[index]))
        if any(abs(location-r['coordinate_cells']) < 1.e-8
               and normal == r['normal_upstream_to_downstream'] for r in candidates):
            continue
        up_points, down_points = location-normal*offsets, location+normal*offsets
        if min(up_points.min(), down_points.min()) < 0 or max(up_points.max(), down_points.max()) > len(q)-1:
            continue
        def side_average(points):
            return np.array([np.interp(points, coordinates, field).mean() for field in q.T])
        up, down = side_average(up_points), side_average(down_points)
        candidate = dict(cell_index=int(index), coordinate_cells=float(location),
            x=float((location+.5)*grid_spacing), normal_upstream_to_downstream=normal,
            compression=float(compression[index]),
            upstream_density=float(up[0]), upstream_velocity=float(up[1]), upstream_pressure=float(up[2]),
            downstream_density=float(down[0]), downstream_velocity=float(down[1]), downstream_pressure=float(down[2]),
            **classify_states(up, down, normal, settings))
        candidates.append(candidate)
    return candidates
