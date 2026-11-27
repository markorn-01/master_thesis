"""NumPy preflight for disjoint equal-mask EI source layouts.

Shared budget helpers for new multi-source pilots. Historical two/three-source
helpers remain unchanged. Equal mask counts allow one shared mass parameter
to match the existing reference's prescribed per-source mass/energy budget.
"""
import numpy as np

from experiments.wind_bubble.two_bubble_source_budget import (
    INJECTION_RADIUS, budget_plan, discrete_rates,
)


def validate_sources(sources):
    points = np.asarray(sources, dtype=float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not len(points)
            or not np.isfinite(points).all()):
        raise ValueError('Require a nonempty list of finite three-dimensional source positions.')
    clearance = np.minimum(points, 1-points).min()
    if clearance <= INJECTION_RADIUS:
        raise ValueError('Injection spheres must lie strictly inside the unit domain.')
    distances = np.linalg.norm(points[:, None]-points[None, :], axis=-1)
    pairs = distances[np.triu_indices(len(points), 1)]
    if len(pairs) and pairs.min() <= 2*INJECTION_RADIUS:
        raise ValueError('Injection spheres must not overlap or touch.')
    return points


def injection_counts(num_cells, sources):
    if (isinstance(num_cells, (bool, np.bool_)) or not isinstance(num_cells, (int, np.integer))
            or num_cells < 16 or num_cells % 16):
        raise ValueError('Require a multiple of 16 cells per axis, at least 16.')
    points = validate_sources(sources)
    x = (np.arange(num_cells)+.5)/num_cells
    limit = (INJECTION_RADIUS-.5/num_cells)**2
    counts = [int(((x[:, None, None]-p[0])**2+(x[None, :, None]-p[1])**2
                   +(x[None, None, :]-p[2])**2 <= limit).sum()) for p in points]
    if not all(counts) or len(set(counts)) != 1:
        raise ValueError('This shared-parameter pilot requires nonempty equal-sized injection masks.')
    return counts


def source_budget(num_cells, reference=None, mass_rate=.01, velocity=10., *, sources, actual_counts=None):
    counts = injection_counts(num_cells, sources)
    if actual_counts is not None and list(actual_counts) != counts:
        raise ValueError('Actual source masks differ from the NumPy preflight.')
    raw_mass, _ = discrete_rates(num_cells, counts, mass_rate, velocity)
    scale = 1.
    target_mass = target_energy = None
    if reference is not None:
        # Validate the old equal two-source reference, not a guessed rate.
        checked = budget_plan(reference['num_cells'], reference,
                              reference['wind_mass_loss_rate_per_source'], velocity)
        rates = checked['discrete_mass_rates']
        if not np.allclose(rates, rates[0], rtol=1e-12, atol=0):
            raise ValueError('Require an equal-strength two-source budget reference.')
        target_mass = float(rates[0])
        target_energy = float(checked['discrete_energy_rates'][0])
        scale = target_mass/raw_mass[0]
    adjusted = float(mass_rate*scale)
    mass, energy = discrete_rates(num_cells, counts, adjusted, velocity)
    if target_mass is not None:
        np.testing.assert_allclose(mass, target_mass, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(energy, target_energy, rtol=1e-12, atol=1e-12)
    count = len(counts)
    return dict(num_cells=int(num_cells), num_sources=count, injection_cell_counts=counts,
                parameter_scale=float(scale), requested_mass_loss_rate_per_source=float(mass_rate),
                wind_mass_loss_rate_per_source=adjusted, wind_terminal_velocity=float(velocity),
                nominal_wind_luminosity_per_source=.5*velocity**2*adjusted,
                nominal_total_wind_luminosity=count*.5*velocity**2*adjusted,
                discrete_mass_rates=mass.tolist(), discrete_energy_rates=energy.tolist(),
                budget_matched=reference is not None,
                energy_rate_definition='Prescribed mechanical source coefficient before state-dependent EI pressure conversion; not net gas-energy deposition.',
                reference_scope=f'Each of {count} sources matches one source of the equal two-wind reference; '
                                f'total prescribed budget is {count/2:g} times its total.')


def weighted_budget(base, strengths=None):
    count = len(base['injection_cell_counts'])
    values = np.ones(count) if strengths is None else np.asarray(strengths, dtype=float)
    if values.shape != (count,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f'Require {count} finite positive source strengths.')
    mass, velocity = base['wind_mass_loss_rate_per_source'], base['wind_terminal_velocity']
    return dict(source_strengths=values.tolist(), source_mass_loss_rates=(mass*values).tolist(),
                nominal_wind_luminosities=(.5*velocity**2*mass*values).tolist(),
                discrete_mass_rates=(np.asarray(base['discrete_mass_rates'])*values).tolist(),
                discrete_energy_rates=(np.asarray(base['discrete_energy_rates'])*values).tolist(),
                equal_source_reference_preserved=False,
                equal_source_strengths=bool(np.all(values == 1)),
                per_source_reference_budget_preserved=bool(base['budget_matched'] and np.all(values == 1)))
