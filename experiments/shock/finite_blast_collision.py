"""Symmetric finite-energy hot patches in a uniform ideal gas.

This is a deliberately simple planar experiment, not a reproduction of the
Woodward-Colella gamma=1.4 reflecting-wall benchmark or of wind injection.
Cell averages preserve the specified physical initial thermal energy at every
resolution. No continuing energy source or replenished post-shock reservoir.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from astronomix import (
    CARTESIAN, SimulationConfig, SimulationParams, construct_primitive_state,
    finalize_config, get_registered_variables, time_integration,
)
from astronomix.option_classes.simulation_config import SnapshotSettings


@dataclass(frozen=True)
class BlastSettings:
    gamma: float = 5/3
    density: float = 1.
    ambient_pressure: float = 1.
    hot_pressure: float = 100.
    centers: tuple = (.4, .6)
    half_width: float = .02
    t_end: float = .06
    events: int = 301


def initial_profile(resolution, settings=BlastSettings()):
    if resolution < 64 or resolution % 4:
        raise ValueError('Require a multiple of four, at least 64 cells.')
    if (settings.gamma <= 1 or settings.density <= 0 or settings.ambient_pressure <= 0
            or settings.hot_pressure <= settings.ambient_pressure
            or len(settings.centers) != 2 or not 0 < settings.half_width < .1
            or not settings.half_width < settings.centers[0] < .5-settings.half_width
            or not np.isclose(settings.centers[1], 1-settings.centers[0])):
        raise ValueError('Require two separated symmetric finite hot patches and positive gas states.')
    edges = np.arange(resolution+1)/resolution
    fraction = np.zeros(resolution)
    for center in settings.centers:
        fraction += np.maximum(0., np.minimum(edges[1:], center+settings.half_width)
            - np.maximum(edges[:-1], center-settings.half_width))*resolution
    q = np.empty((resolution, 3)); q[:, 0] = settings.density; q[:, 1] = 0.
    q[:, 2] = settings.ambient_pressure+fraction*(settings.hot_pressure-settings.ambient_pressure)
    return q


def evolve_blasts(resolution, settings=BlastSettings()):
    q = initial_profile(resolution, settings)
    if not np.isfinite(settings.t_end) or settings.t_end <= 0 or settings.events < 5:
        raise ValueError('Require positive finite end time and at least five states.')
    config = SimulationConfig(geometry=CARTESIAN, dimensionality=1, num_cells=resolution,
        box_size=1., return_snapshots=True, num_snapshots=settings.events,
        use_specific_snapshot_timepoints=True,
        snapshot_settings=SnapshotSettings(return_states=True, return_final_state=True),
        print_elapsed_time=False, progress_bar=False)
    registry = get_registered_variables(config)
    initial = construct_primitive_state(config=config, registered_variables=registry,
        density=jnp.asarray(q[:, 0], jnp.float32), velocity_x=jnp.asarray(q[:, 1], jnp.float32),
        gas_pressure=jnp.asarray(q[:, 2], jnp.float32))
    config = finalize_config(config, initial.shape)
    target = np.linspace(0., settings.t_end, settings.events, dtype=np.float32)
    params = SimulationParams(gamma=settings.gamma, t_end=settings.t_end,
        snapshot_timepoints=jnp.asarray(target), C_cfl=.4)
    output = time_integration(initial, config, params, registry)
    jax.block_until_ready(output.states)
    times = np.asarray(output.time_points)
    q = np.asarray(output.states)[:, [registry.density_index, registry.velocity_index, registry.pressure_index], :]
    profiles = np.moveaxis(q, 1, -1)
    # The stored-event scheduler permits a small absolute float32 endpoint
    # tolerance. Use the recorded times, not the nominal cadence, for motion.
    np.testing.assert_allclose(times, target, rtol=0, atol=2.e-6)
    if not np.all(np.diff(times) > 0):
        raise AssertionError('Actual saved times must increase strictly.')
    if not np.isfinite(profiles).all() or (profiles[:, :, [0, 2]] <= 0).any():
        raise AssertionError('Nonpositive or nonfinite finite-blast evolution.')
    return profiles, times, dict(iterations=int(output.num_iterations),
        riemann_solver=config.riemann_solver, limiter=config.limiter,
        time_integrator=config.time_integrator, reconstruction_order=config.reconstruction_order,
        cfl=float(params.C_cfl), boundary='open', dtype=str(profiles.dtype),
        maximum_requested_actual_time_difference=float(abs(times-target).max()))
