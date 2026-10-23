"""Single-state hydrodynamic shock analysis shared by experiment runners.

Accepts an unpadded 3D state and returns the established offline diagnostic
fields as NumPy arrays. This host-side adapter is not a JIT callback: a future
online integration must separate device computations from host conversion.
Alternative estimators are retained here for exact offline compatibility.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from astronomix.option_classes.simulation_config import CARTESIAN

from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix._physics_modules._shock_finder._shock_mach import (
    _calculate_mach_at_surface, _calculate_plateau_mach_at_surface,
    _calculate_extrapolated_mach_at_surface,
)
from astronomix._physics_modules._shock_finder._shock_zones import (
    get_profile_aware_post_pre_shock_values,
    get_post_pre_sample_relative_changes,
    get_plateau_aware_post_pre_shock_values,
)
from astronomix._physics_modules._shock_finder._energy_dissipation import calculate_thermal_energy_flux


def analyze_shock_state(state, config, registered_variables, helper_data, *, center=None):
    """Analyze one unpadded Cartesian hydrodynamic state.

    ``center`` defines only the radial diagnostic origin; it does not affect
    shock detection. By default it is the center of the configured box.
    ``mach_numbers`` preserves the original estimator; the primary improved
    estimate is explicitly named ``profile_aware_mach_numbers``.
    """
    if config.mhd or config.dimensionality != 3 or config.geometry != CARTESIAN:
        raise ValueError("Single-state analysis requires Cartesian 3D hydrodynamics.")
    expected_shape = tuple(helper_data.geometric_centers.shape[:-1])
    if tuple(state.shape[1:]) != expected_shape:
        raise ValueError("State must be unpadded and match the helper-data grid.")
    result = find_shocks_pfrommer(
        jnp.asarray(state),
        config,
        registered_variables,
        helper_data,
    )
    profile_mach_numbers = _calculate_mach_at_surface(
        jnp.asarray(state),
        result.shock_surface_cells,
        result.shock_zones,
        result.shock_direction,
        result.shock_surface_offsets,
        config,
        registered_variables,
        profile_aware=True,
    )
    plateau_mach_numbers = _calculate_plateau_mach_at_surface(
        jnp.asarray(state),
        result.shock_surface_cells,
        result.shock_zones,
        result.shock_direction,
        result.shock_surface_offsets,
        config,
        registered_variables,
    )
    extrapolated_mach_numbers = _calculate_extrapolated_mach_at_surface(
        jnp.asarray(state),
        result.shock_surface_cells,
        result.shock_zones,
        result.shock_direction,
        result.shock_surface_offsets,
        config,
        registered_variables,
    )
    pressure = jnp.asarray(state)[registered_variables.pressure_index]
    density = jnp.asarray(state)[registered_variables.density_index]
    (
        _,
        _,
        _,
        _,
        profile_sampling_valid,
        profile_post_distance,
        profile_pre_distance,
    ) = get_profile_aware_post_pre_shock_values(
        result.shock_direction,
        result.shock_zones,
        pressure,
        density,
        max_steps=8,
        center_offsets=(
            result.shock_direction
            * result.shock_surface_offsets[jnp.newaxis, ...]
        ),
    )
    (
        profile_post_pressure_change,
        profile_pre_pressure_change,
        profile_post_density_change,
        profile_pre_density_change,
        profile_flatness_valid,
    ) = get_post_pre_sample_relative_changes(
        result.shock_direction,
        pressure,
        density,
        profile_post_distance,
        profile_pre_distance,
        center_offsets=(
            result.shock_direction
            * result.shock_surface_offsets[jnp.newaxis, ...]
        ),
    )
    profile_flatness_valid = profile_sampling_valid & profile_flatness_valid
    (
        _,
        _,
        _,
        _,
        plateau_sampling_valid,
        plateau_post_distance,
        plateau_pre_distance,
        plateau_post_pressure_change,
        plateau_pre_pressure_change,
        plateau_post_density_change,
        plateau_pre_density_change,
    ) = get_plateau_aware_post_pre_shock_values(
        result.shock_direction,
        result.shock_zones,
        pressure,
        density,
        max_steps=8,
        center_offsets=(
            result.shock_direction
            * result.shock_surface_offsets[jnp.newaxis, ...]
        ),
    )
    profile_thermal_energy_flux = calculate_thermal_energy_flux(
        primitive_state=jnp.asarray(state),
        shock_surface=result.shock_surface_cells,
        shock_zones=result.shock_zones,
        shock_direction=result.shock_direction,
        surface_offsets=result.shock_surface_offsets,
        mach_numbers=profile_mach_numbers,
        config=config,
        registered_variables=registered_variables,
        profile_aware_sampling=True,
    )
    surface_mask = np.asarray(result.shock_surface_cells, dtype=bool)
    surface_offsets = np.asarray(result.shock_surface_offsets)
    shock_direction = np.moveaxis(np.asarray(result.shock_direction), 0, -1)
    centers = np.asarray(helper_data.geometric_centers)
    refined_centers = centers + (
        float(config.grid_spacing)
        * surface_offsets[..., np.newaxis]
        * shock_direction
    )
    star_position = (
        np.broadcast_to(np.asarray(config.box_size, dtype=float), (3,)) / 2.0
        if center is None else np.asarray(center, dtype=float)
    )
    if star_position.shape != (3,) or not np.isfinite(star_position).all():
        raise ValueError("center must contain three finite coordinates.")
    refined_radii = np.linalg.norm(
        refined_centers - star_position,
        axis=-1,
    )
    return {
        "surface_mask": surface_mask,
        "refined_centers": refined_centers,
        "refined_radii": refined_radii,
        "shock_direction": shock_direction,
        "mach_numbers": np.asarray(result.mach_numbers),
        "thermal_energy_flux": np.asarray(result.thermal_energy_flux),
        "profile_aware_mach_numbers": np.asarray(
            profile_mach_numbers
        ),
        "plateau_aware_mach_numbers": np.asarray(
            plateau_mach_numbers
        ),
        "extrapolated_mach_numbers": np.asarray(
            extrapolated_mach_numbers
        ),
        "profile_sampling_valid": np.asarray(
            profile_flatness_valid, dtype=bool
        ),
        "profile_post_distance": np.asarray(profile_post_distance),
        "profile_pre_distance": np.asarray(profile_pre_distance),
        "profile_post_pressure_change": np.asarray(
            profile_post_pressure_change
        ),
        "profile_pre_pressure_change": np.asarray(
            profile_pre_pressure_change
        ),
        "profile_post_density_change": np.asarray(
            profile_post_density_change
        ),
        "profile_pre_density_change": np.asarray(
            profile_pre_density_change
        ),
        "plateau_sampling_valid": np.asarray(
            plateau_sampling_valid, dtype=bool
        ),
        "plateau_post_distance": np.asarray(plateau_post_distance),
        "plateau_pre_distance": np.asarray(plateau_pre_distance),
        "plateau_post_pressure_change": np.asarray(
            plateau_post_pressure_change
        ),
        "plateau_pre_pressure_change": np.asarray(
            plateau_pre_pressure_change
        ),
        "plateau_post_density_change": np.asarray(
            plateau_post_density_change
        ),
        "plateau_pre_density_change": np.asarray(
            plateau_pre_density_change
        ),
        "profile_aware_thermal_energy_flux": np.asarray(
            profile_thermal_energy_flux
        ),
    }
