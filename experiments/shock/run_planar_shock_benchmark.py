"""Benchmark the hydrodynamic shock finder on a controlled planar shock.

The input is a synthetic, numerically broadened stationary normal shock.  Its
constant upstream and downstream states are generated from the exact
Rankine--Hugoniot relations, so the true Mach number is known independently of
the finder.  This first benchmark intentionally does not evolve the fluid: it
isolates shock detection, normal estimation, surface localization, and Mach
sampling from errors introduced by a time integrator or a curved flow.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from astronomix import (  # noqa: E402
    CARTESIAN,
    SimulationConfig,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
)
# noqa is required because the repository root is inserted above.
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import (  # noqa: E402
    find_shocks_pfrommer,
)
from astronomix._physics_modules._shock_finder._shock_zones import (  # noqa: E402
    get_adaptive_post_pre_shock_values,
    get_extrapolated_post_pre_shock_values,
    get_plateau_aware_post_pre_shock_values,
    get_post_pre_sample_relative_changes,
    get_profile_aware_post_pre_shock_values,
    get_ramp_exit_aware_post_pre_shock_values,
)
from astronomix._physics_modules._shock_finder._shock_mach import (  # noqa: E402
    _calculate_extrapolated_mach_at_surface,
    _calculate_mach_at_surface,
    _calculate_plateau_mach_at_surface,
    _calculate_ramp_exit_mach_at_surface,
)


GAMMA = 5.0 / 3.0
BOX_SIZE = 1.0
SHOCK_POSITION = 0.5
MACH_MIN = 1.3


def rankine_hugoniot_states(
    mach: float,
    gamma: float = GAMMA,
    upstream_density: float = 1.0,
    upstream_pressure: float = 1.0,
) -> dict[str, float]:
    """Return exact states for a stationary normal hydrodynamic shock.

    The upstream gas is on the positive-x side and flows toward negative x.
    The downstream gas is on the negative-x side.  Velocities are expressed
    in the stationary-shock frame.
    """
    if mach <= 1.0:
        raise ValueError("mach must be greater than one")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    if upstream_density <= 0.0 or upstream_pressure <= 0.0:
        raise ValueError("upstream density and pressure must be positive")

    mach_squared = mach**2
    density_ratio = (
        (gamma + 1.0) * mach_squared
        / ((gamma - 1.0) * mach_squared + 2.0)
    )
    pressure_ratio = (
        2.0 * gamma * mach_squared - (gamma - 1.0)
    ) / (gamma + 1.0)
    upstream_sound_speed = np.sqrt(
        gamma * upstream_pressure / upstream_density
    )
    upstream_velocity = -mach * upstream_sound_speed
    downstream_density = upstream_density * density_ratio
    downstream_pressure = upstream_pressure * pressure_ratio
    downstream_velocity = upstream_velocity / density_ratio

    return {
        "upstream_density": float(upstream_density),
        "upstream_pressure": float(upstream_pressure),
        "upstream_velocity": float(upstream_velocity),
        "downstream_density": float(downstream_density),
        "downstream_pressure": float(downstream_pressure),
        "downstream_velocity": float(downstream_velocity),
        "density_ratio": float(density_ratio),
        "pressure_ratio": float(pressure_ratio),
        "upstream_sound_speed": float(upstream_sound_speed),
    }


def _smooth_ramp(
    coordinate: jnp.ndarray,
    center: float,
    width: float,
) -> jnp.ndarray:
    """Return a compact smoothstep from zero to one across ``width``."""
    fraction = jnp.clip((coordinate - center) / width + 0.5, 0.0, 1.0)
    return fraction**2 * (3.0 - 2.0 * fraction)


def normalized_vector(components) -> np.ndarray:
    """Return a three-component unit vector."""
    vector = np.asarray(components, dtype=float)
    if vector.shape != (3,):
        raise ValueError("normal must contain exactly three components")
    magnitude = np.linalg.norm(vector)
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError("normal must be finite and nonzero")
    return vector / magnitude


def build_planar_shock_state(
    resolution: int,
    mach: float,
    ramp_cells: float,
    normal=(1.0, 0.0, 0.0),
):
    """Construct a 3D planar shock with exact constant plateau states."""
    if resolution < 12:
        raise ValueError("resolution must be at least 12")
    if ramp_cells <= 0.0:
        raise ValueError("ramp_cells must be positive")

    config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=resolution,
        mhd=False,
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    centers = helper_data.geometric_centers
    unit_normal = normalized_vector(normal)
    box_center = jnp.full((3,), SHOCK_POSITION)
    signed_distance = jnp.sum(
        (centers - box_center) * jnp.asarray(unit_normal),
        axis=-1,
    )
    grid_spacing = BOX_SIZE / resolution
    upstream_weight = _smooth_ramp(
        signed_distance,
        0.0,
        ramp_cells * grid_spacing,
    )
    states = rankine_hugoniot_states(mach)

    def blend(downstream_value: float, upstream_value: float):
        return downstream_value + upstream_weight * (
            upstream_value - downstream_value
        )

    density = blend(
        states["downstream_density"], states["upstream_density"]
    )
    pressure = blend(
        states["downstream_pressure"], states["upstream_pressure"]
    )
    normal_velocity = blend(
        states["downstream_velocity"], states["upstream_velocity"]
    )
    zeros = jnp.zeros_like(density)
    primitive_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=normal_velocity * unit_normal[0],
        velocity_y=(
            normal_velocity * unit_normal[1]
            if unit_normal[1] != 0.0
            else zeros
        ),
        velocity_z=(
            normal_velocity * unit_normal[2]
            if unit_normal[2] != 0.0
            else zeros
        ),
        gas_pressure=pressure,
    )
    config = finalize_config(config, primitive_state.shape)
    return (
        primitive_state,
        config,
        registered_variables,
        helper_data,
        states,
        unit_normal,
    )


def benchmark_resolution(
    resolution: int,
    mach: float,
    ramp_cells: float,
    normal=(1.0, 0.0, 0.0),
    diagnostic_outside_offset: float = 0.25,
) -> dict[str, float | int]:
    """Run the unchanged finder and summarize its planar-shock recovery."""
    state, config, variables, helper_data, states, unit_normal = (
        build_planar_shock_state(
            resolution=resolution,
            mach=mach,
            ramp_cells=ramp_cells,
            normal=normal,
        )
    )
    result = find_shocks_pfrommer(
        state,
        config,
        variables,
        helper_data,
        mach_min=MACH_MIN,
    )
    profile_mach_field = _calculate_mach_at_surface(
        state,
        result.shock_surface_cells,
        result.shock_zones,
        result.shock_direction,
        result.shock_surface_offsets,
        config,
        variables,
        profile_aware=True,
    )
    plateau_mach_field = _calculate_plateau_mach_at_surface(
        state,
        result.shock_surface_cells,
        result.shock_zones,
        result.shock_direction,
        result.shock_surface_offsets,
        config,
        variables,
    )
    ramp_exit_mach_field = _calculate_ramp_exit_mach_at_surface(
        state,
        result.shock_surface_cells,
        result.shock_zones,
        result.shock_direction,
        result.shock_surface_offsets,
        config,
        variables,
    )
    extrapolated_mach_field = _calculate_extrapolated_mach_at_surface(
        state,
        result.shock_surface_cells,
        result.shock_zones,
        result.shock_direction,
        result.shock_surface_offsets,
        config,
        variables,
    )

    surface = np.asarray(result.shock_surface_cells, dtype=bool)
    if not surface.any():
        raise RuntimeError(
            f"No shock surface was detected at resolution {resolution}."
        )
    mach_values = np.asarray(result.mach_numbers)[surface]
    valid_mach = mach_values[np.isfinite(mach_values) & (mach_values > 0.0)]
    if valid_mach.size == 0:
        raise RuntimeError(
            f"No valid Mach samples were produced at resolution {resolution}."
        )
    profile_mach_values = np.asarray(profile_mach_field)[surface]
    valid_profile_mach = profile_mach_values[
        np.isfinite(profile_mach_values) & (profile_mach_values > 0.0)
    ]
    if valid_profile_mach.size == 0:
        raise RuntimeError(
            "No valid profile-aware Mach samples were produced at "
            f"resolution {resolution}."
        )
    plateau_mach_values = np.asarray(plateau_mach_field)[surface]
    valid_plateau_mach = plateau_mach_values[
        np.isfinite(plateau_mach_values) & (plateau_mach_values > 0.0)
    ]
    if valid_plateau_mach.size == 0:
        raise RuntimeError(
            "No valid plateau-aware Mach samples were produced at "
            f"resolution {resolution}."
        )
    ramp_exit_mach_values = np.asarray(ramp_exit_mach_field)[surface]
    valid_ramp_exit_mach = ramp_exit_mach_values[
        np.isfinite(ramp_exit_mach_values) & (ramp_exit_mach_values > 0.0)
    ]
    if valid_ramp_exit_mach.size == 0:
        raise RuntimeError(
            "No valid ramp-exit-aware Mach samples were produced at "
            f"resolution {resolution}."
        )
    extrapolated_mach_values = np.asarray(extrapolated_mach_field)[surface]
    valid_extrapolated_mach = extrapolated_mach_values[
        np.isfinite(extrapolated_mach_values)
        & (extrapolated_mach_values > 0.0)
    ]
    if valid_extrapolated_mach.size == 0:
        raise RuntimeError(
            "No valid extrapolated Mach samples were produced at "
            f"resolution {resolution}."
        )

    centers = np.asarray(helper_data.geometric_centers)
    direction = np.moveaxis(np.asarray(result.shock_direction), 0, -1)
    offsets = np.asarray(result.shock_surface_offsets)
    refined_centers = centers + (
        config.grid_spacing * offsets[..., np.newaxis] * direction
    )
    plane_center = np.full(3, SHOCK_POSITION)
    signed_surface_distance = np.sum(
        (refined_centers[surface] - plane_center) * unit_normal,
        axis=-1,
    )
    surface_direction = direction[surface]
    normal_alignment = np.sum(surface_direction * unit_normal, axis=-1)
    p16, median, p84 = np.percentile(valid_mach, [16.0, 50.0, 84.0])
    profile_p16, profile_median, profile_p84 = np.percentile(
        valid_profile_mach, [16.0, 50.0, 84.0]
    )
    plateau_p16, plateau_median, plateau_p84 = np.percentile(
        valid_plateau_mach, [16.0, 50.0, 84.0]
    )
    ramp_p16, ramp_median, ramp_p84 = np.percentile(
        valid_ramp_exit_mach, [16.0, 50.0, 84.0]
    )
    extrapolated_p16, extrapolated_median, extrapolated_p84 = np.percentile(
        valid_extrapolated_mach, [16.0, 50.0, 84.0]
    )

    pressure = state[variables.pressure_index]
    density = state[variables.density_index]
    temperature = pressure / density
    (
        pressure_post,
        pressure_pre,
        _,
        _,
        valid_samples,
        post_distance,
        pre_distance,
    ) = get_adaptive_post_pre_shock_values(
        result.shock_direction,
        result.shock_zones,
        pressure,
        temperature,
        max_steps=8,
        outside_offset=diagnostic_outside_offset,
        center_offsets=(
            result.shock_direction
            * result.shock_surface_offsets[jnp.newaxis, ...]
        ),
    )
    valid_sample_surface = surface & np.asarray(valid_samples, dtype=bool)
    sampled_pressure_post = np.asarray(pressure_post)[valid_sample_surface]
    sampled_pressure_pre = np.asarray(pressure_pre)[valid_sample_surface]
    sampled_pressure_ratio = sampled_pressure_post / sampled_pressure_pre
    diagnostic_mach = np.sqrt(
        (
            sampled_pressure_ratio * (GAMMA + 1.0)
            + (GAMMA - 1.0)
        )
        / (2.0 * GAMMA)
    )
    sampled_post_distance = np.asarray(post_distance)[valid_sample_surface]
    sampled_pre_distance = np.asarray(pre_distance)[valid_sample_surface]
    (
        profile_pressure_post,
        profile_pressure_pre,
        _,
        _,
        profile_samples_valid,
        profile_post_distance,
        profile_pre_distance,
    ) = get_profile_aware_post_pre_shock_values(
        result.shock_direction,
        result.shock_zones,
        pressure,
        temperature,
        max_steps=8,
        center_offsets=(
            result.shock_direction
            * result.shock_surface_offsets[jnp.newaxis, ...]
        ),
    )
    profile_sample_surface = surface & np.asarray(
        profile_samples_valid, dtype=bool
    )
    sampled_profile_pressure_post = np.asarray(profile_pressure_post)[
        profile_sample_surface
    ]
    sampled_profile_pressure_pre = np.asarray(profile_pressure_pre)[
        profile_sample_surface
    ]
    sampled_profile_pressure_ratio = (
        sampled_profile_pressure_post / sampled_profile_pressure_pre
    )
    sampled_profile_post_distance = np.asarray(profile_post_distance)[
        profile_sample_surface
    ]
    sampled_profile_pre_distance = np.asarray(profile_pre_distance)[
        profile_sample_surface
    ]
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
    profile_flatness_mask = (
        profile_sample_surface
        & np.asarray(profile_flatness_valid, dtype=bool)
    )
    profile_flat_mask = (
        profile_flatness_mask
        & (np.asarray(profile_post_pressure_change) <= 0.01)
        & (np.asarray(profile_pre_pressure_change) <= 0.01)
        & (np.asarray(profile_post_density_change) <= 0.01)
        & (np.asarray(profile_pre_density_change) <= 0.01)
    )

    def profile_flatness_values(field):
        return np.asarray(field)[profile_flatness_mask]
    (
        _,
        _,
        _,
        _,
        plateau_samples_valid,
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
    plateau_sample_surface = surface & np.asarray(
        plateau_samples_valid, dtype=bool
    )

    def plateau_values(field):
        return np.asarray(field)[plateau_sample_surface]

    (
        _,
        _,
        _,
        _,
        ramp_samples_valid,
        ramp_post_distance,
        ramp_pre_distance,
        ramp_post_pressure_change,
        ramp_pre_pressure_change,
        ramp_post_density_change,
        ramp_pre_density_change,
    ) = get_ramp_exit_aware_post_pre_shock_values(
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
    ramp_sample_surface = surface & np.asarray(
        ramp_samples_valid, dtype=bool
    )

    def ramp_values(field):
        return np.asarray(field)[ramp_sample_surface]

    (
        _,
        _,
        _,
        _,
        extrapolated_samples_valid,
        extrapolated_post_distance,
        extrapolated_pre_distance,
        extrapolated_post_pressure_change,
        extrapolated_pre_pressure_change,
        extrapolated_post_density_change,
        extrapolated_pre_density_change,
    ) = get_extrapolated_post_pre_shock_values(
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
    extrapolated_sample_surface = surface & np.asarray(
        extrapolated_samples_valid, dtype=bool
    )

    def extrapolated_values(field):
        return np.asarray(field)[extrapolated_sample_surface]

    return {
        "resolution": int(resolution),
        "grid_spacing": float(config.grid_spacing),
        "true_mach": float(mach),
        "ramp_cells": float(ramp_cells),
        "normal_x": float(unit_normal[0]),
        "normal_y": float(unit_normal[1]),
        "normal_z": float(unit_normal[2]),
        "surface_cell_count": int(surface.sum()),
        "valid_mach_fraction": float(valid_mach.size / mach_values.size),
        "mach_median": float(median),
        "mach_p16": float(p16),
        "mach_p84": float(p84),
        "mach_relative_bias": float(median / mach - 1.0),
        "mach_relative_scatter": float((p84 - p16) / (2.0 * mach)),
        "profile_valid_mach_fraction": float(
            valid_profile_mach.size / profile_mach_values.size
        ),
        "profile_mach_median": float(profile_median),
        "profile_mach_p16": float(profile_p16),
        "profile_mach_p84": float(profile_p84),
        "profile_mach_relative_bias": float(profile_median / mach - 1.0),
        "profile_mach_relative_scatter": float(
            (profile_p84 - profile_p16) / (2.0 * mach)
        ),
        "plateau_valid_mach_fraction": float(
            valid_plateau_mach.size / plateau_mach_values.size
        ),
        "plateau_mach_median": float(plateau_median),
        "plateau_mach_p16": float(plateau_p16),
        "plateau_mach_p84": float(plateau_p84),
        "plateau_mach_relative_bias": float(plateau_median / mach - 1.0),
        "plateau_mach_relative_scatter": float(
            (plateau_p84 - plateau_p16) / (2.0 * mach)
        ),
        "plateau_post_sample_distance_cells_median": float(
            np.median(plateau_values(plateau_post_distance))
        ),
        "plateau_pre_sample_distance_cells_median": float(
            np.median(plateau_values(plateau_pre_distance))
        ),
        "plateau_post_pressure_change_median": float(
            np.median(plateau_values(plateau_post_pressure_change))
        ),
        "plateau_pre_pressure_change_median": float(
            np.median(plateau_values(plateau_pre_pressure_change))
        ),
        "plateau_post_density_change_median": float(
            np.median(plateau_values(plateau_post_density_change))
        ),
        "plateau_pre_density_change_median": float(
            np.median(plateau_values(plateau_pre_density_change))
        ),
        "ramp_exit_valid_mach_fraction": float(
            valid_ramp_exit_mach.size / ramp_exit_mach_values.size
        ),
        "ramp_exit_mach_median": float(ramp_median),
        "ramp_exit_mach_p16": float(ramp_p16),
        "ramp_exit_mach_p84": float(ramp_p84),
        "ramp_exit_mach_relative_bias": float(ramp_median / mach - 1.0),
        "ramp_exit_mach_relative_scatter": float(
            (ramp_p84 - ramp_p16) / (2.0 * mach)
        ),
        "ramp_exit_post_sample_distance_cells_median": float(
            np.median(ramp_values(ramp_post_distance))
        ),
        "ramp_exit_pre_sample_distance_cells_median": float(
            np.median(ramp_values(ramp_pre_distance))
        ),
        "ramp_exit_post_pressure_change_median": float(
            np.median(ramp_values(ramp_post_pressure_change))
        ),
        "ramp_exit_pre_pressure_change_median": float(
            np.median(ramp_values(ramp_pre_pressure_change))
        ),
        "ramp_exit_post_density_change_median": float(
            np.median(ramp_values(ramp_post_density_change))
        ),
        "ramp_exit_pre_density_change_median": float(
            np.median(ramp_values(ramp_pre_density_change))
        ),
        "extrapolated_valid_mach_fraction": float(
            valid_extrapolated_mach.size / extrapolated_mach_values.size
        ),
        "extrapolated_mach_median": float(extrapolated_median),
        "extrapolated_mach_p16": float(extrapolated_p16),
        "extrapolated_mach_p84": float(extrapolated_p84),
        "extrapolated_mach_relative_bias": float(
            extrapolated_median / mach - 1.0
        ),
        "extrapolated_mach_relative_scatter": float(
            (extrapolated_p84 - extrapolated_p16) / (2.0 * mach)
        ),
        "extrapolated_post_sample_distance_cells_median": float(
            np.median(extrapolated_values(extrapolated_post_distance))
        ),
        "extrapolated_pre_sample_distance_cells_median": float(
            np.median(extrapolated_values(extrapolated_pre_distance))
        ),
        "extrapolated_post_pressure_change_median": float(
            np.median(
                extrapolated_values(extrapolated_post_pressure_change)
            )
        ),
        "extrapolated_pre_pressure_change_median": float(
            np.median(extrapolated_values(extrapolated_pre_pressure_change))
        ),
        "extrapolated_post_density_change_median": float(
            np.median(extrapolated_values(extrapolated_post_density_change))
        ),
        "extrapolated_pre_density_change_median": float(
            np.median(extrapolated_values(extrapolated_pre_density_change))
        ),
        "profile_sampled_pressure_ratio_median": float(
            np.median(sampled_profile_pressure_ratio)
        ),
        "profile_sampled_post_pressure_fraction": float(
            np.median(sampled_profile_pressure_post)
            / states["downstream_pressure"]
        ),
        "profile_sampled_pre_pressure_fraction": float(
            np.median(sampled_profile_pressure_pre)
            / states["upstream_pressure"]
        ),
        "profile_post_sample_distance_cells_median": float(
            np.median(sampled_profile_post_distance)
        ),
        "profile_pre_sample_distance_cells_median": float(
            np.median(sampled_profile_pre_distance)
        ),
        "profile_flat_sampling_fraction": float(
            np.count_nonzero(profile_flat_mask) / surface.sum()
        ),
        "profile_post_pressure_change_median": float(
            np.median(profile_flatness_values(profile_post_pressure_change))
        ),
        "profile_pre_pressure_change_median": float(
            np.median(profile_flatness_values(profile_pre_pressure_change))
        ),
        "profile_post_density_change_median": float(
            np.median(profile_flatness_values(profile_post_density_change))
        ),
        "profile_pre_density_change_median": float(
            np.median(profile_flatness_values(profile_pre_density_change))
        ),
        "diagnostic_outside_offset": float(diagnostic_outside_offset),
        "diagnostic_mach_median": float(np.median(diagnostic_mach)),
        "diagnostic_mach_relative_bias": float(
            np.median(diagnostic_mach) / mach - 1.0
        ),
        "expected_pressure_ratio": float(states["pressure_ratio"]),
        "sampled_pressure_ratio_median": float(
            np.median(sampled_pressure_ratio)
        ),
        "sampled_post_pressure_fraction": float(
            np.median(sampled_pressure_post)
            / states["downstream_pressure"]
        ),
        "sampled_pre_pressure_fraction": float(
            np.median(sampled_pressure_pre) / states["upstream_pressure"]
        ),
        "post_sample_distance_cells_median": float(
            np.median(sampled_post_distance)
        ),
        "pre_sample_distance_cells_median": float(
            np.median(sampled_pre_distance)
        ),
        "surface_signed_distance_median": float(
            np.median(signed_surface_distance)
        ),
        "surface_position_error_cells": float(
            np.median(signed_surface_distance) / config.grid_spacing
        ),
        "normal_alignment_median": float(np.median(normal_alignment)),
        "normal_angular_error_degrees": float(
            np.degrees(
                np.arccos(np.clip(np.median(normal_alignment), -1.0, 1.0))
            )
        ),
    }


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    """Write one comparison table and a compact diagnostic figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "planar_shock_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    resolution = np.asarray([row["resolution"] for row in rows])
    median = np.asarray([row["mach_median"] for row in rows])
    p16 = np.asarray([row["mach_p16"] for row in rows])
    p84 = np.asarray([row["mach_p84"] for row in rows])
    true_mach = float(rows[0]["true_mach"])
    profile_median = np.asarray(
        [row["profile_mach_median"] for row in rows]
    )
    profile_p16 = np.asarray([row["profile_mach_p16"] for row in rows])
    profile_p84 = np.asarray([row["profile_mach_p84"] for row in rows])
    plateau_median = np.asarray(
        [row["plateau_mach_median"] for row in rows]
    )
    plateau_p16 = np.asarray([row["plateau_mach_p16"] for row in rows])
    plateau_p84 = np.asarray([row["plateau_mach_p84"] for row in rows])
    ramp_median = np.asarray(
        [row["ramp_exit_mach_median"] for row in rows]
    )
    ramp_p16 = np.asarray([row["ramp_exit_mach_p16"] for row in rows])
    ramp_p84 = np.asarray([row["ramp_exit_mach_p84"] for row in rows])
    extrapolated_median = np.asarray(
        [row["extrapolated_mach_median"] for row in rows]
    )
    extrapolated_p16 = np.asarray(
        [row["extrapolated_mach_p16"] for row in rows]
    )
    extrapolated_p84 = np.asarray(
        [row["extrapolated_mach_p84"] for row in rows]
    )
    unit_normal = np.asarray(
        [rows[0]["normal_x"], rows[0]["normal_y"], rows[0]["normal_z"]]
    )
    position_error = np.asarray(
        [row["surface_position_error_cells"] for row in rows]
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].errorbar(
        resolution,
        median,
        yerr=(median - p16, p84 - median),
        marker="o",
        capsize=4,
        label="current finder",
    )
    axes[0].axhline(
        true_mach,
        color="black",
        linestyle="--",
        label=f"true Mach = {true_mach:g}",
    )
    axes[0].errorbar(
        resolution,
        profile_median,
        yerr=(
            profile_median - profile_p16,
            profile_p84 - profile_median,
        ),
        marker="s",
        capsize=4,
        label="profile-aware estimator",
    )
    axes[0].errorbar(
        resolution,
        plateau_median,
        yerr=(
            plateau_median - plateau_p16,
            plateau_p84 - plateau_median,
        ),
        marker="^",
        capsize=4,
        label="plateau-aware estimator",
    )
    axes[0].errorbar(
        resolution,
        ramp_median,
        yerr=(ramp_median - ramp_p16, ramp_p84 - ramp_median),
        marker="D",
        capsize=4,
        label="ramp-exit-aware estimator",
    )
    axes[0].errorbar(
        resolution,
        extrapolated_median,
        yerr=(
            extrapolated_median - extrapolated_p16,
            extrapolated_p84 - extrapolated_median,
        ),
        marker="X",
        capsize=4,
        label="shock-front-extrapolated estimator",
    )
    axes[0].set(
        xlabel="cells per axis",
        ylabel="Mach number",
        title="Mach recovery",
    )
    axes[0].legend()
    axes[1].plot(resolution, position_error, "o-")
    axes[1].axhline(0.0, color="black", linestyle="--")
    axes[1].set(
        xlabel="cells per axis",
        ylabel="surface-position error [cells]",
        title="Surface localization",
    )
    for axis in axes:
        axis.grid(alpha=0.25)

    figure.suptitle(
        "Planar hydrodynamic shock benchmark "
        f"(Mach {true_mach:g}, normal={np.array2string(unit_normal, precision=3)})"
    )
    figure_path = output_dir / "planar_shock_benchmark.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    print(f"Saved benchmark table: {csv_path.resolve()}")
    print(f"Saved benchmark figure: {figure_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mach", type=float, default=5.0)
    parser.add_argument("--ramp-cells", type=float, default=3.0)
    parser.add_argument(
        "--diagnostic-outside-offset",
        type=float,
        default=0.25,
        help=(
            "Diagnostic distance beyond the first shock-zone exit, in cells; "
            "this does not change the production finder result."
        ),
    )
    parser.add_argument(
        "--normal",
        nargs=3,
        type=float,
        default=[1.0, 0.0, 0.0],
        metavar=("NX", "NY", "NZ"),
        help="Planar shock normal; it is normalized internally.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[32, 64, 128],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/planar_shock_benchmark"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for resolution in args.resolutions:
        row = benchmark_resolution(
            resolution=resolution,
            mach=args.mach,
            ramp_cells=args.ramp_cells,
            normal=args.normal,
            diagnostic_outside_offset=args.diagnostic_outside_offset,
        )
        rows.append(row)
        print(
            f"N={resolution}: Mach={row['mach_median']:.6f} "
            f"(bias={100.0 * row['mach_relative_bias']:+.3f}%, "
            f"p16-p84={row['mach_p16']:.6f}-{row['mach_p84']:.6f}); "
            f"profile={row['profile_mach_median']:.6f} "
            f"(bias={100.0 * row['profile_mach_relative_bias']:+.3f}%); "
            f"plateau={row['plateau_mach_median']:.6f} "
            f"(bias={100.0 * row['plateau_mach_relative_bias']:+.3f}%); "
            f"ramp-exit={row['ramp_exit_mach_median']:.6f} "
            f"(bias={100.0 * row['ramp_exit_mach_relative_bias']:+.3f}%); "
            f"extrapolated={row['extrapolated_mach_median']:.6f} "
            f"(bias={100.0 * row['extrapolated_mach_relative_bias']:+.3f}%)"
        )
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
