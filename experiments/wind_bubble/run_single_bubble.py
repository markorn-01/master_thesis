"""Minimal 3D single-star wind-bubble experiment.

This is a first-week smoke test, not yet a physically calibrated production
run.  It uses dimensionless code units to check that

1. the existing 3D stellar-wind source runs,
2. multiple primitive-state snapshots are returned, and
3. an approximately spherical expanding bubble appears.

Run from the repository root:

    python3 experiments/wind_bubble/run_single_bubble.py

For a cheaper test:

    python3 experiments/wind_bubble/run_single_bubble.py --num-cells 16

The script writes an early/middle/late central-slice figure to
``outputs/single_bubble/`` by default.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astronomix import (
    CARTESIAN,
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix._modules._stellar_wind.stellar_wind_options import (
    EI,
    WindConfig,
    WindParams,
)
from astronomix._modules._stellar_wind.weaver import Weaver
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import (
    find_shocks_pfrommer,
)
from astronomix.option_classes.simulation_config import SnapshotSettings


# These are deliberately simple dimensionless code-unit values.  They must be
# replaced by a documented physical/code-unit conversion before Weaver
# validation.
BOX_SIZE = 1.0
AMBIENT_DENSITY = 1.0
AMBIENT_PRESSURE = 1.0e-2
WIND_MASS_LOSS_RATE = 1.0e-2
WIND_FINAL_VELOCITY = 10.0
GAMMA = 5.0 / 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-cells",
        type=int,
        default=32,
        help="Cells along each axis (default: 32, giving a 32^3 grid).",
    )
    parser.add_argument(
        "--num-snapshots",
        type=int,
        default=10,
        help="Number of in-memory snapshots (default: 10).",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=0.05,
        help="End time in dimensionless code units (default: 0.05).",
    )
    parser.add_argument(
        "--num-injection-cells",
        type=int,
        default=2,
        help="Injection radius in grid cells (default: 2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/single_bubble/central_slices.png"),
        help="Output path for the diagnostic figure.",
    )
    parser.add_argument(
        "--shock-output",
        type=Path,
        default=Path("outputs/single_bubble/shock_diagnostics.png"),
        help="Output path for shock overlays and radial histograms.",
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        default=Path("outputs/single_bubble/forward_shock_history.png"),
        help="Output path for the candidate forward-shock history plot.",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=Path("outputs/single_bubble/forward_shock_history.csv"),
        help="Output path for the candidate forward-shock measurements.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_cells < 8:
        raise ValueError("--num-cells must be at least 8.")
    if args.num_snapshots < 3:
        raise ValueError("--num-snapshots must be at least 3.")
    if args.t_end <= 0:
        raise ValueError("--t-end must be positive.")
    if not 1 <= args.num_injection_cells < args.num_cells // 4:
        raise ValueError(
            "--num-injection-cells must be positive and smaller than "
            "one quarter of --num-cells."
        )


def build_problem(args: argparse.Namespace):
    """Construct the uniform ambient medium and enable a centred 3D EI wind."""
    config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=args.num_cells,
        mhd=False,
        return_snapshots=True,
        num_snapshots=args.num_snapshots,
        snapshot_settings=SnapshotSettings(
            return_states=True,
            return_final_state=True,
        ),
        wind_config=WindConfig(
            stellar_wind=True,
            num_injection_cells=args.num_injection_cells,
            wind_injection_scheme=EI,
            trace_wind_density=False,
        ),
    )
    params = SimulationParams(
        gamma=GAMMA,
        t_end=args.t_end,
        wind_params=WindParams(
            wind_mass_loss_rate=WIND_MASS_LOSS_RATE,
            wind_final_velocity=WIND_FINAL_VELOCITY,
        ),
    )

    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    spatial_shape = (args.num_cells,) * 3

    density = jnp.full(spatial_shape, AMBIENT_DENSITY)
    pressure = jnp.full(spatial_shape, AMBIENT_PRESSURE)
    zero_velocity = jnp.zeros(spatial_shape)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=zero_velocity,
        velocity_y=zero_velocity,
        velocity_z=zero_velocity,
        gas_pressure=pressure,
    )
    config = finalize_config(config, initial_state.shape)
    return initial_state, config, params, registered_variables, helper_data


def plot_central_slices(
    states: np.ndarray,
    times: np.ndarray,
    density_index: int,
    pressure_index: int,
    output_path: Path,
) -> None:
    """Plot density and pressure through the central z-plane at three times."""
    selected = np.array([0, len(times) // 2, len(times) - 1])
    midplane = states.shape[-1] // 2

    density_slices = [
        states[i, density_index, :, :, midplane] for i in selected
    ]
    pressure_slices = [
        states[i, pressure_index, :, :, midplane] for i in selected
    ]

    # A common colour scale makes expansion between snapshots visually honest.
    density_log = [np.log10(np.maximum(field, 1.0e-30)) for field in density_slices]
    pressure_log = [
        np.log10(np.maximum(field, 1.0e-30)) for field in pressure_slices
    ]
    density_limits = (
        min(field.min() for field in density_log),
        max(field.max() for field in density_log),
    )
    pressure_limits = (
        min(field.min() for field in pressure_log),
        max(field.max() for field in pressure_log),
    )

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    density_image = None
    pressure_image = None

    for column, snapshot_index in enumerate(selected):
        density_image = axes[0, column].imshow(
            density_log[column].T,
            origin="lower",
            extent=(0.0, BOX_SIZE, 0.0, BOX_SIZE),
            vmin=density_limits[0],
            vmax=density_limits[1],
            cmap="viridis",
        )
        pressure_image = axes[1, column].imshow(
            pressure_log[column].T,
            origin="lower",
            extent=(0.0, BOX_SIZE, 0.0, BOX_SIZE),
            vmin=pressure_limits[0],
            vmax=pressure_limits[1],
            cmap="magma",
        )
        axes[0, column].set_title(f"t = {times[snapshot_index]:.4f}")
        axes[1, column].set_xlabel("x [code length]")
        for row in range(2):
            axes[row, column].set_aspect("equal")
        if column == 0:
            axes[0, column].set_ylabel("y [code length]")
            axes[1, column].set_ylabel("y [code length]")

    figure.colorbar(
        density_image,
        ax=axes[0, :],
        label=r"$\log_{10}(\rho)$ [code units]",
        shrink=0.9,
    )
    figure.colorbar(
        pressure_image,
        ax=axes[1, :],
        label=r"$\log_{10}(p)$ [code units]",
        shrink=0.9,
    )
    figure.suptitle("Single-star wind bubble: central z-plane")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_shock_diagnostics(
    states: np.ndarray,
    times: np.ndarray,
    registered_variables,
    injection_radius: float,
    output_path: Path,
) -> None:
    """Run the 2D shock finder on central slices and plot radial detections.

    The current shock-surface raycaster is not implemented in 3D.  This
    first-week diagnostic therefore analyses the central z-plane of each 3D
    snapshot.  It must not be interpreted as a reconstructed 3D surface.
    """
    selected = np.array([0, len(times) // 2, len(times) - 1])
    midplane = states.shape[-1] // 2

    slice_config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=2,
        box_size=BOX_SIZE,
        num_cells=states.shape[-1],
        mhd=False,
    )
    slice_variables = get_registered_variables(slice_config)
    slice_helper_data = get_helper_data(slice_config)
    empty_2d_state = construct_primitive_state(
        config=slice_config,
        registered_variables=slice_variables,
        density=jnp.ones(states.shape[-3:-1]),
        velocity_x=jnp.zeros(states.shape[-3:-1]),
        velocity_y=jnp.zeros(states.shape[-3:-1]),
        gas_pressure=jnp.ones(states.shape[-3:-1]),
    )
    slice_config = finalize_config(slice_config, empty_2d_state.shape)
    centers = np.asarray(slice_helper_data.geometric_centers)
    radii = np.asarray(slice_helper_data.r)

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    print("\n=== Central-slice shock-finder diagnostics ===")
    print("Note: the current shock-surface raycaster is not implemented in 3D.")

    for column, snapshot_index in enumerate(selected):
        density_2d = states[
            snapshot_index, registered_variables.density_index, :, :, midplane
        ]
        velocity_x_2d = states[
            snapshot_index,
            registered_variables.velocity_index.x,
            :,
            :,
            midplane,
        ]
        velocity_y_2d = states[
            snapshot_index,
            registered_variables.velocity_index.y,
            :,
            :,
            midplane,
        ]
        pressure_2d = states[
            snapshot_index, registered_variables.pressure_index, :, :, midplane
        ]
        slice_state = construct_primitive_state(
            config=slice_config,
            registered_variables=slice_variables,
            density=jnp.asarray(density_2d),
            velocity_x=jnp.asarray(velocity_x_2d),
            velocity_y=jnp.asarray(velocity_y_2d),
            gas_pressure=jnp.asarray(pressure_2d),
        )
        result = find_shocks_pfrommer(
            slice_state,
            slice_config,
            slice_variables,
            slice_helper_data,
        )
        surface_mask = np.asarray(result.shock_surface_cells, dtype=bool)
        surface_radii = radii[surface_mask]
        surface_mach = np.asarray(result.mach_numbers)[surface_mask]

        pressure = states[
            snapshot_index,
            registered_variables.pressure_index,
            :,
            :,
            midplane,
        ]
        pressure_log = np.log10(np.maximum(pressure, 1.0e-30))
        axes[0, column].imshow(
            pressure_log.T,
            origin="lower",
            extent=(0.0, BOX_SIZE, 0.0, BOX_SIZE),
            cmap="magma",
        )

        slab_x = centers[..., 0][surface_mask]
        slab_y = centers[..., 1][surface_mask]
        axes[0, column].scatter(
            slab_x,
            slab_y,
            s=13,
            facecolors="none",
            edgecolors="cyan",
            linewidths=0.8,
            label="shock surface",
        )
        injection_circle = plt.Circle(
            (BOX_SIZE / 2.0, BOX_SIZE / 2.0),
            injection_radius,
            fill=False,
            color="white",
            linestyle="--",
            linewidth=1.2,
            label="injection radius",
        )
        axes[0, column].add_patch(injection_circle)
        axes[0, column].set_title(f"t = {times[snapshot_index]:.4f}")
        axes[0, column].set_xlabel("x [code length]")
        axes[0, column].set_aspect("equal")
        if column == 0:
            axes[0, column].set_ylabel("y [code length]")
            axes[0, column].legend(loc="upper right", fontsize=8)

        radial_limit = BOX_SIZE / 2.0
        bins = np.linspace(
            0.0, radial_limit, max(20, slice_config.num_cells.x // 2)
        )
        axes[1, column].hist(
            surface_radii,
            bins=bins,
            color="tab:blue",
            alpha=0.8,
        )
        axes[1, column].axvspan(
            0.0,
            injection_radius,
            color="tab:red",
            alpha=0.15,
            label="injection region",
        )
        axes[1, column].axvline(
            injection_radius,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
        )
        axes[1, column].set_xlim(0.0, radial_limit)
        axes[1, column].set_xlabel("radius from star [code length]")
        if column == 0:
            axes[1, column].set_ylabel("shock-surface cell count")
            axes[1, column].legend(loc="upper right", fontsize=8)

        outside_injection = surface_radii > injection_radius
        mach_detected = surface_mach[outside_injection]
        radius_detected = surface_radii[outside_injection]
        print(
            f"t={times[snapshot_index]:.6f}: "
            f"{surface_mask.sum()} surface cells, "
            f"{outside_injection.sum()} outside injection region"
        )
        if radius_detected.size:
            print(
                "  outside-injection radius range: "
                f"{radius_detected.min():.6f} .. {radius_detected.max():.6f}; "
                f"median Mach: {np.median(mach_detected):.3f}"
            )
        else:
            print("  no shock-surface cells detected outside injection region")

    figure.suptitle(
        "Central-slice shock detections (cyan) and radial distributions"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def measure_forward_shock_history(
    states: np.ndarray,
    times: np.ndarray,
    registered_variables,
    injection_radius: float,
    grid_spacing: float,
    plot_path: Path,
    csv_path: Path,
) -> None:
    """Measure the sole outside-injection radial band in every central slice.

    This is a provisional forward-shock classifier for the current EI test:
    all detected surface cells outside the injection radius are treated as one
    physical forward-shock candidate.  It must be replaced by explicit radial
    peak selection before analysing simulations containing multiple shocks.
    """
    num_cells = states.shape[-1]
    midplane = num_cells // 2
    slice_shape = (num_cells, num_cells)
    slice_config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=2,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        mhd=False,
    )
    slice_variables = get_registered_variables(slice_config)
    slice_helper_data = get_helper_data(slice_config)
    empty_2d_state = construct_primitive_state(
        config=slice_config,
        registered_variables=slice_variables,
        density=jnp.ones(slice_shape),
        velocity_x=jnp.zeros(slice_shape),
        velocity_y=jnp.zeros(slice_shape),
        gas_pressure=jnp.ones(slice_shape),
    )
    slice_config = finalize_config(slice_config, empty_2d_state.shape)
    radii = np.asarray(slice_helper_data.r)
    weaver = Weaver(
        v_inf=WIND_FINAL_VELOCITY,
        M_dot=WIND_MASS_LOSS_RATE,
        rho_0=AMBIENT_DENSITY,
        p_0=AMBIENT_PRESSURE,
        gamma=GAMMA,
    )

    rows = []
    print("\n=== Candidate forward-shock history ===")
    for snapshot_index, time in enumerate(times):
        slice_state = construct_primitive_state(
            config=slice_config,
            registered_variables=slice_variables,
            density=jnp.asarray(
                states[
                    snapshot_index,
                    registered_variables.density_index,
                    :,
                    :,
                    midplane,
                ]
            ),
            velocity_x=jnp.asarray(
                states[
                    snapshot_index,
                    registered_variables.velocity_index.x,
                    :,
                    :,
                    midplane,
                ]
            ),
            velocity_y=jnp.asarray(
                states[
                    snapshot_index,
                    registered_variables.velocity_index.y,
                    :,
                    :,
                    midplane,
                ]
            ),
            gas_pressure=jnp.asarray(
                states[
                    snapshot_index,
                    registered_variables.pressure_index,
                    :,
                    :,
                    midplane,
                ]
            ),
        )
        result = find_shocks_pfrommer(
            slice_state,
            slice_config,
            slice_variables,
            slice_helper_data,
        )
        surface_mask = np.asarray(result.shock_surface_cells, dtype=bool)
        candidate_mask = surface_mask & (radii > injection_radius)
        candidate_radii = radii[candidate_mask]
        candidate_mach = np.asarray(result.mach_numbers)[candidate_mask]

        if candidate_radii.size:
            radius_p16, radius_median, radius_p84 = np.percentile(
                candidate_radii, [16.0, 50.0, 84.0]
            )
            mach_median = float(np.median(candidate_mach))
        else:
            radius_p16 = radius_median = radius_p84 = np.nan
            mach_median = np.nan

        weaver_radius = (
            float(weaver.get_outer_shock_radius(float(time)))
            if time > 0.0
            else 0.0
        )
        # The injection source occupies a finite sphere.  Do not interpret a
        # comparison as resolved until the analytic shock is at least two grid
        # cells beyond that numerical source region.
        resolved_for_weaver = weaver_radius >= injection_radius + 2.0 * grid_spacing
        relative_error = (
            (float(radius_median) - weaver_radius) / weaver_radius
            if resolved_for_weaver and np.isfinite(radius_median)
            else np.nan
        )

        row = {
            "time": float(time),
            "radius_median": float(radius_median),
            "radius_p16": float(radius_p16),
            "radius_p84": float(radius_p84),
            "radial_spread_p84_minus_p16": float(radius_p84 - radius_p16),
            "surface_cell_count": int(candidate_radii.size),
            "mach_median": mach_median,
            "weaver_outer_radius": weaver_radius,
            "resolved_for_weaver": resolved_for_weaver,
            "relative_error_vs_weaver": relative_error,
        }
        rows.append(row)
        print(
            f"t={row['time']:.6f}: radius={row['radius_median']:.6f}, "
            f"p16..p84={row['radius_p16']:.6f}..{row['radius_p84']:.6f}, "
            f"Weaver={weaver_radius:.6f}, "
            f"relative error={relative_error:.3f}, "
            f"resolved={resolved_for_weaver}"
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    valid = np.array([np.isfinite(row["radius_median"]) for row in rows])
    history_times = np.array([row["time"] for row in rows])
    radius_median = np.array([row["radius_median"] for row in rows])
    radius_p16 = np.array([row["radius_p16"] for row in rows])
    radius_p84 = np.array([row["radius_p84"] for row in rows])
    weaver_radius = np.array([row["weaver_outer_radius"] for row in rows])
    resolved = np.array([row["resolved_for_weaver"] for row in rows])

    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(
        history_times[valid],
        radius_median[valid],
        marker="o",
        color="tab:blue",
        label="median detected radius",
    )
    axis.fill_between(
        history_times[valid],
        radius_p16[valid],
        radius_p84[valid],
        color="tab:blue",
        alpha=0.2,
        label="16th–84th percentile",
    )
    axis.plot(
        history_times,
        weaver_radius,
        color="black",
        linestyle="--",
        label=r"Weaver outer shock ($R\propto t^{3/5}$)",
    )
    if np.any(resolved):
        first_resolved_time = history_times[np.flatnonzero(resolved)[0]]
        axis.axvspan(
            history_times.min(),
            first_resolved_time,
            color="grey",
            alpha=0.12,
            label="unresolved near injection region",
        )
    axis.axhline(
        injection_radius,
        color="tab:red",
        linestyle="--",
        label="injection radius",
    )
    axis.set(
        xlabel="time [code units]",
        ylabel="radius from star [code length]",
        title="Candidate forward-shock radius and Weaver prediction",
    )
    axis.grid(alpha=0.25)
    axis.legend()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_args(args)
    (
        initial_state,
        config,
        params,
        registered_variables,
        _helper_data,
    ) = build_problem(args)

    injection_radius = args.num_injection_cells * float(config.grid_spacing)
    wind_luminosity = (
        0.5 * WIND_MASS_LOSS_RATE * WIND_FINAL_VELOCITY**2
    )

    print("=== Single-star wind-bubble smoke test ===")
    print(f"Grid                 : {args.num_cells}^3")
    print(f"Initial state shape   : {initial_state.shape}")
    print(f"Grid spacing          : {float(config.grid_spacing):.6f}")
    print(f"Injection radius      : {injection_radius:.6f}")
    print(f"Ambient density       : {AMBIENT_DENSITY}")
    print(f"Ambient pressure      : {AMBIENT_PRESSURE}")
    print(f"Wind mass-loss rate   : {WIND_MASS_LOSS_RATE}")
    print(f"Wind terminal velocity: {WIND_FINAL_VELOCITY}")
    print(f"Wind luminosity       : {wind_luminosity}")
    print(
        "Units                 : dimensionless code units "
        "(preliminary same-unit Weaver scaling only)"
    )

    snapshots = time_integration(
        initial_state,
        config,
        params,
        registered_variables,
    )

    states = np.asarray(snapshots.states)
    times = np.asarray(snapshots.time_points)
    print(f"Returned states shape : {states.shape}")
    print(f"Snapshot times        : {times}")
    print(f"Number of iterations  : {int(snapshots.num_iterations)}")

    if not np.all(np.isfinite(states)):
        raise RuntimeError("The simulation produced NaN or infinite snapshot values.")
    if states.shape[0] != args.num_snapshots:
        raise RuntimeError(
            f"Expected {args.num_snapshots} snapshots, received {states.shape[0]}."
        )

    plot_central_slices(
        states=states,
        times=times,
        density_index=registered_variables.density_index,
        pressure_index=registered_variables.pressure_index,
        output_path=args.output,
    )
    print(f"Saved diagnostic figure: {args.output.resolve()}")

    plot_shock_diagnostics(
        states=states,
        times=times,
        registered_variables=registered_variables,
        injection_radius=injection_radius,
        output_path=args.shock_output,
    )
    print(f"Saved shock diagnostics : {args.shock_output.resolve()}")

    measure_forward_shock_history(
        states=states,
        times=times,
        registered_variables=registered_variables,
        injection_radius=injection_radius,
        grid_spacing=float(config.grid_spacing),
        plot_path=args.history_output,
        csv_path=args.history_csv,
    )
    print(f"Saved shock history     : {args.history_output.resolve()}")
    print(f"Saved history table     : {args.history_csv.resolve()}")


if __name__ == "__main__":
    main()
