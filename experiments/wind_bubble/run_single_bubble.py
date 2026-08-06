"""Run and analyse the canonical 3D single-star wind-bubble experiment.

The pipeline evolves a centred stellar-wind bubble, runs the full 3D Pfrommer
shock finder, separates radial reverse/forward shock candidates, compares the
forward candidate with the Weaver solution, and verifies the final candidates
using spherically averaged physical profiles.

The current parameters use dimensionless code units and are not yet a
physically calibrated production setup.

Run from the repository root:

    python3 experiments/wind_bubble/run_single_bubble.py

Results are written to ``outputs/single_bubble_long/`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from astronomix._physics_modules._shock_finder._gradients import (
    _calculate_velocity_divergence,
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
DEFAULT_NUM_CELLS = 64
DEFAULT_NUM_SNAPSHOTS = 20
DEFAULT_T_END = 0.20
DEFAULT_NUM_INJECTION_CELLS = 4
DEFAULT_OUTPUT_DIR = Path("outputs/single_bubble_long")

# Reverse-shock verification thresholds.  Keeping these values together makes
# the physical classification explicit and easier to revisit in convergence
# studies instead of hiding decisions inside plotting code.
MIN_JUMP_RATIO = 1.05
MAX_VELOCITY_RATIO = 0.95
MIN_SHOCK_MACH = 1.3
MAX_INWARD_NORMAL_ALIGNMENT = -0.5
MIN_PERSISTENT_SNAPSHOTS = 3
MIN_INJECTION_SEPARATION_CELLS = 2.0
SURFACE_NEIGHBORHOOD_CELLS = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-cells",
        type=int,
        default=DEFAULT_NUM_CELLS,
        help="Cells along each axis (default: 64, giving a 64^3 grid).",
    )
    parser.add_argument(
        "--num-snapshots",
        type=int,
        default=DEFAULT_NUM_SNAPSHOTS,
        help="Number of in-memory snapshots (default: 20).",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=DEFAULT_T_END,
        help="End time in dimensionless code units (default: 0.20).",
    )
    parser.add_argument(
        "--num-injection-cells",
        type=int,
        default=DEFAULT_NUM_INJECTION_CELLS,
        help="Injection radius in grid cells (default: 4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for every generated figure and table.",
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


def analyze_shocks_3d(
    states: np.ndarray,
    config: SimulationConfig,
    registered_variables,
    helper_data,
):
    """Run the full 3D Pfrommer finder once for every saved snapshot."""
    shock_results = []
    print("\n=== Full 3D shock finding ===")
    for snapshot_index, state in enumerate(states):
        result = find_shocks_pfrommer(
            jnp.asarray(state),
            config,
            registered_variables,
            helper_data,
        )
        surface_mask = np.asarray(result.shock_surface_cells, dtype=bool)
        surface_offsets = np.asarray(result.shock_surface_offsets)
        shock_direction = np.moveaxis(
            np.asarray(result.shock_direction), 0, -1
        )
        centers = np.asarray(helper_data.geometric_centers)
        refined_centers = centers + (
            float(config.grid_spacing)
            * surface_offsets[..., np.newaxis]
            * shock_direction
        )
        star_position = np.full(3, BOX_SIZE / 2.0)
        refined_radii = np.linalg.norm(
            refined_centers - star_position,
            axis=-1,
        )
        shock_results.append(
            {
                "surface_mask": surface_mask,
                "refined_centers": refined_centers,
                "refined_radii": refined_radii,
                "shock_direction": shock_direction,
                "mach_numbers": np.asarray(result.mach_numbers),
                "thermal_energy_flux": np.asarray(result.thermal_energy_flux),
            }
        )
        print(
            f"snapshot {snapshot_index:02d}: "
            f"{surface_mask.sum()} shock-surface cells"
        )
    return shock_results


def plot_shock_diagnostics(
    states: np.ndarray,
    times: np.ndarray,
    shock_results,
    config: SimulationConfig,
    registered_variables,
    helper_data,
    injection_radius: float,
    output_path: Path,
) -> None:
    """Plot central-slab views and radial distributions of 3D detections."""
    selected = np.array([0, len(times) // 2, len(times) - 1])
    midplane = states.shape[-1] // 2
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    print("\n=== 3D shock-finder diagnostics ===")

    for column, snapshot_index in enumerate(selected):
        surface_mask = shock_results[snapshot_index]["surface_mask"]
        surface_radii = shock_results[snapshot_index]["refined_radii"][
            surface_mask
        ]
        surface_mach = shock_results[snapshot_index]["mach_numbers"][
            surface_mask
        ]

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

        # The finder and radial statistics use the complete 3D mask.  Only the
        # overlay is restricted to a three-cell-thick central slab so it can be
        # displayed on the 2D pressure image.
        slab_start = max(0, midplane - 1)
        slab_stop = min(surface_mask.shape[-1], midplane + 2)
        slab_mask = surface_mask[:, :, slab_start:slab_stop]
        slab_centers = shock_results[snapshot_index]["refined_centers"][
            :, :, slab_start:slab_stop, :
        ]
        slab_x = slab_centers[..., 0][slab_mask]
        slab_y = slab_centers[..., 1][slab_mask]
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
            0.0, radial_limit, max(20, config.num_cells.x // 2)
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
        "3D shock detections (central slab) and 3D radial distributions"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def split_radial_shock_candidates(
    surface_radii: np.ndarray,
    injection_radius: float,
    grid_spacing: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split detected radii into reverse- and forward-shock candidates.

    The outermost populated radial band is always the forward-shock candidate.
    An inner band is labelled as a reverse-shock candidate only when it is
    separated from the outer band by a gap at least two grid cells wide and
    both sides contain enough cells to reject isolated detections.

    Returns ``(reverse_radii, forward_radii, separation_radius)``.  When only
    one credible band exists, ``reverse_radii`` and ``separation_radius`` are
    empty/NaN and all radii belong to the forward candidate.
    """
    candidate_radii = np.asarray(surface_radii, dtype=float)
    candidate_radii = candidate_radii[
        np.isfinite(candidate_radii) & (candidate_radii > injection_radius)
    ]
    if candidate_radii.size == 0:
        return np.array([], dtype=float), candidate_radii, np.nan

    sorted_radii = np.sort(candidate_radii)
    radial_gaps = np.diff(sorted_radii)
    minimum_gap = 2.0 * grid_spacing
    minimum_band_cells = max(8, int(np.ceil(0.01 * sorted_radii.size)))

    credible_splits = []
    for gap_index in np.flatnonzero(radial_gaps >= minimum_gap):
        inner_count = gap_index + 1
        outer_count = sorted_radii.size - inner_count
        if (
            inner_count >= minimum_band_cells
            and outer_count >= minimum_band_cells
        ):
            credible_splits.append(int(gap_index))

    if not credible_splits:
        return np.array([], dtype=float), sorted_radii, np.nan

    # If more than one credible gap exists, the widest gap provides the most
    # conservative separation between the two dominant physical surfaces.
    split_index = max(credible_splits, key=lambda index: radial_gaps[index])
    reverse_radii = sorted_radii[: split_index + 1]
    forward_radii = sorted_radii[split_index + 1 :]
    separation_radius = 0.5 * (
        sorted_radii[split_index] + sorted_radii[split_index + 1]
    )
    return reverse_radii, forward_radii, float(separation_radius)


def _radial_band_statistics(
    band_radii: np.ndarray,
    all_surface_radii: np.ndarray,
    all_surface_mach: np.ndarray,
) -> dict[str, float | int]:
    """Return robust radius and Mach statistics for one radial band."""
    if band_radii.size == 0:
        return {
            "radius_median": np.nan,
            "radius_p16": np.nan,
            "radius_p84": np.nan,
            "surface_cell_count": 0,
            "mach_median": np.nan,
        }

    lower = float(np.min(band_radii))
    upper = float(np.max(band_radii))
    band_mask = (all_surface_radii >= lower) & (all_surface_radii <= upper)
    radius_p16, radius_median, radius_p84 = np.percentile(
        band_radii, [16.0, 50.0, 84.0]
    )
    return {
        "radius_median": float(radius_median),
        "radius_p16": float(radius_p16),
        "radius_p84": float(radius_p84),
        "surface_cell_count": int(band_radii.size),
        "mach_median": float(np.median(all_surface_mach[band_mask])),
    }


def measure_shock_histories(
    times: np.ndarray,
    shock_results,
    injection_radius: float,
    grid_spacing: float,
    plot_path: Path,
    csv_path: Path,
) -> list[dict]:
    """Separate and track reverse/forward radial candidates in 3D snapshots."""
    weaver = Weaver(
        v_inf=WIND_FINAL_VELOCITY,
        M_dot=WIND_MASS_LOSS_RATE,
        rho_0=AMBIENT_DENSITY,
        p_0=AMBIENT_PRESSURE,
        gamma=GAMMA,
    )

    rows = []
    print("\n=== Candidate reverse/forward shock histories ===")
    for snapshot_index, time in enumerate(times):
        surface_mask = shock_results[snapshot_index]["surface_mask"]
        surface_radii = shock_results[snapshot_index]["refined_radii"][
            surface_mask
        ]
        surface_mach = shock_results[snapshot_index]["mach_numbers"][surface_mask]
        outside_mask = surface_radii > injection_radius
        outside_radii = surface_radii[outside_mask]
        outside_mach = surface_mach[outside_mask]

        reverse_radii, forward_radii, separation_radius = (
            split_radial_shock_candidates(
                outside_radii,
                injection_radius=injection_radius,
                grid_spacing=grid_spacing,
            )
        )
        reverse = _radial_band_statistics(
            reverse_radii, outside_radii, outside_mach
        )
        forward = _radial_band_statistics(
            forward_radii, outside_radii, outside_mach
        )

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
            (float(forward["radius_median"]) - weaver_radius) / weaver_radius
            if resolved_for_weaver
            and np.isfinite(float(forward["radius_median"]))
            else np.nan
        )

        row = {
            "time": float(time),
            "two_bands_detected": bool(reverse_radii.size),
            "separation_radius": separation_radius,
            "reverse_radius_median": reverse["radius_median"],
            "reverse_radius_p16": reverse["radius_p16"],
            "reverse_radius_p84": reverse["radius_p84"],
            "reverse_surface_cell_count": reverse["surface_cell_count"],
            "reverse_mach_median": reverse["mach_median"],
            "forward_radius_median": forward["radius_median"],
            "forward_radius_p16": forward["radius_p16"],
            "forward_radius_p84": forward["radius_p84"],
            "forward_surface_cell_count": forward["surface_cell_count"],
            "forward_mach_median": forward["mach_median"],
            "weaver_outer_radius": weaver_radius,
            "resolved_for_weaver": resolved_for_weaver,
            "relative_error_vs_weaver": relative_error,
        }
        rows.append(row)
        print(
            f"t={row['time']:.6f}: "
            f"reverse={row['reverse_radius_median']:.6f} "
            f"(N={row['reverse_surface_cell_count']}), "
            f"forward={row['forward_radius_median']:.6f} "
            f"(N={row['forward_surface_cell_count']}), "
            f"Weaver={weaver_radius:.6f}, "
            f"relative error={relative_error:.3f}, "
            f"two bands={row['two_bands_detected']}"
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    history_times = np.array([row["time"] for row in rows])
    forward_median = np.array([row["forward_radius_median"] for row in rows])
    forward_p16 = np.array([row["forward_radius_p16"] for row in rows])
    forward_p84 = np.array([row["forward_radius_p84"] for row in rows])
    reverse_median = np.array([row["reverse_radius_median"] for row in rows])
    reverse_p16 = np.array([row["reverse_radius_p16"] for row in rows])
    reverse_p84 = np.array([row["reverse_radius_p84"] for row in rows])
    weaver_radius = np.array([row["weaver_outer_radius"] for row in rows])
    resolved = np.array([row["resolved_for_weaver"] for row in rows])
    valid_forward = np.isfinite(forward_median)
    valid_reverse = np.isfinite(reverse_median)

    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(
        history_times[valid_forward],
        forward_median[valid_forward],
        marker="o",
        color="tab:blue",
        label="forward-shock candidate",
    )
    axis.fill_between(
        history_times[valid_forward],
        forward_p16[valid_forward],
        forward_p84[valid_forward],
        color="tab:blue",
        alpha=0.2,
        label="forward 16th–84th percentile",
    )
    if np.any(valid_reverse):
        axis.plot(
            history_times[valid_reverse],
            reverse_median[valid_reverse],
            marker="o",
            color="tab:orange",
            label="reverse-shock candidate",
        )
        axis.fill_between(
            history_times[valid_reverse],
            reverse_p16[valid_reverse],
            reverse_p84[valid_reverse],
            color="tab:orange",
            alpha=0.2,
            label="reverse 16th–84th percentile",
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
        title="3D reverse/forward shock candidates and Weaver prediction",
    )
    axis.grid(alpha=0.25)
    axis.legend()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return rows


def _spherical_bin_median(
    values: np.ndarray,
    bin_indices: np.ndarray,
    num_bins: int,
    selection: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate a median value in each spherical radial shell."""
    if selection is None:
        selection = np.ones(values.shape, dtype=bool)
    profile = np.full(num_bins, np.nan)
    for bin_index in range(num_bins):
        shell = selection & (bin_indices == bin_index) & np.isfinite(values)
        if np.any(shell):
            profile[bin_index] = np.median(values[shell])
    return profile


def classify_reverse_shock_evidence(
    measurements: dict[str, float],
    persistent_detection: bool,
    resolved_from_injection: bool,
) -> dict:
    """Classify an inner surface from normal-shock signatures.

    The inner surface is considered consistent with a reverse shock only when
    the wind is supersonic on its inner/upstream side, subsonic on its
    outer/downstream side, the gas is compressed and heated, the radial flow
    decelerates, the local flow is compressive, and the temperature-derived
    shock normal points inward.  Persistence and numerical separation from the
    injection region are required to avoid classifying a source-boundary
    artifact as a physical shock.
    """
    required_criteria = {
        "persistent_detection": bool(persistent_detection),
        "resolved_from_injection": bool(resolved_from_injection),
        "upstream_flow_is_supersonic": measurements["upstream_flow_mach"] > 1.0,
        "downstream_flow_is_subsonic": measurements["downstream_flow_mach"] < 1.0,
        "density_increases": measurements["density_ratio"] > MIN_JUMP_RATIO,
        "pressure_increases": measurements["pressure_ratio"] > MIN_JUMP_RATIO,
        "temperature_increases": measurements["temperature_ratio"]
        > MIN_JUMP_RATIO,
        "radial_flow_decelerates": measurements["velocity_ratio"]
        < MAX_VELOCITY_RATIO,
        "flow_is_compressive": measurements["minimum_divergence"] < 0.0,
        "profile_jump_is_supersonic": measurements["profile_jump_mach"]
        >= MIN_SHOCK_MACH,
        "shock_normal_points_inward": measurements[
            "median_radial_normal_alignment"
        ]
        < MAX_INWARD_NORMAL_ALIGNMENT,
    }
    diagnostic_checks = {
        "adaptive_finder_mach_is_consistent": measurements[
            "peak_surface_mach"
        ]
        >= MIN_SHOCK_MACH,
    }
    verified = all(required_criteria.values())
    finder_mach_warning = verified and not all(diagnostic_checks.values())
    if not verified:
        classification = "not_yet_verified_as_reverse_shock"
    elif finder_mach_warning:
        classification = (
            "consistent_with_reverse_shock_but_finder_mach_needs_"
            "adaptive_sampling"
        )
    else:
        classification = "consistent_with_reverse_shock"

    return {
        "classification": classification,
        "verified": verified,
        "criteria": required_criteria,
        "diagnostic_checks": diagnostic_checks,
        "limitations": (
            [
                "Adaptive boundary sampling did not recover a supersonic "
                "reverse-shock Mach number. Inspect the local shock-zone "
                "topology before energy-dissipation work."
            ]
            if finder_mach_warning
            else []
        ),
        "measurements": measurements,
    }


def _longest_consecutive_true_run(values: list[bool]) -> int:
    """Return the longest consecutive run of true values."""
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def evaluate_reverse_shock(
    final_shock_result,
    history_rows: list[dict],
    bin_centers: np.ndarray,
    profiles: dict[str, np.ndarray],
    injection_radius: float,
    grid_spacing: float,
) -> dict:
    """Measure physical evidence for the final inner shock surface."""
    final_history_row = history_rows[-1]
    reverse_radius = float(final_history_row["reverse_radius_median"])
    if not np.isfinite(reverse_radius):
        return {
            "classification": "no_inner_surface_detected",
            "verified": False,
            "criteria": {},
            "diagnostic_checks": {},
            "limitations": ["No separated inner surface was detected."],
            "measurements": {},
        }

    inner_bins = np.flatnonzero(bin_centers < reverse_radius)
    outer_bins = np.flatnonzero(bin_centers > reverse_radius)
    if not inner_bins.size or not outer_bins.size:
        raise RuntimeError("The reverse-shock radius is outside the radial bins.")
    upstream_index = int(inner_bins[-1])
    downstream_index = int(outer_bins[0])

    surface_mask = final_shock_result["surface_mask"]
    refined_radii = final_shock_result["refined_radii"]
    neighborhood = (
        surface_mask
        & (
            np.abs(refined_radii - reverse_radius)
            <= SURFACE_NEIGHBORHOOD_CELLS * grid_spacing
        )
    )
    if not np.any(neighborhood):
        raise RuntimeError("No detected surface cells surround the inner radius.")

    refined_centers = final_shock_result["refined_centers"]
    radial_vectors = refined_centers - BOX_SIZE / 2.0
    radial_norm = np.linalg.norm(radial_vectors, axis=-1, keepdims=True)
    radial_unit = radial_vectors / np.maximum(radial_norm, 1.0e-30)
    radial_alignment = np.sum(
        final_shock_result["shock_direction"] * radial_unit,
        axis=-1,
    )
    local_indices = slice(upstream_index, downstream_index + 1)

    measurements = {
        "reverse_radius": reverse_radius,
        "upstream_shell_radius": float(bin_centers[upstream_index]),
        "downstream_shell_radius": float(bin_centers[downstream_index]),
        "density_ratio": float(
            profiles["density"][downstream_index]
            / profiles["density"][upstream_index]
        ),
        "pressure_ratio": float(
            profiles["pressure"][downstream_index]
            / profiles["pressure"][upstream_index]
        ),
        "temperature_ratio": float(
            profiles["temperature_proxy"][downstream_index]
            / profiles["temperature_proxy"][upstream_index]
        ),
        "velocity_ratio": float(
            profiles["radial_velocity"][downstream_index]
            / profiles["radial_velocity"][upstream_index]
        ),
        "upstream_flow_mach": float(
            profiles["radial_flow_mach"][upstream_index]
        ),
        "downstream_flow_mach": float(
            profiles["radial_flow_mach"][downstream_index]
        ),
        "minimum_divergence": float(
            np.nanmin(profiles["velocity_divergence"][local_indices])
        ),
        "peak_surface_mach": float(
            np.nanmax(final_shock_result["mach_numbers"][neighborhood])
        ),
        "median_radial_normal_alignment": float(
            np.nanmedian(radial_alignment[neighborhood])
        ),
    }
    pressure_ratio = measurements["pressure_ratio"]
    measurements["profile_jump_mach"] = float(
        np.sqrt(
            (
                pressure_ratio * (GAMMA + 1.0)
                + (GAMMA - 1.0)
            )
            / (2.0 * GAMMA)
        )
    )

    detections = [
        bool(row["two_bands_detected"])
        and np.isfinite(float(row["reverse_radius_median"]))
        for row in history_rows
    ]
    consecutive_detections = _longest_consecutive_true_run(detections)
    measurements["consecutive_detection_snapshots"] = consecutive_detections
    measurements["distance_beyond_injection_cells"] = (
        reverse_radius - injection_radius
    ) / grid_spacing

    return classify_reverse_shock_evidence(
        measurements,
        persistent_detection=(
            consecutive_detections >= MIN_PERSISTENT_SNAPSHOTS
        ),
        resolved_from_injection=(
            reverse_radius
            >= injection_radius
            + MIN_INJECTION_SEPARATION_CELLS * grid_spacing
        ),
    )


def calculate_radial_verification_profiles(
    final_state: np.ndarray,
    final_shock_result,
    registered_variables,
    helper_data,
    config: SimulationConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Calculate spherically averaged profiles for the final snapshot."""
    density = final_state[registered_variables.density_index]
    pressure = final_state[registered_variables.pressure_index]
    velocity_x = final_state[registered_variables.velocity_index.x]
    velocity_y = final_state[registered_variables.velocity_index.y]
    velocity_z = final_state[registered_variables.velocity_index.z]

    centers = np.asarray(helper_data.geometric_centers)
    radii = np.asarray(helper_data.r)
    radial_vectors = centers - BOX_SIZE / 2.0
    inverse_radius = 1.0 / np.maximum(radii, 1.0e-30)
    radial_velocity = (
        velocity_x * radial_vectors[..., 0]
        + velocity_y * radial_vectors[..., 1]
        + velocity_z * radial_vectors[..., 2]
    ) * inverse_radius
    temperature_proxy = pressure / np.maximum(density, 1.0e-30)
    sound_speed = np.sqrt(
        GAMMA * pressure / np.maximum(density, 1.0e-30)
    )
    radial_flow_mach = np.abs(radial_velocity) / np.maximum(sound_speed, 1.0e-30)
    velocity_divergence = np.asarray(
        _calculate_velocity_divergence(
            jnp.asarray(final_state), config, registered_variables
        )
    )

    surface_mask = final_shock_result["surface_mask"]
    surface_mach = final_shock_result["mach_numbers"]
    grid_spacing = float(config.grid_spacing)
    radial_limit = BOX_SIZE / 2.0
    bin_edges = np.arange(0.0, radial_limit + grid_spacing, grid_spacing)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_indices = np.digitize(radii, bin_edges) - 1
    num_bins = len(bin_centers)

    profiles = {
        "density": _spherical_bin_median(density, bin_indices, num_bins),
        "pressure": _spherical_bin_median(pressure, bin_indices, num_bins),
        "temperature_proxy": _spherical_bin_median(
            temperature_proxy, bin_indices, num_bins
        ),
        "radial_velocity": _spherical_bin_median(
            radial_velocity, bin_indices, num_bins
        ),
        "velocity_divergence": _spherical_bin_median(
            velocity_divergence, bin_indices, num_bins
        ),
        "radial_flow_mach": _spherical_bin_median(
            radial_flow_mach, bin_indices, num_bins
        ),
        "surface_mach": _spherical_bin_median(
            surface_mach,
            bin_indices,
            num_bins,
            selection=surface_mask,
        ),
    }
    shell_cell_count = np.bincount(
        bin_indices[(bin_indices >= 0) & (bin_indices < num_bins)].ravel(),
        minlength=num_bins,
    )
    return bin_centers, profiles, shell_cell_count


def write_radial_verification_profiles(
    csv_path: Path,
    bin_centers: np.ndarray,
    profiles: dict[str, np.ndarray],
    shell_cell_count: np.ndarray,
) -> None:
    """Write the calculated radial profiles as a reproducible data table."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["radius", "shell_cell_count", *profiles]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for bin_index, radius in enumerate(bin_centers):
            writer.writerow(
                {
                    "radius": float(radius),
                    "shell_cell_count": int(shell_cell_count[bin_index]),
                    **{
                        name: float(profile[bin_index])
                        for name, profile in profiles.items()
                    },
                }
            )


def plot_radial_verification_profiles(
    final_state: np.ndarray,
    final_shock_result,
    history_rows: list[dict],
    registered_variables,
    helper_data,
    config: SimulationConfig,
    injection_radius: float,
    output_path: Path,
    csv_path: Path,
    verification_path: Path,
) -> dict:
    """Verify and plot the final reverse and forward shock surfaces."""
    bin_centers, profiles, shell_cell_count = (
        calculate_radial_verification_profiles(
            final_state=final_state,
            final_shock_result=final_shock_result,
            registered_variables=registered_variables,
            helper_data=helper_data,
            config=config,
        )
    )
    write_radial_verification_profiles(
        csv_path=csv_path,
        bin_centers=bin_centers,
        profiles=profiles,
        shell_cell_count=shell_cell_count,
    )
    grid_spacing = float(config.grid_spacing)

    verification = evaluate_reverse_shock(
        final_shock_result=final_shock_result,
        history_rows=history_rows,
        bin_centers=bin_centers,
        profiles=profiles,
        injection_radius=injection_radius,
        grid_spacing=grid_spacing,
    )
    verification_path.write_text(
        json.dumps(verification, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n=== Reverse-shock verification ===")
    print("Classification          :", verification["classification"])
    for name, passed in verification["criteria"].items():
        print(f"{name:32}: {'PASS' if passed else 'FAIL'}")
    for name, passed in verification["diagnostic_checks"].items():
        print(f"{name:32}: {'PASS' if passed else 'WARNING'}")

    final_history_row = history_rows[-1]
    reverse_radius = float(final_history_row["reverse_radius_median"])
    forward_radius = float(final_history_row["forward_radius_median"])
    reverse_label = (
        "verified reverse shock"
        if verification["verified"]
        else "reverse-shock candidate"
    )
    markers = [
        (injection_radius, "injection", "tab:red", "--"),
        (reverse_radius, reverse_label, "tab:orange", "-"),
        (forward_radius, "forward-shock candidate", "tab:blue", "-"),
    ]

    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    panels = [
        ("density", r"density $\rho$", True),
        ("pressure", "pressure p", True),
        ("temperature_proxy", r"temperature proxy $p/\rho$", True),
        ("radial_velocity", r"radial velocity $v_r$", False),
        ("velocity_divergence", r"velocity divergence $\nabla\cdot v$", False),
    ]
    for axis, (name, ylabel, logarithmic) in zip(axes.flat[:5], panels):
        axis.plot(bin_centers, profiles[name], color="black")
        if logarithmic:
            axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    mach_axis = axes.flat[5]
    mach_axis.plot(
        bin_centers,
        profiles["radial_flow_mach"],
        color="black",
        label=r"flow $|v_r|/c_s$",
    )
    mach_axis.scatter(
        bin_centers,
        profiles["surface_mach"],
        color="tab:purple",
        s=24,
        label="shock-finder Mach",
    )
    mach_axis.axhline(1.0, color="grey", linestyle=":", linewidth=1.0)
    mach_axis.set_ylabel("Mach number")
    mach_axis.legend(fontsize=8)
    mach_axis.grid(alpha=0.25)

    for axis in axes.flat:
        for radius, label, color, linestyle in markers:
            if np.isfinite(radius):
                axis.axvline(
                    radius,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.2,
                    label=label,
                )
        axis.set_xlim(0.0, BOX_SIZE / 2.0)
        axis.set_xlabel("radius from star [code length]")
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("Final-snapshot spherical profiles for shock verification")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return verification


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    central_slices_path = output_dir / "central_slices.png"
    shock_diagnostics_path = output_dir / "shock_diagnostics.png"
    shock_history_plot_path = output_dir / "shock_histories.png"
    shock_history_csv_path = output_dir / "shock_histories.csv"
    verification_plot_path = output_dir / "radial_verification_profiles.png"
    verification_csv_path = output_dir / "radial_verification_profiles.csv"
    reverse_verification_path = output_dir / "reverse_shock_verification.json"

    (
        initial_state,
        config,
        params,
        registered_variables,
        helper_data,
    ) = build_problem(args)

    injection_radius = args.num_injection_cells * float(config.grid_spacing)
    wind_luminosity = (
        0.5 * WIND_MASS_LOSS_RATE * WIND_FINAL_VELOCITY**2
    )

    print("=== 3D single-star wind-bubble analysis ===")
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
        output_path=central_slices_path,
    )
    print(f"Saved diagnostic figure: {central_slices_path.resolve()}")

    shock_results = analyze_shocks_3d(
        states=states,
        config=config,
        registered_variables=registered_variables,
        helper_data=helper_data,
    )

    plot_shock_diagnostics(
        states=states,
        times=times,
        shock_results=shock_results,
        config=config,
        registered_variables=registered_variables,
        helper_data=helper_data,
        injection_radius=injection_radius,
        output_path=shock_diagnostics_path,
    )
    print(f"Saved shock diagnostics : {shock_diagnostics_path.resolve()}")

    history_rows = measure_shock_histories(
        times=times,
        shock_results=shock_results,
        injection_radius=injection_radius,
        grid_spacing=float(config.grid_spacing),
        plot_path=shock_history_plot_path,
        csv_path=shock_history_csv_path,
    )
    print(f"Saved shock history     : {shock_history_plot_path.resolve()}")
    print(f"Saved history table     : {shock_history_csv_path.resolve()}")

    plot_radial_verification_profiles(
        final_state=states[-1],
        final_shock_result=shock_results[-1],
        history_rows=history_rows,
        registered_variables=registered_variables,
        helper_data=helper_data,
        config=config,
        injection_radius=injection_radius,
        output_path=verification_plot_path,
        csv_path=verification_csv_path,
        verification_path=reverse_verification_path,
    )
    print(f"Saved verification plot : {verification_plot_path.resolve()}")
    print(f"Saved verification table: {verification_csv_path.resolve()}")
    print(f"Saved verification result: {reverse_verification_path.resolve()}")


if __name__ == "__main__":
    main()
