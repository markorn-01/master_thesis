"""Run final-state convergence tests for the 3D stellar-wind bubble.

The production wind-bubble script retains full states at every requested
snapshot because it is designed for time-history analysis.  That allocation
scales as ``num_snapshots * num_variables * resolution**3``.  This runner
instead keeps only the final primitive state, analyses one resolution in each
Python process, writes compact metrics, and releases all JAX allocations before
starting the next resolution.

The physical injection radius is fixed across resolutions.  With the default
box and injection radius this means 4, 8, 16, and 32 injection cells at
64^3, 128^3, 256^3, and 512^3, respectively.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import jax
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
    ON_DEVICE,
    TO_DISK,
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.setup_helpers import (  # noqa: E402
    latest_checkpoint_step,
    restart_from_latest_checkpoint,
)
from astronomix._modules._stellar_wind.stellar_wind_options import (  # noqa: E402
    EI,
    WindConfig,
    WindParams,
)
from astronomix._modules._stellar_wind.weaver import Weaver  # noqa: E402
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import (  # noqa: E402
    find_shocks_pfrommer,
)
from experiments.wind_bubble.run_single_bubble import (  # noqa: E402
    AMBIENT_DENSITY,
    AMBIENT_PRESSURE,
    BOX_SIZE,
    DEFAULT_T_END,
    GAMMA,
    WIND_FINAL_VELOCITY,
    WIND_MASS_LOSS_RATE,
    calculate_radial_verification_profiles,
    split_radial_shock_candidates,
    write_radial_verification_profiles,
)


DEFAULT_RESOLUTIONS = (64, 128, 256)
DEFAULT_INJECTION_RADIUS = 0.0625
DEFAULT_OUTPUT_DIR = Path("outputs/wind_bubble_convergence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=list(DEFAULT_RESOLUTIONS),
        help="Cells per axis (default: 64 128 256).",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=DEFAULT_T_END,
        help="Final simulation time (default: 0.20).",
    )
    parser.add_argument(
        "--injection-radius",
        type=float,
        default=DEFAULT_INJECTION_RADIUS,
        help="Fixed physical injection radius (default: 0.0625).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--worker-resolution",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild summary files from existing per-resolution metrics.",
    )
    parser.add_argument(
        "--checkpoint-segments",
        type=int,
        default=0,
        help=(
            "Write this many evenly spaced restart checkpoints per run. "
            "Zero disables checkpointing (default: 0)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each incomplete resolution from its latest checkpoint.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail before simulation setup unless JAX has a GPU backend.",
    )
    parser.add_argument(
        "--memory-analysis",
        action="store_true",
        help="Print XLA's compiled memory estimate (without checkpoints only).",
    )
    return parser.parse_args()


def require_gpu_backend() -> None:
    """Fail clearly when a cluster job accidentally uses a CPU-only JAX."""
    backend = jax.default_backend()
    devices = jax.devices()
    if backend != "gpu" or not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(
            "A GPU was required, but JAX selected "
            f"backend={backend!r} with devices={devices!r}. Install the CUDA "
            "JAX wheel and run this command inside a Slurm GPU allocation."
        )
    print(f"JAX backend: {backend}; devices: {devices}", flush=True)


def injection_cells_for_resolution(
    resolution: int,
    injection_radius: float = DEFAULT_INJECTION_RADIUS,
) -> int:
    """Return the integer cell radius that preserves ``injection_radius``."""
    if resolution < 8:
        raise ValueError("Resolution must be at least 8 cells per axis.")
    if not 0.0 < injection_radius < BOX_SIZE / 4.0:
        raise ValueError("Injection radius must lie between 0 and L_box / 4.")

    cells_exact = injection_radius * resolution / BOX_SIZE
    cells = int(round(cells_exact))
    if cells < 1 or not np.isclose(cells_exact, cells, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"Injection radius {injection_radius} is not an integer number "
            f"of cells at resolution {resolution}."
        )
    return cells


def _finite_band_statistics(
    band_radii: np.ndarray,
    surface_radii: np.ndarray,
    surface_mach: np.ndarray,
    radial_alignment: np.ndarray,
) -> dict[str, float | int]:
    """Measure radius, Mach, and direction statistics for one radial band."""
    if band_radii.size == 0:
        return {
            "radius_median": np.nan,
            "radius_p16": np.nan,
            "radius_p84": np.nan,
            "normalized_radial_spread": np.nan,
            "surface_cell_count": 0,
            "valid_mach_fraction": np.nan,
            "mach_median": np.nan,
            "mach_p16": np.nan,
            "mach_p84": np.nan,
            "mach_coefficient_of_variation": np.nan,
            "median_radial_alignment": np.nan,
        }

    band_mask = (surface_radii >= band_radii.min()) & (
        surface_radii <= band_radii.max()
    )
    band_mach = surface_mach[band_mask]
    valid_mach = band_mach[np.isfinite(band_mach) & (band_mach > 0.0)]
    radius_p16, radius_median, radius_p84 = np.percentile(
        band_radii, [16.0, 50.0, 84.0]
    )

    stats: dict[str, float | int] = {
        "radius_median": float(radius_median),
        "radius_p16": float(radius_p16),
        "radius_p84": float(radius_p84),
        "normalized_radial_spread": float(
            (radius_p84 - radius_p16) / radius_median
        ),
        "surface_cell_count": int(band_mask.sum()),
        "valid_mach_fraction": float(valid_mach.size / band_mach.size),
        "median_radial_alignment": float(
            np.median(radial_alignment[band_mask])
        ),
    }
    if valid_mach.size:
        stats.update(
            {
                "mach_median": float(np.median(valid_mach)),
                "mach_p16": float(np.percentile(valid_mach, 16.0)),
                "mach_p84": float(np.percentile(valid_mach, 84.0)),
                "mach_coefficient_of_variation": float(
                    np.std(valid_mach) / np.mean(valid_mach)
                ),
            }
        )
    else:
        stats.update(
            {
                "mach_median": np.nan,
                "mach_p16": np.nan,
                "mach_p84": np.nan,
                "mach_coefficient_of_variation": np.nan,
            }
        )
    return stats


def _prefix(prefix: str, values: dict) -> dict:
    return {f"{prefix}_{name}": value for name, value in values.items()}


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio, or NaN when it is not well defined."""
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan
    if abs(denominator) <= np.finfo(float).tiny:
        return np.nan
    return float(numerator / denominator)


def _radial_profile_jump_statistics(
    shock_radius: float,
    bin_centers: np.ndarray,
    profiles: dict[str, np.ndarray],
    shock_kind: str,
) -> dict[str, float]:
    """Measure upstream/downstream jumps around one radial shock surface.

    The unshocked upstream gas lies inside a reverse shock but outside a
    forward shock.  This orientation is important: all reported ratios are
    downstream divided by upstream, independent of shock type.
    """
    if shock_kind not in {"forward", "reverse"}:
        raise ValueError("shock_kind must be 'forward' or 'reverse'.")

    empty = {
        "upstream_shell_radius": np.nan,
        "downstream_shell_radius": np.nan,
        "density_ratio": np.nan,
        "pressure_ratio": np.nan,
        "temperature_ratio": np.nan,
        "velocity_ratio": np.nan,
        "upstream_radial_velocity": np.nan,
        "downstream_radial_velocity": np.nan,
        "upstream_flow_mach": np.nan,
        "downstream_flow_mach": np.nan,
        "minimum_divergence": np.nan,
        "jump_mach": np.nan,
    }
    if not np.isfinite(shock_radius):
        return empty

    inner_bins = np.flatnonzero(bin_centers < shock_radius)
    outer_bins = np.flatnonzero(bin_centers > shock_radius)
    if not inner_bins.size or not outer_bins.size:
        return empty

    inner_index = int(inner_bins[-1])
    outer_index = int(outer_bins[0])
    if shock_kind == "reverse":
        upstream_index, downstream_index = inner_index, outer_index
    else:
        upstream_index, downstream_index = outer_index, inner_index

    density_ratio = _safe_ratio(
        profiles["density"][downstream_index],
        profiles["density"][upstream_index],
    )
    pressure_ratio = _safe_ratio(
        profiles["pressure"][downstream_index],
        profiles["pressure"][upstream_index],
    )
    temperature_ratio = _safe_ratio(
        profiles["temperature_proxy"][downstream_index],
        profiles["temperature_proxy"][upstream_index],
    )
    upstream_velocity = float(profiles["radial_velocity"][upstream_index])
    downstream_velocity = float(
        profiles["radial_velocity"][downstream_index]
    )
    local_divergence = profiles["velocity_divergence"][
        min(inner_index, outer_index) : max(inner_index, outer_index) + 1
    ]
    finite_divergence = local_divergence[np.isfinite(local_divergence)]

    jump_mach = np.nan
    if np.isfinite(pressure_ratio) and pressure_ratio > 0.0:
        jump_mach = float(
            np.sqrt(
                (
                    pressure_ratio * (GAMMA + 1.0)
                    + (GAMMA - 1.0)
                )
                / (2.0 * GAMMA)
            )
        )

    return {
        "upstream_shell_radius": float(bin_centers[upstream_index]),
        "downstream_shell_radius": float(bin_centers[downstream_index]),
        "density_ratio": density_ratio,
        "pressure_ratio": pressure_ratio,
        "temperature_ratio": temperature_ratio,
        "velocity_ratio": _safe_ratio(
            downstream_velocity, upstream_velocity
        ),
        "upstream_radial_velocity": upstream_velocity,
        "downstream_radial_velocity": downstream_velocity,
        "upstream_flow_mach": float(
            profiles["radial_flow_mach"][upstream_index]
        ),
        "downstream_flow_mach": float(
            profiles["radial_flow_mach"][downstream_index]
        ),
        "minimum_divergence": float(np.min(finite_divergence))
        if finite_divergence.size
        else np.nan,
        "jump_mach": jump_mach,
    }


def _surface_samples(result, helper_data, grid_spacing: float):
    """Gather only detected surface samples from the full JAX result arrays."""
    surface_mask = np.asarray(result.shock_surface_cells, dtype=bool)
    flat_indices_np = np.flatnonzero(surface_mask.ravel())
    if flat_indices_np.size == 0:
        raise RuntimeError("No shock-surface cells were detected.")
    flat_indices = jnp.asarray(flat_indices_np)

    centers = jnp.reshape(helper_data.geometric_centers, (-1, 3))[
        flat_indices
    ]
    direction = jnp.reshape(
        jnp.moveaxis(result.shock_direction, 0, -1), (-1, 3)
    )[flat_indices]
    offsets = jnp.ravel(result.shock_surface_offsets)[flat_indices]
    refined_centers = centers + grid_spacing * offsets[:, None] * direction
    displacement = refined_centers - BOX_SIZE / 2.0
    radii = jnp.linalg.norm(displacement, axis=-1)
    radial_unit = displacement / jnp.maximum(radii[:, None], 1.0e-30)
    alignment = jnp.sum(direction * radial_unit, axis=-1)
    mach = jnp.ravel(result.mach_numbers)[flat_indices]

    return (
        np.asarray(radii),
        np.asarray(mach),
        np.asarray(alignment),
        int(flat_indices_np.size),
    )


def run_one_resolution(
    resolution: int,
    t_end: float,
    injection_radius: float,
    output_dir: Path,
    checkpoint_segments: int = 0,
    resume: bool = False,
    memory_analysis: bool = False,
) -> dict:
    """Run and analyse one final-state-only wind-bubble simulation."""
    injection_cells = injection_cells_for_resolution(
        resolution, injection_radius
    )
    grid_spacing = BOX_SIZE / resolution

    resolution_dir = output_dir / f"n{resolution:03d}"
    checkpoint_dir = resolution_dir / "checkpoints"
    checkpointing = checkpoint_segments > 0

    config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=resolution,
        mhd=False,
        return_snapshots=False,
        donate_state=True,
        memory_analysis=memory_analysis,
        snapshot_storage_mode=TO_DISK if checkpointing else ON_DEVICE,
        snapshot_storage_path=str(checkpoint_dir.resolve())
        if checkpointing
        else None,
        num_snapshots=checkpoint_segments if checkpointing else 10,
        wind_config=WindConfig(
            stellar_wind=True,
            num_injection_cells=injection_cells,
            wind_injection_scheme=EI,
            trace_wind_density=False,
        ),
    )
    params = SimulationParams(
        gamma=GAMMA,
        t_end=t_end,
        wind_params=WindParams(
            wind_mass_loss_rate=WIND_MASS_LOSS_RATE,
            wind_final_velocity=WIND_FINAL_VELOCITY,
        ),
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    shape = (resolution,) * 3
    density = jnp.full(shape, AMBIENT_DENSITY)
    pressure = jnp.full(shape, AMBIENT_PRESSURE)
    zero_velocity = jnp.zeros(shape)
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

    restart_state = None
    latest_step = latest_checkpoint_step(checkpoint_dir) if checkpointing else None
    if resume and latest_step is not None:
        initial_state, params, restart_state = restart_from_latest_checkpoint(
            checkpoint_dir,
            params,
        )
        print(
            f"Resuming {resolution}^3 from checkpoint {latest_step} "
            f"at t={float(params.t_start):.6f}.",
            flush=True,
        )
    elif latest_step is not None:
        raise RuntimeError(
            f"Checkpoint data already exists in {checkpoint_dir}. Use --resume "
            "or choose a different --output-dir."
        )

    start_time = time.perf_counter()
    if float(params.t_start) >= t_end:
        final_state = initial_state
        print("Latest checkpoint is already at the requested end time.", flush=True)
    else:
        final_state = time_integration(
            initial_state,
            config,
            params,
            registered_variables,
            restart_state=restart_state,
        )
    if not bool(jnp.all(jnp.isfinite(final_state))):
        raise RuntimeError(f"The {resolution}^3 run produced non-finite values.")

    result = find_shocks_pfrommer(
        final_state,
        config,
        registered_variables,
        helper_data,
    )
    surface_radii, surface_mach, alignment, surface_count = (
        _surface_samples(result, helper_data, grid_spacing)
    )
    outside = np.isfinite(surface_radii) & (
        surface_radii > injection_radius
    )
    outside_radii = surface_radii[outside]
    outside_mach = surface_mach[outside]
    outside_alignment = alignment[outside]
    reverse_radii, forward_radii, separation_radius = (
        split_radial_shock_candidates(
            outside_radii,
            injection_radius=injection_radius,
            grid_spacing=grid_spacing,
        )
    )
    if forward_radii.size == 0:
        raise RuntimeError(
            f"No forward shock was detected outside the injection region "
            f"at {resolution}^3."
        )

    reverse = _finite_band_statistics(
        reverse_radii,
        outside_radii,
        outside_mach,
        outside_alignment,
    )
    forward = _finite_band_statistics(
        forward_radii,
        outside_radii,
        outside_mach,
        outside_alignment,
    )
    bin_centers, profiles, shell_cell_count = (
        calculate_radial_verification_profiles(
            final_state=np.asarray(final_state),
            final_shock_result={
                "surface_mask": np.asarray(
                    result.shock_surface_cells, dtype=bool
                ),
                "mach_numbers": np.asarray(result.mach_numbers),
            },
            registered_variables=registered_variables,
            helper_data=helper_data,
            config=config,
        )
    )
    reverse_profile = _radial_profile_jump_statistics(
        shock_radius=float(reverse["radius_median"]),
        bin_centers=bin_centers,
        profiles=profiles,
        shock_kind="reverse",
    )
    forward_profile = _radial_profile_jump_statistics(
        shock_radius=float(forward["radius_median"]),
        bin_centers=bin_centers,
        profiles=profiles,
        shock_kind="forward",
    )
    weaver = Weaver(
        v_inf=WIND_FINAL_VELOCITY,
        M_dot=WIND_MASS_LOSS_RATE,
        rho_0=AMBIENT_DENSITY,
        p_0=AMBIENT_PRESSURE,
        gamma=GAMMA,
    )
    weaver_radius = float(weaver.get_outer_shock_radius(t_end))
    forward_radius = float(forward["radius_median"])

    metrics = {
        "resolution": resolution,
        "grid_spacing": grid_spacing,
        "num_injection_cells": injection_cells,
        "injection_radius": injection_cells * grid_spacing,
        "t_end": t_end,
        "retained_snapshot_count": 0,
        "final_state_bytes": int(np.prod(final_state.shape) * final_state.dtype.itemsize),
        "surface_cell_count": surface_count,
        "outside_injection_surface_cell_count": int(outside.sum()),
        "two_bands_detected": bool(reverse_radii.size),
        "separation_radius": float(separation_radius),
        **_prefix("reverse", reverse),
        **_prefix("forward", forward),
        **_prefix("reverse_profile", reverse_profile),
        **_prefix("forward_profile", forward_profile),
        "reverse_distance_beyond_injection_cells": float(
            (float(reverse["radius_median"]) - injection_radius)
            / grid_spacing
        )
        if reverse_radii.size
        else np.nan,
        "weaver_outer_radius": weaver_radius,
        "relative_error_vs_weaver": float(
            (forward_radius - weaver_radius) / weaver_radius
        ),
        "elapsed_seconds": time.perf_counter() - start_time,
    }

    resolution_dir.mkdir(parents=True, exist_ok=True)
    write_radial_verification_profiles(
        resolution_dir / "radial_profiles.csv",
        bin_centers,
        profiles,
        shell_cell_count,
    )
    metrics_path = resolution_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def _load_metrics(resolutions: list[int], output_dir: Path) -> list[dict]:
    return [
        json.loads(
            (output_dir / f"n{resolution:03d}" / "metrics.json").read_text()
        )
        for resolution in resolutions
    ]


def write_summary(metrics: list[dict], output_dir: Path) -> None:
    """Write the cross-resolution CSV and diagnostic comparison figure."""
    if not metrics:
        raise ValueError("At least one metrics row is required.")
    metrics = sorted(metrics, key=lambda row: row["resolution"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "wind_bubble_convergence.csv"
    fieldnames = list(metrics[0])
    for row in metrics[1:]:
        fieldnames.extend(name for name in row if name not in fieldnames)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    resolution = np.asarray([row["resolution"] for row in metrics])
    reverse_radius = np.asarray(
        [row["reverse_radius_median"] for row in metrics], dtype=float
    )
    forward_radius = np.asarray(
        [row["forward_radius_median"] for row in metrics], dtype=float
    )
    weaver_radius = np.asarray(
        [row["weaver_outer_radius"] for row in metrics], dtype=float
    )
    weaver_error = 100.0 * np.asarray(
        [row["relative_error_vs_weaver"] for row in metrics], dtype=float
    )

    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes[0, 0].plot(resolution, forward_radius, "o-", label="forward")
    axes[0, 0].plot(resolution, reverse_radius, "o-", label="reverse")
    axes[0, 0].plot(
        resolution, weaver_radius, "k--", label="Weaver forward"
    )
    axes[0, 0].set(ylabel="radius", title="Shock radii")
    axes[0, 0].legend()

    axes[0, 1].plot(resolution, weaver_error, "o-")
    axes[0, 1].axhline(0.0, color="black", linewidth=1.0)
    axes[0, 1].set(ylabel="relative error [%]", title="Weaver comparison")

    for prefix, label in (("forward", "forward"), ("reverse", "reverse")):
        median = np.asarray(
            [row[f"{prefix}_mach_median"] for row in metrics], dtype=float
        )
        p16 = np.asarray(
            [row[f"{prefix}_mach_p16"] for row in metrics], dtype=float
        )
        p84 = np.asarray(
            [row[f"{prefix}_mach_p84"] for row in metrics], dtype=float
        )
        valid = np.isfinite(median) & np.isfinite(p16) & np.isfinite(p84)
        if np.any(valid):
            axes[1, 0].errorbar(
                resolution[valid],
                median[valid],
                yerr=np.vstack(
                    (median[valid] - p16[valid], p84[valid] - median[valid])
                ),
                marker="o",
                capsize=3,
                label=label,
            )
    axes[1, 0].set(ylabel="Mach number", title="Surface Mach distribution")
    axes[1, 0].legend()

    axes[1, 1].plot(
        resolution,
        [row["forward_mach_coefficient_of_variation"] for row in metrics],
        "o-",
        label="forward Mach CV",
    )
    axes[1, 1].plot(
        resolution,
        [row["forward_normalized_radial_spread"] for row in metrics],
        "o-",
        label="forward radial spread",
    )
    axes[1, 1].set(ylabel="normalized spread", title="Angular/radial scatter")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("cells per axis")
        axis.grid(alpha=0.25)
    figure.suptitle("3D wind-bubble resolution convergence")
    figure_path = output_dir / "wind_bubble_convergence.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print(f"\nSaved summary: {csv_path.resolve()}")
    print(f"Saved plot   : {figure_path.resolve()}")


def run_parent(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    if existing_pythonpath:
        environment["PYTHONPATH"] += os.pathsep + existing_pythonpath

    for resolution in args.resolutions:
        injection_cells = injection_cells_for_resolution(
            resolution, args.injection_radius
        )
        print(
            f"\n=== Running {resolution}^3 wind bubble "
            f"(R_inj={injection_cells} cells) ===",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-resolution",
                str(resolution),
                "--t-end",
                str(args.t_end),
                "--injection-radius",
                str(args.injection_radius),
                "--output-dir",
                str(args.output_dir),
                "--checkpoint-segments",
                str(args.checkpoint_segments),
                *(["--resume"] if args.resume else []),
                *(["--require-gpu"] if args.require_gpu else []),
                *(["--memory-analysis"] if args.memory_analysis else []),
            ],
            check=True,
            cwd=REPOSITORY_ROOT,
            env=environment,
        )

    write_summary(_load_metrics(args.resolutions, args.output_dir), args.output_dir)


def main() -> None:
    args = parse_args()
    if args.t_end <= 0.0:
        raise ValueError("--t-end must be positive.")
    if args.checkpoint_segments < 0:
        raise ValueError("--checkpoint-segments cannot be negative.")
    if args.resume and args.checkpoint_segments == 0:
        raise ValueError("--resume requires --checkpoint-segments greater than zero.")
    if args.memory_analysis and args.checkpoint_segments:
        raise ValueError(
            "--memory-analysis and disk checkpointing cannot currently be "
            "combined; run a smaller non-checkpointed estimate first."
        )
    if args.require_gpu:
        require_gpu_backend()
    if args.worker_resolution is not None:
        run_one_resolution(
            resolution=args.worker_resolution,
            t_end=args.t_end,
            injection_radius=args.injection_radius,
            output_dir=args.output_dir,
            checkpoint_segments=args.checkpoint_segments,
            resume=args.resume,
            memory_analysis=args.memory_analysis,
        )
    elif args.aggregate_only:
        write_summary(
            _load_metrics(args.resolutions, args.output_dir), args.output_dir
        )
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
