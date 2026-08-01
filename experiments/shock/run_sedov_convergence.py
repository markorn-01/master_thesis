"""Run a reproducible 3D Sedov resolution-convergence study.

Each resolution runs in a separate Python process so JAX allocations and
compiled executables are released before the next, larger grid begins.
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
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import (  # noqa: E402
    find_shocks_pfrommer,
)
from astronomix.option_classes.simulation_config import HLLC, MINMOD  # noqa: E402


BOX_SIZE = 1.0
EXPLOSION_CENTER = np.array([0.5, 0.5, 0.5])
GAMMA = 5.0 / 3.0
EXPLOSION_ENERGY = 1.0
AMBIENT_DENSITY = 1.0
AMBIENT_PRESSURE = 1.0e-4
INJECTION_RADIUS = 0.05
T_END = 0.05
MACH_MIN = 1.3
SEDOV_XI = 1.15167


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[32, 64, 128],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sedov_convergence"),
    )
    parser.add_argument("--worker-resolution", type=int, default=None)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Build the summary from existing per-resolution metrics.",
    )
    return parser.parse_args()


def analytic_radius() -> float:
    return SEDOV_XI * (
        EXPLOSION_ENERGY * T_END**2 / AMBIENT_DENSITY
    ) ** 0.2


def run_one_resolution(resolution: int, output_dir: Path) -> dict:
    grid_spacing = BOX_SIZE / resolution
    if INJECTION_RADIUS < 1.5 * grid_spacing:
        raise ValueError(
            f"Injection radius spans fewer than 1.5 cells at {resolution}^3."
        )

    config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=3,
        riemann_solver=HLLC,
        limiter=MINMOD,
        box_size=BOX_SIZE,
        num_cells=resolution,
        mhd=False,
    )
    params = SimulationParams(gamma=GAMMA, t_end=T_END)
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    centers = helper_data.geometric_centers
    displacement = centers - jnp.asarray(EXPLOSION_CENTER)
    radii = jnp.sqrt(jnp.sum(displacement**2, axis=-1))
    injection_mask = radii < INJECTION_RADIUS
    injection_volume = jnp.sum(injection_mask) * grid_spacing**3
    pressure_excess = (
        EXPLOSION_ENERGY * (GAMMA - 1.0) / injection_volume
    )

    shape = (resolution,) * 3
    density = jnp.full(shape, AMBIENT_DENSITY)
    pressure = jnp.full(shape, AMBIENT_PRESSURE)
    pressure += jnp.where(injection_mask, pressure_excess, 0.0)
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

    start_time = time.perf_counter()
    final_state = time_integration(
        initial_state,
        config,
        params,
        registered_variables,
    )
    if not bool(jnp.all(jnp.isfinite(final_state))):
        raise RuntimeError(f"The {resolution}^3 run produced non-finite values.")

    result = find_shocks_pfrommer(
        final_state,
        config,
        registered_variables,
        helper_data,
        mach_min=MACH_MIN,
    )
    elapsed_seconds = time.perf_counter() - start_time

    surface = np.asarray(result.shock_surface_cells, dtype=bool)
    radii_np = np.asarray(radii)
    centers_np = np.asarray(centers)
    shock_direction = np.moveaxis(
        np.asarray(result.shock_direction), 0, -1
    )
    surface_offsets = np.asarray(result.shock_surface_offsets)
    refined_centers = centers_np + (
        grid_spacing * surface_offsets[..., np.newaxis] * shock_direction
    )
    refined_displacement = refined_centers - EXPLOSION_CENTER
    refined_radii = np.linalg.norm(refined_displacement, axis=-1)
    surface_radii = refined_radii[surface]
    if surface_radii.size == 0:
        raise RuntimeError(f"No shock surface was detected at {resolution}^3.")

    radius_p16, radius_median, radius_p84 = np.percentile(
        surface_radii, [16.0, 50.0, 84.0]
    )
    displacement_np = centers_np - EXPLOSION_CENTER
    radial_norm = np.linalg.norm(displacement_np, axis=-1, keepdims=True)
    radial_unit = displacement_np / np.maximum(radial_norm, 1.0e-30)
    radial_alignment = np.sum(
        shock_direction[surface] * radial_unit[surface], axis=-1
    )
    expected_radius = analytic_radius()

    metrics = {
        "resolution": resolution,
        "grid_spacing": grid_spacing,
        "injection_cells": int(np.asarray(injection_mask).sum()),
        "deposited_energy": float(
            pressure_excess * injection_volume / (GAMMA - 1.0)
        ),
        "surface_cell_count": int(surface.sum()),
        "radius_median": float(radius_median),
        "radius_p16": float(radius_p16),
        "radius_p84": float(radius_p84),
        "normalized_radial_spread": float(
            (radius_p84 - radius_p16) / radius_median
        ),
        "analytic_radius": expected_radius,
        "relative_radius_error": float(
            (radius_median - expected_radius) / expected_radius
        ),
        "median_radial_alignment": float(np.median(radial_alignment)),
        "elapsed_seconds": elapsed_seconds,
    }

    resolution_dir = output_dir / f"n{resolution:03d}"
    resolution_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = resolution_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def run_worker(resolution: int, output_dir: Path) -> None:
    run_one_resolution(resolution, output_dir)


def run_parent(resolutions: list[int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    if existing_pythonpath:
        environment["PYTHONPATH"] += os.pathsep + existing_pythonpath

    for resolution in resolutions:
        print(f"\n=== Running {resolution}^3 Sedov case ===", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-resolution",
                str(resolution),
                "--output-dir",
                str(output_dir),
            ],
            check=True,
            cwd=REPOSITORY_ROOT,
            env=environment,
        )

    metrics = [
        json.loads(
            (output_dir / f"n{resolution:03d}" / "metrics.json").read_text()
        )
        for resolution in resolutions
    ]
    write_summary(metrics, output_dir)


def write_summary(metrics: list[dict], output_dir: Path) -> None:
    metrics = sorted(metrics, key=lambda row: row["resolution"])
    csv_path = output_dir / "sedov_convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    resolution = np.array([row["resolution"] for row in metrics])
    spacing = np.array([row["grid_spacing"] for row in metrics])
    error = np.abs([row["relative_radius_error"] for row in metrics])
    spread = np.array([row["normalized_radial_spread"] for row in metrics])
    alignment = np.array([row["median_radial_alignment"] for row in metrics])

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].loglog(spacing, error, "o-")
    axes[0].invert_xaxis()
    axes[0].set(
        xlabel=r"grid spacing $\Delta x$",
        ylabel="absolute relative radius error",
        title="Radius convergence",
    )
    axes[1].plot(resolution, spread, "o-")
    axes[1].set(
        xlabel="cells per axis",
        ylabel=r"$(R_{84}-R_{16})/R_{50}$",
        title="Normalized radial spread",
    )
    axes[2].plot(resolution, alignment, "o-")
    axes[2].set(
        xlabel="cells per axis",
        ylabel="median radial alignment",
        title="Direction alignment",
        ylim=(min(0.95, alignment.min() - 0.005), 1.0),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("3D Sedov shock-finder resolution convergence")
    figure_path = output_dir / "sedov_convergence.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print(f"\nSaved summary: {csv_path.resolve()}")
    print(f"Saved plot   : {figure_path.resolve()}")


def main() -> None:
    args = parse_args()
    if args.worker_resolution is not None:
        run_worker(args.worker_resolution, args.output_dir)
    elif args.aggregate_only:
        metrics = [
            json.loads(
                (
                    args.output_dir
                    / f"n{resolution:03d}"
                    / "metrics.json"
                ).read_text()
            )
            for resolution in args.resolutions
        ]
        write_summary(metrics, args.output_dir)
    else:
        run_parent(args.resolutions, args.output_dir)


if __name__ == "__main__":
    main()
