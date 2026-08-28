"""Aggregate temporal wind-bubble energy results across resolutions.

Run from the repository root after copying the per-resolution result folders::

    python -m experiments.wind_bubble.aggregate_energy_convergence

The script reads ``shock_energy_histories.csv`` and
``reverse_shock_verification.json`` from each input directory and writes a
compact convergence table plus a four-panel comparison figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESOLUTIONS = (64, 128, 256)
DEFAULT_INPUT_ROOT = Path("outputs")
DEFAULT_OUTPUT_DIR = Path("outputs/single_bubble_energy_convergence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate single-bubble energy convergence results."
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=list(DEFAULT_RESOLUTIONS),
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _float_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    try:
        return np.asarray([float(row[name]) for row in rows], dtype=float)
    except KeyError as error:
        raise ValueError(f"Missing required energy-history column: {name}") from error


def load_resolution_result(resolution: int, input_root: Path) -> dict:
    """Load and validate one temporal energy result directory."""
    result_dir = input_root / f"single_bubble_energy_n{resolution:03d}"
    history_path = result_dir / "shock_energy_histories.csv"
    verification_path = result_dir / "reverse_shock_verification.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing energy history: {history_path}")
    if not verification_path.is_file():
        raise FileNotFoundError(
            f"Missing reverse-shock verification: {verification_path}"
        )

    with history_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"Energy history is empty: {history_path}")

    time = _float_column(rows, "time")
    if np.any(~np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        raise ValueError(f"Snapshot times must be finite and increasing: {history_path}")

    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    return {
        "resolution": resolution,
        "result_dir": result_dir,
        "rows": rows,
        "time": time,
        "verification": verification,
    }


def summarize_results(results: list[dict]) -> list[dict]:
    """Return one final-snapshot convergence row per resolution."""
    if not results:
        raise ValueError("At least one resolution result is required.")

    summary: list[dict] = []
    previous_energy = np.nan
    for result in sorted(results, key=lambda item: item["resolution"]):
        final = result["rows"][-1]
        verification = result["verification"]
        energy = float(final["combined_cumulative_dissipated_energy"])
        relative_change = (
            (energy - previous_energy) / previous_energy
            if np.isfinite(previous_energy) and previous_energy != 0.0
            else np.nan
        )
        measurements = verification.get("measurements", {})
        summary.append(
            {
                "resolution": result["resolution"],
                "snapshot_count": len(result["rows"]),
                "final_time": float(final["time"]),
                "reverse_classification": verification.get(
                    "classification", "unknown"
                ),
                "reverse_profile_jump_mach": measurements.get(
                    "profile_jump_mach", np.nan
                ),
                "reverse_valid_flux_fraction": float(
                    final["reverse_valid_flux_fraction"]
                ),
                "forward_valid_flux_fraction": float(
                    final["forward_valid_flux_fraction"]
                ),
                "reverse_surface_area_vs_sphere": float(
                    final["reverse_surface_area_vs_sphere"]
                ),
                "forward_surface_area_vs_sphere": float(
                    final["forward_surface_area_vs_sphere"]
                ),
                "reverse_mean_thermal_energy_flux": float(
                    final["reverse_mean_thermal_energy_flux"]
                ),
                "forward_mean_thermal_energy_flux": float(
                    final["forward_mean_thermal_energy_flux"]
                ),
                "reverse_dissipation_rate": float(
                    final["reverse_dissipation_rate"]
                ),
                "forward_dissipation_rate": float(
                    final["forward_dissipation_rate"]
                ),
                "reverse_cumulative_dissipated_energy": float(
                    final["reverse_cumulative_dissipated_energy"]
                ),
                "forward_cumulative_dissipated_energy": float(
                    final["forward_cumulative_dissipated_energy"]
                ),
                "combined_cumulative_dissipated_energy": energy,
                "injected_wind_energy": float(final["injected_wind_energy"]),
                "combined_dissipation_to_injected_energy": float(
                    final["combined_dissipation_to_injected_energy"]
                ),
                "relative_change_from_previous_resolution": relative_change,
            }
        )
        previous_energy = energy
    return summary


def write_summary(summary: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "single_bubble_energy_convergence.csv"
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)
    return path


def plot_convergence(
    results: list[dict], summary: list[dict], output_dir: Path
) -> Path:
    """Plot temporal histories and final cross-resolution convergence."""
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))

    for colour, result in zip(
        colours, sorted(results, key=lambda item: item["resolution"])
    ):
        rows = result["rows"]
        time = result["time"]
        label = rf"${result['resolution']}^3$"
        axes[0, 0].plot(
            time,
            _float_column(rows, "forward_dissipation_rate"),
            marker="o",
            markersize=3,
            color=colour,
            label=label,
        )
        axes[0, 1].plot(
            time,
            _float_column(rows, "reverse_dissipation_rate"),
            marker="o",
            markersize=3,
            color=colour,
            label=label,
        )
        axes[1, 0].plot(
            time,
            _float_column(rows, "combined_cumulative_dissipated_energy"),
            marker="o",
            markersize=3,
            color=colour,
            label=label,
        )

    resolution = np.asarray([row["resolution"] for row in summary])
    energy = np.asarray(
        [row["combined_cumulative_dissipated_energy"] for row in summary]
    )
    fraction = 100.0 * np.asarray(
        [row["combined_dissipation_to_injected_energy"] for row in summary]
    )
    axes[1, 1].plot(resolution, energy, "o-", color="tab:purple")
    for x, y, percentage in zip(resolution, energy, fraction):
        axes[1, 1].annotate(
            f"{percentage:.1f}%",
            (x, y),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
        )

    axes[0, 0].set(title="Forward-shock power", ylabel="dissipation rate")
    axes[0, 1].set(title="Reverse-shock power", ylabel="dissipation rate")
    axes[1, 0].set(
        title="Combined cumulative dissipated energy",
        ylabel="cumulative energy",
    )
    axes[1, 1].set(
        title="Final cumulative-energy convergence",
        xlabel="cells per axis",
        ylabel="cumulative energy at $t=0.2$",
        xticks=resolution,
    )
    axes[1, 1].margins(y=0.15)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xlabel("time")
        axis.legend(title="resolution")
    for axis in axes.flat:
        axis.grid(alpha=0.25)

    figure.suptitle("Single wind-bubble shock-energy convergence")
    path = output_dir / "single_bubble_energy_convergence.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    results = [
        load_resolution_result(resolution, args.input_root)
        for resolution in args.resolutions
    ]
    summary = summarize_results(results)
    csv_path = write_summary(summary, args.output_dir)
    figure_path = plot_convergence(results, summary, args.output_dir)
    print(f"Saved convergence table: {csv_path.resolve()}")
    print(f"Saved convergence plot : {figure_path.resolve()}")


if __name__ == "__main__":
    main()
