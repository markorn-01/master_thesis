"""Compare MHD wind-bubble validation histories without rerunning simulations.

Example
-------
python -m experiments.wind_bubble.compare_mhd_runs \
  --run beta100=outputs/single_bubble_mhd_beta100_n064/mhd_validation.csv \
  --run beta1=outputs/single_bubble_mhd_beta1_n064/mhd_validation.csv \
  --output-dir outputs/mhd_comparison_n064
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = (
    "time",
    "max_abs_magnetic_divergence",
    "mean_magnetic_energy_density",
    "pressure_weighted_extent_parallel_to_field",
    "pressure_weighted_extent_perpendicular_to_field",
    "pressure_weighted_axis_aspect_ratio",
)
DEFAULT_OUTPUT_DIR = Path("outputs/mhd_comparison_n064")


def parse_run_specification(value: str) -> tuple[str, Path]:
    """Parse ``LABEL=CSV_PATH`` while allowing equals signs inside the path."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must have the form LABEL=CSV_PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("run label and CSV path must be non-empty")
    return label.strip(), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_specification,
        required=True,
        metavar="LABEL=CSV_PATH",
        help="Labeled mhd_validation.csv input; provide at least two.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if len(args.run) < 2:
        parser.error("provide at least two --run inputs")
    labels = [label for label, _ in args.run]
    if len(labels) != len(set(labels)):
        parser.error("--run labels must be unique")
    return args


def read_mhd_history(path: Path) -> list[dict[str, float]]:
    """Read and validate one MHD history table."""
    if not path.is_file():
        raise FileNotFoundError(f"MHD validation table not found: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in (reader.fieldnames or ())
        ]
        if missing:
            raise ValueError(
                f"{path} lacks required weighted-morphology columns: "
                + ", ".join(missing)
            )
        rows = [
            {column: float(row[column]) for column in REQUIRED_COLUMNS}
            for row in reader
        ]
    if not rows:
        raise ValueError(f"MHD validation table is empty: {path}")
    times = np.array([row["time"] for row in rows])
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError(
            f"Snapshot times must be finite and strictly increasing: {path}"
        )
    return rows


def write_combined_history(
    histories: list[tuple[str, Path, list[dict[str, float]]]],
    output_path: Path,
) -> None:
    """Write a tidy, long-form table containing every labeled history."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("run_label", "source_csv", *REQUIRED_COLUMNS)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for label, source, rows in histories:
            for row in rows:
                writer.writerow(
                    {"run_label": label, "source_csv": str(source), **row}
                )


def plot_comparison(
    histories: list[tuple[str, Path, list[dict[str, float]]]],
    output_path: Path,
) -> None:
    """Plot stability and continuous morphology diagnostics across runs."""
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for label, _, rows in histories:
        values = {
            column: np.array([row[column] for row in rows])
            for column in REQUIRED_COLUMNS
        }
        times = values["time"]
        magnetic_energy = values["mean_magnetic_energy_density"]
        rms_magnetic_field = np.sqrt(2.0 * magnetic_energy)
        normalized_divergence = (
            values["max_abs_magnetic_divergence"] / rms_magnetic_field
        )
        axes[0, 0].semilogy(
            times,
            np.maximum(normalized_divergence, 1.0e-30),
            marker="o",
            label=label,
        )
        axes[0, 1].plot(
            times,
            magnetic_energy / magnetic_energy[0],
            marker="o",
            label=label,
        )
        line = axes[1, 0].plot(
            times,
            values["pressure_weighted_extent_parallel_to_field"],
            label=f"{label}: parallel",
        )[0]
        axes[1, 0].plot(
            times,
            values["pressure_weighted_extent_perpendicular_to_field"],
            linestyle="--",
            color=line.get_color(),
            label=f"{label}: perpendicular",
        )
        axes[1, 1].plot(
            times,
            values["pressure_weighted_axis_aspect_ratio"],
            marker="o",
            label=label,
        )

    axes[0, 0].set(
        title="Field-normalized magnetic divergence",
        ylabel=r"max $|\nabla\cdot B|/B_{\mathrm{rms}}$ [length$^{-1}$]",
    )
    axes[0, 1].set(
        title="Magnetic-energy evolution",
        ylabel=r"$\langle B^2/2\rangle / \langle B^2/2\rangle_{t=0}$",
    )
    axes[1, 0].set(
        title="Pressure-weighted bubble extents",
        ylabel="RMS extent [code length]",
    )
    axes[1, 1].set(
        title="Pressure-weighted morphology",
        ylabel=r"$R_\parallel/R_\perp$",
    )
    axes[1, 1].axhline(1.0, color="black", linestyle=":", linewidth=1)
    for axis in axes.flat:
        axis.set_xlabel("time [code units]")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("MHD wind-bubble comparison")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    histories = [
        (label, path, read_mhd_history(path)) for label, path in args.run
    ]
    combined_path = args.output_dir / "mhd_comparison.csv"
    figure_path = args.output_dir / "mhd_comparison.png"
    write_combined_history(histories, combined_path)
    plot_comparison(histories, figure_path)
    print(f"Saved combined table: {combined_path.resolve()}")
    print(f"Saved comparison figure: {figure_path.resolve()}")


if __name__ == "__main__":
    main()
