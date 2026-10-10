"""Compare controlled forward-shock Mach diagnostics across resolutions.

Example
-------
python -m experiments.wind_bubble.compare_forward_mach_resolutions \
  --run n064=outputs/run_n064/forward_mach_diagnostics.csv \
  --run n128=outputs/run_n128/forward_mach_diagnostics.csv \
  --output-dir outputs/forward_mach_resolution_comparison
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.wind_bubble.compare_mhd_runs import (
    parse_run_specification,
)


REQUIRED_COLUMNS = (
    "time",
    "resolved_for_weaver",
    "relative_error_vs_weaver",
    "forward_mach_median",
    "forward_profile_aware_mach_median",
    "forward_kinematic_mach",
    "weaver_forward_mach",
    "forward_current_sampled_pressure_ratio",
    "forward_profile_aware_sampled_pressure_ratio",
    "forward_kinematic_expected_pressure_ratio",
)
SUMMARY_COLUMNS = (
    "run_label",
    "source_csv",
    "matched_start_time",
    "matched_end_time",
    "sample_count",
    "mean_radius_relative_error_vs_weaver",
    "mean_current_mach_relative_error_vs_weaver",
    "mean_profile_aware_mach_relative_error_vs_weaver",
    "mean_kinematic_mach_relative_error_vs_weaver",
    "mean_profile_aware_mach_relative_error_vs_kinematic",
    "mean_current_pressure_ratio_relative_error_vs_kinematic",
    "mean_profile_pressure_ratio_relative_error_vs_kinematic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_specification,
        required=True,
        metavar="LABEL=CSV_PATH",
        help="Labeled forward_mach_diagnostics.csv; provide at least two.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) < 2:
        parser.error("provide at least two --run inputs")
    labels = [label for label, _ in args.run]
    if len(labels) != len(set(labels)):
        parser.error("--run labels must be unique")
    return args


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def read_diagnostic(path: Path) -> list[dict[str, float | bool]]:
    """Read and validate one forward-shock diagnostic history."""
    if not path.is_file():
        raise FileNotFoundError(f"Forward Mach diagnostic not found: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in (reader.fieldnames or ())
        ]
        if missing:
            raise ValueError(
                f"{path} lacks required diagnostic columns: "
                + ", ".join(missing)
            )
        rows = []
        for source in reader:
            row: dict[str, float | bool] = {
                column: float(source[column])
                for column in REQUIRED_COLUMNS
                if column != "resolved_for_weaver"
            }
            row["resolved_for_weaver"] = _parse_bool(
                source["resolved_for_weaver"]
            )
            rows.append(row)
    if not rows:
        raise ValueError(f"Forward Mach diagnostic is empty: {path}")
    times = np.asarray([float(row["time"]) for row in rows])
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError(
            f"Snapshot times must be finite and strictly increasing: {path}"
        )
    if not any(bool(row["resolved_for_weaver"]) for row in rows):
        raise ValueError(f"No Weaver-resolved samples found: {path}")
    return rows


def matched_time_window(
    histories: list[tuple[str, Path, list[dict[str, float | bool]]]],
) -> tuple[float, float]:
    """Return the common time interval covered by resolved samples."""
    starts = []
    ends = []
    for _, _, rows in histories:
        resolved_times = [
            float(row["time"])
            for row in rows
            if bool(row["resolved_for_weaver"])
        ]
        starts.append(min(resolved_times))
        ends.append(max(resolved_times))
    start = max(starts)
    end = min(ends)
    if start > end:
        raise ValueError("Resolved histories do not overlap in time.")
    return start, end


def select_matched_rows(
    rows: list[dict[str, float | bool]],
    start: float,
    end: float,
) -> list[dict[str, float | bool]]:
    """Select resolved rows inside a common time interval.

    Snapshot times can overshoot their requested output cadence slightly.  A
    small fraction of the run's median cadence keeps equivalent boundary
    snapshots together without admitting the preceding nominal snapshot.
    """
    times = np.asarray([float(row["time"]) for row in rows])
    cadence = np.median(np.diff(times)) if times.size > 1 else 0.0
    tolerance = max(1.0e-12, 0.1 * float(cadence))
    return [
        row
        for row in rows
        if bool(row["resolved_for_weaver"])
        and float(row["time"]) >= start - tolerance
        and float(row["time"]) <= end + tolerance
    ]


def _mean_relative_difference(
    rows: list[dict[str, float | bool]],
    value_key: str,
    reference_key: str,
) -> float:
    values = np.asarray([float(row[value_key]) for row in rows])
    references = np.asarray([float(row[reference_key]) for row in rows])
    valid = np.isfinite(values) & np.isfinite(references) & (references != 0.0)
    return float(np.mean((values[valid] - references[valid]) / references[valid]))


def summarize_histories(
    histories: list[tuple[str, Path, list[dict[str, float | bool]]]],
) -> list[dict[str, str | float | int]]:
    """Summarize each run over the common resolved time interval."""
    start, end = matched_time_window(histories)
    summaries = []
    for label, source, rows in histories:
        matched = select_matched_rows(rows, start, end)
        summaries.append(
            {
                "run_label": label,
                "source_csv": str(source),
                "matched_start_time": start,
                "matched_end_time": end,
                "sample_count": len(matched),
                "mean_radius_relative_error_vs_weaver": float(
                    np.mean(
                        [
                            float(row["relative_error_vs_weaver"])
                            for row in matched
                        ]
                    )
                ),
                "mean_current_mach_relative_error_vs_weaver": (
                    _mean_relative_difference(
                        matched,
                        "forward_mach_median",
                        "weaver_forward_mach",
                    )
                ),
                "mean_profile_aware_mach_relative_error_vs_weaver": (
                    _mean_relative_difference(
                        matched,
                        "forward_profile_aware_mach_median",
                        "weaver_forward_mach",
                    )
                ),
                "mean_kinematic_mach_relative_error_vs_weaver": (
                    _mean_relative_difference(
                        matched,
                        "forward_kinematic_mach",
                        "weaver_forward_mach",
                    )
                ),
                "mean_profile_aware_mach_relative_error_vs_kinematic": (
                    _mean_relative_difference(
                        matched,
                        "forward_profile_aware_mach_median",
                        "forward_kinematic_mach",
                    )
                ),
                "mean_current_pressure_ratio_relative_error_vs_kinematic": (
                    _mean_relative_difference(
                        matched,
                        "forward_current_sampled_pressure_ratio",
                        "forward_kinematic_expected_pressure_ratio",
                    )
                ),
                "mean_profile_pressure_ratio_relative_error_vs_kinematic": (
                    _mean_relative_difference(
                        matched,
                        "forward_profile_aware_sampled_pressure_ratio",
                        "forward_kinematic_expected_pressure_ratio",
                    )
                ),
            }
        )
    return summaries


def write_summary(
    summaries: list[dict[str, str | float | int]], output_path: Path
) -> None:
    """Write one row per resolution using a stable, machine-readable schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=SUMMARY_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summaries)


def plot_comparison(
    histories: list[tuple[str, Path, list[dict[str, float | bool]]]],
    output_path: Path,
) -> None:
    """Plot estimator and resolution effects over the common time interval."""
    start, end = matched_time_window(histories)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, (label, _, rows) in enumerate(histories):
        matched = select_matched_rows(rows, start, end)
        values = {
            column: np.asarray([float(row[column]) for row in matched])
            for column in REQUIRED_COLUMNS
            if column != "resolved_for_weaver"
        }
        color = colors[index % len(colors)]
        times = values["time"]
        axes[0, 0].plot(
            times,
            values["forward_mach_median"],
            color=color,
            marker="o",
            label=label,
        )
        axes[0, 1].plot(
            times,
            values["forward_profile_aware_mach_median"],
            color=color,
            marker="s",
            label=label,
        )
        axes[1, 0].plot(
            times,
            values["forward_kinematic_mach"],
            color=color,
            marker="^",
            label=label,
        )
        profile_mach_error = (
            values["forward_profile_aware_mach_median"]
            / values["forward_kinematic_mach"]
            - 1.0
        )
        profile_pressure_error = (
            values["forward_profile_aware_sampled_pressure_ratio"]
            / values["forward_kinematic_expected_pressure_ratio"]
            - 1.0
        )
        axes[1, 1].plot(
            times,
            100.0 * profile_mach_error,
            color=color,
            marker="s",
            label=f"{label}: Mach",
        )
        axes[1, 1].plot(
            times,
            100.0 * profile_pressure_error,
            color=color,
            marker="x",
            linestyle="--",
            label=f"{label}: pressure jump",
        )

    reference_rows = select_matched_rows(histories[-1][2], start, end)
    reference_times = np.asarray(
        [float(row["time"]) for row in reference_rows]
    )
    reference_mach = np.asarray(
        [float(row["weaver_forward_mach"]) for row in reference_rows]
    )
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.plot(
            reference_times,
            reference_mach,
            color="black",
            linestyle=":",
            label="Weaver",
        )
        axis.set_ylabel("Mach number")
        axis.legend(fontsize=8)
    axes[0, 0].set_title("Current pressure-jump estimator")
    axes[0, 1].set_title("Profile-aware pressure-jump estimator")
    axes[1, 0].set_title("Tracked propagation-speed Mach")
    axes[1, 1].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[1, 1].set(
        title="Profile-aware deficit relative to tracked speed",
        ylabel="relative difference [%]",
    )
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel("time [code units]")
        axis.grid(alpha=0.25)
    figure.suptitle("Controlled forward-shock resolution comparison")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    histories = [
        (label, path, read_diagnostic(path)) for label, path in args.run
    ]
    summaries = summarize_histories(histories)
    summary_path = args.output_dir / "forward_mach_resolution_summary.csv"
    figure_path = args.output_dir / "forward_mach_resolution_comparison.png"
    write_summary(summaries, summary_path)
    plot_comparison(histories, figure_path)
    print(f"Saved resolution summary: {summary_path.resolve()}")
    print(f"Saved comparison figure : {figure_path.resolve()}")


if __name__ == "__main__":
    main()
