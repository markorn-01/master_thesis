"""Post-process a wind-bubble shock history into a Mach diagnostic.

This utility reuses completed simulations.  It reads the current and
profile-aware pressure-jump Mach histories, derives the tracked-speed Mach and
the corresponding ideal normal-shock pressure ratios, and writes a compact
table and comparison figure without rerunning the hydrodynamics.
"""

import argparse
import csv
from pathlib import Path

from experiments.wind_bubble.run_single_bubble import (
    _forward_mach_diagnostic_values,
    write_forward_mach_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history-csv",
        type=Path,
        required=True,
        help="Existing shock_histories.csv containing both Mach estimators.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: the input CSV directory).",
    )
    return parser.parse_args()


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def read_history_rows(path: Path) -> list[dict]:
    """Read only the fields needed for the compact diagnostic."""
    rows = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        for source in csv.DictReader(csv_file):
            current_mach = float(source["forward_mach_median"])
            profile_mach = float(
                source["forward_profile_aware_mach_median"]
            )
            weaver_mach = float(source["weaver_forward_mach"])
            radial_velocity = float(source["forward_radial_velocity"])
            rows.append(
                {
                    "time": float(source["time"]),
                    "resolved_for_weaver": _parse_bool(
                        source["resolved_for_weaver"]
                    ),
                    "forward_confidence_label": source[
                        "forward_confidence_label"
                    ],
                    "forward_radius_median": float(
                        source["forward_radius_median"]
                    ),
                    "weaver_outer_radius": float(
                        source["weaver_outer_radius"]
                    ),
                    "relative_error_vs_weaver": float(
                        source["relative_error_vs_weaver"]
                    ),
                    "forward_mach_median": current_mach,
                    "forward_profile_aware_mach_median": profile_mach,
                    "weaver_forward_mach": weaver_mach,
                    **_forward_mach_diagnostic_values(
                        radial_velocity=radial_velocity,
                        current_mach=current_mach,
                        profile_aware_mach=profile_mach,
                        weaver_mach=weaver_mach,
                    ),
                }
            )
    if not rows:
        raise ValueError(f"No history rows found in {path}.")
    return rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.history_csv.parent
    rows = read_history_rows(args.history_csv)
    csv_path = output_dir / "forward_mach_diagnostics.csv"
    plot_path = output_dir / "forward_mach_diagnostics.png"
    write_forward_mach_diagnostics(
        history_rows=rows,
        csv_path=csv_path,
        plot_path=plot_path,
    )
    print(f"Saved Mach diagnostic: {plot_path.resolve()}")
    print(f"Saved Mach table     : {csv_path.resolve()}")


if __name__ == "__main__":
    main()
