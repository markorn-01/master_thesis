"""Tests for temporal wind-bubble energy convergence aggregation."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.wind_bubble.aggregate_energy_convergence import (
    load_resolution_result,
    summarize_results,
)


FIELDNAMES = [
    "time",
    "reverse_valid_flux_fraction",
    "reverse_surface_area_vs_sphere",
    "reverse_mean_thermal_energy_flux",
    "reverse_dissipation_rate",
    "forward_valid_flux_fraction",
    "forward_surface_area_vs_sphere",
    "forward_mean_thermal_energy_flux",
    "forward_dissipation_rate",
    "reverse_cumulative_dissipated_energy",
    "forward_cumulative_dissipated_energy",
    "combined_cumulative_dissipated_energy",
    "injected_wind_energy",
    "combined_dissipation_to_injected_energy",
]


class EnergyConvergenceAggregationTests(unittest.TestCase):
    def _write_result(self, root: Path, resolution: int, energy: float) -> None:
        result_dir = root / f"single_bubble_energy_n{resolution:03d}"
        result_dir.mkdir()
        rows = []
        for time, scale in ((0.0, 0.0), (0.2, 1.0)):
            rows.append(
                {
                    "time": time,
                    "reverse_valid_flux_fraction": 1.0,
                    "reverse_surface_area_vs_sphere": 1.05,
                    "reverse_mean_thermal_energy_flux": 0.2 * scale,
                    "reverse_dissipation_rate": 0.1 * scale,
                    "forward_valid_flux_fraction": 1.0,
                    "forward_surface_area_vs_sphere": 1.04,
                    "forward_mean_thermal_energy_flux": 0.3 * scale,
                    "forward_dissipation_rate": 0.2 * scale,
                    "reverse_cumulative_dissipated_energy": 0.25 * energy,
                    "forward_cumulative_dissipated_energy": 0.75 * energy,
                    "combined_cumulative_dissipated_energy": energy * scale,
                    "injected_wind_energy": 0.1 * scale,
                    "combined_dissipation_to_injected_energy": (
                        energy / 0.1 if scale else float("nan")
                    ),
                }
            )
        with (result_dir / "shock_energy_histories.csv").open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        (result_dir / "reverse_shock_verification.json").write_text(
            json.dumps(
                {
                    "classification": "consistent_with_reverse_shock",
                    "measurements": {"profile_jump_mach": 2.0},
                }
            ),
            encoding="utf-8",
        )

    def test_summary_is_sorted_and_reports_relative_energy_change(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_result(root, 128, 0.03)
            self._write_result(root, 64, 0.02)
            results = [
                load_resolution_result(128, root),
                load_resolution_result(64, root),
            ]

            summary = summarize_results(results)

        self.assertEqual([row["resolution"] for row in summary], [64, 128])
        self.assertEqual(summary[0]["snapshot_count"], 2)
        self.assertAlmostEqual(
            summary[1]["relative_change_from_previous_resolution"], 0.5
        )

    def test_non_increasing_snapshot_times_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_result(root, 64, 0.02)
            path = (
                root
                / "single_bubble_energy_n064"
                / "shock_energy_histories.csv"
            )
            with path.open(encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            rows[1]["time"] = rows[0]["time"]
            with path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "finite and increasing"):
                load_resolution_result(64, root)


if __name__ == "__main__":
    unittest.main()
