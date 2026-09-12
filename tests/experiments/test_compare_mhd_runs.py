"""Tests for CPU-only MHD validation-history comparison."""

import argparse
import csv
from pathlib import Path
import tempfile
import unittest

from experiments.wind_bubble.compare_mhd_runs import (
    REQUIRED_COLUMNS,
    RUN_MARKERS,
    marker_for_run,
    parse_run_specification,
    read_mhd_history,
    write_combined_history,
)


class MHDComparisonTests(unittest.TestCase):
    def write_history(self, path: Path, times=(0.0, 0.1)) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            for index, time in enumerate(times):
                writer.writerow(
                    {
                        "time": time,
                        "max_abs_magnetic_divergence": 1.0e-7 * index,
                        "mean_magnetic_energy_density": 0.01 + 0.001 * index,
                        "pressure_weighted_extent_parallel_to_field": (
                            0.1 + 0.01 * index
                        ),
                        "pressure_weighted_extent_perpendicular_to_field": 0.1,
                        "pressure_weighted_axis_aspect_ratio": 1.0 + 0.1 * index,
                    }
                )

    def test_parse_run_specification(self):
        label, path = parse_run_specification("beta1=outputs/run.csv")
        self.assertEqual(label, "beta1")
        self.assertEqual(path, Path("outputs/run.csv"))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_run_specification("outputs/run.csv")

    def test_each_run_gets_a_distinct_repeatable_marker(self):
        markers = [marker_for_run(index) for index in range(len(RUN_MARKERS))]
        self.assertEqual(len(markers), len(set(markers)))
        self.assertEqual(marker_for_run(len(RUN_MARKERS)), RUN_MARKERS[0])

    def test_read_and_combine_histories(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "history.csv"
            combined = directory / "combined.csv"
            self.write_history(source)

            rows = read_mhd_history(source)
            self.assertEqual([row["time"] for row in rows], [0.0, 0.1])
            write_combined_history([("beta1", source, rows)], combined)

            with combined.open(newline="") as handle:
                combined_rows = list(csv.DictReader(handle))
            self.assertEqual(len(combined_rows), 2)
            self.assertEqual(combined_rows[0]["run_label"], "beta1")

    def test_rejects_old_schema_without_weighted_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.csv"
            path.write_text("time,max_abs_magnetic_divergence\n0,0\n")
            with self.assertRaisesRegex(ValueError, "weighted-morphology"):
                read_mhd_history(path)


if __name__ == "__main__":
    unittest.main()
