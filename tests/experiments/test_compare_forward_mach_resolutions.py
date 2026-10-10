"""Tests for controlled forward-shock Mach resolution comparisons."""

import csv
from pathlib import Path
import tempfile
import unittest

from experiments.wind_bubble.compare_forward_mach_resolutions import (
    REQUIRED_COLUMNS,
    matched_time_window,
    plot_comparison,
    read_diagnostic,
    summarize_histories,
    write_summary,
)


class ForwardMachResolutionComparisonTests(unittest.TestCase):
    def write_diagnostic(
        self,
        path: Path,
        resolved_times: tuple[float, ...],
        scale: float,
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            for time in resolved_times:
                weaver = 10.0 / (1.0 + time)
                kinematic = scale * weaver
                profile = 0.9 * kinematic
                current = 0.8 * kinematic
                writer.writerow(
                    {
                        "time": time,
                        "resolved_for_weaver": True,
                        "relative_error_vs_weaver": scale - 1.0,
                        "forward_mach_median": current,
                        "forward_profile_aware_mach_median": profile,
                        "forward_kinematic_mach": kinematic,
                        "weaver_forward_mach": weaver,
                        "forward_current_sampled_pressure_ratio": 0.7,
                        "forward_profile_aware_sampled_pressure_ratio": 0.8,
                        "forward_kinematic_expected_pressure_ratio": 1.0,
                    }
                )

    def test_summarizes_only_common_resolved_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            coarse_path = directory / "coarse.csv"
            fine_path = directory / "fine.csv"
            self.write_diagnostic(coarse_path, (0.04, 0.10, 0.20), 0.95)
            self.write_diagnostic(fine_path, (0.02, 0.10, 0.20), 0.99)
            histories = [
                ("n064", coarse_path, read_diagnostic(coarse_path)),
                ("n128", fine_path, read_diagnostic(fine_path)),
            ]

            self.assertEqual(matched_time_window(histories), (0.04, 0.2))
            summaries = summarize_histories(histories)

            self.assertEqual(summaries[0]["sample_count"], 3)
            self.assertEqual(summaries[1]["sample_count"], 2)
            self.assertAlmostEqual(
                summaries[0]["mean_kinematic_mach_relative_error_vs_weaver"],
                -0.05,
            )
            self.assertAlmostEqual(
                summaries[1][
                    "mean_profile_aware_mach_relative_error_vs_kinematic"
                ],
                -0.1,
            )

            summary_path = directory / "summary.csv"
            figure_path = directory / "comparison.png"
            write_summary(summaries, summary_path)
            plot_comparison(histories, figure_path)
            self.assertTrue(summary_path.is_file())
            self.assertTrue(figure_path.is_file())

    def test_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            path.write_text("time,resolved_for_weaver\n0.1,True\n")

            with self.assertRaisesRegex(ValueError, "required diagnostic"):
                read_diagnostic(path)


if __name__ == "__main__":
    unittest.main()
