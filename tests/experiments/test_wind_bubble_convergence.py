"""Tests for the memory-conscious wind-bubble convergence helpers."""

import unittest

import numpy as np

from experiments.wind_bubble.run_wind_bubble_convergence import (
    _finite_band_statistics,
    injection_cells_for_resolution,
)


class InjectionScalingTests(unittest.TestCase):
    def test_fixed_physical_radius_scales_injection_cells(self):
        expected = {64: 4, 128: 8, 256: 16, 512: 32}
        actual = {
            resolution: injection_cells_for_resolution(resolution)
            for resolution in expected
        }
        self.assertEqual(actual, expected)

    def test_nonrepresentable_radius_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not an integer number"):
            injection_cells_for_resolution(64, injection_radius=0.06)


class BandStatisticsTests(unittest.TestCase):
    def test_statistics_use_only_the_selected_radial_band(self):
        surface_radii = np.array([0.09, 0.10, 0.27, 0.28, 0.29])
        surface_mach = np.array([1.6, 1.8, 5.0, 6.0, 7.0])
        alignment = np.array([-0.9, -1.0, 0.9, 1.0, 0.95])

        result = _finite_band_statistics(
            band_radii=surface_radii[:2],
            surface_radii=surface_radii,
            surface_mach=surface_mach,
            radial_alignment=alignment,
        )

        self.assertEqual(result["surface_cell_count"], 2)
        self.assertAlmostEqual(result["radius_median"], 0.095)
        self.assertAlmostEqual(result["mach_median"], 1.7)
        self.assertAlmostEqual(result["median_radial_alignment"], -0.95)
        self.assertEqual(result["valid_mach_fraction"], 1.0)

    def test_empty_band_returns_nan_metrics(self):
        result = _finite_band_statistics(
            band_radii=np.array([]),
            surface_radii=np.array([0.2]),
            surface_mach=np.array([2.0]),
            radial_alignment=np.array([1.0]),
        )

        self.assertEqual(result["surface_cell_count"], 0)
        self.assertTrue(np.isnan(result["radius_median"]))
        self.assertTrue(np.isnan(result["mach_median"]))


if __name__ == "__main__":
    unittest.main()
