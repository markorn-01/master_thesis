"""Tests for the memory-conscious wind-bubble convergence helpers."""

import unittest

import numpy as np

from experiments.wind_bubble.run_wind_bubble_convergence import (
    _finite_band_statistics,
    _radial_profile_jump_statistics,
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


class RadialProfileJumpTests(unittest.TestCase):
    def setUp(self):
        self.bin_centers = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
        self.profiles = {
            "density": np.array([1.0, 2.0, 4.0, 6.0, 3.0]),
            "pressure": np.array([1.0, 3.0, 12.0, 15.0, 5.0]),
            "temperature_proxy": np.array([1.0, 1.5, 3.0, 2.5, 1.0]),
            "radial_velocity": np.array([3.0, 2.0, 1.0, 1.5, 0.5]),
            "radial_flow_mach": np.array([2.5, 2.0, 0.5, 0.7, 0.2]),
            "velocity_divergence": np.array(
                [0.0, -4.0, -2.0, -3.0, -1.0]
            ),
        }

    def test_reverse_shock_has_upstream_shell_inside(self):
        result = _radial_profile_jump_statistics(
            shock_radius=0.125,
            bin_centers=self.bin_centers,
            profiles=self.profiles,
            shock_kind="reverse",
        )

        self.assertAlmostEqual(result["upstream_shell_radius"], 0.10)
        self.assertAlmostEqual(result["downstream_shell_radius"], 0.15)
        self.assertAlmostEqual(result["density_ratio"], 2.0)
        self.assertAlmostEqual(result["pressure_ratio"], 4.0)
        self.assertAlmostEqual(result["minimum_divergence"], -4.0)

    def test_forward_shock_has_upstream_shell_outside(self):
        result = _radial_profile_jump_statistics(
            shock_radius=0.225,
            bin_centers=self.bin_centers,
            profiles=self.profiles,
            shock_kind="forward",
        )

        self.assertAlmostEqual(result["upstream_shell_radius"], 0.25)
        self.assertAlmostEqual(result["downstream_shell_radius"], 0.20)
        self.assertAlmostEqual(result["density_ratio"], 2.0)
        self.assertAlmostEqual(result["pressure_ratio"], 3.0)
        self.assertAlmostEqual(result["minimum_divergence"], -3.0)

    def test_missing_shock_returns_nan_metrics(self):
        result = _radial_profile_jump_statistics(
            shock_radius=np.nan,
            bin_centers=self.bin_centers,
            profiles=self.profiles,
            shock_kind="reverse",
        )

        self.assertTrue(np.isnan(result["density_ratio"]))
        self.assertTrue(np.isnan(result["jump_mach"]))


if __name__ == "__main__":
    unittest.main()
