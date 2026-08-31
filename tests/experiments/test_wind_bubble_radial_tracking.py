"""Tests for radial reverse/forward shock-candidate separation."""

import unittest

import numpy as np

from experiments.wind_bubble.run_single_bubble import (
    _cumulative_trapezoid_over_detections,
    _radial_band_statistics,
    _shock_tracking_confidence,
    _surface_area_weights,
    _surface_dissipation_statistics,
    _temporal_tracking_diagnostics,
    _weaver_forward_shock_mach,
    classify_reverse_shock_evidence,
    split_radial_shock_candidates,
)


class RadialShockCandidateTests(unittest.TestCase):
    def setUp(self):
        self.injection_radius = 0.0625
        self.grid_spacing = 0.015625

    def split(self, radii):
        return split_radial_shock_candidates(
            np.asarray(radii),
            injection_radius=self.injection_radius,
            grid_spacing=self.grid_spacing,
        )

    def test_empty_detection_returns_two_empty_bands(self):
        reverse, forward, separation = self.split([])
        self.assertEqual(reverse.size, 0)
        self.assertEqual(forward.size, 0)
        self.assertTrue(np.isnan(separation))

    def test_single_outer_band_is_forward_shock(self):
        outer = np.linspace(0.27, 0.29, 100)
        reverse, forward, separation = self.split(outer)
        self.assertEqual(reverse.size, 0)
        np.testing.assert_allclose(forward, outer)
        self.assertTrue(np.isnan(separation))

    def test_two_well_separated_bands_are_labelled_by_radius(self):
        inner = np.linspace(0.085, 0.095, 40)
        outer = np.linspace(0.27, 0.29, 160)
        reverse, forward, separation = self.split(np.concatenate((outer, inner)))
        np.testing.assert_allclose(reverse, inner)
        np.testing.assert_allclose(forward, outer)
        self.assertGreater(separation, inner.max())
        self.assertLess(separation, outer.min())

    def test_isolated_inner_detections_do_not_create_reverse_shock(self):
        isolated_inner = np.array([0.085, 0.087])
        outer = np.linspace(0.27, 0.29, 100)
        reverse, forward, separation = self.split(
            np.concatenate((isolated_inner, outer))
        )
        self.assertEqual(reverse.size, 0)
        self.assertEqual(forward.size, 102)
        self.assertTrue(np.isnan(separation))


class ReverseShockVerificationTests(unittest.TestCase):
    def setUp(self):
        self.measurements = {
            "upstream_flow_mach": 2.0,
            "downstream_flow_mach": 0.5,
            "density_ratio": 2.0,
            "pressure_ratio": 4.0,
            "temperature_ratio": 2.0,
            "velocity_ratio": 0.4,
            "minimum_divergence": -10.0,
            "peak_surface_mach": 2.1,
            "profile_jump_mach": 1.8,
            "median_radial_normal_alignment": -0.95,
        }

    def classify(self, **overrides):
        measurements = {**self.measurements, **overrides}
        return classify_reverse_shock_evidence(
            measurements,
            persistent_detection=True,
            resolved_from_injection=True,
        )

    def test_complete_reverse_shock_signature_is_verified(self):
        result = self.classify()

        self.assertTrue(result["verified"])
        self.assertEqual(
            result["classification"],
            "consistent_with_reverse_shock",
        )
        self.assertTrue(all(result["criteria"].values()))

    def test_invalid_adaptive_mach_is_reported_but_not_misclassified(self):
        result = self.classify(peak_surface_mach=1.0)

        self.assertTrue(result["verified"])
        self.assertFalse(
            result["diagnostic_checks"]["adaptive_finder_mach_is_consistent"]
        )
        self.assertTrue(result["limitations"])

    def test_outward_normal_rejects_reverse_shock_classification(self):
        result = self.classify(median_radial_normal_alignment=0.95)

        self.assertFalse(result["verified"])
        self.assertFalse(result["criteria"]["shock_normal_points_inward"])

    def test_subsonic_upstream_rejects_reverse_shock_classification(self):
        result = self.classify(upstream_flow_mach=0.8)

        self.assertFalse(result["verified"])
        self.assertFalse(result["criteria"]["upstream_flow_is_supersonic"])


class TemporalShockTrackingTests(unittest.TestCase):
    def setUp(self):
        self.grid_spacing = 0.015625

    def test_band_statistics_include_mach_spread_and_alignment(self):
        radii = np.array([0.10, 0.11, 0.12, 0.28, 0.29])
        mach = np.array([1.8, 2.0, np.nan, 5.8, 6.0])
        alignment = np.array([-0.99, -0.97, -0.98, 0.98, 0.99])

        result = _radial_band_statistics(
            band_radii=radii[:3],
            all_surface_radii=radii,
            all_surface_mach=mach,
            all_radial_alignment=alignment,
        )

        self.assertAlmostEqual(result["radius_median"], 0.11)
        self.assertAlmostEqual(result["valid_mach_fraction"], 2.0 / 3.0)
        self.assertAlmostEqual(result["mach_median"], 1.9)
        self.assertLess(result["median_radial_alignment"], -0.95)
        self.assertGreater(result["normalized_radial_spread"], 0.0)

    def test_missing_track_is_reacquired_without_changing_identity(self):
        initialized = _temporal_tracking_diagnostics(
            current_radius=0.10,
            previous_radius=np.nan,
            current_radius_uncertainty=0.005,
            previous_radius_uncertainty=np.nan,
            current_time=0.10,
            previous_time=np.nan,
            previous_radial_velocity=np.nan,
            snapshots_since_detection=1,
            grid_spacing=self.grid_spacing,
        )
        missing = _temporal_tracking_diagnostics(
            current_radius=np.nan,
            previous_radius=0.10,
            current_radius_uncertainty=np.nan,
            previous_radius_uncertainty=0.005,
            current_time=0.20,
            previous_time=0.10,
            previous_radial_velocity=np.nan,
            snapshots_since_detection=1,
            grid_spacing=self.grid_spacing,
        )
        reacquired = _temporal_tracking_diagnostics(
            current_radius=0.12,
            previous_radius=0.10,
            current_radius_uncertainty=0.005,
            previous_radius_uncertainty=0.005,
            current_time=0.30,
            previous_time=0.10,
            previous_radial_velocity=np.nan,
            snapshots_since_detection=2,
            grid_spacing=self.grid_spacing,
        )

        self.assertEqual(initialized["track_status"], "initialized")
        self.assertEqual(missing["track_status"], "missing")
        self.assertEqual(reacquired["track_status"], "reacquired")
        self.assertTrue(reacquired["continuity_ok"])

    def test_implausible_radius_jump_is_flagged(self):
        result = _temporal_tracking_diagnostics(
            current_radius=0.40,
            previous_radius=0.10,
            current_radius_uncertainty=0.005,
            previous_radius_uncertainty=0.005,
            current_time=0.20,
            previous_time=0.10,
            previous_radial_velocity=0.5,
            snapshots_since_detection=1,
            grid_spacing=self.grid_spacing,
        )

        self.assertEqual(result["track_status"], "discontinuous")
        self.assertFalse(result["continuity_ok"])

    def test_constant_velocity_prediction_is_independent_of_snapshot_spacing(self):
        short_interval = _temporal_tracking_diagnostics(
            current_radius=0.15,
            previous_radius=0.10,
            current_radius_uncertainty=0.005,
            previous_radius_uncertainty=0.005,
            current_time=0.20,
            previous_time=0.10,
            previous_radial_velocity=0.5,
            snapshots_since_detection=1,
            grid_spacing=self.grid_spacing,
        )
        long_interval = _temporal_tracking_diagnostics(
            current_radius=0.20,
            previous_radius=0.10,
            current_radius_uncertainty=0.005,
            previous_radius_uncertainty=0.005,
            current_time=0.30,
            previous_time=0.10,
            previous_radial_velocity=0.5,
            snapshots_since_detection=2,
            grid_spacing=self.grid_spacing,
        )

        self.assertTrue(short_interval["continuity_ok"])
        self.assertTrue(long_interval["continuity_ok"])
        self.assertAlmostEqual(short_interval["prediction_residual"], 0.0)
        self.assertAlmostEqual(long_interval["prediction_residual"], 0.0)
        self.assertAlmostEqual(short_interval["radial_velocity"], 0.5)
        self.assertAlmostEqual(long_interval["radial_velocity"], 0.5)

    def test_adiabatic_weaver_forward_mach_uses_similarity_speed(self):
        mach = _weaver_forward_shock_mach(
            radius=0.291671956,
            time=0.2,
            ambient_density=1.0,
            ambient_pressure=0.01,
            gamma=5.0 / 3.0,
        )

        self.assertAlmostEqual(mach, 6.7778437, places=6)

    def test_reverse_shock_quality_checks_use_inward_normal(self):
        statistics = {
            "radius_median": 0.11,
            "normalized_radial_spread": 0.05,
            "surface_cell_count": 100,
            "valid_mach_fraction": 1.0,
            "median_radial_alignment": -0.99,
        }
        tracking = {
            "detected": True,
            "continuity_ok": True,
        }
        result = _shock_tracking_confidence(
            shock_kind="reverse",
            statistics=statistics,
            tracking=tracking,
            injection_radius=0.0625,
            grid_spacing=self.grid_spacing,
            ordering_ok=True,
        )

        self.assertTrue(result["normal_orientation_ok"])
        self.assertTrue(result["resolved_from_injection"])
        self.assertEqual(result["confidence_label"], "high")

    def test_forward_shock_rejects_inward_normal(self):
        statistics = {
            "radius_median": 0.28,
            "normalized_radial_spread": 0.05,
            "surface_cell_count": 100,
            "valid_mach_fraction": 1.0,
            "median_radial_alignment": -0.99,
        }
        tracking = {
            "detected": True,
            "continuity_ok": True,
        }
        result = _shock_tracking_confidence(
            shock_kind="forward",
            statistics=statistics,
            tracking=tracking,
            injection_radius=0.0625,
            grid_spacing=self.grid_spacing,
            ordering_ok=True,
        )

        self.assertFalse(result["normal_orientation_ok"])
        self.assertLess(result["confidence_score"], 1.0)


class ShockEnergyIntegrationTests(unittest.TestCase):
    def test_axis_aligned_surface_weights_equal_grid_face_area(self):
        directions = np.tile([1.0, 0.0, 0.0], (6, 1))

        weights = _surface_area_weights(directions, grid_spacing=0.25)

        np.testing.assert_allclose(weights, np.full(6, 0.25**2))

    def test_oblique_surface_weights_include_projection_correction(self):
        direction = np.array([[1.0, 1.0, 1.0]]) / np.sqrt(3.0)

        weights = _surface_area_weights(direction, grid_spacing=0.5)

        np.testing.assert_allclose(weights, [0.5**2 * np.sqrt(3.0)])

    def test_duplicate_projected_surface_patch_is_counted_once(self):
        directions = np.tile([1.0, 0.0, 0.0], (2, 1))
        indices = np.array([[4, 2, 3], [5, 2, 3]])

        weights = _surface_area_weights(
            directions,
            grid_spacing=0.25,
            surface_indices=indices,
        )

        self.assertAlmostEqual(weights[0], 0.25**2)
        self.assertTrue(np.isnan(weights[1]))

    def test_surface_flux_is_integrated_with_area_weights(self):
        flux = np.array([2.0, 3.0, 100.0])
        directions = np.tile([1.0, 0.0, 0.0], (3, 1))
        selection = np.array([True, True, False])

        result = _surface_dissipation_statistics(
            surface_flux=flux,
            surface_direction=directions,
            grid_spacing=0.5,
            selection=selection,
            radius_median=1.0,
        )

        self.assertEqual(result["surface_sample_count"], 2)
        self.assertEqual(result["valid_flux_sample_count"], 2)
        self.assertAlmostEqual(result["surface_area"], 0.5)
        self.assertAlmostEqual(result["mean_thermal_energy_flux"], 2.5)
        self.assertAlmostEqual(result["dissipation_rate"], 1.25)

    def test_missing_flux_is_reported_as_incomplete_coverage(self):
        result = _surface_dissipation_statistics(
            surface_flux=np.array([2.0, 0.0]),
            surface_direction=np.array([[1.0, 0.0, 0.0]] * 2),
            grid_spacing=0.5,
            selection=np.array([True, True]),
            radius_median=1.0,
        )

        self.assertAlmostEqual(result["valid_flux_fraction"], 0.5)
        self.assertAlmostEqual(result["surface_area"], 0.5)
        self.assertAlmostEqual(result["flux_covered_surface_area"], 0.25)
        self.assertAlmostEqual(result["dissipation_rate"], 0.5)

    def test_cumulative_energy_does_not_bridge_missing_detections(self):
        cumulative, interval_valid = _cumulative_trapezoid_over_detections(
            times=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            rates=np.array([np.nan, 2.0, 4.0, np.nan, 8.0]),
        )

        np.testing.assert_allclose(
            cumulative[1:],
            np.array([0.0, 3.0, 3.0, 3.0]),
        )
        np.testing.assert_array_equal(
            interval_valid,
            np.array([False, False, True, False, False]),
        )


if __name__ == "__main__":
    unittest.main()
