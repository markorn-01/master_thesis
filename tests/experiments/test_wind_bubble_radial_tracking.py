"""Tests for radial reverse/forward shock-candidate separation."""

import unittest

import numpy as np

from experiments.wind_bubble.run_single_bubble import (
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
        reverse, forward, separation = self.split(
            np.concatenate((outer, inner))
        )
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
            result["diagnostic_checks"][
                "adaptive_finder_mach_is_consistent"
            ]
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


if __name__ == "__main__":
    unittest.main()
