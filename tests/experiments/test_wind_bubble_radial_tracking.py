"""Tests for radial reverse/forward shock-candidate separation."""

import unittest

import numpy as np

from experiments.wind_bubble.run_single_bubble import (
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


if __name__ == "__main__":
    unittest.main()
