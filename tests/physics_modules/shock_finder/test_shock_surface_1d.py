"""Regression for the final contiguous zone skipped by the 1D selector."""
import unittest

import jax.numpy as jnp
import numpy as np

from astronomix._physics_modules._shock_finder._shock_surface import _find_shock_surface_1d


class OneDimensionalSurfaceTests(unittest.TestCase):
    def select(self, divergence, zones):
        return np.flatnonzero(np.asarray(_find_shock_surface_1d(
            jnp.asarray(divergence, dtype=jnp.float32), jnp.asarray(zones, dtype=bool))))

    def test_single_zone_is_not_skipped(self):
        np.testing.assert_array_equal(self.select([0, -1, -3, -1, 0], [0, 1, 1, 1, 0]), [2])

    def test_every_zone_including_last_has_one_compression_minimum(self):
        np.testing.assert_array_equal(self.select([0, -2, -4, 0, -5, -3, 0],
                                                 [0, 1, 1, 0, 1, 1, 0]), [2, 4])

    def test_empty_zone_does_not_create_a_surface(self):
        self.assertEqual(len(self.select([0, -5, 0, 0], [0, 0, 0, 0])), 0)

    def test_outside_zone_minimum_is_not_selected_and_tie_keeps_one_cell(self):
        np.testing.assert_array_equal(self.select([-100, -3, -3, -100], [0, 1, 1, 0]), [1])


if __name__ == '__main__':
    unittest.main()
