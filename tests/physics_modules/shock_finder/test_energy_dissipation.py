"""Tests for the Mach-based local thermal-energy-flux calculation."""

import unittest

import jax.numpy as jnp
import numpy as np

from astronomix._physics_modules._shock_finder._energy_dissipation import (
    _thermalization_efficiency,
)


class ThermalizationEfficiencyTests(unittest.TestCase):
    def test_subsonic_and_sonic_values_have_zero_efficiency(self):
        efficiency = _thermalization_efficiency(
            jnp.array([0.0, 0.8, 1.0]),
            gamma=5.0 / 3.0,
        )

        np.testing.assert_array_equal(np.asarray(efficiency), np.zeros(3))

    def test_supersonic_efficiency_is_finite_and_non_negative(self):
        efficiency = np.asarray(
            _thermalization_efficiency(
                jnp.array([1.3, 2.0, 5.0, 10.0]),
                gamma=5.0 / 3.0,
            )
        )

        self.assertTrue(np.isfinite(efficiency).all())
        self.assertTrue((efficiency >= 0.0).all())
        self.assertTrue((np.diff(efficiency) > 0.0).all())

    def test_strong_shock_limit_for_gamma_five_thirds(self):
        efficiency = float(
            _thermalization_efficiency(
                jnp.array(1.0e4),
                gamma=5.0 / 3.0,
            )
        )

        self.assertAlmostEqual(efficiency, 9.0 / 16.0, places=5)


if __name__ == "__main__":
    unittest.main()
