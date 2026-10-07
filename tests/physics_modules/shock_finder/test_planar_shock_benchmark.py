"""Tests for exact states and the first planar shock-finder benchmark."""

import unittest

import numpy as np

from experiments.shock.run_planar_shock_benchmark import (
    benchmark_resolution,
    normalized_vector,
    rankine_hugoniot_states,
)


class PlanarShockBenchmarkTests(unittest.TestCase):
    def test_rankine_hugoniot_states_conserve_fluxes(self):
        gamma = 5.0 / 3.0
        states = rankine_hugoniot_states(mach=5.0, gamma=gamma)

        rho_1 = states["upstream_density"]
        pressure_1 = states["upstream_pressure"]
        velocity_1 = states["upstream_velocity"]
        rho_2 = states["downstream_density"]
        pressure_2 = states["downstream_pressure"]
        velocity_2 = states["downstream_velocity"]

        mass_flux_1 = rho_1 * velocity_1
        mass_flux_2 = rho_2 * velocity_2
        momentum_flux_1 = rho_1 * velocity_1**2 + pressure_1
        momentum_flux_2 = rho_2 * velocity_2**2 + pressure_2
        enthalpy_1 = gamma / (gamma - 1.0) * pressure_1 / rho_1
        enthalpy_2 = gamma / (gamma - 1.0) * pressure_2 / rho_2
        energy_per_mass_1 = 0.5 * velocity_1**2 + enthalpy_1
        energy_per_mass_2 = 0.5 * velocity_2**2 + enthalpy_2

        self.assertAlmostEqual(mass_flux_1, mass_flux_2, places=12)
        self.assertAlmostEqual(momentum_flux_1, momentum_flux_2, places=12)
        self.assertAlmostEqual(energy_per_mass_1, energy_per_mass_2, places=12)

    def test_mach_five_planar_snapshot_is_detected(self):
        metrics = benchmark_resolution(
            resolution=24,
            mach=5.0,
            ramp_cells=3.0,
        )

        self.assertGreater(metrics["surface_cell_count"], 0)
        self.assertGreater(metrics["valid_mach_fraction"], 0.99)
        self.assertGreater(metrics["mach_median"], 1.3)
        self.assertLess(metrics["normal_angular_error_degrees"], 1.0e-3)

    def test_face_diagonal_planar_snapshot_is_detected(self):
        metrics = benchmark_resolution(
            resolution=24,
            mach=5.0,
            ramp_cells=3.0,
            normal=(1.0, 1.0, 0.0),
        )

        self.assertGreater(metrics["surface_cell_count"], 0)
        self.assertGreater(metrics["valid_mach_fraction"], 0.99)
        self.assertGreater(metrics["mach_median"], 1.3)
        self.assertLess(metrics["normal_angular_error_degrees"], 1.0)

    def test_normal_is_normalized(self):
        np.testing.assert_allclose(
            normalized_vector((1.0, 1.0, 1.0)),
            np.full(3, 1.0 / np.sqrt(3.0)),
        )

    def test_zero_normal_is_rejected(self):
        with self.assertRaises(ValueError):
            normalized_vector((0.0, 0.0, 0.0))

    def test_invalid_mach_is_rejected(self):
        with self.assertRaises(ValueError):
            rankine_hugoniot_states(mach=1.0)


if __name__ == "__main__":
    unittest.main()
