import unittest

import jax
import jax.numpy as jnp
import numpy as np

from astronomix._physics_modules._shock_finder._mhd_shock import (
    calculate_fast_magnetosonic_speed,
    calculate_fast_shock_diagnostics,
    calculate_field_obliquity_degrees,
    estimate_normal_shock_speed,
)


class TestMHDShockDiagnostics(unittest.TestCase):
    def test_field_obliquity_parallel_antiparallel_and_perpendicular(self):
        normals = jnp.array(
            [
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        magnetic = jnp.array(
            [
                [2.0, -2.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
            ]
        )

        angles = calculate_field_obliquity_degrees(magnetic, normals)

        np.testing.assert_allclose(angles, np.array([0.0, 0.0, 90.0]))

    def test_zero_magnetic_field_has_undefined_obliquity(self):
        angle = calculate_field_obliquity_degrees(
            jnp.zeros((3,)),
            jnp.array([1.0, 0.0, 0.0]),
        )

        self.assertTrue(np.isnan(float(angle)))

    def test_fast_speed_reduces_to_sound_speed_without_magnetic_field(self):
        gamma = 5.0 / 3.0
        pressure = 1.0 / gamma

        speed = calculate_fast_magnetosonic_speed(
            density=jnp.array(1.0),
            pressure=jnp.array(pressure),
            magnetic_field=jnp.zeros((3,)),
            shock_direction=jnp.array([1.0, 0.0, 0.0]),
            gamma_gas=gamma,
        )

        self.assertAlmostEqual(float(speed), 1.0, places=6)

    def test_parallel_and_perpendicular_fast_speed_limits(self):
        gamma = 5.0 / 3.0
        pressure = 1.0 / gamma
        magnetic = jnp.array([2.0, 0.0, 0.0])

        parallel = calculate_fast_magnetosonic_speed(
            1.0,
            pressure,
            magnetic,
            jnp.array([1.0, 0.0, 0.0]),
            gamma_gas=gamma,
        )
        perpendicular = calculate_fast_magnetosonic_speed(
            1.0,
            pressure,
            magnetic,
            jnp.array([0.0, 1.0, 0.0]),
            gamma_gas=gamma,
        )

        self.assertAlmostEqual(float(parallel), 2.0, places=6)
        self.assertAlmostEqual(float(perpendicular), np.sqrt(5.0), places=6)

    def test_fast_speed_is_rotation_invariant(self):
        gamma = 5.0 / 3.0
        pressure = 1.0 / gamma
        diagonal = 1.0 / np.sqrt(2.0)

        axis_aligned = calculate_fast_magnetosonic_speed(
            1.0,
            pressure,
            jnp.array([2.0, 0.0, 0.0]),
            jnp.array([1.0, 0.0, 0.0]),
            gamma_gas=gamma,
        )
        rotated = calculate_fast_magnetosonic_speed(
            1.0,
            pressure,
            jnp.array([np.sqrt(2.0), np.sqrt(2.0), 0.0]),
            jnp.array([diagonal, diagonal, 0.0]),
            gamma_gas=gamma,
        )

        self.assertAlmostEqual(float(axis_aligned), float(rotated), places=6)

    def test_normal_shock_speed_uses_mass_conservation(self):
        speed, valid = estimate_normal_shock_speed(
            density_upstream=1.0,
            density_downstream=2.0,
            velocity_upstream=jnp.array([3.0, 0.0, 0.0]),
            velocity_downstream=jnp.array([1.0, 0.0, 0.0]),
            shock_direction=jnp.array([1.0, 0.0, 0.0]),
        )

        self.assertTrue(bool(valid))
        self.assertAlmostEqual(float(speed), -1.0, places=6)

    def test_equal_density_makes_shock_speed_invalid(self):
        speed, valid = estimate_normal_shock_speed(
            density_upstream=1.0,
            density_downstream=1.0,
            velocity_upstream=jnp.array([2.0, 0.0, 0.0]),
            velocity_downstream=jnp.array([1.0, 0.0, 0.0]),
            shock_direction=jnp.array([1.0, 0.0, 0.0]),
        )

        self.assertFalse(bool(valid))
        self.assertTrue(np.isnan(float(speed)))

    def test_consistent_states_are_classified_as_fast_shock(self):
        diagnostics = calculate_fast_shock_diagnostics(
            density_upstream=1.0,
            pressure_upstream=0.6,
            velocity_upstream=jnp.array([3.0, 0.0, 0.0]),
            magnetic_upstream=jnp.array([0.0, 0.5, 0.0]),
            density_downstream=2.0,
            pressure_downstream=6.0,
            velocity_downstream=jnp.array([1.0, 0.0, 0.0]),
            magnetic_downstream=jnp.array([0.0, 1.0, 0.0]),
            shock_direction=jnp.array([1.0, 0.0, 0.0]),
        )

        self.assertTrue(bool(diagnostics.shock_speed_valid))
        self.assertTrue(bool(diagnostics.density_compression))
        self.assertTrue(bool(diagnostics.entropy_increase))
        self.assertTrue(bool(diagnostics.upstream_superfast))
        self.assertTrue(bool(diagnostics.downstream_subfast))
        self.assertTrue(bool(diagnostics.fast_shock))
        self.assertGreater(float(diagnostics.upstream_fast_mach), 1.0)
        self.assertLess(float(diagnostics.downstream_fast_mach), 1.0)

    def test_subfast_upstream_state_is_rejected(self):
        diagnostics = calculate_fast_shock_diagnostics(
            density_upstream=1.0,
            pressure_upstream=0.6,
            velocity_upstream=jnp.array([0.5, 0.0, 0.0]),
            magnetic_upstream=jnp.array([0.0, 0.5, 0.0]),
            density_downstream=2.0,
            pressure_downstream=6.0,
            velocity_downstream=jnp.array([0.25, 0.0, 0.0]),
            magnetic_downstream=jnp.array([0.0, 1.0, 0.0]),
            shock_direction=jnp.array([1.0, 0.0, 0.0]),
        )

        self.assertFalse(bool(diagnostics.upstream_superfast))
        self.assertFalse(bool(diagnostics.fast_shock))

    def test_invalid_sampling_is_never_classified(self):
        diagnostics = calculate_fast_shock_diagnostics(
            density_upstream=1.0,
            pressure_upstream=0.6,
            velocity_upstream=jnp.array([3.0, 0.0, 0.0]),
            magnetic_upstream=jnp.array([0.0, 0.5, 0.0]),
            density_downstream=2.0,
            pressure_downstream=6.0,
            velocity_downstream=jnp.array([1.0, 0.0, 0.0]),
            magnetic_downstream=jnp.array([0.0, 1.0, 0.0]),
            shock_direction=jnp.array([1.0, 0.0, 0.0]),
            sampling_valid=False,
        )

        self.assertFalse(bool(diagnostics.fast_shock))

    def test_complete_diagnostic_is_jittable(self):
        diagnostic_function = jax.jit(calculate_fast_shock_diagnostics)

        diagnostics = diagnostic_function(
            density_upstream=jnp.array(1.0),
            pressure_upstream=jnp.array(0.6),
            velocity_upstream=jnp.array([3.0, 0.0, 0.0]),
            magnetic_upstream=jnp.array([0.0, 0.5, 0.0]),
            density_downstream=jnp.array(2.0),
            pressure_downstream=jnp.array(6.0),
            velocity_downstream=jnp.array([1.0, 0.0, 0.0]),
            magnetic_downstream=jnp.array([0.0, 1.0, 0.0]),
            shock_direction=jnp.array([1.0, 0.0, 0.0]),
        )

        self.assertTrue(bool(diagnostics.fast_shock))


if __name__ == "__main__":
    unittest.main()
