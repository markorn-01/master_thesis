"""Unit tests for three-dimensional shock-surface refinement."""

import jax
import jax.numpy as jnp
import numpy as np
import unittest

from astronomix._physics_modules._shock_finder._shock_surface import (
    _find_shock_surface_3d,
)


def _constant_direction(shape, axis, sign=1.0):
    direction = jnp.zeros((3, *shape))
    return direction.at[axis].set(sign)


class ShockSurface3DTests(unittest.TestCase):
    def test_planar_zone_selects_single_minimum_divergence_layer(self):
        for axis in (0, 1, 2):
            with self.subTest(axis=axis):
                shape = (9, 8, 7)
                shock_zones = jnp.zeros(shape, dtype=jnp.bool_)
                div_v = jnp.zeros(shape)

                zone_slices = [slice(None)] * 3
                for layer, divergence in zip((3, 4, 5), (-2.0, -8.0, -3.0)):
                    zone_slices[axis] = layer
                    index = tuple(zone_slices)
                    shock_zones = shock_zones.at[index].set(True)
                    div_v = div_v.at[index].set(divergence)

                surface = _find_shock_surface_3d(
                    div_v,
                    shock_zones,
                    _constant_direction(shape, axis),
                )

                expected = jnp.zeros(shape, dtype=jnp.bool_)
                zone_slices[axis] = 4
                expected = expected.at[tuple(zone_slices)].set(True)
                np.testing.assert_array_equal(
                    np.asarray(surface), np.asarray(expected)
                )

    def test_empty_zone_produces_no_surface(self):
        shape = (6, 7, 8)
        surface = _find_shock_surface_3d(
            jnp.zeros(shape),
            jnp.zeros(shape, dtype=jnp.bool_),
            _constant_direction(shape, axis=0),
        )

        self.assertEqual(surface.dtype, jnp.bool_)
        self.assertEqual(surface.shape, shape)
        self.assertFalse(np.asarray(surface).any())

    def test_boundary_zone_does_not_wrap_to_opposite_boundary(self):
        shape = (7, 6, 5)
        shock_zones = jnp.zeros(shape, dtype=jnp.bool_)
        div_v = jnp.zeros(shape)
        shock_zones = shock_zones.at[0:3, :, :].set(True)
        div_v = div_v.at[0, :, :].set(-9.0)
        div_v = div_v.at[1, :, :].set(-4.0)
        div_v = div_v.at[2, :, :].set(-2.0)

        surface = _find_shock_surface_3d(
            div_v,
            shock_zones,
            _constant_direction(shape, axis=0),
        )

        expected = jnp.zeros(shape, dtype=jnp.bool_).at[0, :, :].set(True)
        np.testing.assert_array_equal(np.asarray(surface), np.asarray(expected))
        self.assertFalse(np.asarray(surface[-1]).any())

    def test_equal_minimum_plateau_is_reduced_to_one_layer(self):
        shape = (8, 5, 4)
        shock_zones = jnp.zeros(shape, dtype=jnp.bool_).at[2:6, :, :].set(True)
        div_v = jnp.zeros(shape)
        div_v = div_v.at[2, :, :].set(-2.0)
        div_v = div_v.at[3, :, :].set(-7.0)
        div_v = div_v.at[4, :, :].set(-7.0)
        div_v = div_v.at[5, :, :].set(-3.0)

        surface = _find_shock_surface_3d(
            div_v,
            shock_zones,
            _constant_direction(shape, axis=0, sign=1.0),
        )

        expected = jnp.zeros(shape, dtype=jnp.bool_).at[3, :, :].set(True)
        np.testing.assert_array_equal(np.asarray(surface), np.asarray(expected))

    def test_spherical_zone_selects_a_thin_closed_shell(self):
        shape = (21, 21, 21)
        coordinates = jnp.indices(shape, dtype=jnp.float32)
        displacement = coordinates - 10.0
        radius = jnp.sqrt(jnp.sum(displacement**2, axis=0))

        shock_zones = (radius >= 4.0) & (radius <= 7.0)
        # Maximum compression is centred on radius five.  Its precise sampled
        # radius varies with direction on a Cartesian mesh.
        div_v = jnp.where(shock_zones, -10.0 + (radius - 5.0) ** 2, 0.0)
        safe_radius = jnp.where(radius > 0.0, radius, 1.0)
        direction = displacement / safe_radius

        surface = _find_shock_surface_3d(
            div_v,
            shock_zones,
            direction,
        )
        surface_array = np.asarray(surface)
        surface_radii = np.asarray(radius)[surface_array]

        self.assertGreater(surface_radii.size, 100)
        self.assertTrue(np.asarray(shock_zones)[surface_array].all())
        self.assertLess(abs(float(np.median(surface_radii)) - 5.0), 0.35)
        self.assertLess(float(np.percentile(surface_radii, 90)), 5.8)
        # All six axial directions must be represented, guarding against an
        # implementation that accidentally raycasts along only one axis.
        for axis in range(3):
            self.assertTrue(surface_array.take(5, axis=axis).any())
            self.assertTrue(surface_array.take(15, axis=axis).any())

    def test_function_is_jittable(self):
        shape = (6, 6, 6)
        shock_zones = jnp.zeros(shape, dtype=jnp.bool_).at[2:5, :, :].set(True)
        div_v = jnp.zeros(shape)
        div_v = div_v.at[2, :, :].set(-1.0)
        div_v = div_v.at[3, :, :].set(-5.0)
        div_v = div_v.at[4, :, :].set(-2.0)
        direction = _constant_direction(shape, axis=0)

        surface = jax.jit(_find_shock_surface_3d)(div_v, shock_zones, direction)

        self.assertEqual(surface.shape, shape)
        self.assertEqual(surface.dtype, jnp.bool_)
        self.assertEqual(int(jnp.sum(surface)), shape[1] * shape[2])


if __name__ == "__main__":
    unittest.main()
