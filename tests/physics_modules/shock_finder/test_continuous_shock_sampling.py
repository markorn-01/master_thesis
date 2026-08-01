"""Tests for interpolation along the complete local shock normal."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from astronomix._physics_modules._shock_finder._shock_zones import (
    get_post_pre_shock_values,
)


class ContinuousShockSamplingTests(unittest.TestCase):
    def test_axis_aligned_sampling_has_expected_orientation(self):
        field = jnp.arange(9.0)
        direction = jnp.ones((1, 9))

        post, pre, _, _ = get_post_pre_shock_values(
            direction,
            field,
            field,
            max_steps=2,
        )

        self.assertAlmostEqual(float(post[4]), 2.0)
        self.assertAlmostEqual(float(pre[4]), 6.0)

    def test_diagonal_sampling_interpolates_linear_2d_field(self):
        x, y = jnp.meshgrid(jnp.arange(9.0), jnp.arange(9.0), indexing="ij")
        field_a = 2.0 * x + 3.0 * y
        field_b = -1.0 * x + 4.0 * y
        direction = jnp.broadcast_to(
            jnp.array([3.0 / 5.0, 4.0 / 5.0])[:, None, None],
            (2, 9, 9),
        )
        distance = 1.5

        a_post, a_pre, b_post, b_pre = get_post_pre_shock_values(
            direction,
            field_a,
            field_b,
            max_steps=distance,
        )

        center = (4, 4)
        a_change = distance * (2.0 * 3.0 / 5.0 + 3.0 * 4.0 / 5.0)
        b_change = distance * (-1.0 * 3.0 / 5.0 + 4.0 * 4.0 / 5.0)
        self.assertAlmostEqual(
            float(a_post[center]), float(field_a[center] - a_change), places=5
        )
        self.assertAlmostEqual(
            float(a_pre[center]), float(field_a[center] + a_change), places=5
        )
        self.assertAlmostEqual(
            float(b_post[center]), float(field_b[center] - b_change), places=5
        )
        self.assertAlmostEqual(
            float(b_pre[center]), float(field_b[center] + b_change), places=5
        )

    def test_sampling_uses_all_three_direction_components(self):
        x, y, z = jnp.meshgrid(
            jnp.arange(9.0),
            jnp.arange(9.0),
            jnp.arange(9.0),
            indexing="ij",
        )
        field = x + 2.0 * y + 4.0 * z
        unit_component = 1.0 / np.sqrt(3.0)
        direction = jnp.full((3, 9, 9, 9), unit_component)

        post, pre, _, _ = get_post_pre_shock_values(
            direction,
            field,
            field,
            max_steps=1,
        )

        center = (4, 4, 4)
        expected_change = 7.0 / np.sqrt(3.0)
        self.assertAlmostEqual(
            float(post[center]), float(field[center] - expected_change), places=5
        )
        self.assertAlmostEqual(
            float(pre[center]), float(field[center] + expected_change), places=5
        )

    def test_function_is_jittable(self):
        field = jnp.arange(49.0).reshape(7, 7)
        direction = jnp.broadcast_to(
            jnp.array([1.0, 1.0])[:, None, None] / jnp.sqrt(2.0),
            (2, 7, 7),
        )

        sample = jax.jit(
            lambda d, a: get_post_pre_shock_values(
                d, a, a, max_steps=1.25
            )
        )
        outputs = sample(direction, field)

        self.assertEqual(len(outputs), 4)
        for output in outputs:
            self.assertEqual(output.shape, field.shape)
            self.assertTrue(np.all(np.isfinite(np.asarray(output))))

    def test_sampling_clamps_instead_of_wrapping_at_boundaries(self):
        field = jnp.arange(6.0)
        direction = jnp.ones((1, 6))

        post, pre, _, _ = get_post_pre_shock_values(
            direction,
            field,
            field,
            max_steps=2,
        )

        # The post-shock sample from cell zero lies outside the lower boundary
        # and must use the nearest boundary value, not wrap to the far side.
        self.assertAlmostEqual(float(post[0]), 0.0)
        self.assertAlmostEqual(float(pre[-1]), 5.0)

    def test_sampling_origin_can_be_shifted_to_subcell_surface(self):
        field = jnp.arange(9.0)
        direction = jnp.ones((1, 9))
        center_offsets = jnp.full((1, 9), 0.25)

        post, pre, _, _ = get_post_pre_shock_values(
            direction,
            field,
            field,
            max_steps=2.0,
            center_offsets=center_offsets,
        )

        self.assertAlmostEqual(float(post[4]), 2.25, places=5)
        self.assertAlmostEqual(float(pre[4]), 6.25, places=5)


if __name__ == "__main__":
    unittest.main()
