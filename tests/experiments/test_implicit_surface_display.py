"""Analytic checks for display fitting and its known information limit."""
import unittest

import numpy as np

from experiments.wind_bubble.implicit_surface_display import (
    DisplaySettings, implicit_samples, plane_field, trim_supported, local_hull_support,
)


class ImplicitDisplayTests(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[x, y, 0.] for x in range(-3, 4) for y in range(-3, 4)])
        self.n = np.tile([0., 0., 1.], (len(self.p), 1))

    def test_plane_field_recovers_known_signed_distance_and_unknown_support(self):
        values, _, _, _ = plane_field(np.array([[0., 0., .2], [.2, .3, -.3], [20., 0., 0.]]),
                                      self.p, self.n, 1.)
        np.testing.assert_allclose(values[:2], [.2, -.3], atol=1e-12)
        self.assertTrue(np.isnan(values[-1]))

    def test_analytic_plane_mesh_and_original_arrays_preserved(self):
        before = self.p.copy(); normals = self.n.copy()
        v, f, groups, _ = implicit_samples(self.p, self.n, np.zeros(len(self.p), int), 1.)
        self.assertGreater(len(f), 0)
        np.testing.assert_allclose(v[:, 2], 0., atol=1e-8)
        np.testing.assert_array_equal(groups, 0)
        np.testing.assert_array_equal(before, self.p)
        np.testing.assert_array_equal(normals, self.n)

    def test_distinct_groups_keep_their_two_known_planes(self):
        p = np.vstack([self.p, self.p+[0., 0., .1]])
        v, f, labels, _ = implicit_samples(p, np.vstack([self.n, self.n]),
                                          np.repeat([0, 1], len(self.p)), 1.)
        self.assertGreater(len(f), 0)
        self.assertTrue((np.ptp(labels[f], axis=1) == 0).all())
        for group, height in [(0, 0.), (1, .1)]:
            np.testing.assert_allclose(v[labels == group, 2], height, atol=1e-6)

    def test_identical_normals_on_unresolved_layers_expose_blended_midplane(self):
        p = np.vstack([self.p, self.p+[0., 0., .1]])
        # A zero at the middle is not one of the stipulated true sheets. This
        # failure must remain visible, rather than declaring a universal solver.
        values, _, _, _ = plane_field(np.array([[0., 0., .05]]), p, np.vstack([self.n, self.n]), 1.)
        np.testing.assert_allclose(values, 0., atol=1e-12)
        self.assertGreater(min(abs(.05), abs(.05-.1)), .02)

    def test_support_trim_rejects_an_unsupported_triangle_interior(self):
        p = np.array([[0., 0., 0.], [4., 0., 0.], [0., 4., 0.]])
        n = np.tile([0., 0., 1.], (3, 1))
        f, report = trim_supported(p, [[0, 1, 2]], p, n, 1.)
        self.assertEqual(len(f), 0)
        self.assertEqual(report['unsupported_faces'], 1)

    def test_hull_support_distinguishes_interpolation_from_boundary_extrapolation(self):
        p = np.array([[x, y, 0.] for x in range(4) for y in range(4)])
        n = np.tile([0., 0., 1.], (len(p), 1))
        accepted = local_hull_support(np.array([[1.5, 1.5, 0.], [-.1, 1.5, 0.], [0., 1.5, 0.]]), p, n, 1.)
        np.testing.assert_array_equal(accepted, [True, False, True])

    def test_empty_excluded_and_resource_limits(self):
        v, f, _, _ = implicit_samples(self.p, self.n, np.full(len(self.p), -1), 1.)
        self.assertEqual(len(v), 0); self.assertEqual(len(f), 0)
        with self.assertRaisesRegex(ValueError, 'limit'):
            implicit_samples(self.p, self.n, np.zeros(len(self.p), int), 1., DisplaySettings(max_grid_nodes=8))
        with self.assertRaises(ValueError): DisplaySettings(step_cells=0)


if __name__ == '__main__':
    unittest.main()
