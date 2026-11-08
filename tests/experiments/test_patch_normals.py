import unittest
import numpy as np
from experiments.wind_bubble.compare_patch_normals import plane_normals


class GeometricNormals(unittest.TestCase):
    def setUp(self):
        x, y = np.meshgrid(np.arange(-2., 3), np.arange(-2., 3))
        self.p = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))

    def test_plane_recovers_geometry_and_saved_sign(self):
        saved = np.tile([.5, 0, -np.sqrt(3)/2], (len(self.p), 1))
        normals, valid, _, _ = plane_normals(self.p, saved, np.zeros(len(saved)), 4.)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(normals, np.tile([0, 0, -1], (len(saved), 1)), atol=1e-12)

    def test_rotation(self):
        angle = .6
        rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        expected = rotation @ [0, 0, 1]
        normals, valid, _, _ = plane_normals(self.p @ rotation.T, np.tile(expected, (25, 1)), np.zeros(25), 4.)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(normals, np.tile(expected, (25, 1)), atol=1e-12)

    def test_collinear_keeps_saved_and_marks_invalid(self):
        p = np.column_stack((np.arange(8), np.zeros((8, 2))))
        saved = np.tile([0., 0, 1], (8, 1))
        normals, valid, _, _ = plane_normals(p, saved, np.zeros(8), 20.)
        self.assertFalse(valid.any())
        np.testing.assert_array_equal(normals, saved)

    def test_separate_groups_do_not_mix_crossing_planes(self):
        p = np.concatenate([self.p, self.p[:, [0, 2, 1]]])
        saved = np.concatenate([np.tile([0., 0, 1], (25, 1)), np.tile([0., 1, 0], (25, 1))])
        labels = np.repeat([0, 1], 25)
        normals, valid, counts, _ = plane_normals(p, saved, labels, 10.)
        self.assertTrue(valid.all()); self.assertTrue((counts == 25).all())
        np.testing.assert_allclose(normals, saved, atol=1e-12)

    def test_insufficient_points_and_excluded_group(self):
        saved = np.tile([0., 0, 1], (25, 1)); labels = np.zeros(25, int); labels[0] = -1
        normals, valid, counts, _ = plane_normals(self.p, saved, labels, .1)
        self.assertFalse(valid.any()); self.assertEqual(counts[0], 0)
        np.testing.assert_array_equal(normals, saved)
