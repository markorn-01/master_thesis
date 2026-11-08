"""Known geometry checks for descriptive patch angles."""
import unittest
import numpy as np
from experiments.wind_bubble.inspect_mesh_patch import triangle_angles


class PatchAngles(unittest.TestCase):
    def test_planar_triangle(self):
        p = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        n = np.tile([0., 0, 3], (3, 1))
        tangent, spread = triangle_angles(p, n, np.array([[0, 1, 2]]))
        np.testing.assert_allclose(tangent, 0)
        np.testing.assert_allclose(spread, 0)

    def test_common_tilt_is_not_normal_disagreement(self):
        p = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        n = np.tile([.5, 0, np.sqrt(3)/2], (3, 1))
        tangent, spread = triangle_angles(p, n, np.array([[0, 1, 2]]))
        np.testing.assert_allclose(tangent, 30)
        np.testing.assert_allclose(spread, 0, atol=1e-6)

    def test_degenerate_edge_rejected(self):
        with self.assertRaises(ValueError):
            triangle_angles(np.zeros((3, 3)), np.tile([0., 0, 1], (3, 1)), np.array([[0, 1, 2]]))
