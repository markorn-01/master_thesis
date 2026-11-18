"""Final cleanup must preserve group identity and surviving triangle geometry."""
import unittest

import numpy as np

from experiments.wind_bubble.finalize_adaptive_sequence import triangle_keys
from experiments.wind_bubble.finalize_surface_display import remove_defects


class FinalCombinedCleanupTests(unittest.TestCase):
    def test_geometry_key_survives_compaction_and_winding_changes(self):
        p = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        old = triangle_keys(p, [[0, 1, 2]], np.full(3, 4))
        new = triangle_keys(p[[2, 0, 1]], [[0, 2, 1]], np.full(3, 4))
        self.assertEqual(old, new)

    def test_identical_geometry_does_not_erase_group_identity(self):
        p = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        self.assertNotEqual(triangle_keys(p, [[0, 1, 2]], np.full(3, 0)),
                            triangle_keys(p, [[0, 1, 2]], np.full(3, 2)))

    def test_cross_group_intersections_are_removed_while_unrelated_group_survives(self):
        p = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.],
                      [.5, .5, -1.], [.5, .5, 1.], [1.5, .5, 0.],
                      [4., 0., 0.], [5., 0., 0.], [4., 1., 0.]])
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        groups = np.repeat([0, 2, 1], 3)
        before = p.copy()
        v, f, g, report = remove_defects(p, faces, groups)
        self.assertEqual(report['removed_faces'], 2)
        self.assertEqual(triangle_keys(v, f, g), triangle_keys(p, faces[2:], groups))
        np.testing.assert_array_equal(p, before)


if __name__ == '__main__':
    unittest.main()
