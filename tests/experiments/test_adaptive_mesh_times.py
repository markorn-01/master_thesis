"""A failed geometry/accuracy check must remain visible as raw samples."""
import unittest

import numpy as np

from experiments.wind_bubble.check_adaptive_mesh_times import display_faces


class AdaptiveMeshTimeTests(unittest.TestCase):
    def setUp(self):
        self.faces = np.array([[0, 1, 2], [3, 4, 5]])
        self.groups = np.array([2, 2, 2, 7, 7, 7])
        self.metrics = dict(self_intersecting=False, nonmanifold_vertices=0,
                            nonmanifold_edges=0, inconsistent_shared_edge_winding=0)

    def test_only_groups_meeting_the_fixed_accuracy_target_are_shown_as_meshes(self):
        faces, approved = display_faces(self.faces, self.groups,
            [dict(group=2, accuracy_passed=True), dict(group=7, accuracy_passed=False)], self.metrics)
        np.testing.assert_array_equal(faces, [[0, 1, 2]])
        self.assertEqual(approved, [2])

    def test_global_intersection_keeps_all_groups_as_points(self):
        faces, approved = display_faces(self.faces, self.groups,
            [dict(group=2, accuracy_passed=True), dict(group=7, accuracy_passed=True)],
            dict(self.metrics, self_intersecting=True))
        self.assertEqual(len(faces), 0); self.assertEqual(approved, [])

    def test_cross_group_triangle_is_rejected(self):
        with self.assertRaises(ValueError):
            display_faces([[0, 1, 3]], self.groups, [], self.metrics)


if __name__ == '__main__':
    unittest.main()
