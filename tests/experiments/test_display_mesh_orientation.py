"""Winding correction must change triangle order only."""
import unittest

import numpy as np

from experiments.wind_bubble.orient_display_mesh import orient_faces


class DisplayMeshOrientationTests(unittest.TestCase):
    def test_reversed_neighbor_is_fixed_without_changing_the_open_patch(self):
        p = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
        faces = np.array([[0, 1, 2], [1, 2, 3]])
        result, report = orient_faces(p, faces)
        np.testing.assert_array_equal(np.sort(result, axis=1), np.sort(faces, axis=1))
        self.assertEqual(report['changed_faces'], 1)
        self.assertEqual(report['before']['inconsistent_shared_edge_winding'], 1)
        self.assertEqual(report['after']['inconsistent_shared_edge_winding'], 0)
        self.assertEqual(report['before']['boundary_edges'], report['after']['boundary_edges'])

    def test_consistent_mesh_is_returned_exactly_unchanged(self):
        p = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        result, report = orient_faces(p, faces)
        np.testing.assert_array_equal(result, faces)
        self.assertEqual(report['changed_faces'], 0)


if __name__ == '__main__':
    unittest.main()
