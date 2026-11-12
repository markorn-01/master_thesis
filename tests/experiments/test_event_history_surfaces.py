import unittest
import numpy as np
from experiments.wind_bubble.animate_event_history import segment_rows, cutaway_mask


class HistorySurfaces(unittest.TestCase):
    def test_join_uses_group_labels_not_row_order(self):
        labels = np.array([8, -1, 3, 8, -2])
        nodes = [dict(group=3, segment_id=17, surface_samples=1),
                 dict(group=8, segment_id=2, surface_samples=2)]
        np.testing.assert_array_equal(segment_rows(labels, nodes), [2, -1, 17, 2, -1])
        np.testing.assert_array_equal(labels, [8, -1, 3, 8, -2])

    def test_empty_and_unretained_catalogs(self):
        self.assertEqual(segment_rows(np.array([], dtype=int), []).size, 0)
        np.testing.assert_array_equal(segment_rows([-2, -1], []), [-1, -1])

    def test_reject_missing_duplicate_and_wrong_counts(self):
        good = dict(group=0, segment_id=1, surface_samples=2)
        for nodes in [[], [good, good], [dict(good, surface_samples=1)], [dict(good, group=2)]]:
            with self.assertRaises(ValueError): segment_rows([0, 0], nodes)

    def test_cutaway_includes_plane_and_preserves_coordinates(self):
        points = np.array([[1, 2, .5], [1, 2, .51], [1, 2, .49]])
        original = points.copy()
        np.testing.assert_array_equal(cutaway_mask(points), [True, False, True])
        np.testing.assert_array_equal(points, original)
        with self.assertRaises(ValueError): cutaway_mask([[1, 2, np.nan]])
