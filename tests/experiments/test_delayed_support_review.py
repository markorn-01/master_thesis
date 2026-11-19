import unittest

import numpy as np

from experiments.wind_bubble.review_delayed_support import evidence_roles, group_rows


class DelayedSupportReview(unittest.TestCase):
    def test_existing_and_conditional_roles_are_distinct(self):
        self.assertEqual(evidence_roles([1, 2, 7], [2], [1, 7]),
                         {1: 'conditional', 2: 'accepted', 7: 'conditional'})

    def test_reject_omitted_or_promoted_fragment(self):
        for accepted, additional in [([1, 2], [1]), ([1], []), ([1, 2, 3], [])]:
            with self.assertRaises(ValueError):
                evidence_roles([1, 2], accepted, additional)

    def test_original_row_join_preserves_positions(self):
        points = np.array([[.3, .4, .5], [.1, .2, .3], [.3, .4, .5]])
        before = points.copy()
        np.testing.assert_array_equal(group_rows(points, [7, 1, 7], 7, 2), [0, 2])
        np.testing.assert_array_equal(points, before)

    def test_reject_wrong_population_and_invalid_geometry(self):
        for points, labels, count in [(np.zeros((2, 3)), [1, 1], 1),
                                       (np.zeros((2, 3)), [1], 2),
                                       (np.full((2, 3), np.nan), [1, 1], 2)]:
            with self.assertRaises(ValueError):
                group_rows(points, labels, 1, count)


if __name__ == '__main__':
    unittest.main()
