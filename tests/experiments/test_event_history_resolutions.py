import unittest
import numpy as np
from experiments.wind_bubble.compare_event_history_resolutions import group_statistics


class ResolutionStatistics(unittest.TestCase):
    def test_nearest_source_orientation_and_valid_mach(self):
        data = dict(positions=np.array([[.1, 0, 0], [.9, 0, 0], [.8, 0, 0]]),
                    normals=np.array([[2, 0, 0], [1, 0, 0], [3, 0, 0]]),
                    source_positions=np.array([[0, 0, 0], [1, 0, 0]]),
                    mach=np.array([4., 2., 100.]), mach_valid=np.array([True, True, False]))
        nodes = [dict(group=7, segment_id=2, surface_samples=99), dict(group=3, segment_id=0)]
        rows = group_statistics(data, np.array([3, 7, 7]), nodes)
        self.assertEqual(rows[0]['surface_samples'], 2)
        self.assertEqual(rows[0]['median_nearest_source_alignment'], -1)
        self.assertEqual(rows[0]['median_mach'], 2)
        self.assertEqual(rows[1]['median_nearest_source_alignment'], 1)
        np.testing.assert_allclose(rows[0]['median_nearest_source_radius'], .15)

    def test_exclude_unretained_and_nonpositive_mach(self):
        data = dict(positions=np.array([[.1, 0, 0], [.2, 0, 0]]),
                    normals=np.array([[1., 0, 0], [1, 0, 0]]),
                    source_positions=np.array([[0, 0, 0]]), mach=np.array([-1., 100.]),
                    mach_valid=np.array([True, True]))
        rows = group_statistics(data, np.array([0, -1]), [dict(group=0)])
        self.assertEqual(rows[0]['surface_samples'], 1)
        self.assertIsNone(rows[0]['median_mach'])
        enriched = group_statistics(data, np.array([0, -1]),
            [dict(group=0, surface_samples=99, valid_mach_samples=99, median_mach=99)])
        self.assertEqual(set(rows[0]), set(enriched[0]))
        self.assertEqual(enriched[0]['valid_mach_samples'], 0)
        with self.assertRaises(ValueError): group_statistics(data, np.array([0, -1]), [])
