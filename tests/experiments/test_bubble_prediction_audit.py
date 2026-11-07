import unittest
import numpy as np
from experiments.wind_bubble.audit_bubble_predictions import distance_summary,nearest_distances


class BubblePredictionAuditTests(unittest.TestCase):
    def test_missing_targets_are_not_reported_as_zero_distance(self):
        d=nearest_distances(np.zeros((2,3)),np.empty((0,3)))
        self.assertTrue(np.isinf(d).all())
        summary=distance_summary(d)
        self.assertEqual(summary['finite_samples'],0)
        self.assertIsNone(summary['median'])

    def test_empty_selected_population_is_explicit(self):
        d=nearest_distances(np.empty((0,3)),np.zeros((1,3)))
        self.assertEqual(distance_summary(d)['samples'],0)
        self.assertIsNone(distance_summary(d)['median'])

    def test_directional_distances_differ_and_do_not_assign_identity(self):
        a=np.array([[0,0,0],[1,0,0.]])
        b=np.array([[0,0,0.]])
        np.testing.assert_array_equal(nearest_distances(a,b),[0,1])
        np.testing.assert_array_equal(nearest_distances(b,a),[0])


if __name__=='__main__':unittest.main()
