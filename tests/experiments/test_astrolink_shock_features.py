"""Feature-metric and hierarchy/index mapping tests; AstroLink not required."""
import unittest
import numpy as np

from experiments.astrolink.check_two_bubble_clustering import (
    feature_arrays, deepest_partition, restore_labels,
)


class AstroLinkFeatureTests(unittest.TestCase):
    def test_explicit_metric_and_unit_normalization(self):
        p = np.array([[.5,.5,.5],[.6,.5,.5]])
        n = np.array([[2.,0,0],[0,3.,0]])
        features,rows = feature_arrays(p,n,.1,4.)
        np.testing.assert_array_equal(rows,[0,1])
        self.assertAlmostEqual(float(np.sum((features[1]-features[0])**2)),1+16*2)
        spatial,_ = feature_arrays(p,n,.1,0.)
        np.testing.assert_array_equal(features[:,:3],spatial)

    def test_modes_use_common_valid_geometry(self):
        p = np.array([[0.,0,0],[1,0,0],[np.nan,0,0],[2,0,0]])
        n = np.array([[1.,0,0],[0,0,0],[1,0,0],[np.nan,0,0]])
        for weight in (0.,1.,4.):
            features,rows = feature_arrays(p,n,.1,weight)
            np.testing.assert_array_equal(rows,[0])
            self.assertEqual(len(features),1)

    def test_root_does_not_become_a_detected_structure(self):
        labels = deepest_partition([2,0,1],[[0,3]],['1'],3)
        np.testing.assert_array_equal(labels,[-1,-1,-1])

    def test_nested_memberships_and_shuffle_map_to_original_rows(self):
        labels = deepest_partition([2,0,3,1],[[0,4],[0,3],[1,2]],['1','1-1','1-1-1'],4)
        np.testing.assert_array_equal(labels,[2,-1,1,1])
        restored = restore_labels(labels,np.array([2,0,3,1]),np.array([0,2,3,5]),6)
        np.testing.assert_array_equal(restored,[-1,-2,1,2,-2,1])

    def test_rejects_invalid_metric_and_ordering(self):
        for dx,weight in [(0,1),(.1,-1),(.1,np.nan)]:
            with self.assertRaises(ValueError):
                feature_arrays(np.ones((2,3)),np.ones((2,3)),dx,weight)
        with self.assertRaises(ValueError):
            deepest_partition([0,0],[[0,2]],['1'],2)


if __name__ == '__main__':
    unittest.main()
