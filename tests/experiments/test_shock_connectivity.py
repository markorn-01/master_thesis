"""Meaningful edge/label checks for the geometry baseline."""
import unittest
import numpy as np
from experiments.astrolink.shock_connectivity import connectivity_partition


class ConnectivityTests(unittest.TestCase):
    def test_opposite_normals_do_not_connect_close_surfaces(self):
        p=np.array([[0.,0,0],[.01,0,0],[.015,0,0]])
        n=np.array([[0.,0,1],[0,0,2],[0,0,-1]])
        r=connectivity_partition(p,n,.01,2.,30.,1)
        self.assertEqual(len(r['component_sizes']),2)
        self.assertEqual(r['raw_labels'][0],r['raw_labels'][1])
        self.assertNotEqual(r['raw_labels'][1],r['raw_labels'][2])

    def test_distance_and_angle_thresholds_are_independent(self):
        p=np.array([[0.,0,0],[.01,0,0],[1.,0,0]])
        n=np.array([[1.,0,0],[.5,np.sqrt(.75),0],[1,0,0]])
        self.assertEqual(len(connectivity_partition(p,n,.01,2,30,1)['component_sizes']),3)
        self.assertEqual(len(connectivity_partition(p,n,.01,2,70,1)['component_sizes']),2)

    def test_small_components_are_preserved_in_raw_output(self):
        p=np.array([[0.,0,0],[.01,0,0],[1.,0,0],[2,0,0]])
        n=np.array([[1.,0,0],[1,0,0],[1,0,0],[0,0,0]])
        r=connectivity_partition(p,n,.01,2,30,2)
        self.assertEqual(r['labels'][2],-1)
        self.assertGreaterEqual(r['raw_labels'][2],0)
        self.assertEqual(r['labels'][3],-2)
        np.testing.assert_array_equal(sorted(r['component_sizes']),[1,2])

    def test_shuffling_preserves_membership_up_to_label_renumbering(self):
        p=np.array([[0.,0,0],[.01,0,0],[1.,0,0],[1.01,0,0]])
        n=np.tile([0.,0,1],(4,1));perm=np.array([2,0,3,1])
        a=connectivity_partition(p,n,.01,2,30,1)['labels']
        b=connectivity_partition(p[perm],n[perm],.01,2,30,1)['labels']
        restored=np.empty_like(b);restored[perm]=b
        np.testing.assert_array_equal(a[:,None]==a[None,:],restored[:,None]==restored[None,:])

    def test_no_geometry_and_invalid_thresholds(self):
        result=connectivity_partition(np.empty((0,3)),np.empty((0,3)),.01)
        self.assertEqual(len(result['labels']),0)
        with self.assertRaises(ValueError):
            connectivity_partition(np.ones((1,3)),np.ones((1,3)),.01,max_angle_degrees=180)


if __name__=='__main__':
    unittest.main()
