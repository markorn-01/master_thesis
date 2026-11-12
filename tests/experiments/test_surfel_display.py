import unittest
import numpy as np
from experiments.wind_bubble.surfel_display import disk_vertices


class Surfels(unittest.TestCase):
    def test_disks_centered_in_normal_plane_at_fixed_radius(self):
        p=np.array([[1.,2.,3.],[0,0,0]]);n=np.array([[0.,0.,2.],[1.,1.,1.]])
        disks=disk_vertices(p,n,.2)
        np.testing.assert_allclose(disks.mean(axis=1),p,atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(disks-p[:,None,:],axis=-1),.2)
        np.testing.assert_allclose(np.einsum('ijk,ik->ij',disks-p[:,None,:],n),0,atol=1e-14)
    def test_empty_and_input_preservation(self):
        self.assertEqual(disk_vertices(np.empty((0,3)),np.empty((0,3)),.2).shape,(0,10,3))
        p=np.array([[0.,0.,0.]]);n=np.array([[0.,0.,2.]])
        disk_vertices(p,n,.2);np.testing.assert_array_equal(n,[[0,0,2]])
    def test_reject_invalid_geometry(self):
        for normals,radius in [([[0,0,0]],.1),([[0,0,1]],0)]:
            with self.assertRaises(ValueError):disk_vertices([[0,0,0]],normals,radius)
