import unittest
import numpy as np
from experiments.wind_bubble.ball_pivot_mesh import ball_pivot_samples


class BallPivotTests(unittest.TestCase):
    def setUp(self):
        self.p=np.array([[x,y,0.] for x in range(5) for y in range(5)])
        self.n=np.tile([0,0,1.],(len(self.p),1));self.labels=np.zeros(len(self.p),int)

    def test_planar_grid_reconstructs_with_original_vertices_and_open_rim(self):
        p=self.p.copy();n=self.n.copy()
        f,r,_=ball_pivot_samples(p,n,self.labels,1.)
        self.assertEqual(len(f),32);self.assertEqual(r['topology']['boundary_edges'],16)
        self.assertEqual(r['topology']['nonmanifold_edges'],0)
        np.testing.assert_array_equal(p,self.p);np.testing.assert_array_equal(n,self.n)

    def test_separate_near_sheets_never_share_a_face(self):
        p=np.vstack((self.p,self.p+[.35,.35,.1]));n=np.vstack((self.n,self.n));labels=np.repeat([0,1],len(self.p))
        f,r,_=ball_pivot_samples(p,n,labels,1.)
        self.assertGreater(len(f),0);self.assertTrue((np.ptp(labels[f],axis=1)==0).all())

    def test_zero_normals_or_negative_radii_rejected_before_native_call(self):
        with self.assertRaises(ValueError):ball_pivot_samples(self.p,np.zeros_like(self.n),self.labels,1.)
        with self.assertRaises(ValueError):ball_pivot_samples(self.p,self.n,self.labels,1.,[-1])


if __name__=='__main__':unittest.main()
