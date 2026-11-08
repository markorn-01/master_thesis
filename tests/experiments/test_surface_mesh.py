import unittest
import numpy as np
from experiments.wind_bubble.surface_mesh import triangulate_samples
from experiments.wind_bubble.prototype_surface_mesh import controls


class SurfaceMeshTests(unittest.TestCase):
    def setUp(self):
        self.p=np.array([[x,y,0.] for x in range(7) for y in range(7)])
        self.n=np.tile([0,0,1.],(len(self.p),1));self.labels=np.zeros(len(self.p),int)

    def test_planar_grid_is_triangulated_without_moving_vertices(self):
        before=self.p.copy();normals=self.n.copy()
        f,r=triangulate_samples(self.p,self.n,self.labels,1.)
        self.assertEqual(len(f),72);self.assertEqual(len(np.unique(f)),49)
        self.assertEqual(r['topology']['nonmanifold_edges'],0)
        self.assertEqual(r['topology']['inconsistent_shared_edge_winding'],0)
        np.testing.assert_array_equal(self.p,before);np.testing.assert_array_equal(self.n,normals)

    def test_separate_labels_and_wide_gap_are_not_bridged(self):
        result={x['case']:x for x in controls()}
        for name in ['near_sheets_separate_groups','wide_gap_same_group','opposite_normals_same_group']:
            self.assertGreater(result[name]['faces'],0)
            self.assertEqual(result[name]['cross_stipulated_sheet_faces'],0)

    def test_close_aligned_sheets_in_one_group_expose_known_limitation(self):
        r=next(x for x in controls() if x['case']=='near_sheets_same_group_counterexample')
        self.assertGreater(r['cross_stipulated_sheet_faces'],0)

    def test_row_permutation_preserves_triangle_membership(self):
        f,_=triangulate_samples(self.p,self.n,self.labels,1.)
        order=np.random.default_rng(42).permutation(len(self.p))
        other,_=triangulate_samples(self.p[order],self.n[order],self.labels[order],1.)
        self.assertEqual({tuple(sorted(x)) for x in f},{tuple(sorted(x)) for x in order[other]})

    def test_collinear_or_excluded_samples_do_not_create_faces(self):
        p=np.array([[x,0,0.] for x in range(5)]);n=np.tile([0,0,1.],(5,1))
        f,r=triangulate_samples(p,n,[0]*5,1.)
        self.assertEqual(len(f),0)
        f,_=triangulate_samples(self.p,self.n,[-1]*len(self.p),1.)
        self.assertEqual(len(f),0)

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):triangulate_samples(self.p,self.n,self.labels,0)
        n=self.n.copy();n[0]=0
        with self.assertRaises(ValueError):triangulate_samples(self.p,n,self.labels,1)


if __name__=='__main__':unittest.main()
