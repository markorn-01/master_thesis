import unittest
import numpy as np
from experiments.wind_bubble.check_temporal_mesh import group_separation,plane_segments
from experiments.wind_bubble.verify_temporal_mesh import face_components


class TemporalMeshTests(unittest.TestCase):
    def test_plane_crossing_is_interpolated_at_known_points(self):
        p=np.array([[0,0,-1.],[2,0,1.],[0,2,1.]])
        segments,rows,coplanar=plane_segments(p,[[0,1,2]],z=0)
        self.assertEqual(rows.tolist(),[0]);self.assertEqual(coplanar,0)
        self.assertEqual({tuple(x) for x in segments[0]},{(1.,0.),(0.,1.)})

    def test_coplanar_triangles_are_reported_not_silently_filled(self):
        p=np.array([[0,0,.5],[1,0,.5],[0,1,.5]])
        segments,rows,coplanar=plane_segments(p,[[0,1,2]])
        self.assertEqual(len(segments),0);self.assertEqual(coplanar,1)

    def test_separation_bound_is_sufficient_and_can_be_inconclusive(self):
        tri=np.array([[0,0,0],[1,0,0],[0,1,0.]])
        labels=np.repeat([0,1],3);faces=np.array([[0,1,2],[3,4,5]])
        far=np.vstack((tri,tri+[0,0,4]))
        self.assertTrue(group_separation(far,labels,faces)[0]['separation_bound_passed'])
        near=np.vstack((tri,tri+[0,0,.1]))
        self.assertFalse(group_separation(near,labels,faces)[0]['separation_bound_passed'])
        # The near parallel faces are actually disjoint: failing the bound is
        # not evidence of an intersection.

    def test_edge_components_do_not_join_at_a_single_vertex(self):
        result=face_components([[0,1,2],[1,3,2],[0,4,5]])
        self.assertEqual(result['edge_connected_components'],2)
        self.assertEqual(result['largest_component_faces'],2)
        self.assertEqual(result['singleton_face_components'],1)

    def test_one_group_has_no_cross_group_separation_claim(self):
        p=np.array([[0,0,0],[1,0,0],[0,1,0.]])
        self.assertEqual(group_separation(p,[0,0,0],np.array([[0,1,2]])),[])


if __name__=='__main__':unittest.main()
