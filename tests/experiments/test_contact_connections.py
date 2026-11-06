import unittest
import numpy as np
from experiments.astrolink.inspect_contact_connections import joining_edges


class JoiningEdgeTests(unittest.TestCase):
    def test_only_edges_between_eligible_distinct_components(self):
        p = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]],float)
        n = np.tile([0,1,0],(4,1))
        r = joining_edges(p,n,1.,[0,0,1,-2],[[0,1],[1,2],[2,3]])
        np.testing.assert_array_equal(r['pairs'],[[1,2]])
        np.testing.assert_allclose(r['distance_cells'],[1.])
        np.testing.assert_allclose(r['normal_angle_degrees'],[0.])
        np.testing.assert_allclose(r['normal_chord_alignment_a'],[0.])

    def test_parallel_normal_chord_and_normal_scaling(self):
        r = joining_edges([[0,0,0],[0,2,0]],[[0,2,0],[0,4,0]],.5,[0,1],[[0,1]])
        np.testing.assert_allclose(r['distance_cells'],[4.])
        np.testing.assert_allclose(r['normal_chord_alignment_a'],[1.])
        np.testing.assert_allclose(r['normal_chord_alignment_b'],[1.])

    def test_no_edges_returns_empty_measurements(self):
        r = joining_edges([[0,0,0]],[[1,0,0]],1.,[0],[])
        self.assertEqual(r['pairs'].shape,(0,2))
        self.assertEqual(len(r['distance_cells']),0)


if __name__ == '__main__': unittest.main()
