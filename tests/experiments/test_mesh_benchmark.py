import unittest
import numpy as np
from experiments.wind_bubble.mesh_benchmark_cases import cases,evaluate_mesh
from experiments.wind_bubble.ball_pivot_mesh import guard_faces


class MeshBenchmarkTests(unittest.TestCase):
    def test_spherical_cap_closure_is_detected_by_radial_hole_probe(self):
        angle=np.arange(3)*2*np.pi/3;p=np.column_stack((.8*np.cos(angle),.8*np.sin(angle),np.full(3,.6)))
        case=dict(name='sphere_cap_hole',positions=p,truth_sheet=np.zeros(3,int),reference_area=1.)
        self.assertEqual(evaluate_mesh(case,[[0,1,2]])['sampled_hole_intrusion_faces'],1)

    def test_planar_hole_closure_is_detected(self):
        p=np.array([[-3,-3,0],[3,-3,0],[0,4,0.]])
        case=dict(name='planar_hole',positions=p,truth_sheet=np.zeros(3,int),reference_area=None)
        self.assertEqual(evaluate_mesh(case,[[0,1,2]])['sampled_hole_intrusion_faces'],1)

    def test_sheet_truth_is_distinct_from_supplied_group_labels(self):
        case=next(c for c in cases() if c['name']=='near_sheets_same_group_counterexample')
        self.assertEqual(len(np.unique(case['labels'])),1)
        self.assertEqual(len(np.unique(case['truth_sheet'])),2)
        n=len(case['positions'])//2
        self.assertEqual(evaluate_mesh(case,[[0,1,n]])['cross_stipulated_sheet_faces'],1)

    def test_shared_guards_reject_cross_group_and_long_faces(self):
        p=np.array([[0,0,0],[1,0,0],[0,1,0],[4,0,0.]])
        n=np.tile([0,0,1.],(4,1))
        faces,_=guard_faces(p,n,[0,0,1,0],[[0,1,2],[0,2,3]],1)
        self.assertEqual(len(faces),0)
        faces,_=guard_faces(p,n,[0]*4,[[0,2,3]],1)
        self.assertEqual(len(faces),0)

    def test_empty_mesh_is_zero_coverage_not_successful_reconstruction(self):
        case=cases()[0];r=evaluate_mesh(case,[])
        self.assertEqual(r['area_fraction'],0)
        self.assertEqual(r['used_vertices'],0)


if __name__=='__main__':unittest.main()
