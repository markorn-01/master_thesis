import unittest
import numpy as np
from experiments.wind_bubble.review_detection_audit import thresholds,stencil_probe


class DetectionReviewTests(unittest.TestCase):
    def test_thresholds_at_production_minimum_mach(self):
        pressure,temperature=thresholds()
        self.assertAlmostEqual(pressure,1.8625)
        self.assertAlmostEqual(temperature,1.2921782544378695)

    def test_large_jump_does_not_imply_source_independent_sampling(self):
        state=np.ones((5,32,32,32),np.float32)
        state[4,24:]=2
        crop=dict(centers=np.array([[23.5,16.5,16.5]],np.float32)/32,
            direction=np.array([[-1.,0,0]],np.float32),grid_spacing=1/32,
            pressure_index=4,density_index=0,
            log_pressure_jump=np.array([np.log(2)],np.float32),
            log_temperature_jump=np.array([np.log(2)],np.float32))
        original=state.copy()
        result=stencil_probe(state,crop,dict(source_positions=[[.35,.5,.5],[.65,.5,.5]],
            injection_radius=.0625),np.array([True]))
        self.assertTrue(all(r['joint_pass']==1 for r in result))
        self.assertTrue(all(r['pass_with_center_endpoints_and_corners_outside']==0 for r in result))
        np.testing.assert_array_equal(state,original)
