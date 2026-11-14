import unittest
import numpy as np
from experiments.wind_bubble.inspect_weak_source import probe


class WeakSourceProbe(unittest.TestCase):
    def test_selects_local_inward_points_without_changing_inputs(self):
        s=dict(positions=np.array([[.725,.5,.5],[.725,.5,.5],[.81,.5,.5],[.425,.5,.5]]),
               normals=np.array([[-2.,0,0],[1.,0,0],[-1.,0,0],[-1.,0,0]]),
               source_positions=np.array([[.35,.5,.5],[.65,.5,.5]]),injection_radius=.0625)
        original=s['normals'].copy()
        strong,weak=probe(s)
        np.testing.assert_array_equal(weak,[True,False,False,False])
        np.testing.assert_array_equal(strong,[False,False,False,True])
        np.testing.assert_array_equal(s['normals'],original)

    def test_empty_event_has_empty_masks(self):
        s=dict(positions=np.empty((0,3)),normals=np.empty((0,3)),
               source_positions=np.array([[.35,.5,.5],[.65,.5,.5]]),injection_radius=.0625)
        self.assertEqual([len(mask) for mask in probe(s)],[0,0])
