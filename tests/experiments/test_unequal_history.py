import unittest
import numpy as np
from experiments.wind_bubble.analyze_unequal_history import window_catalogs


class UnequalWindows(unittest.TestCase):
    def setup_window(self):
        groups=[[0,1,2],[10,11],[20,21]]
        data=[dict(surface=dict(positions=np.array([[i,0,0] for i in labels],np.float32),
             normals=np.tile(np.array([0,0,1],np.float32),(len(labels),1)))) for labels in groups]
        nodes=[dict(node_id=f'{e}:{g}',group=g,event=e) for e,labels in enumerate(groups) for g in labels]
        def link(e,p,c):return dict(parent_event=e,status='shared',kind='continuation',parents=[f'{e}:{p}'],children=[f'{e+1}:{c}'])
        graph=dict(nodes=nodes,events=[link(0,0,10),link(1,10,20),link(1,11,21)])
        return data,[np.array(g) for g in groups],graph
    def test_outer_continuation_excluded_and_all_other_fragments_kept(self):
        data,labels,graph=self.setup_window()
        catalogs,pairs=window_catalogs(data,labels,graph,0)
        self.assertEqual(pairs,[(11,21)])
        for actual,expected in zip(catalogs,[[1,2],[11],[21]]):np.testing.assert_array_equal(actual['labels'],expected)
        self.assertEqual(catalogs[0]['normals'].dtype,np.dtype(float))
        np.testing.assert_array_equal(data[0]['surface']['positions'],[[0,0,0],[1,0,0],[2,0,0]])
    def test_distance_specific_future_does_not_seed_family(self):
        data,labels,graph=self.setup_window();graph['events'][-1]['status']='distance_specific'
        catalogs,pairs=window_catalogs(data,labels,graph,0)
        self.assertEqual(pairs,[]);self.assertEqual(len(catalogs[1]['labels']),0)
    def test_no_earlier_detection_remains_no_evidence(self):
        data,labels,graph=self.setup_window();labels[0][:]=-1
        graph['events']=graph['events'][1:]
        catalogs,pairs=window_catalogs(data,labels,graph,0)
        self.assertEqual(len(catalogs[0]['labels']),0)
        self.assertEqual(pairs,[(10,20),(11,21)])
