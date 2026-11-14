import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from experiments.wind_bubble.analyze_unequal_history import window_catalogs, validation_scope


class ValidationScope(unittest.TestCase):
    def test_absent_128_reference_is_explicitly_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta=dict(status='complete',source_strengths=[1.,.5],num_cells=128,verify_offline=False)
            self.assertFalse(validation_scope(Path(tmp),meta)['online_offline_verified'])
            meta['verify_offline']=True
            with self.assertRaises(ValueError):validation_scope(Path(tmp),meta)

    def test_64_reference_is_required_and_failure_never_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            meta=dict(status='complete',source_strengths=[1.,.5],num_cells=64)
            with self.assertRaises(ValueError):validation_scope(root,meta)
            (root/'equivalence.json').write_text(json.dumps(dict(passed=False)))
            with self.assertRaises(ValueError):validation_scope(root,meta)
            (root/'equivalence.json').write_text(json.dumps(dict(passed=True)))
            self.assertTrue(validation_scope(root,meta)['online_offline_verified'])


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
