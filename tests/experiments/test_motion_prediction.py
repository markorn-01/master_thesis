import unittest
import numpy as np
from experiments.astrolink.controlled_motion_cases import motion_cases
from experiments.astrolink.check_motion_prediction import evaluate
from experiments.astrolink.motion_prediction import predict_positions


class MotionPredictionTests(unittest.TestCase):
    def setUp(self):self.cases={c['name']:c for c in motion_cases()}

    def results(self,name,gate=.2):
        r,_=evaluate(self.cases[name]);return {x['method']:x for x in r if x['gate']==gate}

    def test_constant_motion_uses_actual_unequal_time_intervals(self):
        r=self.results('constant_translation')
        self.assertEqual(r['geometry']['missed_links'],[[10,20]])
        self.assertTrue(r['centroid_prediction']['exact_correspondence'])
        np.testing.assert_allclose(r['centroid_prediction']['estimates'][0]['velocity'],[0,0,.3])

    def test_prediction_avoids_nearest_position_swap_in_passing_control(self):
        r=self.results('passing_sheets')
        self.assertEqual(r['geometry']['false_links'],[[10,21],[11,20]])
        self.assertTrue(r['centroid_prediction']['exact_correspondence'])

    def test_prediction_can_hurt_after_stop_or_coverage_change(self):
        for name in ['abrupt_stop','changing_sample_coverage']:
            r=self.results(name)
            self.assertTrue(r['geometry']['exact_correspondence'])
            self.assertFalse(r['centroid_prediction']['exact_correspondence'])

    def test_uninitialized_groups_use_marked_geometric_fallback(self):
        for name in ['no_motion_initialization','unresolved_nearby_disappearance']:
            records,predicted=evaluate(self.cases[name])
            self.assertEqual(records[0]['initialized_groups'],0)
            self.assertTrue(all(x['velocity'] is None for x in records[0]['estimates']))
            np.testing.assert_array_equal(predicted,self.cases[name]['positions_1'])

    def test_truth_and_future_positions_do_not_affect_prediction(self):
        case=dict(self.cases['constant_translation']);_,before=evaluate(case)
        case['expected_links']=[];case['positions_2']=case['positions_2']+100
        _,after=evaluate(case);np.testing.assert_array_equal(before,after)

    def test_translation_does_not_fix_split_or_wide_gate_ambiguity(self):
        self.assertEqual(len(self.results('group_split')['centroid_prediction']['missed_links']),2)
        self.assertTrue(self.results('approaching_sheets',.08)['centroid_prediction']['exact_correspondence'])
        self.assertFalse(self.results('approaching_sheets',.2)['centroid_prediction']['exact_correspondence'])

    def test_centroid_translation_cannot_predict_symmetric_expansion(self):
        r=self.results('symmetric_expansion')
        self.assertFalse(r['centroid_prediction']['exact_correspondence'])
        np.testing.assert_allclose(r['centroid_prediction']['estimates'][0]['velocity'],[0,0,0],atol=1e-12)

    def test_reject_nonincreasing_times(self):
        c=self.cases['constant_translation']
        with self.assertRaises(ValueError):
            predict_positions(c['positions_0'],c['normals_0'],c['labels_0'],c['positions_1'],c['normals_1'],c['labels_1'],[0,0,1])


if __name__=='__main__':unittest.main()
