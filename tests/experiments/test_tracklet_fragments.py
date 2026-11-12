import unittest
import numpy as np
from experiments.astrolink.tracklet_fragments import attach_fragments
from experiments.astrolink.check_tracklet_fragments import cases


class DelayedFragments(unittest.TestCase):
    def case(self,name):return next(c for c in cases() if c['name']==name)
    def apply(self,case,predict=True):
        return attach_fragments(*[case[k] for k in ['a','b','c']],case['continuations'],[0,1,2],.07,predict=predict)
    def test_all_eight_fragments_retained(self):
        r=self.apply(self.case('eight_fragments'))
        self.assertEqual(len(r['links']),8);self.assertEqual(r['families'][0]['child_union_coverage'],1)
        self.assertFalse(r['physical_ancestry_established'])
    def test_future_motion_selects_supported_parent(self):
        for name,parent in [('future_motion_positive',0),('future_motion_negative',1)]:
            case=self.case(name)
            self.assertEqual([r['parent_group'] for r in self.apply(case)['links']],[parent])
            self.assertEqual(len(self.apply(case,False)['links']),2)
    def test_identical_observations_cannot_resolve_physical_truth(self):
        a=self.apply(self.case('unobservable_ancestry_0'));b=self.apply(self.case('unobservable_ancestry_1'))
        self.assertEqual(a,b);self.assertEqual(len(a['links']),2)
        self.assertFalse(a['physical_ancestry_established'])
    def test_indistinguishable_families_abstain(self):
        result=self.apply(self.case('coincident_child_families'))
        self.assertEqual(result['links'],[])
        self.assertEqual(result['assignments'][0]['status'],'ambiguous_family')
    def test_opposite_normals_rejected(self):
        result=self.apply(self.case('opposite_normal_distractor'))
        self.assertEqual([r['parent_group'] for r in result['links']],[0])
    def test_irregular_time_ratio_and_input_preservation(self):
        case=self.case('future_motion_positive')
        # Doubling the earlier interval doubles the predicted backward movement.
        arrays=[case[k]['positions'].copy() for k in ['a','b','c']]
        result=attach_fragments(*[case[k] for k in ['a','b','c']],case['continuations'],[-1,1,2],.07)
        self.assertEqual(result['models'][0]['backward_time_ratio'],2)
        for key,original in zip(['a','b','c'],arrays):np.testing.assert_array_equal(case[key]['positions'],original)
    def test_reject_bad_window_and_duplicate_future(self):
        case=self.case('eight_fragments');data=[case[k] for k in ['a','b','c']]
        for times in [[0,0,1],[0,2,1]]:
            with self.assertRaises(ValueError):attach_fragments(*data,case['continuations'],times,.07)
        with self.assertRaises(ValueError):attach_fragments(*data,[(10,20),(10,20)],[0,1,2],.07)
