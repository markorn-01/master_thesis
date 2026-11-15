import copy
import unittest
import numpy as np
from experiments.astrolink.expanded_event_tracking import expanded_tracking
from experiments.astrolink.event_tracking import event_tracking
from experiments.wind_bubble.check_five_fragment_events import new_cases
from experiments.astrolink.controlled_tracking_cases import evaluate_links


class ExpandedEvents(unittest.TestCase):
    def args(self,name):
        case=next(c for c in new_cases() if c['name']==name)
        return case,[case[k] for k in ['positions_a','normals_a','labels_a','positions_b','normals_b','labels_b']]

    def test_five_way_combination_and_separation(self):
        for name in ['five_disjoint_fragments','five_way_separation']:
            case,args=self.args(name)
            result=expanded_tracking(*args,.04)
            self.assertTrue(evaluate_links(result,case['expected_links'])['exact_correspondence'])
            self.assertEqual(len(result['events']),1)

    def test_cap_three_preserves_frozen_result_and_inputs(self):
        _,args=self.args('five_disjoint_fragments');before=copy.deepcopy(args)
        self.assertEqual(expanded_tracking(*args,.04,max_partners=3),event_tracking(*args,.04))
        expanded_tracking(*args,.04)
        for old,new in zip(before,args):np.testing.assert_array_equal(old,new)

    def test_coincident_parents_remain_unresolved(self):
        _,args=self.args('coincident_parents_ambiguous')
        result=expanded_tracking(*args,.04)
        self.assertEqual(result['events'],[])
        self.assertTrue(all(p['status']!='assigned' for p in result['parents']))
        self.assertTrue(any(p['status']=='ambiguous' for p in result['parents']))

    def test_opposite_normal_and_small_unique_patch_not_forced(self):
        for name in ['opposite_normal_fragment','small_unique_fragment']:
            _,args=self.args(name);result=expanded_tracking(*args,.04)
            self.assertFalse(any(len(e['parents'])==5 for e in result['events']))

    def test_invalid_unbounded_cap_rejected(self):
        _,args=self.args('five_disjoint_fragments')
        for cap in [2,6,5.,True]:
            with self.assertRaises(ValueError):expanded_tracking(*args,.04,max_partners=cap)
