import copy
import unittest
import numpy as np
from experiments.astrolink.continuation_status import continuation_status,distance_sensitivity
from experiments.astrolink.controlled_tracking_cases import tracking_cases
from experiments.astrolink.check_controlled_tracking import evaluate
from experiments.astrolink.support_evidence import support_evidence


class ContinuationStatusTests(unittest.TestCase):
    def result(self,name,gate=.2):
        return evaluate(next(c for c in tracking_cases() if c['name']==name),gate)

    def test_translation_candidate_not_identity(self):
        table=continuation_status(self.result('translation_within_gate'))
        self.assertEqual(table['parents'][0]['advisory_status'],'continuation_candidate')
        self.assertFalse(table['parents'][0]['physical_identity_established'])
        self.assertEqual(table['children'][0]['advisory_status'],'continuation_candidate')

    def test_split_ambiguous_without_forcing_identity(self):
        table=continuation_status(self.result('one_group_splits_into_two'))
        self.assertEqual(table['parents'][0]['advisory_status'],'ambiguous')
        self.assertTrue(all(c['advisory_status']=='unmatched' for c in table['children']))

    def test_fragments_are_support_not_merger(self):
        table=continuation_status(self.result('two_fragments_form_one_group'))
        self.assertTrue(all(p['advisory_status']=='multiple_parent_support' for p in table['parents']))
        self.assertTrue(all(p['exclusive_child_support_samples']>0 for p in table['parents']))
        self.assertEqual(table['children'][0]['advisory_status'],'multiple_parent_support')

    def test_known_false_link_stays_visible_and_accepted(self):
        result=self.result('nearby_sheet_disappears');before=copy.deepcopy(result)
        table=continuation_status(result)
        self.assertEqual([p['parent_group'] for p in table['parents'] if p['matcher_accepted']],[0,1])
        self.assertTrue(all(p['redundant_support'] for p in table['parents']))
        self.assertTrue(all(p['advisory_status']=='multiple_parent_support' for p in table['parents']))
        self.assertEqual(result['parents'],before['parents']);self.assertEqual(result['links'],before['links'])
        for key,value in result['supported_child_rows'].items():np.testing.assert_array_equal(value,before['supported_child_rows'][key])

    def test_outside_gate_unmatched_and_distance_warning(self):
        tables=[continuation_status(self.result('translation_beyond_gate',g)) for g in [.2,.35]]
        self.assertEqual(tables[0]['parents'][0]['advisory_status'],'unmatched')
        self.assertTrue(distance_sensitivity(tables)[0])

    def test_fresh_sampling_is_candidate(self):
        self.assertEqual(continuation_status(self.result('fresh_sampling'))['parents'][0]['advisory_status'],'continuation_candidate')

    def test_empty_catalog_and_child_appearance(self):
        empty=np.empty((0,3));p=np.array([[0.,0,0]])
        result=support_evidence(empty,empty,np.empty(0,int),p,np.array([[0.,0,1]]),[2],.2)
        table=continuation_status(result)
        self.assertEqual(table['parents'],[]);self.assertEqual(table['children'][0]['advisory_status'],'unmatched')

    def test_missing_exclusive_diagnostic_is_error(self):
        result=self.result('translation_within_gate');del result['links'][0]['exclusive_child_support_samples']
        with self.assertRaises(KeyError):continuation_status(result)

    def test_parent_eligible_but_child_pool_insufficient(self):
        a=np.array([[0.,0,0]]);b=np.array([[float(x),0,0] for x in range(10)])
        result=support_evidence(a,np.array([[0.,0,1]]),[0],b,np.tile([0.,0,1],(10,1)),[10]*10,.1)
        table=continuation_status(result)
        self.assertEqual(table['parents'][0]['matcher_parent_status'],'eligible')
        self.assertFalse(table['parents'][0]['matcher_accepted'])
        self.assertEqual(table['parents'][0]['advisory_status'],'unmatched')
        self.assertEqual(table['children'][0]['advisory_status'],'unmatched')

    def test_distance_comparison_rejects_different_parents(self):
        a=continuation_status(self.result('translation_within_gate'))
        b=continuation_status(self.result('two_fragments_form_one_group'))
        with self.assertRaises(ValueError):distance_sensitivity([a,b])
