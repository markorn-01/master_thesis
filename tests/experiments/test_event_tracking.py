import copy
import unittest
import numpy as np
from experiments.astrolink.event_tracking import event_tracking,select_events
from experiments.astrolink.controlled_tracking_cases import tracking_cases,evaluate_links


class EventTracking(unittest.TestCase):
    def run_case(self,name,gate=.2):
        case=next(c for c in tracking_cases() if c['name']==name)
        args=[case[k] for k in ['positions_a','normals_a','labels_a','positions_b','normals_b','labels_b']]
        return case,event_tracking(*args,gate)

    def test_separation_and_combination(self):
        for name,kind in [('one_group_splits_into_two','separation'),('two_fragments_form_one_group','combination')]:
            case,result=self.run_case(name)
            self.assertTrue(evaluate_links(result,case['expected_links'])['exact_correspondence'])
            self.assertEqual([e['kind'] for e in result['events']],[kind])

    def test_nearest_continuation_avoids_existing_distractor(self):
        case,result=self.run_case('nearby_sheet_disappears')
        self.assertTrue(evaluate_links(result,case['expected_links'])['exact_correspondence'])
        self.assertFalse(any(e['kind']=='combination' for e in result['candidates']))

    def test_global_assignment_handles_approaching_sheets(self):
        case,result=self.run_case('approaching_aligned_sheets',.35)
        self.assertTrue(evaluate_links(result,case['expected_links'])['exact_correspondence'])

    def test_motion_beyond_gate_still_missed(self):
        case,result=self.run_case('translation_beyond_gate')
        self.assertEqual(evaluate_links(result,case['expected_links'])['missed_links'],[[0,10]])

    def test_exact_tie_is_unresolved(self):
        p=np.array([[x,y,0.] for x in range(3) for y in range(3)])
        pa=np.vstack((p+[0,0,-.1],p+[0,0,.1]));na=np.tile([0.,0,1],(len(pa),1));nb=na[:len(p)]
        result=event_tracking(pa,na,np.repeat([0,1],len(p)),p,nb,np.zeros(len(p),int),.2)
        self.assertEqual(result['events'],[])
        self.assertTrue(all(x['status']=='ambiguous' for x in result['parents']))

    def test_global_selection_beats_greedy_and_keeps_unaffected_event(self):
        events=[dict(parents=[0],children=[10],score=3.),dict(parents=[0],children=[11],score=2.),
                dict(parents=[1],children=[10],score=2.),dict(parents=[2],children=[12],score=1.)]
        result=select_events(events,[0,1,2],[10,11,12])
        self.assertEqual(result['accepted'],[1,2,3]);self.assertEqual(result['optimal_score'],5.)

    def test_empty_catalogs_and_appearance(self):
        empty=np.empty((0,3));p=np.array([[0.,0,0]])
        result=event_tracking(empty,empty,np.empty(0,int),p,np.array([[0.,0,1]]),np.array([1]),.2)
        self.assertEqual(result['events'],[]);self.assertEqual(result['children'][0]['status'],'unmatched')

    def test_input_arrays_unchanged(self):
        case=tracking_cases()[0];keys=['positions_a','normals_a','labels_a','positions_b','normals_b','labels_b']
        args=[case[k].copy() for k in keys];before=copy.deepcopy(args);event_tracking(*args,.2)
        for a,b in zip(args,before):np.testing.assert_array_equal(a,b)

    def test_invalid_parameters(self):
        p=np.array([[0.,0,0]]);n=np.array([[0.,0,1]]);l=np.array([0])
        with self.assertRaises(ValueError):event_tracking(p,n,l,p,n,l,0.)
        with self.assertRaises(ValueError):event_tracking(p,n,l,p,n,l,.2,exclusive_fraction=0.)

    def test_three_fragments_and_reverse_separation(self):
        p=np.array([[x,y,0.] for x in range(3) for y in range(3)])
        n=np.tile([0.,0,1],(9,1));groups=np.repeat([0,1,2],3);one=np.full(9,10)
        forward=event_tracking(p,n,groups,p,n,one,.2)
        backward=event_tracking(p,n,one,p,n,groups,.2)
        self.assertEqual([(e['kind'],e['parents'],e['children']) for e in forward['events']],[('combination',[0,1,2],[10])])
        self.assertEqual([(e['kind'],e['parents'],e['children']) for e in backward['events']],[('separation',[10],[0,1,2])])

    def test_solver_resolves_above_tie_tolerance_and_abstains_below(self):
        for delta,expected in [(1e-7,[1]),(5e-9,[])]:
            candidates=[dict(parents=[0],children=[10],score=1.),dict(parents=[0],children=[11],score=1.+delta)]
            result=select_events(candidates,[0],[10,11])
            self.assertEqual(result['accepted'],expected)
            np.testing.assert_allclose(result['optimal_score'],1.+delta,rtol=0,atol=1e-14)
