import unittest
from experiments.astrolink.continuation_segments import stable_pairs,assign_segments
from experiments.astrolink.controlled_tracking_cases import tracking_cases
from experiments.astrolink.check_controlled_tracking import evaluate


class Segments(unittest.TestCase):
    def pairs(self,name,gates=(.1,.2,.35)):
        case=next(c for c in tracking_cases() if c['name']==name)
        return stable_pairs([evaluate(case,g) for g in gates])

    def test_translation_and_resampling(self):
        for name in ['translation_within_gate','fresh_sampling']:
            self.assertEqual(self.pairs(name),[(0,10)])

    def test_distance_disagreement_breaks_continuation(self):
        self.assertEqual(self.pairs('translation_beyond_gate'),[])

    def test_split_and_multiple_parents_break_continuation(self):
        for name in ['one_group_splits_into_two','two_fragments_form_one_group','nearby_sheet_disappears']:
            self.assertEqual(self.pairs(name),[])

    def test_inherit_one_start_another(self):
        mapping,rows,counter=assign_segments({3:7},[10,11],[(3,10)],8)
        self.assertEqual(mapping,{10:7,11:8});self.assertEqual(counter,9)
        self.assertTrue(rows[0]['inherited']);self.assertFalse(rows[1]['inherited'])

    def test_empty_event_prevents_gap_bridging(self):
        mapping,_,counter=assign_segments({0:0},[],[],1)
        mapping,rows,counter=assign_segments(mapping,[0],[],counter)
        self.assertEqual(mapping,{0:1});self.assertFalse(rows[0]['inherited'])

    def test_reject_non_unique_or_unknown_pairs(self):
        for pairs in [[(0,10),(1,10)],[(0,10),(0,11)],[(2,10)]]:
            with self.assertRaises(ValueError):assign_segments({0:0,1:1},[10,11],pairs,2)

    def test_require_multiple_distance_settings(self):
        with self.assertRaises(ValueError):stable_pairs([])
