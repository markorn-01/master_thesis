import unittest
from experiments.astrolink.controlled_tracking_cases import tracking_cases,evaluate_links
from experiments.astrolink.check_controlled_tracking import evaluate


class ControlledTrackingTests(unittest.TestCase):
    def setUp(self):
        self.cases={case['name']:case for case in tracking_cases()}

    def metric(self,name,gate=.2):
        case=self.cases[name]
        return evaluate_links(evaluate(case,gate),case['expected_links'])

    def test_motion_gate_recovers_fast_sheet_only_when_reached(self):
        self.assertEqual(self.metric('translation_beyond_gate')['missed_links'],[[0,10]])
        self.assertTrue(self.metric('translation_beyond_gate',.35)['exact_correspondence'])

    def test_wider_gate_can_destroy_correspondence_for_aligned_sheets(self):
        self.assertTrue(self.metric('approaching_aligned_sheets',.2)['exact_correspondence'])
        self.assertEqual(len(self.metric('approaching_aligned_sheets',.35)['missed_links']),2)
        self.assertTrue(self.metric('approaching_opposite_normals',.35)['exact_correspondence'])

    def test_nearby_disappearing_sheet_exposes_false_link(self):
        self.assertEqual(self.metric('nearby_sheet_disappears')['false_links'],[[1,10]])
        self.assertTrue(self.metric('distant_sheet_disappears')['exact_correspondence'])

    def test_split_failure_and_fragment_pooling_are_distinct(self):
        self.assertEqual(self.metric('one_group_splits_into_two')['missed_links'],[[0,10],[0,11]])
        self.assertTrue(self.metric('two_fragments_form_one_group')['exact_correspondence'])

    def test_fresh_sampling_does_not_require_same_positions(self):
        self.assertTrue(self.metric('fresh_sampling')['exact_correspondence'])


if __name__=='__main__':unittest.main()
