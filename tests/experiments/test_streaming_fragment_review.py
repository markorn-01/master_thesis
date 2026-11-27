from copy import deepcopy
import unittest

from experiments.wind_bubble.review_streaming_fragments import family_evidence


def channels():
    accepted = {1, 2, 3, 4, 5}
    records = []
    for method in ('delayed_static_family', 'delayed_normal_motion_family'):
        for gate in (1.5, 2.5):
            family = dict(child_group=10, later_group=20, parent_groups=[1, 2, 3, 4, 5, 6],
                          coverage_pass=True, child_union_coverage=.8)
            links = [dict(parent_group=p, child_group=10, already_in_adjacent=p in accepted,
                          conflicting_adjacent_targets=[], physical_ancestry_established=False)
                     for p in range(1, 7)]
            records.append(dict(method=method, gate_reference_cells=gate,
                result=dict(window_status='evaluated', physical_ancestry_established=False,
                            inherited_id_assigned=False, conflicts=[],
                            analysis=dict(families=[family]), conditional_support=links)))
    return records, accepted


class StreamingFragmentReview(unittest.TestCase):
    def test_sixth_fragment_stays_conditional_in_all_channels(self):
        records, accepted = channels()
        before = deepcopy(records)
        result = family_evidence(records, 10, accepted)
        self.assertEqual(result['additional'], [6])
        self.assertEqual(result['later'], 20)
        self.assertEqual(result['roles'], {1:'accepted', 2:'accepted', 3:'accepted',
                                           4:'accepted', 5:'accepted', 6:'conditional'})
        self.assertEqual(records, before)

    def test_missing_or_repeated_channel_cannot_be_consensus(self):
        records, accepted = channels()
        for changed in (records[:3], records[:3]+[deepcopy(records[0])]):
            with self.assertRaises(ValueError):
                family_evidence(changed, 10, accepted)

    def test_disagreeing_later_group_cannot_be_hidden(self):
        records, accepted = channels()
        records[-1]['result']['analysis']['families'][0]['later_group'] = 21
        with self.assertRaises(ValueError):
            family_evidence(records, 10, accepted)

    def test_conditional_link_cannot_be_promoted(self):
        records, accepted = channels()
        records[0]['result']['conditional_support'][-1]['already_in_adjacent'] = True
        with self.assertRaises(ValueError):
            family_evidence(records, 10, accepted)

    def test_inherited_identity_conflict_and_failed_coverage_rejected(self):
        records, accepted = channels()
        for key, value in (('inherited_id_assigned', True), ('physical_ancestry_established', True),
                           ('conflicts', [{'parent_group':6}]), ('window_status', 'pending_future')):
            changed = deepcopy(records); changed[0]['result'][key] = value
            with self.assertRaises(ValueError):
                family_evidence(changed, 10, accepted)
        records[0]['result']['analysis']['families'][0]['coverage_pass'] = False
        with self.assertRaises(ValueError):
            family_evidence(records, 10, accepted)

    def test_missing_or_repeated_family_member_rejected(self):
        records, accepted = channels()
        for parents in ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 6]):
            changed = deepcopy(records)
            changed[0]['result']['analysis']['families'][0]['parent_groups'] = parents
            with self.assertRaises(ValueError):
                family_evidence(changed, 10, accepted)


if __name__ == '__main__':
    unittest.main()
