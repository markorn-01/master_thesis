import unittest

import numpy as np

from experiments.wind_bubble.build_group_history_table import statistics, visible_relationships


class GroupHistoryTable(unittest.TestCase):
    def fixture(self):
        data = dict(positions=np.arange(12).reshape(4, 3), time=np.array(.1),
                    mach=np.array([2., 4., np.nan, 9.]), mach_valid=np.array([True, True, False, True]),
                    upstream_inside_any_source=np.array([True, False, False, True]),
                    downstream_inside_any_source=np.zeros(4, bool))
        node = dict(node_id='e0001:g7', event=1, time=.1, group=7, segment_id=3,
                    previous_group=4, inherited=True, surface_samples=3, valid_mach_samples=2,
                    median_mach=3., upstream_inside_source=1, downstream_inside_source=0)
        return data, np.array([7, 7, 7, -1]), node

    def test_statistics_exclude_invalid_values_and_unretained_samples(self):
        data, labels, node = self.fixture()
        rows, accounting = statistics(data, labels, [node], 1)
        self.assertEqual(rows[0]['median_mach'], 3.)
        self.assertEqual(rows[0]['invalid_mach_samples'], 1)
        self.assertEqual(accounting['detected_samples'], 4)
        self.assertEqual(accounting['retained_samples'], 3)
        self.assertEqual(accounting['unretained_samples'], 1)
        self.assertEqual(accounting['all_upstream_source_overlaps'], 2)

    def test_reject_duplicate_or_missing_groups(self):
        data, labels, node = self.fixture()
        for nodes in ([], [node, node]):
            with self.assertRaises(ValueError):
                statistics(data, labels, nodes, 1)

    def test_initial_empty_event_has_no_fabricated_group_or_mach(self):
        data, _, _ = self.fixture()
        data = {k: v[:0] if np.asarray(v).ndim else np.array(0.) for k, v in data.items()}
        rows, event = statistics(data, np.array([], int), [], 0)
        self.assertEqual(rows, [])
        self.assertEqual(event['detected_samples'], 0)

    def test_wrong_time_or_changed_mach_rejected(self):
        data, labels, node = self.fixture()
        for changed in (dict(node, time=.2), dict(node, median_mach=8.)):
            with self.assertRaises(ValueError):
                statistics(data, labels, [changed], 1)

    def test_conditional_edge_appears_at_c_not_at_a_or_b(self):
        baseline = [dict(available_event=14)]
        support = [dict(parent_event=13, child_event=14, available_event=15)]
        self.assertEqual(visible_relationships(baseline, support, 13), ([], []))
        self.assertEqual(visible_relationships(baseline, support, 14), (baseline, []))
        self.assertEqual(visible_relationships(baseline, support, 15), (baseline, support))


if __name__ == '__main__':
    unittest.main()
