import unittest

from experiments.astrolink.freeze_multiframe_baseline import adjacent_baseline, summarize, synthetic_cases


class FrozenMultiframeBaseline(unittest.TestCase):
    def test_baseline_cannot_access_future_catalog(self):
        case = next(c for c in synthetic_cases() if c['name'] == 'eight_fragments')

        class ForbiddenFuture(dict):
            def __getitem__(self, key):
                raise AssertionError('Future information leaked into adjacent baseline.')

        before = adjacent_baseline(case['catalogs'], .07)
        after = adjacent_baseline(case['catalogs'][:2]+[ForbiddenFuture()], .07)
        self.assertEqual(before, after)

    def test_unknown_truth_is_not_reported_as_perfect_accuracy(self):
        result = dict(links=[], events=[], parents=[dict(status='ambiguous')],
                      children=[dict(status='unmatched')])
        row = summarize(result)
        self.assertNotIn('false_links', row)
        self.assertNotIn('exact_correspondence', row)
        self.assertEqual(row['ambiguous_parents'], 1)
        self.assertEqual(row['unmatched_children'], 1)

    def test_false_and_missed_links_are_separate(self):
        result = dict(links=[dict(parent_group=0, child_group=10, accepted=True),
                             dict(parent_group=1, child_group=10, accepted=False)],
                      events=[], parents=[], children=[])
        row = summarize(result, [(1, 10)])
        self.assertEqual(row['accepted_links'], 1)
        self.assertEqual(row['false_links'], 1)
        self.assertEqual(row['missed_links'], 1)
        self.assertFalse(row['exact_correspondence'])


if __name__ == '__main__':
    unittest.main()
