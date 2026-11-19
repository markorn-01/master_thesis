from copy import deepcopy
import unittest

import numpy as np

from experiments.astrolink.audit_delayed_support import check_window
from experiments.astrolink.delayed_support import delayed_support
from experiments.astrolink.freeze_multiframe_baseline import adjacent_baseline, synthetic_cases


class DelayedSupport(unittest.TestCase):
    def data(self, name, gate=.07):
        case = next(c for c in synthetic_cases() if c['name'] == name)
        current = adjacent_baseline(case['catalogs'], gate)
        future = adjacent_baseline(case['catalogs'][1:], gate) if len(case['catalogs']) == 3 else None
        return case, current, future

    def apply(self, name, predict=True, gate=.07):
        case, current, future = self.data(name, gate)
        result = delayed_support(case['catalogs'], case['times'], current, [future], predict=predict)
        check_window(case['catalogs'], case['times'], current, [future], result)
        return result

    def test_automatically_seed_eight_fragment_family(self):
        result = self.apply('eight_fragments')
        self.assertEqual(result['future_continuations'], [[10, 20]])
        self.assertEqual(len(result['conditional_support']), 8)
        self.assertEqual(result['analysis']['families'][0]['child_union_coverage'], 1.)
        self.assertFalse(result['inherited_id_assigned'])
        for row in result['conditional_support']:
            self.assertNotIn('accepted', row)

    def test_motion_ablation_uses_future_geometry(self):
        for name, parent in [('future_motion_positive', 0), ('future_motion_negative', 1)]:
            motion = self.apply(name)
            static = self.apply(name, predict=False)
            self.assertEqual([r['parent_group'] for r in motion['conditional_support']], [parent])
            self.assertEqual(len(static['conditional_support']), 2)

    def test_no_seed_when_future_displacement_exceeds_gate(self):
        result = self.apply('future_motion_positive', gate=.05)
        self.assertEqual(result['window_status'], 'no_unprotected_future_seed')
        self.assertEqual(result['conditional_support'], [])

    def test_identical_observations_remain_conditional(self):
        a = self.apply('unobservable_ancestry_0')
        b = self.apply('unobservable_ancestry_1')
        self.assertEqual(a, b)
        self.assertEqual(len(a['conditional_support']), 2)
        self.assertFalse(a['physical_ancestry_established'])

    def test_ambiguous_future_cannot_become_truth_seed(self):
        result = self.apply('coincident_child_families')
        self.assertEqual(result['future_continuations'], [])
        self.assertEqual(result['conditional_support'], [])

    def test_existing_continuation_is_protected(self):
        result = self.apply('opposite_normal_distractor')
        self.assertEqual(result['protected_continuations'], [[0, 10]])
        self.assertEqual(result['conditional_support'], [])

    def test_real_seed_requires_both_gate_acceptance(self):
        case, current, future = self.data('eight_fragments')
        absent = deepcopy(future); absent['events'] = []
        result = delayed_support(case['catalogs'], case['times'], current, [future, absent])
        self.assertEqual(result['conditional_support'], [])

    def test_terminal_window_stays_pending(self):
        case, current, _ = self.data('eight_fragments')
        result = delayed_support(case['catalogs'][:2], case['times'][:2], current)
        self.assertEqual(result['window_status'], 'pending_future')
        self.assertEqual(result['accepted_adjacent_events'], current['events'])
        with self.assertRaises(ValueError):
            delayed_support(case['catalogs'][:2], case['times'][:2], current, [current])

    def test_preserves_inputs_and_acceptance_by_value(self):
        case, current, future = self.data('eight_fragments')
        old = [{k: v.copy() for k, v in c.items()} for c in case['catalogs']]
        previous = deepcopy(current)
        result = delayed_support(case['catalogs'], case['times'], current, [future])
        for a, b in zip(old, case['catalogs']):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(current, previous)
        result['accepted_adjacent_events'].append({'changed': True})
        self.assertEqual(current, previous)

    def test_reject_changed_thresholds_and_bad_times(self):
        case, current, future = self.data('eight_fragments')
        changed = deepcopy(current); changed['parameters']['coverage'] = .4
        with self.assertRaises(ValueError):
            delayed_support(case['catalogs'], case['times'], changed, [future])
        with self.assertRaises(ValueError):
            delayed_support(case['catalogs'], [0, 0, 2], current, [future])

    def test_conflict_is_reported_without_overriding_event(self):
        case, current, future = self.data('eight_fragments')
        # Inject a frozen decision to test the evidence-channel policy, not
        # the optimizer. A different accepted target must remain untouched.
        current['events'] = [dict(kind='combination', parents=[0, 1], children=[99])]
        current['links'] = [dict(parent_group=i, child_group=99, accepted=True) for i in (0, 1)]
        result = delayed_support(case['catalogs'], case['times'], current, [future])
        self.assertEqual(result['accepted_adjacent_events'], current['events'])
        self.assertEqual({r['parent_group'] for r in result['conflicts']}, {0, 1})
        check_window(case['catalogs'], case['times'], current, [future], result)

    def test_dense_audit_handles_float32_simulation_geometry(self):
        case, _, _ = self.data('eight_fragments')
        catalogs = [{k: v.astype(np.float32) if k != 'labels' else v.copy()
                     for k, v in data.items()} for data in case['catalogs']]
        for data in catalogs:
            data['normals'][:] = [.1234, .2543, .9989]
        current = adjacent_baseline(catalogs, .07)
        future = adjacent_baseline(catalogs[1:], .07)
        result = delayed_support(catalogs, case['times'], current, [future])
        check_window(catalogs, case['times'], current, [future], result)


if __name__ == '__main__':
    unittest.main()
