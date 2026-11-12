import unittest
from experiments.astrolink.event_history import assemble_history


def comparison(index, parents, children, events, gate):
    return dict(parent_event=index, gate_reference_cells=gate, result=dict(
        parents=[dict(parent_group=p) for p in parents],
        children=[dict(child_group=c) for c in children], events=events))


def event(kind, parents, children):
    return dict(kind=kind, parents=parents, children=children)


class EventHistory(unittest.TestCase):
    def records(self, groups, intervals):
        return [comparison(i, groups[i], groups[i+1], events, gate)
                for i, pair in enumerate(intervals)
                for gate, events in zip([1.5, 2.5], pair)]

    def test_combination_connects_but_does_not_inherit(self):
        groups = [[0, 1], [2], [3]]
        merge = [event('combination', [0, 1], [2])]
        continuation = [event('continuation', [2], [3])]
        graph = assemble_history(groups, [0, 1, 2], self.records(groups, [(merge, merge), (continuation, continuation)]))
        self.assertEqual([n['segment_id'] for n in graph['nodes']], [0, 1, 2, 2])
        self.assertEqual(graph['events'][0]['parents'], ['e0000:g0', 'e0000:g1'])
        self.assertEqual(graph['segment_count'], 3)

    def test_separation_starts_two_segments(self):
        groups = [[0], [1, 2]]; split = [event('separation', [0], [1, 2])]
        graph = assemble_history(groups, [0, 1], self.records(groups, [(split, split)]))
        self.assertEqual([n['segment_id'] for n in graph['nodes']], [0, 1, 2])

    def test_alternative_does_not_prevent_independent_shared_continuation(self):
        groups = [[0, 1, 2], [3, 4]]
        shared = [event('continuation', [0], [3])]
        wide = shared + [event('combination', [1, 2], [4])]
        graph = assemble_history(groups, [0, 1], self.records(groups, [(shared, wide)]))
        self.assertEqual([n['segment_id'] for n in graph['nodes']], [0, 1, 2, 0, 3])
        branch = next(e for e in graph['events'] if e['kind'] == 'combination')
        self.assertEqual(branch['status'], 'distance_specific')
        self.assertEqual(branch['selected_at_gates'], [2.5])

    def test_empty_catalog_prevents_gap_bridging(self):
        groups = [[0], [], [0]]
        graph = assemble_history(groups, [0, 1, 2], self.records(groups, [([], []), ([], [])]))
        self.assertEqual([n['segment_id'] for n in graph['nodes']], [0, 1])
        self.assertEqual(graph['events'], [])

    def test_reject_missing_distance_and_conflicting_events(self):
        groups = [[0], [1, 2]]
        a = [event('continuation', [0], [1]), event('continuation', [0], [2])]
        with self.assertRaises(ValueError):
            assemble_history(groups, [0, 1], self.records(groups, [(a, a)]))
        with self.assertRaises(ValueError):
            assemble_history(groups, [0, 1], [])

    def test_reject_reversed_times(self):
        with self.assertRaises(ValueError): assemble_history([[], []], [1, 0], [])
