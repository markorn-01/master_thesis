import unittest
from unittest.mock import patch
import weakref

import numpy as np

from experiments.astrolink.streaming_history import StreamingShockHistory
from experiments.astrolink.surface_history import HistorySettings, analyze_surface_history


def catalog(event, points=None):
    if points is None:
        points = [[x, y, event*.001] for x in (0., .01) for y in (0., .01)]
    p = np.asarray(points, np.float32).reshape(-1, 3)
    return dict(positions=p, normals=np.tile([0., 0., 1.], (len(p), 1)).astype(np.float32),
                time=float(event)*.1, grid_spacing=.01, source_positions=np.array([[.3, .3, .3]]),
                mach=np.full(len(p), 2., np.float32), mach_valid=np.ones(len(p), bool),
                upstream_inside_any_source=np.zeros(len(p), bool),
                downstream_inside_any_source=np.zeros(len(p), bool))


SETTINGS = HistorySettings(max_partners=5, min_group_samples=1)


class StreamingHistory(unittest.TestCase):
    def test_batch_and_incremental_partitions_segments_and_events_match(self):
        catalogs = [catalog(i) for i in range(5)]
        batch = analyze_surface_history(catalogs, [c['time'] for c in catalogs], SETTINGS)
        stream = StreamingShockHistory(SETTINGS)
        deltas = [stream.push(c) for c in catalogs]
        final = stream.finish()
        for delta, expected in zip(deltas, batch['partitions']):
            for key in expected:
                np.testing.assert_array_equal(delta['partition'][key], expected[key])
        nodes = [n for d in deltas for n in d['nodes']]
        for a, b in zip(nodes, batch['graph']['nodes']):
            self.assertEqual({k: a[k] for k in b}, b)
        self.assertEqual([e for d in deltas for e in d['tracking_events']], batch['graph']['events'])
        self.assertEqual(final['segment_count'], batch['graph']['segment_count'])

    def test_delayed_results_emitted_only_when_C_arrives(self):
        a = catalog(0, [[0, 0, 0], [.04, 0, 0]])
        b = catalog(1, [[x, 0, 0] for x in np.arange(5)*.01])
        c = catalog(2, b['positions']+[0, 0, .001])
        stream = StreamingShockHistory(SETTINGS)
        self.assertEqual(stream.push(a)['support_records'], [])
        before = stream.push(b)
        self.assertEqual(before['support_records'], [])
        self.assertEqual(before['tracking_events'][0]['kind'], 'combination')
        after = stream.push(c)
        self.assertEqual(len(after['support_records']), 4)
        for row in after['support_records']:
            self.assertEqual((row['parent_event'], row['child_event'], row['available_event']), (0, 1, 2))
            self.assertEqual(row['available_time'], c['time'])
            self.assertEqual(len(row['result']['conditional_support']), 2)
            self.assertFalse(row['result']['inherited_id_assigned'])

    def test_finish_keeps_last_transition_pending_and_releases_buffers(self):
        stream = StreamingShockHistory(SETTINGS)
        stream.push(catalog(0)); stream.push(catalog(1))
        final = stream.finish()
        self.assertEqual(len(final['pending_terminal_support']), 4)
        for row in final['pending_terminal_support']:
            self.assertIsNone(row['available_event'])
            self.assertEqual(row['result']['window_status'], 'pending_future')
        self.assertEqual(stream.footprint()['retained_geometry_bytes'], 0)
        self.assertEqual(stream.footprint()['retained_adjacent_intervals'], 0)
        with self.assertRaises(ValueError): stream.push(catalog(2))
        with self.assertRaises(ValueError): stream.finish()

    def test_bounded_state_and_old_arrays_released(self):
        stream = StreamingShockHistory(SETTINGS)
        stream.push(catalog(0))
        old = weakref.ref(stream._recent[0]['geometry']['positions'])
        sizes = []
        for i in range(1, 40):
            delta = stream.push(catalog(i))
            footprint = delta['footprint']; sizes.append(footprint['retained_geometry_bytes'])
            self.assertLessEqual(footprint['retained_geometry_catalogs'], 2)
            self.assertEqual(footprint['retained_adjacent_intervals'], 1)
            self.assertEqual(footprint['retained_group_mapping_entries'], 1)
        self.assertIsNone(old())
        self.assertEqual(len(set(sizes)), 1)
        self.assertEqual(stream.footprint()['max_processing_geometry_catalogs'], 3)

    def test_caller_input_and_emitted_results_cannot_change_retained_state(self):
        stream = StreamingShockHistory(SETTINGS)
        source = catalog(0); stream.push(source)
        source['positions'][:] = 999
        delta = stream.push(catalog(1))
        self.assertEqual(delta['nodes'][0]['segment_id'], 0)
        delta['partition']['labels'][:] = -1
        delta['adjacent_records'][0]['result']['parameters']['coverage'] = .1
        after = stream.push(catalog(2))
        self.assertEqual(after['nodes'][0]['segment_id'], 0)
        for row in after['support_records']:
            self.assertEqual(row['result']['baseline_parameters']['coverage'], .5)

    def test_invalid_push_does_not_advance_the_state(self):
        stream = StreamingShockHistory(SETTINGS); stream.push(catalog(0))
        before = stream.footprint()
        for bad in (catalog(0), dict(catalog(1), grid_spacing=.02),
                    dict(catalog(1), source_positions=np.array([[1., 1., 1.]])),
                    dict(catalog(1), mach_valid=np.ones(4, int))):
            with self.assertRaises(ValueError): stream.push(bad)
            self.assertEqual(before, stream.footprint())
        self.assertEqual(stream.push(catalog(1))['event'], 1)

    def test_algorithm_failure_does_not_commit_a_partial_event(self):
        stream = StreamingShockHistory(SETTINGS)
        stream.push(catalog(0)); stream.push(catalog(1))
        before = stream.footprint()
        with patch('experiments.astrolink.streaming_history.delayed_support', side_effect=RuntimeError('audit failure')):
            with self.assertRaises(RuntimeError): stream.push(catalog(2))
        self.assertEqual(stream.footprint(), before)
        retried = stream.push(catalog(2))
        self.assertEqual(retried['event'], 2)
        self.assertEqual(retried['nodes'][0]['segment_id'], 0)

    def test_empty_event_breaks_continuation_without_fabricating_zero_mach(self):
        stream = StreamingShockHistory(SETTINGS)
        first = stream.push(catalog(0))
        empty = stream.push(catalog(1, []))
        later = stream.push(catalog(2))
        self.assertEqual(empty['group_statistics'], [])
        self.assertEqual(empty['accounting']['detected_samples'], 0)
        self.assertNotEqual(first['nodes'][0]['segment_id'], later['nodes'][0]['segment_id'])

    def test_missing_mach_and_source_flags_match_the_current_observation(self):
        data = catalog(0)
        data['mach'][:] = [1., 2., np.nan, 9.]
        data['mach_valid'][3] = False
        data['upstream_inside_any_source'][0] = True
        delta = StreamingShockHistory(SETTINGS).push(data)
        row = delta['group_statistics'][0]
        self.assertEqual(row['median_mach'], 1.5)
        self.assertEqual(row['invalid_mach_samples'], 2)
        self.assertEqual(row['upstream_source_overlaps'], 1)

    def test_entirely_empty_or_single_event_stream_can_close(self):
        empty = StreamingShockHistory(SETTINGS).finish()
        self.assertEqual(empty['events_received'], 0)
        stream = StreamingShockHistory(SETTINGS); stream.push(catalog(0, []))
        final = stream.finish()
        self.assertEqual(final['segment_count'], 0)
        self.assertEqual(final['pending_terminal_support'], [])


if __name__ == '__main__':
    unittest.main()
