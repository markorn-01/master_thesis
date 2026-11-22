"""Callback product lifecycle and verification against the batch tracker."""
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.wind_bubble.streaming_products import (
    StreamingProductWriter, check_batch_reference, compare_products, load_event,
)


def catalog(event, empty=False):
    points = np.asarray([[.2+x*.01, .2+y*.01, .2+event*.001]
                         for x in range(5) for y in range(5)], np.float32)
    if empty:
        points = points[:0]
    count = len(points)
    return dict(positions=points, normals=np.tile([0., 0., 1.], (count, 1)).astype(np.float32),
                time=np.asarray(event*.1), grid_spacing=np.asarray(.01),
                source_positions=np.array([[.5, .5, .5]]),
                mach=np.full(count, 2., np.float32), mach_valid=np.ones(count, bool),
                density_jump=np.full(count, 2.), sample_positions_valid=np.ones(count, bool),
                upstream_inside_any_source=np.zeros(count, bool),
                downstream_inside_any_source=np.zeros(count, bool))


class StreamingProducts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_sequence(self, name, *, empty_initial=False):
        parent = self.root/name
        parent.mkdir()
        writer = StreamingProductWriter(parent/'tracking')
        self.addCleanup(writer.close, success=False)
        for event in range(3):
            data = catalog(event, empty=empty_initial and event == 0)
            np.savez_compressed(parent/f'surfaces_{event:04d}.npz', **data)
            # A production receive must use its argument, never reload saved catalogs.
            with patch('numpy.load', side_effect=AssertionError('Unexpected callback file read')):
                writer.receive(data)
            self.assertLessEqual(writer.metadata['footprint']['retained_geometry_catalogs'], 2)
        writer.close()
        writer.close()  # Cleanup after the successful close is deliberately idempotent.
        return writer

    def test_online_offline_and_independent_batch_agree(self):
        a, b = [self.write_sequence(name) for name in ('online', 'reference')]
        result = compare_products(a.directory, b.directory)
        self.assertTrue(result['streaming_tracking_match'])
        self.assertTrue(result['tracking_full_records_exact'])
        self.assertEqual(result['compared_group_observations'], 3)
        result = check_batch_reference(b.directory.parent, b.directory)
        self.assertTrue(result['tracking_batch_reference_match'])
        self.assertEqual(result['batch_reference_adjacent_results'], 4)
        self.assertEqual(result['batch_reference_delayed_results'], 8)
        final = json.loads((a.directory/'finish.json').read_text())
        self.assertEqual(final['footprint']['retained_geometry_catalogs'], 0)
        self.assertTrue(all(r['available_event'] is None for r in final['pending_terminal_support']))
        self.assertIsNone(a.tracker)

    def test_empty_initial_event_produces_no_invented_group(self):
        a, b = [self.write_sequence(name, empty_initial=True) for name in ('online', 'reference')]
        self.assertEqual(load_event(a.directory, 0)['group_statistics'], [])
        self.assertEqual(compare_products(a.directory, b.directory)['compared_group_observations'], 2)
        self.assertTrue(check_batch_reference(b.directory.parent, b.directory)['tracking_batch_reference_match'])

    def test_altered_categorical_tracking_result_is_rejected(self):
        a, b = [self.write_sequence(name) for name in ('online', 'reference')]
        record = load_event(b.directory, 1)
        record['nodes'][0]['segment_id'] += 1
        with gzip.open(b.directory/'event_0001.json.gz', 'wt') as stream:
            json.dump(record, stream)
        with self.assertRaises(AssertionError):
            compare_products(a.directory, b.directory)

    def test_altered_csv_is_rejected_even_when_event_json_matches(self):
        a, b = [self.write_sequence(name) for name in ('online', 'reference')]
        path = b.directory/'group_history.csv'
        with path.open() as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        rows[0]['median_mach'] = '999'
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        with self.assertRaises(AssertionError):
            compare_products(a.directory, b.directory)

    def test_altered_source_coordinates_cannot_reuse_fingerprint(self):
        a, b = [self.write_sequence(name) for name in ('online', 'reference')]
        path = b.directory.parent/'surfaces_0000.npz'
        with np.load(path) as saved:
            data = {k: saved[k].copy() for k in saved.files}
        data['positions'][0, 0] += .01
        np.savez_compressed(path, **data)
        with self.assertRaises(AssertionError):
            compare_products(a.directory, b.directory)

    def test_failed_receive_cannot_finalize_as_complete(self):
        writer = StreamingProductWriter(self.root/'tracking')
        writer.receive(catalog(0))
        with self.assertRaises(ValueError):
            writer.receive(catalog(0))
        writer.close()
        self.assertTrue(writer.closed)
        self.assertIsNone(writer.tracker)
        self.assertEqual(json.loads((writer.directory/'metadata.json').read_text())['status'], 'failed')
        self.assertFalse((writer.directory/'finish.json').exists())

    def test_aborted_or_failed_close_does_not_invent_terminal_support(self):
        for mode in ('abort', 'finish_failure'):
            writer = StreamingProductWriter(self.root/mode)
            writer.receive(catalog(0))
            if mode == 'abort':
                writer.close(success=False)
            else:
                with patch.object(writer.tracker, 'finish', side_effect=RuntimeError('failed finish')):
                    with self.assertRaises(RuntimeError):
                        writer.close()
            self.assertTrue(writer.stream.closed)
            self.assertIsNone(writer.tracker)
            self.assertEqual(json.loads((writer.directory/'metadata.json').read_text())['status'], 'failed')
            self.assertFalse((writer.directory/'finish.json').exists())


if __name__ == '__main__':
    unittest.main()
