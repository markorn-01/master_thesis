"""Join callback group statistics to causal visualization tables."""
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.wind_bubble.streaming_group_demo import build_tables
from experiments.wind_bubble.streaming_products import StreamingProductWriter, load_event
from tests.experiments.test_streaming_products import catalog


class StreamingGroupDemo(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory/'simulation'
        self.root.mkdir()
        self.sources = [[.3, .3, .5], [.7, .3, .5], [.5, .7, .4]]
        self.tracking = self.root/'online/tracking'
        self.writer = StreamingProductWriter(self.tracking)
        self.addCleanup(self.writer.close, success=False)
        for event in range(3):
            data = catalog(event)
            data['source_positions'] = np.asarray(self.sources)
            np.savez_compressed(self.root/'online'/f'surfaces_{event:04d}.npz', **data)
            self.writer.receive(data)
        self.writer.close()
        self.metadata = dict(status='complete', backend='unit_test_fixture', temporal_tracking=True,
                             source_positions=self.sources, actual_times=[0., .1, .2], verify_offline=False)
        self.write_metadata()

    def write_metadata(self):
        (self.root/'metadata.json').write_text(json.dumps(self.metadata))

    def rewrite_event(self, index, update):
        record = load_event(self.tracking, index)
        update(record)
        with gzip.open(self.tracking/f'event_{index:04d}.json.gz', 'wt') as stream:
            json.dump(record, stream)

    def test_callback_tables_preserve_rows_segments_and_visibility(self):
        output = self.directory/'tables'
        report = build_tables(self.root, output)
        self.assertEqual(report['origin'], 'hydrodynamic_host_callback')
        self.assertEqual(report['events'], 3)
        self.assertEqual(report['group_observations'], 3)
        self.assertEqual(report['segments'], 1)
        rows = json.loads((output/'group_history.json').read_text())
        self.assertEqual([r['median_mach'] for r in rows], [2., 2., 2.])
        visible = json.loads((output/'visibility.json').read_text())
        self.assertEqual(visible[0]['visible_node_ids'], ['e0000:g0'])
        self.assertEqual(visible[0]['visible_event_ids'], [])
        self.assertEqual(len(visible[1]['visible_node_ids']), 2)
        self.assertEqual(len(visible[1]['visible_event_ids']), 1)
        self.assertEqual(len(visible[2]['visible_node_ids']), 3)
        self.assertEqual(json.loads((output/'conditional_support.json').read_text()), [])
        terminal = json.loads((output/'pending_terminal_support.json').read_text())
        self.assertTrue(all(r['available_event'] is None for r in terminal))

    def test_changed_original_sample_cannot_reuse_its_statistics(self):
        path = self.root/'online/surfaces_0001.npz'
        with np.load(path) as saved:
            data = {k:saved[k].copy() for k in saved.files}
        data['positions'][0, 0] += .001
        np.savez_compressed(path, **data)
        with self.assertRaisesRegex(ValueError, 'statistics differ'):
            build_tables(self.root, self.directory/'tables')

    def test_future_support_cannot_appear_one_event_early(self):
        self.rewrite_event(2, lambda r:r['support_records'][0].update(available_event=1))
        with self.assertRaisesRegex(ValueError, 'available at C'):
            build_tables(self.root, self.directory/'tables')

    def test_duplicate_support_channel_is_rejected(self):
        def repeat(record):
            record['support_records'][1] = record['support_records'][0]
        self.rewrite_event(2, repeat)
        with self.assertRaisesRegex(ValueError, 'repeated delayed'):
            build_tables(self.root, self.directory/'tables')

    def test_saved_catalog_run_is_not_silently_called_online_tracking(self):
        self.metadata['temporal_tracking'] = False
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'selected explicitly'):
            build_tables(self.root, self.directory/'tables')

    def test_failed_requested_equivalence_cannot_become_successful_demo(self):
        self.metadata['verify_offline'] = True
        self.write_metadata()
        (self.root/'equivalence.json').write_text(json.dumps(dict(passed=False)))
        with self.assertRaisesRegex(ValueError, 'verification is missing or failed'):
            build_tables(self.root, self.directory/'tables')

    def test_group_csv_must_match_original_callback_rows(self):
        path = self.tracking/'group_history.csv'
        path.write_text(path.read_text().replace('2.0', '999.0'))
        with self.assertRaisesRegex(ValueError, 'CSV does not reproduce'):
            build_tables(self.root, self.directory/'tables')

    def test_small_percentile_rounding_difference_preserves_recorded_value(self):
        value = float(np.nextafter(np.float32(2.), np.float32(np.inf)))
        self.rewrite_event(1, lambda r:r['group_statistics'][0].update(mach_p84=value))
        path = self.tracking/'group_history.csv'
        with path.open() as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        rows[1]['mach_p84'] = str(value)
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        output = self.directory/'tables'
        report = build_tables(self.root, output)
        saved = json.loads((output/'group_history.json').read_text())
        self.assertEqual(saved[1]['mach_p84'], value)
        self.assertTrue(report['recorded_callback_statistics_preserved_exact'])

    def test_large_percentile_difference_is_rejected(self):
        self.rewrite_event(1, lambda r:r['group_statistics'][0].update(mach_p84=3.))
        with self.assertRaisesRegex(ValueError, 'statistics differ'):
            build_tables(self.root, self.directory/'tables')


if __name__ == '__main__':
    unittest.main()
