import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import experiments.astrolink.event_tracking as matching
from experiments.astrolink.expanded_event_tracking import expanded_tracking
from experiments.wind_bubble.check_five_fragment_events import new_cases
from experiments.wind_bubble.profile_tracking_replay import (
    compare_payloads, payload_digest, product_payload, replay,
)
from tests.experiments.test_streaming_history import catalog


class TrackingReplayProfile(unittest.TestCase):
    def test_expanded_matcher_builds_one_tree_and_selects_only_final_candidates(self):
        # The five-way event needs the extension; its final selection must not
        # be replaced with the earlier maximum-three result.
        case = next(c for c in new_cases() if c['name'] == 'five_disjoint_fragments')
        args = [case[k] for k in ('positions_a', 'normals_a', 'labels_a',
                                 'positions_b', 'normals_b', 'labels_b')]
        with patch.object(matching, 'cKDTree', wraps=matching.cKDTree) as trees, \
                patch.object(matching, 'select_events', wraps=matching.select_events) as selection:
            result = expanded_tracking(*args, .04)
        self.assertEqual(trees.call_count, 1)
        self.assertEqual(selection.call_count, 1)
        self.assertEqual(result['events'][0]['parents'], [0, 1, 2, 3, 4])
        self.assertEqual(result['events'][0]['children'], [10])

    def test_semantic_payload_ignores_container_headers_but_checks_dtype_and_values(self):
        with tempfile.TemporaryDirectory() as folder:
            roots = [Path(folder)/name for name in ('a', 'b')]
            for index, root in enumerate(roots):
                root.mkdir()
                np.savez_compressed(root/'groups_0000.npz', labels=np.array([0, -1], np.int32))
                with gzip.GzipFile(filename=str(root/'event_0000.json.gz'), mode='wb', mtime=index+1) as stream:
                    stream.write(json.dumps({'time': .1, 'events': []}).encode())
            a, b = map(product_payload, roots)
            compare_payloads(a, b)
            self.assertEqual(payload_digest(a), payload_digest(b))
            np.savez_compressed(roots[1]/'groups_0000.npz', labels=np.array([0, -1], np.int64))
            with self.assertRaises(AssertionError):
                compare_payloads(a, product_payload(roots[1]))

    def test_archive_float_tolerance_does_not_relax_categories(self):
        a = {'event': 2, 'median_mach': 2.0, 'status': 'ambiguous'}
        b = dict(a, median_mach=2.0000001)
        compare_payloads(a, b, exact=False)
        with self.assertRaises(AssertionError):
            compare_payloads(a, b)
        with self.assertRaises(AssertionError):
            compare_payloads(a, dict(b, status='assigned'), exact=False)

    def test_csv_scalar_values_use_archive_float_tolerance(self):
        with tempfile.TemporaryDirectory() as folder:
            roots = [Path(folder)/name for name in ('a', 'b')]
            for index, root in enumerate(roots):
                root.mkdir()
                with (root/'group_history.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['time', 'mach_p16', 'median_mach', 'mach_p84', 'group'])
                    writer.writeheader()
                    writer.writerow(dict(time='.1', mach_p16='', median_mach=str(2+index*1e-7), mach_p84='3', group='1'))
            a, b = map(product_payload, roots)
            compare_payloads(a, b, exact=False)
            with self.assertRaises(AssertionError):
                compare_payloads(a, b)

    def test_instrumentation_preserves_products_and_tracks_empty_and_pending_events(self):
        catalogs = [catalog(0, []), catalog(1), catalog(2)]
        _, plain, _, _ = replay(catalogs, expanded_tracking)
        _, measured, rows, calls = replay(catalogs, expanded_tracking, instrument=True)
        compare_payloads(measured, plain)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]['surface_samples'], 0)
        self.assertEqual(calls['adjacent_association'], 4)
        # One completed A-B-C window plus the terminal pending window.
        self.assertEqual(calls['delayed_static'], 4)
        self.assertEqual(calls['delayed_motion'], 4)
        self.assertEqual(measured['finish.json']['footprint']['retained_geometry_catalogs'], 0)
        self.assertEqual(len(measured['finish.json']['pending_terminal_support']), 4)


if __name__ == '__main__':
    unittest.main()
