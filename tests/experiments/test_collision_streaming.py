"""Integration of the optional tracker with the existing collision callback."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.wind_bubble.run_two_bubbles import CollisionWriter, compare_outputs
from tests.experiments.test_streaming_products import catalog


class CollisionStreaming(unittest.TestCase):
    def test_current_catalog_reaches_tracker_without_changing_primary_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            centers = np.zeros((2, 2, 2, 3))
            plain = CollisionWriter(Path(root)/'plain', centers, .01)
            tracked = CollisionWriter(Path(root)/'tracked', centers, .01, track_shocks=True)
            reference = CollisionWriter(Path(root)/'reference', centers, .01, track_shocks=True)
            try:
                for event in range(3):
                    data = catalog(event)
                    with patch('experiments.wind_bubble.run_two_bubbles.make_catalog', return_value=(data, 25)):
                        for writer in (plain, tracked, reference):
                            with patch('numpy.load', side_effect=AssertionError('Unexpected callback readback')):
                                writer.receive(float(data['time']), *[np.ones((2, 2)) for _ in range(12)])
                for writer in (plain, tracked, reference):
                    writer.close()
                self.assertEqual(plain.rows, tracked.rows)
                for file in plain.directory.glob('*.npz'):
                    with np.load(file) as a, np.load(tracked.directory/file.name) as b:
                        self.assertEqual(a.files, b.files)
                        for key in a.files:
                            np.testing.assert_array_equal(a[key], b[key])
                self.assertTrue(compare_outputs(tracked, reference)['streaming_tracking_match'])
                self.assertIsNone(plain.tracking)
                self.assertEqual(tracked.tracking.events, 3)
                with self.assertRaises(AssertionError):
                    compare_outputs(plain, tracked)
            finally:
                for writer in (plain, tracked, reference):
                    writer.close(success=False)

    def test_initialization_failure_closes_primary_history(self):
        with tempfile.TemporaryDirectory() as root:
            opened = []
            original_open = Path.open

            def remember(path, *args, **kwargs):
                stream = original_open(path, *args, **kwargs)
                if path.name == 'history.csv':
                    opened.append(stream)
                return stream

            with patch.object(Path, 'open', remember), patch(
                    'experiments.wind_bubble.streaming_products.StreamingProductWriter',
                    side_effect=RuntimeError('unavailable tracker')):
                with self.assertRaises(RuntimeError):
                    CollisionWriter(Path(root)/'online', np.zeros((2, 2, 2, 3)), .01, track_shocks=True)
            self.assertEqual(len(opened), 1)
            self.assertTrue(opened[0].closed)


if __name__ == '__main__':
    unittest.main()
