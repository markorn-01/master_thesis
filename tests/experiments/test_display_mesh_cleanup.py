"""Geometry defects must be removed without inventing replacement faces."""
import unittest
import csv
from pathlib import Path
import tempfile

import numpy as np

from experiments.wind_bubble.finalize_surface_display import remove_defects
from experiments.wind_bubble.research_surface_display import write_metrics


class DisplayMeshCleanupTests(unittest.TestCase):
    def test_intersecting_faces_are_removed_without_a_replacement(self):
        p = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.],
                      [.5, .5, -1.], [.5, .5, 1.], [1.5, .5, 0.]])
        v, f, _, report = remove_defects(p, [[0, 1, 2], [3, 4, 5]], np.zeros(6, int))
        self.assertEqual(len(f), 0); self.assertEqual(len(v), 0)
        self.assertEqual(report['removed_faces'], 2)

    def test_valid_open_patch_is_preserved_exactly(self):
        p = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        v, f, groups, report = remove_defects(p, faces, np.full(4, 2))
        np.testing.assert_array_equal(v[f], p[faces])
        np.testing.assert_array_equal(groups, 2)
        self.assertEqual(report['removed_faces'], 0)

    def test_truth_only_metric_columns_are_retained_without_inventing_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'metrics.csv'
            write_metrics(path, [dict(case='sphere', faces=10),
                                 dict(case='two_sheets', faces=8, off_true_sheet_probe_fraction=.5)])
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['off_true_sheet_probe_fraction'], '')
            self.assertEqual(rows[1]['off_true_sheet_probe_fraction'], '0.5')


if __name__ == '__main__':
    unittest.main()
