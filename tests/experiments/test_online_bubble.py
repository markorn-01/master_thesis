"""Accounting and surface-integration checks for compact online products."""
import unittest

import numpy as np

from experiments.wind_bubble.run_online_bubble import distribution, summarize


class OnlineBubbleTests(unittest.TestCase):
    def test_histogram_accounts_for_invalid_and_out_of_range_samples(self):
        hist, info = distribution([np.nan, np.inf, -1., 0., 1., 2., 3., 4., 5.], [2., 3., 4.])
        self.assertEqual(info['invalid_count'], 4)
        self.assertEqual(info['valid_count'], 5)
        self.assertEqual(info['underflow'], 1)
        self.assertEqual(info['overflow'], 1)
        self.assertEqual(hist.sum(), 3)  # Last edge is included in the last bin.
        self.assertEqual(info['valid_count'], hist.sum()+info['underflow']+info['overflow'])

    def make_fields(self):
        n = 16
        shape = (n,)*3
        centers = np.stack(np.meshgrid(*([(np.arange(n)+.5)/n]*3), indexing='ij'), axis=-1)
        mask = np.zeros(shape, bool)
        for x in (9, 14):
            mask[x, 6:10, 7:9] = True
        normals = np.zeros((3,)+shape)
        normals[0] = 1.
        return centers, mask, normals

    def test_area_is_deduplicated_per_band_not_across_distinct_surfaces(self):
        centers, mask, normals = self.make_fields()
        shape = mask.shape
        row, mh, rh = summarize(.2, mask, np.zeros(shape), normals,
                                np.full(shape, 5.), np.full(shape, 2.),
                                np.full(shape, 4.), np.ones(shape), np.ones(shape, bool),
                                centers, 1/16, .0625)
        self.assertEqual(row['surface_cells_outside_injection'], 16)
        self.assertEqual(row['reverse_surface_sample_count'], 8)
        self.assertEqual(row['forward_surface_sample_count'], 8)
        self.assertAlmostEqual(row['total_tracked_surface_area'], 16/16**2)
        self.assertAlmostEqual(row['total_tracked_dissipation_rate'], 2*16/16**2)
        self.assertEqual(mh.sum(), 16)
        self.assertEqual(rh.sum(), 16)

    def test_empty_detection_is_missing_power_not_zero(self):
        centers, mask, normals = self.make_fields()
        mask[:] = False
        row, mh, rh = summarize(0., mask, np.zeros(mask.shape), normals,
                                np.zeros(mask.shape), np.zeros(mask.shape),
                                np.ones(mask.shape), np.ones(mask.shape), np.ones(mask.shape, bool),
                                centers, 1/16, .0625)
        self.assertEqual(row['surface_cells_all'], 0)
        self.assertTrue(np.isnan(row['total_tracked_dissipation_rate']))
        self.assertTrue(np.isnan(row['mach_median']))
        self.assertEqual(mh.sum()+rh.sum(), 0)


if __name__ == '__main__':
    unittest.main()
