"""Accounting and surface-integration checks for compact online products."""
import unittest
import tempfile
from pathlib import Path

import numpy as np

from experiments.wind_bubble.run_online_bubble import (
    distribution, summarize, DiagnosticWriter, verify_products, MACH_EDGES, DENSITY_EDGES,
)


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

    def test_sparse_output_preserves_history_and_reproduces_histograms(self):
        centers, mask, normals = self.make_fields()
        shape = mask.shape
        mach = np.full(shape, 5.)
        mach[9, 6, 7] = np.nan  # Keep this geometric sample in the catalog.
        payload = (.2, mask, np.zeros(shape), normals, mach, np.full(shape, 2.),
                   np.full(shape, .8), np.ones(shape), np.ones(shape, bool))
        with tempfile.TemporaryDirectory() as directory:
            writers = [DiagnosticWriter(Path(directory)/name, centers, 1/16, .0625,
                                         save_surfaces=enabled)
                       for name, enabled in [('plain', False), ('sparse', True)]]
            try:
                for writer in writers:
                    writer.receive(*payload)
                for key in writers[0].rows[0]:
                    np.testing.assert_equal(writers[0].rows[0][key], writers[1].rows[0][key])
                self.assertFalse((writers[0].directory/'surfaces_0000.npz').exists())
                with np.load(writers[1].directory/'surfaces_0000.npz') as catalog:
                    self.assertEqual(len(catalog['positions']), writers[1].rows[0]['surface_cells_outside_injection'])
                    self.assertEqual(int((~catalog['mach_valid']).sum()), 1)
                    self.assertTrue(np.all(catalog['density_jump'] < 1))
                    with np.load(writers[1].directory/'histograms_0000.npz') as hist:
                        for field, valid_key, edges, count_key in [
                                ('mach', 'mach_valid', MACH_EDGES, 'mach_counts'),
                                ('density_jump', 'density_jump_valid', DENSITY_EDGES, 'density_jump_counts')]:
                            reconstructed, _ = np.histogram(catalog[field][catalog[valid_key]], bins=edges)
                            np.testing.assert_array_equal(reconstructed, hist[count_key])
                for writer in writers:
                    with np.load(writer.directory/'histograms_0000.npz') as hist:
                        np.testing.assert_array_equal(hist['mach_counts'].sum(), 15)
            finally:
                for writer in writers:
                    writer.close()

    def test_equivalence_checks_sparse_properties_not_only_summaries(self):
        centers, mask, normals = self.make_fields()
        shape = mask.shape
        payload = (.2, mask, np.zeros(shape), normals, np.full(shape, 5.),
                   np.full(shape, 2.), np.full(shape, 4.), np.ones(shape), np.ones(shape, bool))
        with tempfile.TemporaryDirectory() as directory:
            writers = [DiagnosticWriter(Path(directory)/name, centers, 1/16, .0625,
                                         save_surfaces=True) for name in ['online', 'offline']]
            try:
                for writer in writers:
                    writer.receive(*payload)
                result = verify_products(*writers)
                self.assertTrue(result['sparse_surfaces_match'])
                target = writers[1].directory/'surfaces_0000.npz'
                with np.load(target) as archive:
                    modified = {key: archive[key] for key in archive.files}
                modified['positions'][0, 0] += .1
                np.savez_compressed(target, **modified)
                with self.assertRaises(AssertionError):
                    verify_products(*writers)
            finally:
                for writer in writers:
                    writer.close()

    def test_temporal_products_preserve_primary_diagnostics_and_are_verified(self):
        centers, mask, normals = self.make_fields()
        shape = mask.shape
        fields = (mask, np.zeros(shape), normals, np.full(shape, 5.),
                  np.full(shape, 2.), np.full(shape, 4.), np.ones(shape), np.ones(shape, bool))
        with tempfile.TemporaryDirectory() as directory:
            writers = [DiagnosticWriter(Path(directory)/name, centers, 1/16, .0625,
                                         track_shocks=enabled, integrate_energy=enabled)
                       for name, enabled in [('plain', False), ('online', True), ('offline', True)]]
            try:
                for writer in writers:
                    for time in (.1, .3):
                        writer.receive(time, *fields)
                for a, b in zip(writers[0].rows, writers[1].rows):
                    self.assertEqual(len(a), 41)
                    for key in a:
                        np.testing.assert_equal(a[key], b[key])
                for i in range(2):
                    with np.load(writers[0].directory/f'histograms_{i:04d}.npz') as a, np.load(writers[1].directory/f'histograms_{i:04d}.npz') as b:
                        for key in a.files:
                            np.testing.assert_array_equal(a[key], b[key])
                self.assertFalse((writers[0].directory/'tracks.csv').exists())
                self.assertFalse((writers[0].directory/'energy.csv').exists())
                online, offline = writers[1:]
                self.assertTrue((online.directory/'tracks.csv').exists())
                self.assertTrue((online.directory/'energy.csv').exists())
                self.assertEqual(len(online.temporal.track_rows), 4)
                self.assertAlmostEqual(online.temporal.energy_rows[-1]['combined_cumulative_dissipated_energy'], .025)
                verified = verify_products(online, offline)
                self.assertTrue(verified['tracking_matches'])
                self.assertTrue(verified['energy_matches'])
                self.assertTrue(verified['offline_batch_energy_matches'])
                offline.temporal.track_rows[-1]['track_status'] = 'discontinuous'
                with self.assertRaises(AssertionError):
                    verify_products(online, offline)
                offline.temporal.track_rows[-1]['track_status'] = online.temporal.track_rows[-1]['track_status']
                offline.temporal.energy_rows[-1]['combined_cumulative_dissipated_energy'] += 1.
                with self.assertRaises(AssertionError):
                    verify_products(online, offline)
            finally:
                for writer in writers:
                    writer.close()


if __name__ == '__main__':
    unittest.main()
