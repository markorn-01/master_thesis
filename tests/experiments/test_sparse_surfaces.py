"""Sparse catalogs preserve geometry, invalid samples and histogram population."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from experiments.wind_bubble.sparse_surfaces import sparse_surface_catalog, sampling_position_flags


class SparseSurfaceTests(unittest.TestCase):
    def make_payload(self):
        n = 16
        shape = (n,)*3
        centers = np.stack(np.meshgrid(*([(np.arange(n)+.5)/n]*3), indexing='ij'), axis=-1)
        mask = np.zeros(shape, bool)
        mask[8, 8, 8] = True  # Inside injection, excluded.
        mask[12, 8, 8] = True
        mask[14, 8, 8] = True
        direction = np.zeros((3,)+shape)
        direction[0], direction[1] = .6, .8
        offsets = np.zeros(shape)
        offsets[12, 8, 8] = .25
        mach, flux, rho2, rho1 = [np.ones(shape) for _ in range(4)]
        mach[12, 8, 8], flux[12, 8, 8], rho2[12, 8, 8] = 5., 2., .8
        mach[14, 8, 8], flux[14, 8, 8] = np.nan, np.nan
        valid = np.ones(shape, bool)
        valid[14, 8, 8] = False
        return (.2, mask, offsets, direction, mach, flux, rho2, rho1, valid,
                centers, 1/n, .0625)

    def test_refined_geometry_matches_original_cells_and_normals(self):
        payload = self.make_payload()
        result = sparse_surface_catalog(*payload)
        np.testing.assert_array_equal(result['grid_indices'], [[12, 8, 8], [14, 8, 8]])
        expected = payload[9][12, 8, 8] + .25/16*np.array([.6, .8, 0.])
        np.testing.assert_allclose(result['positions'][0], expected)
        np.testing.assert_allclose(result['normals'], [[.6, .8, 0.], [.6, .8, 0.]])
        self.assertTrue(np.all(result['radii'] > .0625))

    def test_invalid_properties_are_retained_and_subunity_ratio_is_valid(self):
        result = sparse_surface_catalog(*self.make_payload())
        self.assertEqual(len(result['positions']), 2)
        self.assertEqual(result['density_jump'][0], .8)
        self.assertTrue(result['density_jump_valid'][0])
        self.assertTrue(np.isnan(result['density_jump'][1]))
        np.testing.assert_array_equal(result['mach_valid'], [True, False])
        np.testing.assert_array_equal(result['sampling_valid'], [True, False])
        np.testing.assert_array_equal(result['density_jump_valid'], [True, False])
        np.testing.assert_array_equal(result['thermalization_flux_valid'], [True, False])

    def test_empty_event_has_consistent_array_shapes_and_round_trips(self):
        payload = list(self.make_payload())
        payload[1][:] = False
        result = sparse_surface_catalog(*payload)
        self.assertEqual(result['positions'].shape, (0, 3))
        self.assertEqual(result['normals'].shape, (0, 3))
        self.assertEqual(result['grid_indices'].shape, (0, 3))
        self.assertEqual(result['mach'].shape, (0,))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'surfaces_0000.npz'
            np.savez_compressed(target, **result)
            with np.load(target, allow_pickle=False) as loaded:
                for key, value in result.items():
                    np.testing.assert_array_equal(loaded[key], value)

    def test_source_overlap_uses_endpoint_not_surface_and_includes_boundary(self):
        flags = sampling_position_flags(np.array([[.6875, .5, .5]]),
            np.array([[-1., 0., 0.]]), np.array([.5]), np.array([1.]), .125, .0625)
        self.assertTrue(flags['sample_positions_valid'][0])
        self.assertTrue(flags['upstream_inside_injection'][0])  # r=0.0625 exactly
        self.assertFalse(flags['downstream_inside_injection'][0])
        np.testing.assert_allclose(flags['upstream_positions'][0], [.5625, .5, .5])
        np.testing.assert_allclose(flags['downstream_positions'][0], [.75, .5, .5])

    def test_missing_or_invalid_distances_are_unknown_not_clean_samples(self):
        pos = np.array([[.7, .5, .5], [.8, .5, .5]])
        normals = np.array([[-1., 0., 0.], [-1., 0., 0.]])
        for post, pre in [(None, None), (np.array([np.nan, -1.]), np.ones(2))]:
            flags = sampling_position_flags(pos, normals, post, pre, .125, .0625)
            self.assertFalse(flags['sample_positions_valid'].any())
            self.assertTrue(np.isnan(flags['upstream_radii']).all())
            self.assertFalse(flags['upstream_inside_injection'].any())
        with self.assertRaises(ValueError):
            sampling_position_flags(pos, normals, np.ones(2), None, .125, .0625)

    def test_endpoint_flags_do_not_filter_or_change_catalog_measurements(self):
        payload = self.make_payload()
        before = sparse_surface_catalog(*payload)
        shape = payload[1].shape
        after = sparse_surface_catalog(*payload, post_distance=np.ones(shape), pre_distance=np.ones(shape))
        for key in ('positions', 'grid_indices', 'mach', 'thermalization_flux', 'density_jump',
                    'sampling_valid', 'density_jump_valid'):
            np.testing.assert_equal(before[key], after[key])
        self.assertTrue(after['sample_positions_valid'].all())
        self.assertFalse(after['sampling_valid'][-1])  # Geometric and thermodynamic validity differ.
        self.assertFalse(before['sample_positions_valid'].any())


if __name__ == '__main__':
    unittest.main()
