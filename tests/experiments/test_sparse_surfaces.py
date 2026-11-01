"""Sparse catalogs preserve geometry, invalid samples and histogram population."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from experiments.wind_bubble.sparse_surfaces import sparse_surface_catalog


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


if __name__ == '__main__':
    unittest.main()
