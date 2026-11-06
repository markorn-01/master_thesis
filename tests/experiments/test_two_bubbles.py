"""Injection regression and source-aware collision-output checks."""
import argparse
import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from astronomix._modules._stellar_wind.stellar_wind import _wind_injection
from astronomix.option_classes.simulation_config import finalize_config, FINITE_DIFFERENCE
from experiments.wind_bubble.run_single_bubble import build_problem
from experiments.wind_bubble.run_two_bubbles import (
    SOURCES, INJECTION_RADIUS, make_catalog, CollisionWriter, compare_outputs,
)


class TwoSourceInjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state, cls.config, cls.params, cls.registry, cls.helper = build_problem(
            argparse.Namespace(num_cells=32, num_snapshots=3, t_end=.01,
                               num_injection_cells=2, plasma_beta=None))

    def configuration(self, positions):
        c = self.config._replace(wind_config=self.config.wind_config._replace(source_positions=positions))
        return finalize_config(c, self.state.shape)

    def inject(self, positions):
        return np.asarray(_wind_injection(self.state, jnp.asarray(.001),
                          self.configuration(positions), self.params, self.helper, self.registry))

    def test_explicit_center_preserves_original_source_exactly(self):
        np.testing.assert_array_equal(self.inject(()), self.inject(((.5, .5, .5),)))

    def test_two_sources_equal_sum_of_disjoint_single_source_increments(self):
        a, b = [self.inject((source,)) for source in SOURCES]
        two, original = self.inject(SOURCES), np.asarray(self.state)
        np.testing.assert_allclose(two, original+(a-original)+(b-original), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(two, two[:, ::-1, :, :], rtol=1e-6, atol=1e-8)

    def test_discrete_mass_and_energy_match_existing_ei_prescription(self):
        result, original = self.inject(SOURCES), np.asarray(self.state)
        dx = float(self.config.grid_spacing)
        centers = np.asarray(self.helper.geometric_centers)
        count = sum(int((np.linalg.norm(centers-s, axis=-1) <= INJECTION_RADIUS-dx/2).sum()) for s in np.asarray(SOURCES))
        volume = 4*np.pi*INJECTION_RADIUS**3/3
        mass = ((result-original)[self.registry.density_index]).sum()*dx**3
        energy = ((result-original)[self.registry.pressure_index]).sum()*dx**3/(float(self.params.gamma)-1)
        self.assertGreater(count, 0)
        np.testing.assert_allclose(mass, .001*.01*count*dx**3/volume, rtol=1e-4)
        np.testing.assert_allclose(energy, .001*.5*count*dx**3/volume, rtol=1e-5)

    def test_rejects_overlapping_or_outside_sources(self):
        for positions in [((.5,.5,.5),(.55,.5,.5)), ((.02,.5,.5),), ((np.nan,.5,.5),)]:
            with self.assertRaises(ValueError):
                self.configuration(positions)

    def test_rejects_unsupported_solver(self):
        c = self.configuration(SOURCES)._replace(solver_mode=FINITE_DIFFERENCE)
        with self.assertRaisesRegex(ValueError, 'finite-volume EI'):
            finalize_config(c, self.state.shape)


class CollisionProductTests(unittest.TestCase):
    def fields(self):
        n = 32
        centers = np.stack(np.meshgrid(*([(np.arange(n)+.5)/n]*3), indexing='ij'), axis=-1)
        shape = (n,)*3
        mask = np.zeros(shape, bool)
        mask[[11,16,20],16,16] = True
        normals = np.zeros((3,)+shape)
        normals[0] = 1
        fields = (mask, np.zeros(shape), normals, np.full(shape, 5.), np.ones(shape),
                  np.full(shape, .8), np.ones(shape), np.ones(shape, bool),
                  np.full(shape, 4.), np.full(shape, 4.))
        return fields, centers

    def test_excludes_both_sources_but_keeps_interaction_midplane(self):
        fields, centers = self.fields()
        catalog, total = make_catalog(.1, fields, centers, 1/32)
        self.assertEqual(total, 3)
        self.assertEqual(len(catalog['positions']), 1)
        self.assertTrue(catalog['upstream_inside_sources'][0,1])
        self.assertTrue(catalog['downstream_inside_sources'][0,0])
        self.assertNotIn('radii', catalog)
        self.assertNotIn('track_id', catalog)
        self.assertEqual(int(catalog['schema_version']), 3)
        self.assertEqual(float(catalog['density_jump'][0]), .8)

    def test_source_exclusion_uses_refined_position_and_unknown_endpoints(self):
        fields, centers = self.fields()
        fields[1][11,16,16] = 3.
        fields[-1][11,16,16] = np.nan
        catalog, _ = make_catalog(.1, fields, centers, 1/32)
        self.assertEqual(len(catalog['positions']), 2)
        self.assertFalse(catalog['sample_positions_valid'][0])
        self.assertFalse(catalog['upstream_inside_any_source'][0])

    def test_writer_histograms_and_equivalence_detect_altered_catalog(self):
        fields, centers = self.fields()
        with tempfile.TemporaryDirectory() as root:
            writers = [CollisionWriter(Path(root)/name, centers, 1/32) for name in ('online','offline')]
            try:
                for writer in writers:
                    writer.receive(.1, *fields, np.ones((32,32)), np.ones((32,32)))
                self.assertTrue(compare_outputs(*writers)['passed'])
                self.assertEqual(writers[0].rows[0]['upstream_inside_any_source'], 1)
                with np.load(writers[0].directory/'histograms_0000.npz') as h:
                    self.assertEqual(h['mach_counts'].sum(), 1)
                    self.assertEqual(h['density_jump_counts'].sum(), 1)
                p = writers[1].directory/'surfaces_0000.npz'
                with np.load(p) as a:
                    changed = {k: a[k] for k in a.files}
                changed['positions'][0,0] += .01
                np.savez_compressed(p, **changed)
                with self.assertRaises(AssertionError):
                    compare_outputs(*writers)
                with self.assertRaises(ValueError):
                    writers[0].receive(.1, *fields, np.ones((32,32)), np.ones((32,32)))
            finally:
                for writer in writers:
                    writer.close()


if __name__ == '__main__':
    unittest.main()
