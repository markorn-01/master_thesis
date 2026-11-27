"""Four-source geometry, actual EI injection and shared-workflow checks."""
import argparse
import copy
import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from astronomix._modules._stellar_wind.stellar_wind import _wind_injection
from astronomix.option_classes.simulation_config import finalize_config
from experiments.wind_bubble import multi_source_setup as multi
from experiments.wind_bubble.four_bubble_setup import SOURCES, injection_counts, source_budget, weighted_budget
from experiments.wind_bubble.run_four_bubbles import argument_parser, preflight, run
from experiments.wind_bubble.run_single_bubble import build_problem
from experiments.wind_bubble.run_two_bubbles import make_catalog
from experiments.wind_bubble.three_bubble_setup import SOURCES as THREE_SOURCES, source_budget as three_budget
from tests.experiments.test_two_bubble_source_budget import reference_metadata


class FourSourceSetup(unittest.TestCase):
    def test_regular_noncoplanar_tetrahedron_with_disjoint_interior_sources(self):
        points = multi.validate_sources(SOURCES)
        np.testing.assert_array_equal(points.mean(axis=0), [.5, .5, .5])
        distances = np.linalg.norm(points[:, None]-points[None, :], axis=-1)
        pairs = distances[np.triu_indices(4, 1)]
        np.testing.assert_allclose(pairs, np.sqrt(2)/4, rtol=0, atol=1e-15)
        self.assertGreater(abs(np.linalg.det(points[1:]-points[0])), 0)
        self.assertGreater(pairs.min(), 2*multi.INJECTION_RADIUS)
        self.assertGreater(np.minimum(points, 1-points).min(), multi.INJECTION_RADIUS)

    def test_same_per_source_budget_and_explicit_increased_total(self):
        reference = reference_metadata()
        for n in (32, 64, 128):
            plan = source_budget(n, reference)
            self.assertEqual(len(set(injection_counts(n))), 1)
            np.testing.assert_allclose(plan['discrete_mass_rates'], reference['discrete_mass_rates'][0], rtol=1e-12)
            np.testing.assert_allclose(plan['discrete_energy_rates'], reference['discrete_energy_rates'][0], rtol=1e-12)
            self.assertAlmostEqual(sum(plan['discrete_energy_rates']), 2*sum(reference['discrete_energy_rates']))
        with self.assertRaises(ValueError):
            source_budget(32, reference, actual_counts=[1, 2, 3, 4])

    def test_generic_helpers_agree_with_existing_three_source_budget(self):
        reference = reference_metadata()
        for n in (32, 64):
            new = multi.source_budget(n, reference, sources=THREE_SOURCES)
            old = three_budget(n, reference)
            for key in ('injection_cell_counts', 'wind_mass_loss_rate_per_source',
                        'discrete_mass_rates', 'discrete_energy_rates', 'nominal_total_wind_luminosity'):
                np.testing.assert_allclose(new[key], old[key], rtol=1e-12)

    def test_rejects_unsupported_masks_invalid_geometry_and_bad_reference(self):
        for sources in ([], [[.5, .5]], [[np.nan, .5, .5]], [[.02, .5, .5]],
                        [[.5, .5, .5], [.55, .5, .5]], [[.5, .5, .5], [.625, .5, .5]]):
            with self.assertRaises(ValueError):
                multi.validate_sources(sources)
        for n in (True, 32., 0, 31, 16):
            with self.assertRaises(ValueError):
                injection_counts(n)
        reference = copy.deepcopy(reference_metadata())
        reference['discrete_energy_rates'][0] *= 2
        with self.assertRaises(ValueError):
            source_budget(64, reference)

    def test_source_strengths_preserve_the_per_source_convention(self):
        base = source_budget(64, reference_metadata())
        equal = weighted_budget(base)
        self.assertTrue(equal['per_source_reference_budget_preserved'])
        weighted = weighted_budget(base, [1, .5, .75, .25])
        np.testing.assert_allclose(weighted['discrete_energy_rates'],
                                   np.asarray(base['discrete_energy_rates'])*[1, .5, .75, .25])
        self.assertFalse(weighted['per_source_reference_budget_preserved'])
        for values in ([1, 1, 1], [1, 1, 1, 0], [1, 1, 1, np.nan]):
            with self.assertRaises(ValueError):
                weighted_budget(base, values)

    def test_preflight_defines_pilot_without_creating_output(self):
        args = argument_parser().parse_args(['--preflight-only', '--track-shocks', '--verify-offline'])
        plan = preflight(args)
        self.assertEqual(plan['num_sources'], 4)
        self.assertEqual(plan['num_analysis'], 17)
        self.assertEqual(len(plan['requested_times']), 17)
        self.assertTrue(plan['weighted_budget']['per_source_reference_budget_preserved'])
        self.assertGreater(plan['minimum_injection_boundary_clearance'], 0)
        for count, end in ((2, .16), (17, -1), (17, np.nan)):
            with self.assertRaises(ValueError):
                preflight(argparse.Namespace(**{**vars(args), 'num_analysis': count, 't_end': end}))

    def test_runner_reuses_existing_online_and_offline_paths(self):
        args = argument_parser().parse_args(['--track-shocks', '--verify-offline'])
        with patch('experiments.wind_bubble.run_two_bubbles.run', return_value=['done']) as runner:
            self.assertEqual(run(args), ['done'])
        self.assertIs(runner.call_args.args[0], args)
        self.assertEqual(runner.call_args.kwargs['source_positions'], SOURCES)
        self.assertIs(runner.call_args.kwargs['budget_builder'], source_budget)
        self.assertIs(runner.call_args.kwargs['strength_builder'], weighted_budget)


class FourSourceInjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state, cls.config, cls.params, cls.registry, cls.helper = build_problem(argparse.Namespace(
            num_cells=32, num_snapshots=3, t_end=.02, num_injection_cells=2, plasma_beta=None))

    def configuration(self, sources, strengths):
        return finalize_config(self.config._replace(wind_config=self.config.wind_config._replace(
            source_positions=sources, source_strengths=strengths)), self.state.shape)

    def test_actual_disjoint_injection_equals_sum_of_individual_sources(self):
        before = self.state.at[self.registry.velocity_index.x].set(.2)
        weights = (1., .8, .6, .4)
        def inject(sources, strengths):
            return np.asarray(_wind_injection(before, jnp.asarray(.001),
                self.configuration(sources, strengths), self.params, self.helper, self.registry))
        increments = [inject((source,), (weight,))-np.asarray(before)
                      for source, weight in zip(SOURCES, weights)]
        np.testing.assert_allclose(inject(SOURCES, weights), np.asarray(before)+sum(increments), rtol=1e-6, atol=1e-8)

    def test_actual_mask_counts_and_budget_matched_mass_energy_at_rest(self):
        plan = source_budget(32, reference_metadata())
        params = self.params._replace(wind_params=self.params.wind_params._replace(
            wind_mass_loss_rate=plan['wind_mass_loss_rate_per_source']))
        dt = .001
        config = self.configuration(SOURCES, (1.,)*4)
        after = np.asarray(_wind_injection(self.state, jnp.asarray(dt), config, params, self.helper, self.registry))
        increment = after-np.asarray(self.state)
        centers, dx = np.asarray(self.helper.geometric_centers), float(config.grid_spacing)
        for i, source in enumerate(SOURCES):
            mask = np.linalg.norm(centers-np.asarray(source), axis=-1) <= multi.INJECTION_RADIUS-dx/2
            self.assertEqual(int(mask.sum()), plan['injection_cell_counts'][i])
            mass = increment[self.registry.density_index][mask].sum()*dx**3/dt
            heat = increment[self.registry.pressure_index][mask].sum()*dx**3/dt/(float(params.gamma)-1)
            np.testing.assert_allclose(mass, plan['discrete_mass_rates'][i], rtol=1e-4)
            np.testing.assert_allclose(heat, plan['discrete_energy_rates'][i], rtol=1e-5)

    def test_catalog_excludes_all_four_sources_without_assigning_source_identity(self):
        n = 32
        shape = (n,)*3
        centers = np.asarray(self.helper.geometric_centers)
        mask = np.zeros(shape, bool)
        for index in ((11, 11, 11), (19, 19, 11), (19, 11, 19), (11, 19, 19), (15, 15, 15)):
            mask[index] = True
        normals = np.zeros((3,)+shape); normals[0] = 1
        fields = (mask, np.zeros(shape), normals, np.full(shape, 3.), np.ones(shape),
                  np.full(shape, 3.), np.ones(shape), np.ones(shape, bool),
                  np.ones(shape), np.ones(shape))
        result, total = make_catalog(.1, fields, centers, 1/n, SOURCES)
        self.assertEqual(total, 5)
        self.assertEqual(len(result['positions']), 1)
        self.assertEqual(result['source_distances'].shape, (1, 4))
        self.assertEqual(result['upstream_inside_sources'].shape, (1, 4))
        self.assertNotIn('source_id', result)
        self.assertNotIn('track_id', result)


if __name__ == '__main__':
    unittest.main()
