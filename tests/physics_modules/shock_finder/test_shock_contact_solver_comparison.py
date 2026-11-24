import copy
import unittest
from dataclasses import replace

import jax
import numpy as np

from astronomix.test_setups.reference_solutions.riemann_solver import _exact_riemann_ideal_gas
from experiments.shock.colliding_shocks_reference import conserved, flux
from experiments.shock.compare_shock_contact_solvers import (
    TIMES_TO_COMPARE, conservation_metrics, decision, evolve_case, motion_checks,
)
from experiments.shock.run_evolved_collision import make_slab_analyzer
from experiments.shock.shock_contact_solver_reference import ShockContactReference
from experiments.wind_bubble.collision_front_motion import MotionSettings, pressure_ridges


class ShockContactSolverComparisonTests(unittest.TestCase):
    def test_scaled_reference_satisfies_both_jumps_and_independent_exact_riemann_oracle(self):
        reference = ShockContactReference()
        self.assertLess(float(np.max(abs(reference.flux_residuals()))), 1.e-12)
        x = np.array([.2, .65, .7, .71, .75, .9]); time = .1
        with jax.experimental.enable_x64():
            actual = np.stack([np.asarray(v) for v in _exact_riemann_ideal_gas(
                *reference.contact_state, *reference.upstream, reference.gamma,
                x, time, reference.origin)], axis=-1)
        np.testing.assert_allclose(actual, reference.point_values(x, time), rtol=1.e-12, atol=1.e-12)

    def test_exact_cell_averages_include_boundary_fluxes_at_all_resolutions(self):
        reference = ShockContactReference()
        for n in (64, 128, 256):
            for contact_only in (False, True):
                initial = conserved(reference.cell_averages(n, 0., contact_only)).mean(axis=0)
                right = reference.shocked if contact_only else reference.upstream
                for time in (.04, .08, .16):
                    expected = initial+time*(flux(reference.contact_state)-flux(right))
                    actual = conserved(reference.cell_averages(n, time, contact_only)).mean(axis=0)
                    np.testing.assert_allclose(actual, expected, rtol=1.e-12, atol=1.e-12)

    def test_exact_contact_changes_density_without_pressure_or_velocity_jump(self):
        reference = ShockContactReference()
        for n in (64, 128):
            q = reference.cell_averages(n, .08, contact_only=True)
            np.testing.assert_allclose(q[:, 1], reference.shocked[1], atol=1.e-13)
            np.testing.assert_allclose(q[:, 2], reference.shocked[2], atol=1.e-13)
            self.assertGreater(q[:, 0].max()/q[:, 0].min(), 90)

    def test_conservation_budget_includes_open_boundary_fluxes(self):
        reference = ShockContactReference(); times = np.linspace(0, .16, 41)
        q = np.array([reference.cell_averages(128, t) for t in times])
        self.assertGreater(abs(conserved(q[-1]).mean(axis=0)-conserved(q[0]).mean(axis=0)).max(), .1)
        metrics = conservation_metrics(q, times, q)
        self.assertLess(metrics['maximum_relative_conserved_integral_error'], 1.e-12)
        self.assertEqual(metrics['maximum_relative_exact_integral_error'], 0.)

    def test_measured_speed_check_recovers_known_separated_front(self):
        reference = ShockContactReference(); times = np.linspace(0, .2, 201)
        profiles = np.array([reference.cell_averages(256, t) for t in times])
        rows = motion_checks(profiles, times, reference)
        late = [r for r in rows if r['time'] >= .12 and r['fit_events'] == 33 and r['compatible_10pct']]
        self.assertTrue(late)
        self.assertLess(max(r['relative_speed_error'] for r in late), .05)

    def test_default_ridge_windows_are_unchanged_and_full_domain_is_explicit(self):
        n = 256; x = (np.arange(n)+.5)/n
        q = np.ones((5, n)); q[4] = 1+np.tanh((x-.65)*n)
        self.assertEqual(pressure_ridges(q, x), [])
        expanded = replace(MotionSettings(), ridge_windows=((.02, .98),))
        self.assertEqual(len(pressure_ridges(q, x, expanded)), 1)

    def test_both_solver_options_evolve_positive_gas_and_reject_a_pure_contact(self):
        reference = ShockContactReference(); evaluate = make_slab_analyzer(64)
        for scheme in ('HLL', 'HLLC'):
            q, times, config = evolve_case(reference, 64, scheme, True, events=41, t_end=.04)
            self.assertGreater(config['num_iterations'], 0)
            self.assertTrue(np.isfinite(q).all())
            self.assertGreater(float(abs(q[-1]-q[0]).max()), .5)
            self.assertTrue((q[..., [0, 2]] > 0).all())
            self.assertEqual(config['scheme'], scheme)
            exact = np.array([reference.cell_averages(64, float(t), True) for t in times])
            self.assertLess(conservation_metrics(q, times, exact)['maximum_relative_conserved_integral_error'], 5.e-5)
            for event in (0, 20, 40):
                self.assertFalse(evaluate(q[event])['shock_surface'].any())

    def test_trial_gate_does_not_promote_small_improvements_or_failed_negatives(self):
        rows = []
        for n in (128, 256):
            for scheme, error in (('HLL', .2), ('HLLC', .1)):
                rows.extend(dict(num_cells=n, scheme=scheme, time=t, contact_only=False,
                    peak_density_relative_error=error) for t in TIMES_TO_COMPARE)
                rows.append(dict(num_cells=n, scheme=scheme, time=.16, contact_only=True,
                    detected_surface_cells=0))
        motion = [dict(num_cells=n, scheme='HLLC', fit_events=33, time=.16, compatible_10pct=True)
                  for n in (128, 256)]
        stability = [dict(maximum_relative_conserved_integral_error=1.e-6)]
        self.assertTrue(decision(rows, motion, stability, (128, 256))['supports_one_later_bubble_trial'])
        smaller = copy.deepcopy(rows)
        for r in smaller:
            if not r['contact_only'] and r['scheme'] == 'HLLC':
                r['peak_density_relative_error'] = .18
        self.assertFalse(decision(smaller, motion, stability, (128, 256))['supports_one_later_bubble_trial'])
        rows[-1]['detected_surface_cells'] = 1
        self.assertFalse(decision(rows, motion, stability, (128, 256))['supports_one_later_bubble_trial'])

    def test_invalid_solver_or_reference_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            ShockContactReference(contact_density=-1)
        with self.assertRaises(ValueError):
            ShockContactReference().positions(1.)
        with self.assertRaises(ValueError):
            evolve_case(ShockContactReference(), 64, 'invented')


if __name__ == '__main__':
    unittest.main()
