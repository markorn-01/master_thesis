import unittest

import numpy as np

from experiments.shock.colliding_shocks_reference import SymmetricShockCollision
from experiments.shock.check_colliding_shocks import make_analyzer
from experiments.shock.run_evolved_collision import (
    evolve, front_metrics, hydrodynamic_metrics, known_structure, make_slab_analyzer,
    pressure_fronts,
)


class EvolvedCollisionTests(unittest.TestCase):
    def test_finite_mach_and_strong_limit_are_different_reference_targets(self):
        finite = SymmetricShockCollision(3.)
        self.assertAlmostEqual(finite.outgoing_mach, 1.9148542155126766)
        strong = SymmetricShockCollision(1000.)
        self.assertAlmostEqual(strong.outgoing_mach, np.sqrt(5), places=5)
        self.assertAlmostEqual(strong.central[2]/strong.incoming_left[2], 6., places=4)
        self.assertAlmostEqual(strong.central[0]/strong.incoming_left[0], 2.5, places=5)
        self.assertAlmostEqual(strong.outgoing_speed/strong.incoming_speed, .5, places=5)

    def test_thin_slab_matches_full_cube_interior_for_same_reference_profile(self):
        reference = SymmetricShockCollision()
        q, _ = reference.cell_averages(32, reference.time_at_separation('outgoing', 8, 32))
        full = make_analyzer(32)(q); thin = make_slab_analyzer(32)(q)
        for key in ('shock_zone', 'shock_surface', 'surface_offsets', 'mach', 'divergence'):
            np.testing.assert_allclose(thin[key], full[key][:, 16, 16], rtol=1.e-6, atol=1.e-6)
        np.testing.assert_array_equal(thin['normal_x'], full['direction'][:, 16, 16, 0])

    def test_gas_is_really_evolved_and_boundary_flux_balance_is_checked(self):
        reference = SymmetricShockCollision()
        profiles, times, solver = evolve(reference, 64, events=9)
        self.assertEqual(profiles.shape, (9, 64, 3))
        self.assertGreater(solver['num_iterations'], 0)
        self.assertTrue(np.isfinite(profiles).all())
        self.assertGreater(float(abs(profiles[-1]-profiles[0]).max()), 1.)
        metrics = hydrodynamic_metrics(profiles[-1], reference, float(times[-1]))
        self.assertLess(metrics['maximum_boundary_flux_balance_error'], 5.e-5)
        measured = pressure_fronts(profiles[-1], 1/64)
        truth = reference.structure(float(times[-1]))['positions']
        self.assertLess(float(abs(measured-truth).max()*64), 3.)

    def test_one_midpoint_surface_cannot_be_scored_as_two_reflected_fronts(self):
        n = 64; fields = dict(shock_surface=np.zeros(n, bool), normal_x=np.ones(n),
            surface_offsets=np.zeros(n), mach=np.full(n, 2.), compression=np.ones(n, bool),
            gradient_alignment=np.ones(n, bool), minimum_jump=np.ones(n, bool), shock_zone=np.ones(n, bool))
        fields['shock_surface'][31] = True
        fields['surface_offsets'][31] = .5
        reference = SymmetricShockCollision()
        structure = reference.structure(reference.time_at_separation('outgoing', 2, n))
        rows = front_metrics(fields, structure, n)
        self.assertFalse(any(r['located'] for r in rows))

    def test_float32_collision_time_is_scored_as_meeting_not_two_distinct_fronts(self):
        reference = SymmetricShockCollision(10.)
        structure = known_structure(reference, float(np.float32(reference.collision_time)))
        self.assertEqual(structure['phase'], 'meeting')

    def test_invalid_resolution_and_insufficient_time_series_fail_early(self):
        reference = SymmetricShockCollision()
        for n, events in ((31, 9), (65, 9), (64, 4)):
            with self.assertRaises(ValueError):
                evolve(reference, n, events=events)


if __name__ == '__main__':
    unittest.main()
