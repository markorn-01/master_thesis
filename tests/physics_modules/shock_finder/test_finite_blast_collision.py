import unittest

import numpy as np

from experiments.shock.finite_blast_collision import BlastSettings, initial_profile, evolve_blasts
from experiments.shock.colliding_shocks_reference import SymmetricShockCollision, conserved
from experiments.shock.blast_front_evidence import analyze, EvidenceSettings, candidates, summaries
from experiments.shock.run_finite_blast_collision import conservation, at_time
from experiments.shock.review_finite_blast_collision import local_stage_counts, STAGES


class FiniteBlastTests(unittest.TestCase):
    def test_initial_energy_is_resolution_independent_and_symmetry_is_exact(self):
        settings = BlastSettings()
        expected = (settings.ambient_pressure+4*settings.half_width*(settings.hot_pressure
                    - settings.ambient_pressure))/(settings.gamma-1)
        for n in (256, 512, 1024):
            q = initial_profile(n, settings)
            np.testing.assert_allclose(q, q[::-1]*[1., -1., 1.], atol=1.e-10)
            self.assertAlmostEqual(float(conserved(q).mean(axis=0)[2]), expected, places=10)
            self.assertEqual(float(conserved(q).mean(axis=0)[0]), settings.density)

    def test_real_evolution_has_no_continuing_source_and_conserves_before_boundary_arrival(self):
        settings = BlastSettings(t_end=.02, events=11)
        profiles, times, solver = evolve_blasts(128, settings)
        self.assertEqual(profiles.shape, (11, 128, 3))
        self.assertGreater(solver['iterations'], 0)
        self.assertGreater(float(abs(profiles[-1]-profiles[0]).max()), 1.)
        result = conservation(profiles, settings)
        self.assertTrue(result['boundaries_undisturbed'])
        self.assertLess(result['maximum_normalized_integral_change'], 5.e-5)

    def test_independent_motion_distinguishes_above_and_below_threshold_exact_shocks(self):
        for incident, expected in [(3., 'jump_supported_above_threshold'), (1.2, 'jump_supported_below_threshold')]:
            reference = SymmetricShockCollision(incident)
            times = reference.collision_time+np.arange(41)*.002
            profiles = np.array([reference.cell_averages(1024, float(t))[0] for t in times])
            rows, _, _ = analyze(profiles, times, reference.collision_time)
            supported = [r for r in rows if r['evidence'].startswith('jump_supported')]
            self.assertGreater(len(supported), 10)
            self.assertEqual({r['evidence'] for r in supported}, {expected})
            self.assertEqual({r['side'] for r in supported}, {'left', 'right'})

    def test_constant_pressure_contact_cannot_supply_pressure_front_evidence(self):
        n = 128; times = np.arange(41)*.001; q = np.ones((41, n, 3)); q[:, :, 1] = 0.
        q[:, :n//2, 0] = 4.
        rows, pairs, tracks = analyze(q, times, 0.)
        self.assertFalse(rows); self.assertFalse(pairs); self.assertFalse(tracks)

    def test_dominant_pressure_peak_rejects_competition_instead_of_following_wiggles(self):
        n = 1024; x = (np.arange(n)+.5)/n
        q = np.column_stack([np.ones(n), np.zeros(n),
            1+3*np.tanh((x-.43)*n/2)+.03*np.tanh((x-.46)*n/2)])
        q[:, 2] += 4.
        left = [r for r in candidates(q, 1/n, EvidenceSettings()) if r['side'] == 'left']
        self.assertEqual(len(left), 1)
        self.assertLess(abs(left[0]['x']-.43), 2/n)
        q[:, 2] = 10+3*np.tanh((x-.43)*n/2)+3*np.tanh((x-.46)*n/2)
        self.assertFalse([r for r in candidates(q, 1/n, EvidenceSettings()) if r['side'] == 'left'])

    def test_fading_summary_requires_persistent_below_events_after_above_events(self):
        track = [dict(event=i, time=i*.001, x=.499-.002*i,
            normal_upstream_to_downstream=1, side='left') for i in range(9)]
        rows = [dict(track=0, event=i, time=i*.001, evidence='jump_supported_above_threshold'
            if i in (0, 1, 2, 8) else 'jump_supported_below_threshold') for i in range(9)]
        self.assertFalse(summaries(rows, [track], 1/1024)[0]['ordered_above_then_below'])
        rows[-1]['evidence'] = 'jump_supported_below_threshold'
        self.assertTrue(summaries(rows, [track], 1/1024)[0]['ordered_above_then_below'])

    def test_conserved_time_interpolation_does_not_interpolate_primitive_energy_incorrectly(self):
        profiles = np.array([[[1., 1., 1.]], [[2., -1., 3.]]])
        actual = at_time(profiles, np.array([0., 2.]), 1.)
        expected = .5*(conserved(profiles[0])+conserved(profiles[1]))
        np.testing.assert_allclose(actual, expected)

    def test_invalid_initial_geometry_and_unordered_times_are_rejected(self):
        with self.assertRaises(ValueError):
            initial_profile(64, BlastSettings(centers=(.4, .7)))
        with self.assertRaises(ValueError):
            initial_profile(63)
        with self.assertRaises(ValueError):
            analyze(np.ones((4, 16, 3)), np.array([0., .2, .1, .3]), 0.)
        with self.assertRaises(ValueError):
            analyze(np.ones((4, 3)), np.arange(4), 0.)

    def test_local_stage_counts_require_intersection_at_the_same_cell(self):
        data = {k:np.zeros((1, 16), bool) for k in STAGES}
        data['compression'][0, 7] = True
        data['gradient_alignment'][0, 8] = True
        row = local_stage_counts(data, 0, .5, 16)
        self.assertEqual(row['compression'], 1)
        self.assertEqual(row['gradient_alignment'], 1)
        self.assertEqual(row['compression_and_alignment'], 0)

    def test_minimum_jump_failure_is_kept_separate_from_gradient_failure(self):
        data = {k:np.zeros((1, 16), bool) for k in STAGES}
        data['compression'][:] = data['gradient_alignment'][:] = True
        data['temperature_jump_pass'][0, 7] = True
        row = local_stage_counts(data, 0, .5, 16)
        self.assertEqual(row['compression_and_alignment'], 4)
        self.assertEqual(row['before_jump_temperature_pass'], 1)
        self.assertEqual(row['before_jump_pressure_pass'], 0)
        self.assertEqual(row['before_jump_both_jumps_pass'], 0)


if __name__ == '__main__':
    unittest.main()
