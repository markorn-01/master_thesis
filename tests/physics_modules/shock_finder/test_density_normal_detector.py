import unittest

import numpy as np

from experiments.shock.density_normal_detector import (
    DensityNormalSettings, analyze_line, classify_states,
)
from experiments.shock.shock_contact_reference import ShockContactReference


class DensityNormalDetectorTests(unittest.TestCase):
    def test_exact_isolated_shock_has_correct_normal_and_mach(self):
        for orientation in (-1, 1):
            profile, layout = ShockContactReference().control(64, 'shock', phase=.37, orientation=orientation)
            before = profile.copy()
            accepted = [r for r in analyze_line(profile, 1/64) if r['accepted']]
            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted[0]['normal_upstream_to_downstream'], -orientation)
            self.assertAlmostEqual(accepted[0]['mass_inferred_mach'], 3.)
            self.assertLess(abs(accepted[0]['x']-layout['shock_x'])*64, 1.)
            np.testing.assert_array_equal(profile, before)

    def test_separated_shock_contact_retains_the_known_shock(self):
        profile, layout = ShockContactReference().profile(64, 8.)
        accepted = [r for r in analyze_line(profile, 1/64) if r['accepted']]
        self.assertTrue(any(abs(r['x']-layout['shock_x'])*64 < 2 for r in accepted))
        for r in accepted:
            self.assertAlmostEqual(r['mass_inferred_mach'], 3.)

    def test_contacts_alone_never_pass_the_kinematic_classifier(self):
        for factor in (.25, 4., 8.):
            for width in (1, 3, 5):
                profile, _ = ShockContactReference(contact_density_factor=factor).control(64, 'contact', width)
                rows = analyze_line(profile, 1/64)
                self.assertFalse(any(r['kinematic_candidate'] or r['accepted'] for r in rows))

    def test_isentropic_compression_is_not_an_entropy_admissible_shock(self):
        gamma = 5/3
        rho = 11**(1/gamma)
        c1, c2 = np.sqrt(gamma), np.sqrt(gamma*11/rho)
        drop = 2*(c2-c1)/(gamma-1)
        result = classify_states([1., 0., 1.], [rho, drop, 11.], -1)
        self.assertTrue(result['kinematic_candidate'])
        self.assertFalse(result['accepted'])
        self.assertAlmostEqual(result['sampled_log_entropy_change'], 0.)

    def test_mixed_contact_sample_keeps_candidate_and_rejection_distinct(self):
        reference = ShockContactReference()
        result = classify_states(reference.upstream, reference.contact_gas, -1)
        self.assertTrue(result['kinematic_candidate'])
        self.assertFalse(result['entropy_increase'])
        self.assertFalse(result['accepted'])
        # The original physical shock still exists: these side samples are not
        # its immediate downstream state. This rejection must remain visible.
        self.assertLess(result['sampled_log_entropy_change'], 0.)

    def test_galilean_boost_preserves_classification_and_inferred_mach(self):
        profile, _ = ShockContactReference().control(64, 'shock')
        original = analyze_line(profile, 1/64)
        profile[:, 1] += 17.
        boosted = analyze_line(profile, 1/64)
        self.assertEqual([r['accepted'] for r in original], [r['accepted'] for r in boosted])
        np.testing.assert_allclose([r['mass_inferred_mach'] for r in original],
                                   [r['mass_inferred_mach'] for r in boosted])

    def test_configured_strength_floor_is_distinct_from_entropy_validity(self):
        reference = ShockContactReference(mach=1.2)
        result = classify_states(reference.upstream, reference.downstream, -1)
        self.assertTrue(result['entropy_increase'])
        self.assertFalse(result['kinematic_candidate'])
        self.assertFalse(result['accepted'])

    def test_opposite_density_directions_at_a_tied_compression_peak_are_kept(self):
        profile, _ = ShockContactReference(contact_density_factor=.25).profile(32, .5)
        rows = analyze_line(profile, 1/32)
        near_shock = [r for r in rows if abs(r['x']-.5) < 1.e-8]
        self.assertEqual({r['normal_upstream_to_downstream'] for r in near_shock}, {-1, 1})

    def test_acceptance_gate_stops_when_extra_detections_include_smooth_compression(self):
        from experiments.shock.check_density_normal_detector import acceptance_decision
        rows = [dict(kind='shock', filter_cells=1, baseline_outcome='recovered',
            alternative_shock_accepted=True, alternative_mach_relative_error=0.,
            alternative_accepted_count=1, alternative_kinematic_count=1),
            dict(kind='pair', baseline_outcome='no_zone_candidates',
                alternative_shock_accepted=True, alternative_shock_located_kinematically=True,
                alternative_accepted_count=1, alternative_kinematic_count=1),
            dict(kind='negative_control', baseline_outcome='smooth_control_rejected',
                alternative_accepted_count=1, alternative_kinematic_count=1)]
        result = acceptance_decision(rows)
        self.assertFalse(result['passed'])
        self.assertEqual(result['decision'], 'reject_alternative_do_not_apply_to_bubble')
        self.assertFalse(result['gates']['no_contact_or_smooth_control_accepted'])

    def test_invalid_inputs_and_averaging_thickness_fail_early(self):
        profile, _ = ShockContactReference().control(32, 'shock')
        with self.assertRaises(ValueError):
            analyze_line(profile, 0.)
        with self.assertRaises(ValueError):
            analyze_line(profile, 1/32, DensityNormalSettings(shock_thickness_cells=2))
        with self.assertRaises(ValueError):
            classify_states([0, 0, 1], [3, 2, 11], -1)
        with self.assertRaises(ValueError):
            classify_states([1, 0, 1], [3, 2, 11], 0)


if __name__ == '__main__':
    unittest.main()
