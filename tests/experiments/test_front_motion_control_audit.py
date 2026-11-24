import unittest

import numpy as np

from experiments.wind_bubble.audit_front_motion_control import (
    ERRORS, one_side_fit, rejection_summary, slope_resolution_allowance, upstream_sonic_samples,
)


class FrontMotionControlAuditTests(unittest.TestCase):
    def test_speed_fit_does_not_span_the_two_collision_sides(self):
        track = [dict(event=i, time=i*.001, x=.51-.002*i) for i in range(17)]
        fit, reason = one_side_fit(track, 8, 4)
        self.assertIsNone(fit)
        self.assertEqual(reason, 'crosses_collision_midplane')
        track = [dict(event=i, time=i*.001, x=.45+.001*i) for i in range(17)]
        fit, reason = one_side_fit(track, 8, 4)
        self.assertEqual(reason, 'available')
        self.assertAlmostEqual(fit[0], 1.)

    def test_half_cell_position_allowance_bounds_the_ols_slope_perturbation(self):
        times = np.linspace(0, .016, 33); width = 1/128
        t = times-times.mean()
        worst_errors = .5*width*np.sign(t)
        slope_error = t @ worst_errors/(t @ t)
        self.assertAlmostEqual(slope_error, slope_resolution_allowance(times, width))

    def test_sonic_sample_is_supersonic_for_known_shock_speed_and_zero_for_contact_speed(self):
        x = (np.arange(128)+.5)/128
        q = np.zeros((5, 128)); q[0] = q[4] = 1.
        sources = np.array([[.1, .1, .1]])
        positions, mach, _ = upstream_sonic_samples(q, x, .49, np.array([1., 0, 0]),
            -3*np.sqrt(5/3), sources, 1/128, 'central_left')
        self.assertTrue(len(positions))
        self.assertTrue((positions < .5).all())
        np.testing.assert_allclose(mach, 3.)
        q[1] = 2.
        _, mach, _ = upstream_sonic_samples(q, x, .49, np.array([1., 0, 0]),
            2., sources, 1/128, 'central_left')
        np.testing.assert_allclose(mach, 0.)

    def test_failure_counts_do_not_promote_failed_full_checks(self):
        row = dict(region='outer_control', necessary_shock_conditions='True',
                   compatible_05pct='False', compatible_10pct='False', compatible_20pct='False',
                   entropy_signal_error='.2', **{key: '.01' for key in ERRORS})
        row['density_ratio_error'] = '.32'
        result = rejection_summary([row])[0]
        self.assertEqual(result['necessary_conditions_pass'], 1)
        self.assertEqual(result['density_ratio_error_fails_10pct'], 1)
        self.assertEqual(result['compatible_pairs_10pct'], 0)


if __name__ == '__main__':
    unittest.main()
