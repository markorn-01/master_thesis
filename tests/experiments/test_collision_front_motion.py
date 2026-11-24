import unittest

import numpy as np

from experiments.shock.colliding_shocks_reference import conserved, primitive, SymmetricShockCollision
from experiments.wind_bubble.collision_front_motion import (
    analyze_motion, bilinear_band, measured_speed_metrics, strip_fields,
)


def moving_jump(left, right, speed, n=256, times=None, origin=.445):
    times = np.linspace(0, .008, 41) if times is None else np.asarray(times)
    edges = np.arange(n+1)/n
    result = []
    for time in times:
        f = np.clip((origin+speed*time-edges[:-1])*n, 0, 1)
        q = primitive(f[:, None]*conserved(left) + (1-f[:, None])*conserved(right))
        canonical = np.zeros((5, n))
        canonical[[0, 1, 4]] = q.T
        result.append(canonical)
    return np.asarray(result), times, (np.arange(n)+.5)/n


class CollisionFrontMotionTests(unittest.TestCase):
    def test_measured_speed_is_an_independent_test_not_mass_inferred(self):
        reference = SymmetricShockCollision()
        up = np.array([1., 0, 0, 0, 1.])
        down = np.r_[reference.incoming_left[:2], 0., 0., reference.incoming_left[2]]
        correct = measured_speed_metrics(up, down, [-1., 0, 0], -reference.incoming_speed)
        self.assertTrue(correct['necessary_shock_conditions'])
        for key in ('mass_relative_error', 'momentum_relative_error', 'enthalpy_relative_error'):
            self.assertLess(correct[key], 1.e-12)
        wrong = measured_speed_metrics(up, down, [-1., 0, 0], -reference.incoming_speed*.5)
        self.assertGreater(wrong['mass_relative_error'], .1)

    def test_motion_recovers_known_moving_shock_without_using_reference_speed(self):
        reference = SymmetricShockCollision()
        lines, times, x = moving_jump(reference.incoming_left, reference.ambient, reference.incoming_speed)
        before = lines.copy()
        observations, pairs, _ = analyze_motion(lines, times, x)
        accepted = [r for r in pairs if r['compatible_10pct']]
        self.assertGreater(len(accepted), 20)
        self.assertGreater(sum(r['compatible_10pct'] for r in observations), 15)
        self.assertLess(abs(np.median([r['measured_x_speed'] for r in accepted])/reference.incoming_speed-1), .05)
        np.testing.assert_array_equal(before, lines)

    def test_reflected_and_boosted_shock_is_not_discarded(self):
        reference = SymmetricShockCollision()
        left, right = reference.ambient.copy(), reference.incoming_right.copy()
        left[1] += 2.; right[1] += 2.
        lines, times, x = moving_jump(left, right, -reference.incoming_speed+2., origin=.48)
        _, pairs, _ = analyze_motion(lines, times, x)
        accepted = [r for r in pairs if r['compatible_10pct']]
        self.assertTrue(accepted)
        self.assertTrue(all(r['normal_x'] > .99 for r in accepted))

    def test_moving_contact_has_no_pressure_front(self):
        lines, times, x = moving_jump([4., 2., 1.], [1., 2., 1.], 2.)
        observations, pairs, _ = analyze_motion(lines, times, x)
        self.assertEqual(observations, [])
        self.assertEqual(pairs, [])

    def test_weak_physical_front_is_separate_from_production_threshold_failure(self):
        reference = SymmetricShockCollision(incoming_mach=1.2)
        times = np.arange(41)*.0005
        lines, times, x = moving_jump(reference.incoming_left, reference.ambient,
            reference.incoming_speed, n=128, times=times, origin=.1)
        observations, _, _ = analyze_motion(lines, times, x)
        accepted = [r for r in observations if r['compatible_10pct']]
        self.assertTrue(accepted)
        self.assertTrue(all(r['compatible_below_detection_threshold_10pct'] for r in accepted))
        self.assertFalse(any(r['compatible_above_detection_threshold_10pct'] for r in accepted))

    def test_smooth_isentropic_compression_is_not_promoted(self):
        x = (np.arange(256)+.5)/256
        times = np.linspace(0, .008, 41)
        lines = []
        for time in times:
            p = 1+5*(1-np.tanh((x-.46-time)*256/2))
            rho = p**(3/5)
            c = np.sqrt((5/3)*p/rho)
            q = np.zeros((5, len(x)))
            q[0], q[1], q[4] = rho, 3*(c-np.sqrt(5/3)), p
            lines.append(q)
        observations, pairs, _ = analyze_motion(lines, times, x)
        self.assertTrue(observations)
        self.assertFalse(any(r['compatible_20pct'] for r in pairs))

    def test_known_two_outgoing_shocks_get_separate_paths(self):
        reference = SymmetricShockCollision()
        times = np.linspace(.079, .088, 41)
        lines = []
        for time in times:
            q, _ = reference.cell_averages(256, time)
            state = np.zeros((5, 256)); state[[0, 1, 4]] = q.T
            lines.append(state)
        _, pairs, _ = analyze_motion(lines, times, (np.arange(256)+.5)/256)
        accepted = [r for r in pairs if r['compatible_10pct']]
        self.assertTrue(any(r['region'] == 'central_left' for r in accepted))
        self.assertTrue(any(r['region'] == 'central_right' for r in accepted))
        self.assertTrue(all((r['upstream_x']-.5)*(r['downstream_x']-.5) > 0 for r in accepted))

    def test_small_strip_interpolates_a_linear_3d_field_exactly(self):
        dx = 1/128
        y = .375+np.array([-1.5, -.5, .5, 1.5])*dx
        z = .5+np.array([-1.5, -.5, .5, 1.5])*dx
        band = np.broadcast_to(2*y[:, None]+3*z[None, :], (3, 5, 128, 4, 4))
        line, dy, dz = strip_fields(band, y, z, dx)
        np.testing.assert_allclose(line, 2*.375+3*.5)
        np.testing.assert_allclose(dy, 2.)
        np.testing.assert_allclose(dz, 3.)
        with self.assertRaises(ValueError):
            bilinear_band(band, y, z, .2, .5)

    def test_invalid_time_order_and_wrong_state_shape_are_rejected(self):
        q = np.ones((10, 5, 16)); x = (np.arange(16)+.5)/16
        with self.assertRaises(ValueError):
            analyze_motion(q, np.zeros(10), x)
        with self.assertRaises(ValueError):
            measured_speed_metrics(np.ones(4), np.ones(5), [1, 0, 0], 1.)


if __name__ == '__main__':
    unittest.main()
