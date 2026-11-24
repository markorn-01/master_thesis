import unittest

import numpy as np

from experiments.shock.colliding_shocks_reference import SymmetricShockCollision
from experiments.wind_bubble.check_outer_control_calibration import (
    density_example, evaluate_window, known_controls, moving_control,
)


class OuterControlCalibrationTests(unittest.TestCase):
    def test_controls_preserve_shocks_and_reject_contact_and_isentropic_profile(self):
        rows = known_controls()
        self.assertEqual(len(rows), 15)
        self.assertTrue(all(r['passed'] for r in rows))
        for row in rows:
            if row['expected_compatible']:
                self.assertLess(abs(row['measured_x_speed']/row['reference_speed']-1), .03)

    def test_weak_shock_is_not_confused_with_above_threshold_detection(self):
        reference = SymmetricShockCollision(incoming_mach=1.2)
        times = np.arange(65)*.0005
        lines, x = moving_control(reference.incoming_left, reference.ambient,
            reference.incoming_speed, times)
        row, pairs = evaluate_window(lines, times, x, 32, 16,
            distances=np.arange(1, 8.001, .25))
        self.assertTrue(row['compatible_10pct'])
        self.assertFalse(row['compatible_above_threshold_10pct'])
        self.assertTrue(pairs)

    def test_downstream_density_example_uses_measured_gas_not_imposed_ratio(self):
        x = (np.arange(128)+.5)/128
        q = np.zeros((5, 128)); q[0] = 1.; q[4] = .01
        q[0, x > .2] = 2.5; q[4, x > .2] = .8
        before = q.copy()
        result = density_example(q, x, .2)
        self.assertEqual(result['sampled_density_ratio'], 2.5)
        self.assertGreater(result['pressure_predicted_density_ratio'], 3.8)
        self.assertGreater(result['relative_density_shortfall'], .3)
        np.testing.assert_array_equal(q, before)

    def test_incomplete_time_window_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_window(np.ones((9, 5, 16)), np.arange(9), np.arange(16), 0, 4)


if __name__ == '__main__':
    unittest.main()
