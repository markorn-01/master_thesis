import unittest

import numpy as np

from experiments.shock.joint_shock_contact_fit import (
    FitSettings, fit_profile, forward_fan, to_conserved,
)
from experiments.shock.shock_contact_reference import ShockContactReference


class JointShockContactFitTests(unittest.TestCase):
    def test_forward_averages_agree_with_independent_reference(self):
        reference = ShockContactReference(mach=2.4, contact_density_factor=5.)
        for width in (1, 3, 5):
            q, truth = reference.profile(32, .35, width, phase=.23)
            parameters = [0., 0., 0., 2.4, np.log(reference.contact_gas[0]),
                          truth['shock_x']*32, .35]
            actual = forward_fan(parameters, 32, width)
            np.testing.assert_allclose(actual, to_conserved(q, 5/3), rtol=1.e-12, atol=1.e-12)

    def test_blind_fit_recovers_hidden_state_with_changed_physics(self):
        reference = ShockContactReference(mach=2.4, contact_density_factor=5.)
        q, truth = reference.profile(32, .35, 3, phase=.23)
        before = q.copy()
        # Only observed values and dx cross the fitting interface.
        result = fit_profile(q, 1/32)
        self.assertTrue(result['joint_reconstruction_supported'])
        self.assertEqual(result['selected_model'], 'joint')
        self.assertAlmostEqual(result['recovered_mach'], 2.4, places=6)
        np.testing.assert_allclose(result['recovered_downstream'], reference.downstream, atol=1.e-7)
        self.assertAlmostEqual(result['recovered_contact_x'], truth['contact_x'], places=6)
        self.assertAlmostEqual(result['recovered_shock_x'], truth['shock_x'], places=6)
        np.testing.assert_array_equal(q, before)

    def test_no_pure_downstream_cell_is_needed_for_restricted_fan(self):
        reference = ShockContactReference()
        q, _ = reference.profile(32, .5)
        self.assertFalse(np.any(np.all(np.isclose(q, reference.downstream), axis=1)))
        result = fit_profile(q, 1/32)
        self.assertTrue(result['joint_reconstruction_supported'])
        self.assertAlmostEqual(result['recovered_downstream'][0], 3., places=6)
        self.assertAlmostEqual(result['recovered_gap_cells'], .5, places=6)
        self.assertGreaterEqual(len(result['all_discrete_fits']), 2*3*3)

    def test_reflection_and_bulk_velocity_do_not_supply_truth(self):
        reference = ShockContactReference()
        q, truth = reference.profile(32, .5, orientation=-1)
        q[:, 1] += 17.
        result = fit_profile(q, 1/32)
        self.assertTrue(result['joint_reconstruction_supported'])
        self.assertAlmostEqual(result['recovered_mach'], 3., places=6)
        self.assertAlmostEqual(result['recovered_shock_x'], truth['shock_x'], places=6)
        self.assertAlmostEqual(result['recovered_downstream'][1], 17-reference.downstream[1], places=6)

    def test_simple_waves_do_not_create_a_hidden_pair(self):
        reference = ShockContactReference()
        for kind in ('shock', 'contact'):
            q, _ = reference.control(32, kind, phase=.37)
            result = fit_profile(q, 1/32)
            # A conservative tie with the redundant pair is allowed on a
            # short observation window; it must never become a usable pair.
            self.assertIn(result['selected_model'], (kind, 'ambiguous'))
            self.assertFalse(result['joint_reconstruction_supported'])

    def test_smooth_simple_wave_selects_isentropic_alternative(self):
        gamma = 5/3; x = (np.arange(32)+.5)/32
        pressure = 1+5*(1-np.tanh((x-.5)*16))
        density = pressure**(1/gamma)
        velocity = 2/(gamma-1)*(np.sqrt(gamma*pressure/density)-np.sqrt(gamma))
        result = fit_profile(np.column_stack([density, velocity, pressure]), 1/32)
        self.assertEqual(result['selected_model'], 'isentropic')
        self.assertFalse(result['joint_reconstruction_supported'])

    def test_constant_state_remains_ambiguous(self):
        q = np.tile([1., 0., 1.], (32, 1))
        result = fit_profile(q, 1/32)
        self.assertEqual(result['selected_model'], 'ambiguous')
        self.assertFalse(result['joint_reconstruction_supported'])

    def test_invalid_observations_fail_before_optimization(self):
        q = np.tile([1., 0., 1.], (32, 1))
        for spacing in (0., np.nan):
            with self.assertRaises(ValueError):
                fit_profile(q, spacing)
        with self.assertRaises(ValueError):
            fit_profile(q[:10], 1/32)
        with self.assertRaises(ValueError):
            fit_profile(q, 1/32, FitSettings(blur_cells=(2,)))
        q[0, 0] = 0.
        with self.assertRaises(ValueError):
            fit_profile(q, 1/32)


if __name__ == '__main__':
    unittest.main()
