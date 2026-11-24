"""Regression and safety checks for the opt-in transition-jump screen."""
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from astronomix._physics_modules._shock_finder._shock_zones import (
    _transition_jump_candidates)
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import find_shocks_pfrommer
from experiments.shock.check_transition_jump_screening import (
    make_comparison_analyzer, smooth_profile, isolated_front_is_accurate)
from experiments.shock.colliding_shocks_reference import SymmetricShockCollision
from experiments.shock.shock_contact_reference import ShockContactReference
from experiments.shock.run_planar_shock_benchmark import build_planar_shock_state


class TransitionJumpScreeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluate = staticmethod(make_comparison_analyzer(64))

    def test_archived_mach_154_front_is_recovered_without_changing_gas(self):
        path = Path('outputs/finite_blast_collision_n0512_n1024_n2048/finite_blast_n2048.npz')
        with np.load(path, allow_pickle=False) as data:
            q = data['profiles'][129].copy()
        before = q.copy(); fields = make_comparison_analyzer(2048)(q)
        n = len(q); x = (np.arange(n)+.5)/n
        for label, expected in (('baseline', False), ('transition', True)):
            f = fields[label]; position = x+f['surface_offsets']*f['normal_x']/n
            selected = f['shock_surface'] & (abs(position-.34974719832085055)*n <= 2)
            self.assertEqual(bool(selected.any()), expected)
            if expected:
                self.assertLess(abs(float(np.median(f['mach'][selected]))/1.544-1), .05)
        np.testing.assert_array_equal(before, q)

    def test_smooth_isentropic_compression_does_not_add_candidates(self):
        for width in (1., 2., 4., 8., 16.):
            fields = self.evaluate(smooth_profile(64, width))
            self.assertFalse(fields['extra_jump'].any())
            self.assertFalse(fields['transition']['shock_surface'].any())

    def test_contacts_and_subthreshold_shocks_are_not_reported(self):
        for orientation in (-1, 1):
            for width in (1, 3, 5):
                q, _ = ShockContactReference(contact_density_factor=8.).control(
                    64, 'contact', width, phase=.37, orientation=orientation)
                self.assertFalse(self.evaluate(q)['transition']['shock_surface'].any())
                q, _ = ShockContactReference(mach=1.2).control(
                    64, 'shock', width, phase=.37, orientation=orientation)
                self.assertFalse(self.evaluate(q)['transition']['shock_surface'].any())

    def test_reconstructed_threshold_and_output_fields_remain_consistent(self):
        for mach in (1.5, 3.):
            q, layout = ShockContactReference(mach=mach).control(64, 'shock', 5, phase=.37)
            fields = self.evaluate(q)['transition']; selected = fields['shock_surface']
            self.assertTrue(selected.any())
            self.assertTrue(np.all(fields['mach'][selected] >= 1.3))
            self.assertLess(abs(float(np.median(fields['mach'][selected]))/mach-1), .05)
            for key in ('mach', 'thermal_flux', 'surface_offsets'):
                np.testing.assert_array_equal(fields[key][~selected], 0)

    def test_opposing_front_search_does_not_cross_the_midplane(self):
        reference = SymmetricShockCollision()
        q, structure = reference.cell_averages(64, reference.time_at_separation('outgoing', 2, 64))
        fields = self.evaluate(q)['transition']; mask = fields['shock_surface']
        self.assertEqual(int(mask.sum()), 2)
        self.assertLess(float(np.max(abs(fields['mach'][mask]/reference.outgoing_mach-1))), .05)

    def test_isolated_midpoint_front_does_not_use_two_front_half_domain_assignment(self):
        n = 64
        fields = dict(shock_surface=np.zeros(n, bool), surface_offsets=np.zeros(n),
            normal_x=np.ones(n), mach=np.zeros(n))
        fields['shock_surface'][31] = True
        fields['surface_offsets'][31] = .5  # Refined position exactly x=0.5.
        fields['mach'][31] = 3.
        structure = dict(positions=np.array([.5]), hot_to_cold_directions=np.array([1.]), mach=3.)
        self.assertTrue(isolated_front_is_accurate(fields, structure, n))
        fields['normal_x'][31] = -1
        self.assertFalse(isolated_front_is_accurate(fields, structure, n))

    def test_unbounded_compression_and_unavailable_domain_exit_cannot_extend_screen(self):
        n = 32; x = jnp.arange(n, dtype=jnp.float32)
        # No zone exit before the hard search cap or the domain edge.
        zone = jnp.ones(n, dtype=bool)
        direction = jnp.ones((1, n))
        pressure = 1.+x; rho = jnp.ones(n)
        mask = jax.jit(_transition_jump_candidates)(pressure, rho, direction, zone)
        self.assertFalse(np.asarray(mask).any())

    def test_entropy_roundoff_guard_uses_cell_entropy_not_interpolated_primitive_ratio(self):
        q = smooth_profile(64, 1.)
        pressure, rho = jnp.asarray(q[:, 2], jnp.float32), jnp.asarray(q[:, 0], jnp.float32)
        direction = jnp.ones((1, 64)); zone = jnp.zeros(64, dtype=bool).at[30:35].set(True)
        mask = jax.jit(_transition_jump_candidates)(pressure, rho, direction, zone)
        self.assertFalse(np.asarray(mask).any())

    def test_oblique_shocks_and_isentropic_controls_use_continuous_normals(self):
        for normal in ((1., 1., 0.), (1., 1., 1.)):
            state, config, variables, helper, _, _ = build_planar_shock_state(24, 3., 4., normal)
            result = find_shocks_pfrommer(state, config, variables, helper,
                profile_aware_mach=True, transition_aware_detection=True)
            surface = np.asarray(result.shock_surface_cells)
            self.assertTrue(surface.any())
            self.assertLess(abs(float(np.median(np.asarray(result.mach_numbers)[surface]))/3-1), .05)
            self.assertEqual(int(result.num_shocks), int(surface.sum()))
            np.testing.assert_array_equal(np.asarray(result.shock_ids), surface.astype(int))
            # Keep the same compressive velocity and P geometry, but make the
            # cell entropy constant: a longer path must not invent a shock.
            pressure = state[variables.pressure_index]
            smooth = state.at[variables.density_index].set(pressure**(3/5))
            rejected = find_shocks_pfrommer(smooth, config, variables, helper,
                profile_aware_mach=True, transition_aware_detection=True)
            self.assertEqual(int(rejected.num_shocks), 0)
            self.assertFalse(np.asarray(rejected.shock_surface_cells).any())


if __name__ == '__main__':
    unittest.main()
