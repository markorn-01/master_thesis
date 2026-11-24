"""Physical-reference checks and detector controls for a shock beside a contact."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from astronomix.test_setups.reference_solutions.riemann_solver import _exact_riemann_ideal_gas
from experiments.shock.check_colliding_shocks import make_analyzer
from experiments.shock.check_shock_contact import pressure_transition, score
from experiments.shock.colliding_shocks_reference import conserved, flux
from experiments.shock.shock_contact_reference import ShockContactReference


class ShockContactPhysicsTests(unittest.TestCase):
    def test_shock_and_contact_satisfy_flux_balance_and_shock_entropy(self):
        for mach in (2.,3.,5.):
            for factor in (.25,4.,8.):
                reference=ShockContactReference(mach,factor)
                np.testing.assert_allclose(reference.flux_residuals(),0,atol=1e-12,rtol=0)
                up,down=reference.upstream,reference.downstream
                self.assertGreater(down[2]/down[0]**reference.gamma,up[2]/up[0]**reference.gamma)
                np.testing.assert_allclose(reference.contact_gas[1:],down[1:],atol=0)
                self.assertAlmostEqual(reference.contact_gas[0]/down[0],factor)
                self.assertGreater(reference.shock_speed,reference.contact_speed)

    def test_matches_independent_exact_riemann_solver(self):
        # Cross-check the elementary two-wave construction using the existing
        # general nonlinear pressure solve, not the detector's Mach relation.
        x=np.linspace(0,1,257)
        for factor in (.25,8.):
            reference=ShockContactReference(contact_density_factor=factor)
            left,right=reference.contact_gas,reference.upstream
            time=.04;origin=.25
            with jax.experimental.enable_x64():
                rho,u,p=_exact_riemann_ideal_gas(*left,*right,reference.gamma,
                    jnp.asarray(x),time,origin)
                actual=np.stack([np.asarray(rho),np.asarray(u),np.asarray(p)],axis=-1)
            expected=reference.sample(x,time,origin)
            np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-11)

    def test_exact_cell_averages_obey_integrated_conservation(self):
        reference=ShockContactReference()
        origin=.25
        initial=conserved(reference.cell_averages(64,0,origin)).mean(axis=0)
        net_flux=flux(reference.contact_gas)-flux(reference.upstream)
        for time in (.01,.025,.05):
            average=conserved(reference.cell_averages(64,time,origin)).mean(axis=0)
            np.testing.assert_allclose(average,initial+time*net_flux,rtol=1e-13,atol=1e-13)

    def test_filter_preserves_total_conserved_fields(self):
        reference=ShockContactReference()
        exact,_=reference.profile(64,2.)
        for width in (3,5):
            broadened,_=reference.profile(64,2.,width)
            np.testing.assert_allclose(conserved(exact).sum(axis=0),conserved(broadened).sum(axis=0),
                                       rtol=1e-13,atol=1e-11)

    def test_grid_phase_separation_and_reflection_are_consistent(self):
        reference=ShockContactReference()
        for phase in (0.,.5):
            values,layout=reference.profile(64,2.,phase=phase)
            np.testing.assert_allclose(reference.positions(layout['time'],layout['riemann_origin']),
                                       [layout['contact_x'],layout['shock_x']],atol=1e-15)
            self.assertAlmostEqual((layout['shock_x']-layout['contact_x'])*64,2.)
            reflected,opposite=reference.profile(64,2.,phase=phase,orientation=-1)
            np.testing.assert_array_equal(reflected,values[::-1]*[1.,-1.,1.])
            self.assertAlmostEqual(opposite['shock_x'],1-layout['shock_x'])

    def test_no_contact_limit_reproduces_isolated_shock(self):
        reference=ShockContactReference(contact_density_factor=1.)
        for gap in (.5,1.,8.):
            pair,layout=reference.profile(64,gap)
            single,_=reference.control(64,'shock')
            np.testing.assert_allclose(pair,single,rtol=1e-13,atol=1e-13)
            self.assertFalse(layout['has_contact'])

    def test_contact_can_reverse_combined_temperature_change_without_removing_shock(self):
        reference=ShockContactReference(contact_density_factor=8.)
        left,middle,right=reference.states
        self.assertGreater(middle[2]/middle[0],right[2]/right[0]) # The physical shock heats.
        self.assertLess(left[2]/left[0],right[2]/right[0])       # The dense contact is colder.
        self.assertGreater(left[0],middle[0])
        self.assertGreater(middle[0],right[0])

    def test_invalid_reference_inputs_rejected(self):
        for factor in (0.,-1.,np.nan):
            with self.assertRaises(ValueError):ShockContactReference(contact_density_factor=factor)
        reference=ShockContactReference()
        for args in ((32,0.),(32,-1.),(33,2.),(32,20.)):
            with self.assertRaises(ValueError):reference.profile(*args)
        with self.assertRaises(ValueError):reference.profile(32,2.,phase=1.)
        with self.assertRaises(ValueError):reference.profile(32,2.,orientation=0)


class ShockContactFinderControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyze=staticmethod(make_analyzer(32))

    def test_isolated_and_well_separated_shocks_are_recovered(self):
        for orientation in (1,-1):
            for factor in (.25,8.):
                reference=ShockContactReference(contact_density_factor=factor)
                for kind in ('shock','pair'):
                    if kind=='pair': profile,layout=reference.profile(32,8.,orientation=orientation)
                    else: profile,layout=reference.control(32,'shock',orientation=orientation)
                    fields=self.analyze(profile)
                    row,_,_=score(profile,fields,layout,32,kind)
                    self.assertEqual(row['outcome'],'recovered')
                    self.assertGreaterEqual(row['coverage_2cell'],.99)
                    self.assertLess(abs(row['mach_relative_error']),1e-5)

    def test_moving_contacts_have_constant_pressure_and_velocity_and_no_surface(self):
        for factor in (.25,4.,8.):
            reference=ShockContactReference(contact_density_factor=factor)
            for width in (1,3,5):
                profile,layout=reference.control(32,'contact',width)
                np.testing.assert_allclose(profile[:,1:],np.broadcast_to(reference.downstream[1:],profile[:,1:].shape),
                                           rtol=0,atol=1e-13)
                self.assertFalse(pressure_transition(profile).any())
                row,_,_=score(profile,self.analyze(profile),layout,32,'contact')
                self.assertEqual(row['outcome'],'contact_rejected')

    def test_stage_accounting_matches_production_mask_for_close_pair(self):
        reference=ShockContactReference()
        profile,layout=reference.profile(32,.5)
        fields=self.analyze(profile)
        row,_,band=score(profile,fields,layout,32,'pair')
        np.testing.assert_array_equal(fields['shock_zone'],fields['compression']
            & fields['gradient_alignment'] & fields['pressure_jump_pass'] & fields['temperature_jump_pass'])
        self.assertGreater(int(band.sum()),0)
        self.assertEqual(sum(row[f'pressure_cells_pass_pattern_{i:04b}'] for i in range(16)),int(band.sum()))


if __name__=='__main__':
    unittest.main()
