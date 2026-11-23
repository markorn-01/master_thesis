"""Independent reference conservation and known-location detection controls."""
import unittest
import numpy as np

from experiments.shock.colliding_shocks_reference import (
    SymmetricShockCollision,conserved,flux,isolated_control,contact_control,
)
from experiments.shock.check_colliding_shocks import make_analyzer,score_fronts


class ExactCollisionReferenceTests(unittest.TestCase):
    def test_all_shock_branches_conserve_mass_momentum_and_energy(self):
        for mach in (1.5,2.,3.,5.):
            reference=SymmetricShockCollision(mach)
            np.testing.assert_allclose(reference.flux_residuals(),0,rtol=0,atol=1e-12)
            self.assertGreater(reference.outgoing_mach,1)
            self.assertGreater(reference.outgoing_speed,0)
            self.assertGreater(reference.central[0],reference.incoming_left[0])
            self.assertGreater(reference.central[2],reference.incoming_left[2])
            # Conservation alone also admits nonphysical discontinuities.
            # Both incident and outgoing branches must increase entropy.
            entropy_proxy=lambda q:q[2]/q[0]**reference.gamma
            self.assertGreater(entropy_proxy(reference.incoming_left),entropy_proxy(reference.ambient))
            self.assertGreater(entropy_proxy(reference.central),entropy_proxy(reference.incoming_left))

    def test_exact_cell_averages_obey_box_integral_balance_through_collision(self):
        reference=SymmetricShockCollision()
        initial,_=reference.cell_averages(64,0.)
        initial_total=conserved(initial).mean(axis=0)
        net_flux=flux(reference.incoming_left)-flux(reference.incoming_right)
        for time in (.5*reference.collision_time,reference.collision_time,reference.collision_time+.02):
            values,_=reference.cell_averages(64,time)
            np.testing.assert_allclose(conserved(values).mean(axis=0),initial_total+time*net_flux,
                                       rtol=1e-13,atol=1e-13)

    def test_controlled_broadening_preserves_conserved_box_totals(self):
        reference=SymmetricShockCollision()
        time=reference.time_at_separation('outgoing',8,64)
        original,_=reference.cell_averages(64,time,1)
        for width in (3,5):
            filtered,_=reference.cell_averages(64,time,width)
            np.testing.assert_allclose(conserved(filtered).sum(axis=0),conserved(original).sum(axis=0),
                                       rtol=1e-13,atol=1e-12)
            self.assertTrue((filtered[:,0]>0).all())
            self.assertTrue((filtered[:,2]>0).all())

    def test_positions_and_shock_frame_mach_are_known_on_both_sides(self):
        reference=SymmetricShockCollision()
        for phase,normal in [('incoming',[1.,-1.]),('outgoing',[-1.,1.])]:
            time=reference.time_at_separation(phase,8,64)
            structure=reference.structure(time)
            np.testing.assert_allclose(structure['positions'],[.4375,.5625],atol=1e-14)
            np.testing.assert_array_equal(structure['hot_to_cold_directions'],normal)
            self.assertEqual(structure['phase'],phase)
        meeting=reference.structure(reference.collision_time)
        self.assertFalse(meeting['two_distinct_fronts'])
        np.testing.assert_array_equal(meeting['positions'],[.5,.5])

    def test_invalid_inputs_and_out_of_domain_fronts_rejected(self):
        for mach in (1.,-1.,np.nan):
            with self.assertRaises(ValueError):SymmetricShockCollision(mach)
        reference=SymmetricShockCollision()
        with self.assertRaises(ValueError):reference.cell_averages(32,.01,2)
        with self.assertRaises(ValueError):reference.structure(-1.)
        with self.assertRaises(ValueError):reference.structure(reference.collision_time+1.)
        with self.assertRaises(ValueError):reference.time_at_separation('incoming',100,32)


class KnownCollisionDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference=SymmetricShockCollision()
        cls.evaluate=staticmethod(make_analyzer(32))

    def test_well_separated_exact_incoming_and_outgoing_fronts_are_recovered(self):
        for phase in ('incoming','outgoing'):
            time=self.reference.time_at_separation(phase,8,32)
            profile,structure=self.reference.cell_averages(32,time)
            fields=self.evaluate(profile)
            rows,count=score_fronts(fields,structure,32)
            self.assertEqual(len(rows),2)
            self.assertGreater(count,0)
            for row in rows:
                self.assertGreaterEqual(row['coverage_2cell'],.99)
                self.assertIsNotNone(row['mach_relative_error'])
                self.assertLess(abs(row['mach_relative_error']),.05)

    def test_isolated_branch_controls_have_correct_strengths(self):
        for phase in ('incoming','outgoing'):
            profile,structure=isolated_control(self.reference,32,phase)
            rows,count=score_fronts(self.evaluate(profile),structure,32)
            self.assertEqual(len(rows),1)
            self.assertGreater(count,0)
            self.assertGreaterEqual(rows[0]['coverage_2cell'],.99)
            self.assertLess(abs(rows[0]['mach_relative_error']),.05)

    def test_isobaric_contact_produces_no_surface(self):
        for width in (1,3,5):
            profile,_=contact_control(32,width)
            np.testing.assert_allclose(profile[:,2],1.,atol=1e-14)
            fields=self.evaluate(profile)
            self.assertFalse(fields['shock_surface'].any())

    def test_two_fronts_cannot_both_match_a_single_central_surface(self):
        # A positional tolerance must not turn one merged point into two recoveries.
        n=32;shape=(n,)*3;fields=dict(shock_surface=np.zeros(shape,bool),
            direction=np.zeros(shape+(3,)),surface_offsets=np.zeros(shape),mach=np.zeros(shape))
        fields['shock_surface'][15,2:-2,2:-2]=True
        fields['surface_offsets'][15]=.5
        fields['direction'][15,...,0]=1.
        fields['mach'][15]=3.
        structure=dict(phase='incoming',positions=np.array([.49,.51]),mach=3.,hot_to_cold_directions=np.array([1.,-1.]))
        rows,_=score_fronts(fields,structure,n)
        self.assertEqual(sum(r['coverage_2cell'] for r in rows),0.)


if __name__=='__main__':unittest.main()
