"""Candidate evidence must remain separate from full state compatibility."""
import unittest

import numpy as np

from experiments.shock.pressure_velocity_diagnostic import (
    analyze_state,jump_diagnostics,summarize,
)
from experiments.shock.shock_contact_reference import ShockContactReference
from experiments.shock.check_pressure_velocity_diagnostic import negative_controls


class PressureVelocityDiagnosticTests(unittest.TestCase):
    def test_exact_shock_pair_has_consistent_independent_jumps(self):
        ref=ShockContactReference()
        up=np.array([1.,0.,0.,0.,1.]);down=np.r_[ref.downstream[0],ref.downstream[1],0.,0.,ref.downstream[2]]
        result=jump_diagnostics(up,down,[-1.,0.,0.])
        for key in ('pressure_velocity_relative_error','density_jump_relative_error','entropy_signal_relative_error',
                    'momentum_relative_error','enthalpy_relative_error'):
            self.assertLess(result[key],1e-12)
        self.assertAlmostEqual(result['pressure_mach'],3.)

    def test_unresolved_dense_contact_is_candidate_not_full_jump(self):
        profile,_=ShockContactReference().profile(32,.5)
        rows,geometry=analyze_state(profile.T,1/32)
        report=summarize(rows,geometry)
        self.assertEqual(report['pressure_velocity_candidates_10pct'],1)
        self.assertEqual(report['full_jump_candidates_10pct'],0)
        self.assertTrue(all(r['density_jump_relative_error']>1 for r in rows if r['pressure_velocity_10pct']))

    def test_resolved_shock_contact_pair_has_valid_state_pair(self):
        for orientation in (1,-1):
            profile,_=ShockContactReference().profile(64,2.,orientation=orientation)
            rows,geometry=analyze_state(profile.T,1/64)
            self.assertEqual(summarize(rows,geometry)['full_jump_candidates_10pct'],1)
            for r in rows:
                if r['full_jump_10pct']:
                    self.assertAlmostEqual(r['pressure_mach'],3.)

    def test_contacts_alone_are_not_candidates(self):
        for factor in (.25,8.):
            profile,_=ShockContactReference(contact_density_factor=factor).control(32,'contact',5)
            rows,geometry=analyze_state(profile.T,1/32)
            self.assertEqual(len(geometry['indices']),0)
            self.assertEqual(rows,[])

    def test_smooth_compression_can_flag_pv_but_not_full_jump(self):
        pv_flags=0
        for name,state in negative_controls():
            rows,geometry=analyze_state(state,1/state.shape[1])
            result=summarize(rows,geometry)
            self.assertEqual(result['full_jump_candidates_10pct'],0,name)
            pv_flags+=result['pressure_velocity_candidates_10pct']
        self.assertGreater(pv_flags,0) # Explicitly exposes pressure/velocity-only ambiguity.

    def test_galilean_boost_preserves_pair_classification(self):
        profile,_=ShockContactReference().control(32,'shock')
        rows,_=analyze_state(profile.T,1/32)
        profile[:,1]+=17.
        boosted,_=analyze_state(profile.T,1/32)
        self.assertEqual([r['full_jump_10pct'] for r in rows],[r['full_jump_10pct'] for r in boosted])
        np.testing.assert_allclose([r['pressure_mach'] for r in rows],[r['pressure_mach'] for r in boosted])

    def test_three_dimensional_planar_control_is_read_only(self):
        n=16;profile,_=ShockContactReference().control(n,'shock')
        state=np.zeros((5,n,n,n))
        for field,column in ((0,0),(1,1),(4,2)):
            state[field]=profile[:,column,None,None]
        before=state.copy()
        rows,geometry=analyze_state(state,1/n)
        np.testing.assert_array_equal(state,before)
        self.assertEqual(summarize(rows,geometry)['full_jump_candidates_10pct'],(n-4)**2)

    def test_opposing_front_direction_is_not_crossed_by_accepted_window(self):
        from experiments.shock.colliding_shocks_reference import SymmetricShockCollision
        ref=SymmetricShockCollision()
        profile,_=ref.cell_averages(32,ref.time_at_separation('incoming',2,32))
        rows,_=analyze_state(profile.T,1/32)
        self.assertTrue(any(not r['ray_direction_consistent'] for r in rows))
        self.assertTrue(all(not r['pressure_velocity_10pct'] for r in rows if not r['ray_direction_consistent']))


if __name__=='__main__':
    unittest.main()
