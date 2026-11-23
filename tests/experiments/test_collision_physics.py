import unittest
import numpy as np

from experiments.wind_bubble.assess_collision_physics import pair_metrics,source_free_segments


def exact_shock(mach,normal=(1.,0.,0.),boost=(0.,0.,0.)):
    gamma=5/3
    normal=np.asarray(normal,float);normal/=np.linalg.norm(normal)
    boost=np.asarray(boost,float)
    rho1=p1=1.
    r=(gamma+1)*mach**2/((gamma-1)*mach**2+2)
    p=(2*gamma*mach**2-(gamma-1))/(gamma+1)
    u1=mach*np.sqrt(gamma)
    a=np.r_[rho1,u1*normal+boost,p1]
    b=np.r_[rho1*r,u1/r*normal+boost,p1*p]
    return a,b,normal


class CollisionPhysicsTests(unittest.TestCase):
    def test_exact_weak_and_strong_shocks_pass_independent_checks(self):
        for mach in [1.1,2.,5.,10.]:
            a,b,n=exact_shock(mach)
            result=pair_metrics(a,b,n)
            self.assertTrue(result['compatible_05pct'])
            self.assertAlmostEqual(result['upstream_shock_frame_mach'],mach,places=10)
            self.assertAlmostEqual(result['mass_inferred_speed'],0.,places=10)
            for key in ['momentum_relative_error','enthalpy_relative_error','thermal_hugoniot_relative_residual']:
                self.assertAlmostEqual(result[key],0.,places=12)

    def test_rotations_and_galilean_boost_preserve_verdict_and_mach(self):
        a,b,n=exact_shock(3.,normal=(1,2,3),boost=(7,-4,2))
        result=pair_metrics(a,b,n)
        self.assertTrue(result['compatible_05pct'])
        self.assertAlmostEqual(result['upstream_shock_frame_mach'],3.,places=10)
        self.assertAlmostEqual(result['mass_inferred_speed'],np.array([7,-4,2])@n,places=10)

    def test_cold_dense_compression_fails_energy_and_entropy(self):
        a=np.array([.18,.49,0,0,1.10])
        b=np.array([1.48,.08,0,0,1.84])
        result=pair_metrics(a,b,[1,0,0])
        self.assertTrue(result['compressive_contrast'])
        self.assertTrue(result['converging_velocity'])
        self.assertLess(result['internal_energy_change'],0)
        self.assertGreater(result['hugoniot_work'],0)
        self.assertFalse(result['entropy_increase'])
        self.assertFalse(result['necessary_shock_conditions'])

    def test_advected_isobaric_contact_is_not_labeled_shock(self):
        a=np.array([1.,2.,1.,0.,1.])
        b=np.array([4.,2.,1.,0.,1.])
        result=pair_metrics(a,b,[1,0,0])
        self.assertAlmostEqual(result['mass_inferred_speed'],2.)
        self.assertFalse(result['compressive_contrast'])
        self.assertFalse(result['compatible_20pct'])

    def test_isentropic_compression_does_not_pass_entropy_condition(self):
        gamma=5/3;r=1.1
        c1=np.sqrt(gamma);c2=np.sqrt(gamma*r**(gamma-1))
        a=np.array([1.,2.,0.,0.,1.])
        b=np.array([r,2.-2*(c2-c1)/(gamma-1),0.,0.,r**gamma])
        result=pair_metrics(a,b,[1,0,0])
        self.assertTrue(result['compressive_contrast'])
        self.assertFalse(result['entropy_increase'])
        self.assertFalse(result['compatible_20pct'])

    def test_segment_excludes_source_crossing_even_with_outside_endpoints(self):
        sources=np.array([[.5,.5,.5]])
        a=np.array([[.3,.5,.5],[.3,.7,.5]])
        b=np.array([[.7,.5,.5],[.7,.7,.5]])
        np.testing.assert_array_equal(source_free_segments(a,b,sources,.1),[False,True])

    def test_invalid_state_and_normal_rejected(self):
        a,b,n=exact_shock(2.)
        for invalid in [a*np.nan,np.array([-1.,0,0,0,1.])]:
            with self.assertRaises(ValueError):pair_metrics(invalid,b,n)
        with self.assertRaises(ValueError):pair_metrics(a,b,[2,0,0])


if __name__=='__main__':unittest.main()
