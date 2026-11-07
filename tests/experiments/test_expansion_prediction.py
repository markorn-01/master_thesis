import unittest
import numpy as np
from experiments.astrolink.controlled_expansion_cases import expansion_cases
from experiments.astrolink.check_expansion_prediction import evaluate
from experiments.astrolink.expansion_prediction import predict_expanding_positions


class ExpansionPredictionTests(unittest.TestCase):
    def setUp(self):self.cases={c['name']:c for c in expansion_cases()}

    def results(self,name,gate=.2):
        r,_,_=evaluate(self.cases[name]);return {x['method']:x for x in r if x['gate']==gate}

    def test_fixed_center_expansion_is_recovered(self):
        r=self.results('symmetric_expansion')
        self.assertFalse(r['centroid_translation']['exact_correspondence'])
        self.assertTrue(r['isotropic_expansion']['exact_correspondence'])
        estimate=r['isotropic_expansion']['estimates'][0]
        np.testing.assert_allclose(estimate['velocity'],0,atol=1e-12)
        self.assertAlmostEqual(estimate['extent_rate'],.3)

    def test_joint_translation_and_expansion_matches_constructed_positions(self):
        c=self.cases['translating_expanding_sphere'];_,_,p=evaluate(c)
        np.testing.assert_allclose(p,c['positions_2'],rtol=0,atol=1e-12)

    def test_future_positions_and_truth_cannot_influence_prediction(self):
        c=dict(self.cases['translating_expanding_sphere']);_,_,before=evaluate(c)
        c['positions_2']=c['positions_2']+50;c['expected_links']=[]
        _,_,after=evaluate(c);np.testing.assert_array_equal(before,after)

    def test_slowing_expansion_exposes_extrapolation_failure(self):
        r=self.results('decelerating_expansion')
        self.assertTrue(r['geometry']['exact_correspondence'])
        self.assertFalse(r['isotropic_expansion']['exact_correspondence'])

    def test_uniform_scaling_cannot_recover_directional_deformation(self):
        c=self.cases['anisotropic_expansion'];_,_,p=evaluate(c)
        self.assertGreater(float(np.max(np.linalg.norm(p-c['positions_2'],axis=1))),.1)
        self.assertFalse(self.results('anisotropic_expansion',.08)['isotropic_expansion']['exact_correspondence'])

    def test_split_and_coverage_change_remain_failures(self):
        for name in ['group_split','changing_sample_coverage']:
            self.assertFalse(self.results(name)['isotropic_expansion']['exact_correspondence'])

    def test_nonpositive_future_extent_uses_explicit_translation_fallback(self):
        d=np.vstack((np.eye(3),-np.eye(3)));n=np.tile([0,0,1.],(6,1))
        p,e,_=predict_expanding_positions(.5*d,n,[0]*6,.4*d,n,[10]*6,[0,.2,1.2])
        self.assertEqual(e[0]['expansion_reason'],'translation_fallback_nonpositive_predicted_extent')
        self.assertFalse(e[0]['expansion_initialized']);np.testing.assert_allclose(p,.4*d)

    def test_unresolved_extent_does_not_divide_by_zero(self):
        p=np.zeros((2,3));n=np.tile([0,0,1.],(2,1))
        predicted,e,_=predict_expanding_positions(p,n,[0,0],p,n,[10,10],[0,.2,1.2])
        self.assertEqual(e[0]['expansion_reason'],'translation_fallback_unresolved_extent')
        np.testing.assert_array_equal(predicted,p)


if __name__=='__main__':unittest.main()
