import copy
import unittest
import numpy as np
from experiments.wind_bubble.two_bubble_source_budget import SOURCES,INJECTION_RADIUS,budget_plan


def reference_metadata():
    p=budget_plan(64)
    return dict(status='complete',num_cells=64,source_positions=SOURCES,injection_radius=INJECTION_RADIUS,
                injection_prescription='existing EI analytic-volume normalization and radius-minus-half-cell mask',
                **{k:p[k] for k in ['wind_terminal_velocity','wind_mass_loss_rate_per_source','injection_cell_counts',
                                    'discrete_mass_rates','discrete_energy_rates']})


class SourceBudgetTests(unittest.TestCase):
    def test_known_counts_and_unmatched_power_difference(self):
        a,b=budget_plan(64),budget_plan(128)
        self.assertEqual(a['injection_cell_counts'],[176,176]);self.assertEqual(b['injection_cell_counts'],[1780,1780])
        np.testing.assert_allclose(a['discrete_energy_rates'],[.32825707012703415]*2)
        self.assertGreater(b['discrete_energy_rates'][0]/a['discrete_energy_rates'][0],1.26)

    def test_matching_preserves_both_budgets_and_velocity(self):
        ref=reference_metadata();p=budget_plan(128,ref)
        np.testing.assert_allclose(p['discrete_mass_rates'],ref['discrete_mass_rates'],rtol=1e-12)
        np.testing.assert_allclose(p['discrete_energy_rates'],ref['discrete_energy_rates'],rtol=1e-12)
        self.assertEqual(p['wind_terminal_velocity'],10.)
        self.assertLess(p['wind_mass_loss_rate_per_source'],.01)

    def test_default_keeps_original_parameters_and_same_resolution_scale_one(self):
        self.assertEqual(budget_plan(32)['parameter_scale'],1)
        self.assertEqual(budget_plan(64,reference_metadata())['parameter_scale'],1)

    def test_reference_mismatch_and_inconsistent_rates_rejected(self):
        for key,value in [('status','running'),('injection_radius',.08),('wind_terminal_velocity',20.),
                          ('discrete_energy_rates',[.5,.5]),('injection_cell_counts',[177,176]),
                          ('source_positions',((.3,.5,.5),(.7,.5,.5)))]:
            ref=copy.deepcopy(reference_metadata());ref[key]=value
            with self.assertRaises(ValueError):budget_plan(128,ref)

    def test_solver_count_disagreement_rejected(self):
        with self.assertRaises(ValueError):budget_plan(128,actual_counts=[176,176])


if __name__=='__main__':unittest.main()
