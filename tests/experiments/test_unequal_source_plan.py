import unittest
import numpy as np
from experiments.wind_bubble.unequal_sources import strength_plan
from experiments.wind_bubble.two_bubble_source_budget import budget_plan


class UnequalPlan(unittest.TestCase):
    def test_default_preserves_equal_budget(self):
        base=budget_plan(64);plan=strength_plan(base)
        self.assertEqual(plan['discrete_mass_rates'],base['discrete_mass_rates'])
        self.assertTrue(plan['equal_source_reference_preserved'])
    def test_half_strength_scales_mass_and_luminosity(self):
        plan=strength_plan(budget_plan(64),[1,.5])
        for key in ['source_mass_loss_rates','nominal_wind_luminosities','discrete_mass_rates','discrete_energy_rates']:
            self.assertEqual(plan[key][1],.5*plan[key][0])
    def test_reject_invalid_weights(self):
        for values in [[1],[-1,1],[0,1],[1,np.inf]]:
            with self.assertRaises(ValueError):strength_plan(budget_plan(64),values)
