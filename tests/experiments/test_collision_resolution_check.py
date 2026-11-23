import copy
import unittest
import numpy as np

from experiments.wind_bubble.run_collision_resolution_check import resolution_plan
from experiments.wind_bubble.three_bubble_setup import SOURCES,source_budget
from tests.experiments.test_two_bubble_source_budget import reference_metadata


def three_source_reference():
    two=reference_metadata()
    budget=source_budget(64,two)
    return dict(status='complete',num_cells=64,t_end=.16,injection_radius=.0625,
        ambient_density=1.,ambient_pressure=.01,gamma=5/3,wind_terminal_velocity=10.,
        source_positions=SOURCES,source_strengths=[1.,1.,1.],solver_dt_max=.16/17,
        requested_times=np.linspace(0,.16,17,dtype=np.float32).tolist(),
        discrete_mass_rates=budget['discrete_mass_rates'],discrete_energy_rates=budget['discrete_energy_rates'])


class CollisionResolutionCheckTests(unittest.TestCase):
    def test_same_physical_budgets_and_schedule_at_finer_grid(self):
        reference=three_source_reference()
        plan=resolution_plan(reference,reference_metadata())
        self.assertEqual(plan['num_cells'],128)
        self.assertEqual(plan['budget']['injection_cell_counts'],[1736]*3)
        self.assertAlmostEqual(plan['budget']['wind_mass_loss_rate_per_source'],.008110599078341014)
        np.testing.assert_allclose(plan['budget']['discrete_energy_rates'],reference['discrete_energy_rates'],rtol=1e-12)
        np.testing.assert_allclose(plan['budget']['discrete_mass_rates'],reference['discrete_mass_rates'],rtol=1e-12)
        np.testing.assert_array_equal(plan['requested_times'],reference['requested_times'])
        self.assertEqual(plan['solver_dt_max'],reference['solver_dt_max'])

    def test_changes_to_physics_or_geometry_rejected(self):
        for key,value in [('gamma',1.4),('injection_radius',.08),('ambient_pressure',.02),
                          ('source_strengths',[1.,.5,1.]),('source_positions',[[0,0,0]]*3)]:
            reference=copy.deepcopy(three_source_reference());reference[key]=value
            with self.assertRaises(ValueError):resolution_plan(reference,reference_metadata())

    def test_inconsistent_source_budget_rejected(self):
        reference=three_source_reference();reference['discrete_energy_rates']=[1.,1.,1.]
        with self.assertRaises(AssertionError):resolution_plan(reference,reference_metadata())


if __name__=='__main__':unittest.main()
