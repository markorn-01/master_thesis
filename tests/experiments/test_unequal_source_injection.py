"""JAX tests: required locally on CPU and again in the remote GPU environment."""
import argparse
import unittest
import jax.numpy as jnp
import numpy as np
from astronomix._modules._stellar_wind.stellar_wind import _wind_injection
from astronomix.option_classes.simulation_config import finalize_config
from experiments.wind_bubble.run_single_bubble import build_problem
from experiments.wind_bubble.two_bubble_source_budget import SOURCES, budget_plan
from experiments.wind_bubble.unequal_sources import strength_plan


class UnequalInjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state,cls.config,cls.params,cls.registry,cls.helper=build_problem(argparse.Namespace(
            num_cells=32,num_snapshots=3,t_end=.01,num_injection_cells=2,plasma_beta=None))
    def configuration(self,positions=SOURCES,strengths=()):
        return finalize_config(self.config._replace(wind_config=self.config.wind_config._replace(
            source_positions=positions,source_strengths=strengths)),self.state.shape)
    def inject(self,state,config,params=None):
        return np.asarray(_wind_injection(state,jnp.asarray(.001),config,
            self.params if params is None else params,self.helper,self.registry))
    def test_unit_weights_preserve_equal_source_state_exactly(self):
        for state in [self.state,self.state.at[self.registry.velocity_index.x].set(.2)]:
            np.testing.assert_array_equal(self.inject(state,self.configuration()),self.inject(state,self.configuration(strengths=(1.,1.))))
    def test_disjoint_sources_match_individually_scaled_mass_parameters(self):
        before=self.state.at[self.registry.velocity_index.x].set(.2)
        singles=[]
        for source,weight in zip(SOURCES,[1.,.5]):
            params=self.params._replace(wind_params=self.params.wind_params._replace(
                wind_mass_loss_rate=self.params.wind_params.wind_mass_loss_rate*weight))
            singles.append(self.inject(before,self.configuration((source,)),params)-np.asarray(before))
        both=self.inject(before,self.configuration(strengths=(1.,.5)))
        np.testing.assert_allclose(both,np.asarray(before)+sum(singles),rtol=1e-6,atol=1e-8)
    def test_integrated_discrete_budgets_at_rest(self):
        after=self.inject(self.state,self.configuration(strengths=(1.,.5)))
        change=after-np.asarray(self.state);dx=float(self.config.grid_spacing)
        plan=strength_plan(budget_plan(32),[1.,.5])
        centers=np.asarray(self.helper.geometric_centers)
        for i,source in enumerate(SOURCES):
            mask=np.linalg.norm(centers-source,axis=-1)<=.0625-dx/2
            mass=change[self.registry.density_index][mask].sum()*dx**3/.001
            heat=change[self.registry.pressure_index][mask].sum()*dx**3/.001/(float(self.params.gamma)-1)
            np.testing.assert_allclose(mass,plan['discrete_mass_rates'][i],rtol=1e-4)
            np.testing.assert_allclose(heat,plan['discrete_energy_rates'][i],rtol=1e-5)
    def test_invalid_weights_rejected(self):
        for strengths in [(1.,),(1.,-1.),(1.,float('nan'))]:
            with self.assertRaises(ValueError):self.configuration(strengths=strengths)
        with self.assertRaises(ValueError):self.configuration((),(1.,))
