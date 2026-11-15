import argparse
import unittest
import numpy as np
import jax.numpy as jnp
from astronomix._modules._stellar_wind.stellar_wind import _wind_injection
from astronomix.option_classes.simulation_config import finalize_config
from experiments.wind_bubble.run_single_bubble import build_problem
from experiments.wind_bubble.two_bubble_source_budget import budget_plan,INJECTION_RADIUS
from experiments.wind_bubble.three_bubble_setup import SOURCES,injection_counts,source_budget,weighted_budget


class ThreeBubbles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state,cls.config,cls.params,cls.registry,cls.helper=build_problem(argparse.Namespace(
            num_cells=32,num_snapshots=3,t_end=.02,num_injection_cells=2,plasma_beta=None))

    def configuration(self,sources,strengths):
        return finalize_config(self.config._replace(wind_config=self.config.wind_config._replace(
            source_positions=sources,source_strengths=strengths)),self.state.shape)

    def test_three_disjoint_nonaligned_sources_and_equal_masks(self):
        positions=np.asarray(SOURCES);dist=np.linalg.norm(positions[:,None]-positions[None,:],axis=-1)
        self.assertGreater(dist[np.triu_indices(3,1)].min(),2*INJECTION_RADIUS)
        self.assertGreater(np.linalg.norm(np.cross(positions[1]-positions[0],positions[2]-positions[0])),0)
        for n in [32,64,128]:self.assertEqual(len(set(injection_counts(n))),1)

    def test_reference_per_source_budget_preserved(self):
        base=budget_plan(64)
        reference=dict(status='complete',source_positions=((.35,.5,.5),(.65,.5,.5)),injection_radius=INJECTION_RADIUS,
            injection_prescription='existing EI analytic-volume normalization and radius-minus-half-cell mask',**base)
        for n in [32,64,128]:
            plan=source_budget(n,reference)
            np.testing.assert_allclose(plan['discrete_energy_rates'],base['discrete_energy_rates'][0])
            self.assertAlmostEqual(sum(plan['discrete_energy_rates']),1.5*sum(base['discrete_energy_rates']))
        with self.assertRaises(ValueError):source_budget(32,reference,actual_counts=[1,2,3])
        with self.assertRaises(ValueError):weighted_budget(source_budget(32),[1,1])

    def test_three_sources_match_sum_of_individual_injections(self):
        before=self.state.at[self.registry.velocity_index.x].set(.2)
        def inject(sources,weights):
            return np.asarray(_wind_injection(before,jnp.asarray(.001),self.configuration(sources,weights),self.params,self.helper,self.registry))
        changes=[inject((source,),(weight,))-np.asarray(before) for source,weight in zip(SOURCES,[1.,.8,.6])]
        np.testing.assert_allclose(inject(SOURCES,(1.,.8,.6)),np.asarray(before)+sum(changes),rtol=1e-6,atol=1e-8)

    def test_discrete_source_mass_energy_at_rest(self):
        config=self.configuration(SOURCES,(1.,1.,1.))
        after=np.asarray(_wind_injection(self.state,jnp.asarray(.001),config,self.params,self.helper,self.registry))
        delta=after-np.asarray(self.state);centers=np.asarray(self.helper.geometric_centers);dx=float(config.grid_spacing)
        plan=source_budget(32)
        for i,source in enumerate(SOURCES):
            mask=np.linalg.norm(centers-source,axis=-1)<=INJECTION_RADIUS-dx/2
            self.assertEqual(int(mask.sum()),plan['injection_cell_counts'][i])
            mass=delta[self.registry.density_index][mask].sum()*dx**3/.001
            heat=delta[self.registry.pressure_index][mask].sum()*dx**3/.001/(float(self.params.gamma)-1)
            np.testing.assert_allclose(mass,plan['discrete_mass_rates'][i],rtol=1e-4)
            np.testing.assert_allclose(heat,plan['discrete_energy_rates'][i],rtol=1e-5)
