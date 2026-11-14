import argparse
from pathlib import Path
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from experiments.wind_bubble.audit_detection_stages import selected_indices,StateCapture,stage_fields
from experiments.wind_bubble.run_single_bubble import build_problem


class DetectionAuditTests(unittest.TestCase):
    def test_event_selection_does_not_add_schedule_entries(self):
        self.assertEqual(selected_indices([2,1,2],3),[1,2])
        for values in [[],[-1],[3]]:
            with self.assertRaises(ValueError):selected_indices(values,3)

    def test_jitted_observer_writes_only_selected_unmodified_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture=StateCapture(Path(tmp)/'audit',[0.,.1,.2])
            observe=capture.observer([1])
            @jax.jit
            def step(time,state):
                observe(time,state)
                return state+1
            state=jnp.ones((5,4,4,4))
            for time in [0.,.1,.2]:np.testing.assert_array_equal(step(jnp.asarray(time),state),2)
            jax.effects_barrier()
            self.assertEqual(capture.saved,[1])
            np.testing.assert_array_equal(np.load(Path(tmp)/'audit/state_0001.npy'),state)
            self.assertEqual(len(list((Path(tmp)/'audit').glob('*.npy'))),1)

    def test_production_mask_decomposition_on_nonuniform_state(self):
        state,config,_,registry,helper=build_problem(argparse.Namespace(num_cells=16,num_snapshots=3,
            t_end=.02,num_injection_cells=1,plasma_beta=None))
        shape=helper.geometric_centers.shape[:-1]
        physical=jnp.ones((state.shape[0],)+shape)
        x=helper.geometric_centers[...,0]
        physical=physical.at[registry.pressure_index].set(jnp.where(x<.5,10.,.01))
        physical=physical.at[registry.density_index].set(jnp.where(x<.5,4.,1.))
        fields={k:np.asarray(v) for k,v in stage_fields(physical,config,registry,helper).items()}
        np.testing.assert_array_equal(fields['minimum_jump'],fields['pressure_jump_pass']&fields['temperature_jump_pass'])
        np.testing.assert_array_equal(fields['shock_zone'],fields['compression']&fields['gradient_alignment']&fields['minimum_jump'])
        self.assertTrue(fields['minimum_jump'].any())
