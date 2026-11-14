import argparse
import json
from pathlib import Path
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from experiments.wind_bubble.audit_detection_stages import selected_indices,StateCapture,stage_fields,extension_schedule,compare_reference,run
from experiments.wind_bubble.run_single_bubble import build_problem


class DetectionAuditTests(unittest.TestCase):
    def test_missing_reference_state_fails_before_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'reference';(root/'audit').mkdir(parents=True)
            (root/'metadata.json').write_text(json.dumps(dict(status='complete',num_cells=32,
                source_strengths=[1.,.5],requested_times=[0.,.01,.02],t_end=.02)))
            (root/'audit/verification.json').write_text(json.dumps(dict(passed=True,captured_events=[1])))
            args=argparse.Namespace(capture_events=[1,2,3],num_analysis=4,extend_reference=True,
                reference_run=root,t_end=.03,num_cells=32,source_strengths=[1.,.5],output_dir=Path(tmp)/'new')
            with self.assertRaisesRegex(ValueError,'Reference full state missing'):run(args)
            self.assertFalse(args.output_dir.exists())

    def test_prefix_comparison_checks_prefix_and_reports_boolean_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots=[Path(tmp)/name for name in ['long','short']]
            for root,times in zip(roots,[[0.,.1,.2],[0.,.1]]):
                (root/'online').mkdir(parents=True)
                meta=dict(num_cells=32,source_strengths=[1.,.5],source_positions=[[.35,.5,.5],[.65,.5,.5]],
                    discrete_mass_rates=[1.,.5],discrete_energy_rates=[1.,.5],injection_radius=.0625,
                    wind_terminal_velocity=10.,ambient_density=1.,ambient_pressure=.01,actual_times=times)
                (root/'metadata.json').write_text(json.dumps(meta))
                for i in range(2):
                    for prefix in ['surfaces','histograms','slice']:
                        np.savez(root/'online'/f'{prefix}_{i:04d}.npz',time=times[i],count=i)
            result=compare_reference(*roots,prefix=True)
            self.assertIs(result['prefix_only'],True)
            self.assertEqual(result['events'],2)
            with self.assertRaises(AssertionError):compare_reference(*roots)

    def test_extension_preserves_times_bitwise_and_original_dt_cap(self):
        old=np.asarray(jnp.linspace(0,.16,17))
        meta=dict(status='complete',requested_times=old.tolist(),t_end=.16)
        times,dt=extension_schedule(meta,.20,21)
        np.testing.assert_array_equal(times[:17],old)
        self.assertEqual(dt,.16/17)
        np.testing.assert_allclose(times[17:],[.17,.18,.19,.20],rtol=0,atol=1e-7)
        meta['solver_dt_max']=.001
        self.assertEqual(extension_schedule(meta,.20,21)[1],.001)
        with self.assertRaises(ValueError):extension_schedule(meta,.20,20)
        with self.assertRaises(ValueError):extension_schedule(meta,.16,17)

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
