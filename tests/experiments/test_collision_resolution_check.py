import copy
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from experiments.wind_bubble.run_collision_resolution_check import resolution_plan,validate_saved_run,run,sha
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

    def test_saved_run_validation_checks_completion_states_and_core_source_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'audit').mkdir();(root/'online').mkdir()
            reference=three_source_reference()
            source=root/'detector.py';source.write_text('unchanged production source\n')
            actual={**reference,'num_cells':4,'backend':'gpu','requested_times':[0.],
                    'actual_times':[0.],'source_sha256':{str(source):sha(source)}}
            plan=dict(num_cells=4,requested_times=[0.])
            (root/'metadata.json').write_text(json.dumps(actual))
            for prefix in ['surfaces','histograms','slice']:
                np.savez(root/'online'/f'{prefix}_0000.npz',time=0.)
            np.save(root/'audit/state_0000.npy',np.ones((5,4,4,4),dtype=np.float32))
            hashes,changes=validate_saved_run(root,actual,reference,plan,[0])
            self.assertEqual(changes,{})
            self.assertIn(str(root/'audit/state_0000.npy'),hashes)
            wrapper=Path('experiments/wind_bubble/run_collision_resolution_check.py').resolve()
            actual['source_sha256'][str(wrapper)]='0'*64
            _,changes=validate_saved_run(root,actual,reference,plan,[0])
            self.assertEqual(len(changes),1)
            change=next(iter(changes.values()))
            self.assertEqual(change['integration_sha256'],'0'*64)
            self.assertEqual(change['postprocessing_sha256'],sha(wrapper))
            actual['status']='running'
            with self.assertRaisesRegex(ValueError,'completed simulation'):
                validate_saved_run(root,actual,reference,plan,[0])
            actual['status']='complete';source.write_text('changed production source\n')
            with self.assertRaisesRegex(ValueError,'Simulation source changed'):
                validate_saved_run(root,actual,reference,plan,[0])

    def test_audit_only_never_reintegrates_and_passes_a_path_to_the_auditor(self):
        """Regression for the costly post-integration Namespace-versus-Path error."""
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);reference_root=base/'reference';output=base/'saved'
            (reference_root/'audit').mkdir(parents=True);(output/'audit').mkdir(parents=True)
            budget_path=base/'budget.json';budget_path.write_text(json.dumps(reference_metadata()))
            reference=three_source_reference()
            reference['source_budget_reference']=dict(path=str(budget_path))
            (reference_root/'audit/verification.json').write_text(json.dumps(
                dict(passed=True,backend='gpu',state_sha256={},captured_events=[4])))
            actual={**reference,'num_cells':128,'backend':'gpu'}
            (output/'metadata.json').write_text(json.dumps(actual))
            state=output/'audit/state_0004.npy';state.write_bytes(b'boundary-dispatch fixture')
            args=argparse.Namespace(reference_run=reference_root,output_dir=output,require_gpu=False,audit_only=True)
            def audit(directory,events,meta):
                self.assertIsInstance(directory,Path)
                self.assertEqual(directory,output)
                self.assertEqual(events,[4])
                self.assertEqual(meta['num_cells'],128)
                return [],{str(state):sha(state)}
            module='experiments.wind_bubble.run_collision_resolution_check.'
            with patch(module+'check_reference',return_value=(reference,{})), \
                 patch(module+'validate_saved_run',return_value=({},{})), \
                 patch(module+'audit_saved_states',side_effect=audit) as auditor, \
                 patch(module+'assess_physics') as physical, \
                 patch(module+'run_simulation') as integrate, \
                 patch(module+'StateCapture') as capture, \
                 patch(module+'jax.default_backend',return_value='cpu'):
                result=run(args)
            integrate.assert_not_called();capture.assert_not_called()
            auditor.assert_called_once();physical.assert_called_once()
            self.assertTrue(result['resumed_from_saved_states'])
            self.assertTrue(result['passed'])


if __name__=='__main__':unittest.main()
