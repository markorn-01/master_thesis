"""One budget-matched 128-cubed follow-up to the verified collision-layer audit.

This is a resolution experiment, not same-state online/offline equivalence.
"""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

from experiments.wind_bubble.audit_collision_detection import check_reference, audit_saved_states
from experiments.wind_bubble.audit_detection_stages import StateCapture
from experiments.wind_bubble.assess_collision_physics import ROOT, run as assess_physics, sha
from experiments.wind_bubble.run_two_bubbles import run as run_simulation
from experiments.wind_bubble.three_bubble_setup import SOURCES,source_budget,weighted_budget


def resolution_plan(reference,budget_reference):
    """Match physical source budgets, not the grid-dependent injection coefficient."""
    for key,value in [('num_cells',64),('t_end',.16),('injection_radius',.0625),
                      ('ambient_density',1.),('ambient_pressure',.01),('gamma',5/3),
                      ('wind_terminal_velocity',10.)]:
        if reference.get(key)!=value:raise ValueError(f'Unsupported reference {key}.')
    if not np.array_equal(reference['source_positions'],SOURCES):
        raise ValueError('Source geometry differs.')
    if not np.array_equal(reference['source_strengths'],[1.,1.,1.]):
        raise ValueError('Require the original equal-strength three-source case.')
    times=np.asarray(reference['requested_times'],np.float32)
    if times.shape!=(17,) or not np.all(np.diff(times)>0) or times[0]!=0 or abs(times[-1]-.16)>1e-7:
        raise ValueError('Require the unchanged 17-event schedule.')
    budget=source_budget(128,budget_reference)
    for key in ('discrete_mass_rates','discrete_energy_rates'):
        np.testing.assert_allclose(budget[key],reference[key],rtol=1e-12,atol=1e-12)
    return dict(num_cells=128,t_end=reference['t_end'],num_analysis=len(times),
        requested_times=times.tolist(),solver_dt_max=reference['solver_dt_max'],budget=budget)


def run(args):
    reference,inputs=check_reference(args.reference_run)
    report_path=args.reference_run/'audit/verification.json'
    report=json.loads(report_path.read_text())
    if not report['passed'] or report.get('backend')!='gpu':
        raise ValueError('Require the verified matching GPU audit as reference.')
    inputs[str(report_path)]=sha(report_path)
    for name,expected in report['state_sha256'].items():
        path=args.reference_run/'audit'/Path(name).name
        if sha(path)!=expected:raise ValueError(f'Reference captured state changed: {path}')
        inputs[str(path)]=expected
    budget_path=Path(reference['source_budget_reference']['path'])
    plan=resolution_plan(reference,json.loads(budget_path.read_text()))
    if args.output_dir.exists():raise ValueError('Use a fresh output directory.')
    times=np.asarray(plan['requested_times'],np.float32)
    events=report['captured_events']
    capture=StateCapture(args.output_dir/'audit',times)
    simulation=argparse.Namespace(num_cells=128,num_analysis=17,t_end=.16,
        source_strengths=[1.,1.,1.],match_source_budget=budget_path,
        require_gpu=args.require_gpu,verify_offline=False,track_shocks=False,output_dir=args.output_dir)
    run_simulation(simulation,state_observer=capture.observer(events),analysis_times=times,
        solver_dt_max=plan['solver_dt_max'],source_positions=SOURCES,
        budget_builder=source_budget,strength_builder=weighted_budget,
        extra_source_files=[Path(__file__),Path('experiments/wind_bubble/three_bubble_setup.py'),
            Path('experiments/wind_bubble/audit_collision_detection.py'),
            Path('experiments/wind_bubble/assess_collision_physics.py')])
    if sorted(capture.saved)!=events:raise AssertionError('Missing captured event.')
    actual=json.loads((args.output_dir/'metadata.json').read_text())
    for key in ('source_positions','source_strengths','injection_radius','wind_terminal_velocity',
                'ambient_density','ambient_pressure','gamma','discrete_mass_rates','discrete_energy_rates'):
        np.testing.assert_allclose(actual[key],reference[key],rtol=1e-12,atol=1e-12,err_msg=key)
    np.testing.assert_array_equal(actual['requested_times'],times)
    np.testing.assert_allclose(actual['actual_times'],times,rtol=0,atol=1e-6)
    if actual['solver_dt_max']!=reference['solver_dt_max']:raise AssertionError('Solver timestep cap changed.')
    _,hashes=audit_saved_states(simulation,events,actual)
    for path,expected in inputs.items():
        if sha(path)!=expected:raise AssertionError(f'Reference input changed: {path}')
    source_files=[Path(__file__),Path('experiments/wind_bubble/assess_collision_physics.py'),
        Path('experiments/wind_bubble/audit_collision_detection.py'),
        Path('experiments/wind_bubble/audit_detection_stages.py')]
    source_files+=sorted(Path('astronomix/_physics_modules/_shock_finder').glob('*.py'))
    result=dict(passed=True,backend=jax.default_backend(),num_cells=128,reference_num_cells=64,
        captured_events=events,capture_times=times[events].tolist(),source_budgets_match=True,
        physical_parameters_match=True,schedule_changed=False,resolution_changed=True,
        physics_or_thresholds_changed=False,production_mask_decomposition_matches=True,
        exported_indices_and_positions_match=True,state_sha256=hashes,
        full_state_bytes=sum(Path(p).stat().st_size for p in hashes),reference_sha256=inputs,
        source_sha256={str(p):sha(p) for p in source_files},
        same_resolution_reference_replay_performed=False,
        scope='Matching configuration/source-budget and selected-state mask audit at finer resolution; '
              'not numerical convergence or online/offline equivalence.')
    (args.output_dir/'audit/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['passed','num_cells','captured_events','source_budgets_match',
          'physical_parameters_match','physics_or_thresholds_changed']},indent=2))
    assess_physics(argparse.Namespace(root=args.output_dir,output_dir=args.output_dir/'physics'))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--require-gpu',action='store_true')
    run(parser.parse_args())
