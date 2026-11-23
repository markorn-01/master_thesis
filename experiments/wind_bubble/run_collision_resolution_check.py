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


def validate_saved_run(root,actual,reference,plan,events):
    """Require a completed, compatible integration before analyzing its states."""
    if actual.get('status')!='complete' or actual.get('num_cells')!=plan['num_cells']:
        raise ValueError('Require a completed simulation at the planned resolution.')
    for key in ('source_positions','source_strengths','injection_radius','wind_terminal_velocity',
                'ambient_density','ambient_pressure','gamma','discrete_mass_rates','discrete_energy_rates'):
        np.testing.assert_allclose(actual[key],reference[key],rtol=1e-12,atol=1e-12,err_msg=key)
    times=np.asarray(plan['requested_times'],np.float32)
    np.testing.assert_array_equal(actual['requested_times'],times)
    np.testing.assert_allclose(actual['actual_times'],times,rtol=0,atol=1e-6)
    if actual['solver_dt_max']!=reference['solver_dt_max']:raise AssertionError('Solver timestep cap changed.')
    if actual.get('backend')!='gpu':raise ValueError('The saved resolution experiment must have run on GPU.')
    # Only this orchestration script may differ to repair postprocessing. The
    # integrator, detector, source model and analyzer must retain their hashes.
    orchestration_changes={}
    for name,expected in actual['source_sha256'].items():
        path=Path(name.split('master_thesis/')[-1])
        current=sha(path)
        if current!=expected:
            if path.resolve()!=Path(__file__).resolve():
                raise ValueError(f'Simulation source changed; refusing saved-state audit: {path}')
            orchestration_changes[str(path)]=dict(integration_sha256=expected,postprocessing_sha256=current)
    paths=[root/'metadata.json']
    for index in range(len(times)):
        paths.extend(root/'online'/f'{prefix}_{index:04d}.npz' for prefix in ('surfaces','histograms','slice'))
    for index in events:
        path=root/'audit'/f'state_{index:04d}.npy'
        state=np.load(path,mmap_mode='r',allow_pickle=False)
        if state.shape!=(5,plan['num_cells'],plan['num_cells'],plan['num_cells']) or not np.isfinite(state).all():
            raise ValueError(f'Invalid saved physical state: {path}')
        paths.append(path)
    return {str(path):sha(path) for path in paths},orchestration_changes


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
    audit_only=bool(getattr(args,'audit_only',False))
    if args.require_gpu and jax.default_backend()!='gpu':raise RuntimeError('GPU requested but unavailable.')
    if not audit_only and args.output_dir.exists():raise ValueError('Use a fresh output directory.')
    if audit_only and not (args.output_dir/'metadata.json').is_file():
        raise ValueError('--audit-only requires the existing completed simulation directory.')
    times=np.asarray(plan['requested_times'],np.float32)
    events=report['captured_events']
    if not audit_only:
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
    saved_hashes,orchestration_changes=validate_saved_run(args.output_dir,actual,reference,plan,events)
    _,hashes=audit_saved_states(args.output_dir,events,actual)
    for path,expected in {**inputs,**saved_hashes}.items():
        if sha(path)!=expected:raise AssertionError(f'Reference input changed: {path}')
    source_files=[Path(__file__),Path('experiments/wind_bubble/assess_collision_physics.py'),
        Path('experiments/wind_bubble/audit_collision_detection.py'),
        Path('experiments/wind_bubble/audit_detection_stages.py')]
    source_files+=sorted(Path('astronomix/_physics_modules/_shock_finder').glob('*.py'))
    result=dict(passed=True,backend=actual['backend'],postprocessing_backend=jax.default_backend(),
        resumed_from_saved_states=audit_only,orchestration_source_changes=orchestration_changes,
        num_cells=128,reference_num_cells=64,
        captured_events=events,capture_times=times[events].tolist(),source_budgets_match=True,
        physical_parameters_match=True,schedule_changed=False,resolution_changed=True,
        physics_or_thresholds_changed=False,production_mask_decomposition_matches=True,
        exported_indices_and_positions_match=True,state_sha256=hashes,
        full_state_bytes=sum(Path(p).stat().st_size for p in hashes),reference_sha256=inputs,
        saved_simulation_sha256=saved_hashes,
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
    parser.add_argument('--audit-only',action='store_true',
                        help='Finish audits from an existing completed run; do not integrate again.')
    run(parser.parse_args())
