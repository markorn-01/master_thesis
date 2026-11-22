"""Bounded complete-workflow benchmark: identical diagnostics, tracking off/on.

Two sequential fresh workers use the established three-source case. Timed
trials include writer construction, synchronized integration/callbacks and
terminal close. Compilation, warmup, plotting and validation are separate.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

from experiments.wind_bubble.benchmark_online_analysis import parse_gpu_rows, planned_memory

ROOT = Path(__file__).resolve().parents[2]
MODES = ('diagnostics', 'tracked')
REFERENCE = Path('outputs/two_bubbles_iwr_compgpu9_n064/metadata.json')


def source_info():
    from experiments.wind_bubble.streaming_products import tracking_source_files
    paths = tracking_source_files()+[Path(__file__),
        Path('experiments/wind_bubble/benchmark_online_analysis.py'),
        Path('experiments/wind_bubble/run_two_bubbles.py'),
        Path('experiments/wind_bubble/run_three_bubbles.py'),
        Path('experiments/wind_bubble/run_single_bubble.py'),
        Path('experiments/wind_bubble/run_online_bubble.py'),
        Path('experiments/wind_bubble/sparse_surfaces.py'),
        Path('experiments/wind_bubble/three_bubble_setup.py'),
        Path('experiments/wind_bubble/two_bubble_source_budget.py'),
        Path('experiments/shock/analyze_state.py'),
        Path('astronomix/time_stepping/time_integration.py'),
        Path('astronomix/time_stepping/_time_loop.py'),
        Path('astronomix/option_classes/simulation_config.py'),
        Path('astronomix/_modules/_stellar_wind/stellar_wind.py')]
    paths += list(Path('astronomix/_physics_modules/_shock_finder').glob('*.py'))
    relative = sorted({str(p.resolve().relative_to(ROOT)) for p in paths})
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    return dict(commit=commit, sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in relative})


def worker(args):
    import gc
    import jax
    import jax.numpy as jnp
    import numpy as np
    from astronomix.data_classes.simulation_helper_data import get_helper_data, _helper_data_requirements
    from astronomix.option_classes.simulation_config import finalize_config
    from astronomix.time_stepping.time_integration import _prepare_padded_state, _build_initial_loop_state, _integrate_core
    from astronomix.time_stepping._utils import _unpad
    from experiments.wind_bubble.run_single_bubble import build_problem
    from experiments.wind_bubble.run_online_bubble import diagnostic_payload
    from experiments.wind_bubble.run_two_bubbles import CollisionWriter
    from experiments.wind_bubble.three_bubble_setup import SOURCES, source_budget, weighted_budget, injection_counts

    # Import the tracking dependencies in both modes, outside measured execution.
    sources = source_info()
    if args.require_gpu and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU required but unavailable.')
    if jax.default_backend() == 'gpu' and len(jax.devices()) != 1:
        raise RuntimeError('Select exactly one GPU using CUDA_VISIBLE_DEVICES.')
    setup = argparse.Namespace(num_cells=args.num_cells, num_snapshots=args.num_analysis,
        t_end=args.t_end, num_injection_cells=args.num_cells//16, plasma_beta=None)
    state, base, params, registry, helper = build_problem(setup)
    base = base._replace(wind_config=base.wind_config._replace(source_positions=SOURCES,
        source_strengths=(1.,1.,1.)), use_specific_snapshot_timepoints=True,
        print_elapsed_time=False, progress_bar=False, donate_state=False)
    base = finalize_config(base, state.shape)
    times = jnp.linspace(0., args.t_end, args.num_analysis)
    params = params._replace(snapshot_timepoints=times, dt_max=args.t_end/args.num_analysis)
    centers, dx = np.asarray(helper.geometric_centers), float(base.grid_spacing)
    counts = [int((np.linalg.norm(centers-np.asarray(s),axis=-1) <= .0625-dx/2).sum()) for s in SOURCES]
    assert counts == injection_counts(args.num_cells)
    reference = json.loads(REFERENCE.read_text())
    budget = source_budget(args.num_cells, reference, float(params.wind_params.wind_mass_loss_rate),
                           float(params.wind_params.wind_final_velocity), actual_counts=counts)
    params = params._replace(wind_params=params.wind_params._replace(wind_mass_loss_rate=budget['wind_mass_loss_rate_per_source']))
    weighted = weighted_budget(budget, [1.,1.,1.])
    config = base._replace(return_snapshots=False, activate_snapshot_callback=True)
    analysis = base._replace(return_snapshots=False, activate_snapshot_callback=False)
    helper_pad = get_helper_data(config, padded=True, requirements=_helper_data_requirements(config))

    class Sink:
        writer = None
        def receive(self, *payload):
            if self.writer is None:
                raise RuntimeError('No active benchmark writer.')
            self.writer.receive(*payload)
    sink = Sink()

    def callback(at_time, padded, variables):
        physical = _unpad(padded, config)
        payload = diagnostic_payload(physical, analysis, variables, helper, sampling_quality=True)
        jax.debug.callback(sink.receive, at_time, *payload,
            physical[variables.density_index, :, :, args.num_cells//2],
            physical[variables.pressure_index, :, :, args.num_cells//2], ordered=True)

    def integrate(initial, dynamic_params, dynamic_helper):
        padded = _prepare_padded_state(initial, config, dynamic_params, registry)
        loop = _build_initial_loop_state(padded, config, dynamic_params)
        end, loop, store, count = _integrate_core(config, dynamic_params, registry,
            dynamic_helper, callback, loop, initial.shape)
        return _unpad(loop.primitive_state, config), end, count

    jax.block_until_ready((state, params, helper_pad, helper))
    before = time.perf_counter()
    compiled = jax.jit(integrate).lower(state, params, helper_pad).compile()
    compile_seconds = time.perf_counter()-before

    def execute(directory):
        writer = None
        start = time.perf_counter()
        try:
            directory.mkdir(parents=True, exist_ok=False)
            writer = CollisionWriter(directory/'products', centers, dx, sources=SOURCES,
                                     track_shocks=args.mode == 'tracked')
            sink.writer = writer
            constructed = time.perf_counter()
            final, end, iterations = compiled(state, params, helper_pad)
            jax.block_until_ready((final, end, iterations))
            jax.effects_barrier()
            integrated = time.perf_counter()
            writer.close()  # Includes terminal delayed-support records and flush/close.
            closed = time.perf_counter()
            # Validation, output-size scans and final-state host transfer/hash are untimed.
            np.testing.assert_allclose([r['time'] for r in writer.rows], np.asarray(times), rtol=0, atol=1e-6)
            final_np = np.asarray(final)
            if not np.isfinite(final_np).all():
                raise AssertionError('Non-finite final gas state.')
            np.testing.assert_allclose(float(end), args.t_end, rtol=0, atol=1e-6)
            products = directory/'products'
            primary_bytes = sum(p.stat().st_size for p in products.iterdir() if p.is_file())
            tracking_bytes = sum(p.stat().st_size for p in (products/'tracking').rglob('*') if p.is_file())
            return dict(construction_seconds=constructed-start, integration_seconds=integrated-constructed,
                finalization_seconds=closed-integrated, total_workflow_seconds=closed-start,
                primary_output_bytes=primary_bytes, tracking_output_bytes=tracking_bytes,
                output_payload_bytes=primary_bytes+tracking_bytes, events=len(writer.rows),
                group_observations=writer.tracking.groups if writer.tracking is not None else None,
                iterations=int(iterations), final_time=float(end),
                final_state_sha256=hashlib.sha256(final_np.tobytes()).hexdigest())
        finally:
            if writer is not None:
                writer.close(success=False)
            sink.writer = None

    with tempfile.TemporaryDirectory(prefix='warmup-', dir=args.output_dir) as tmp:
        warmup = execute(Path(tmp)/'trial')
    trials = []
    for index in range(args.repeats):
        gc.collect()
        trials.append(execute(args.output_dir/f'trial_{index:02d}'))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = dict(status='complete', mode=args.mode, source=sources, hostname=platform.node(),
        python=platform.python_version(), jax=jax.__version__, numpy=np.__version__,
        backend=jax.default_backend(), device_kinds=[d.device_kind for d in jax.devices()],
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        preallocate=os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),
        num_cells=args.num_cells, num_analysis=args.num_analysis, t_end=args.t_end,
        requested_times=np.asarray(times).tolist(), solver_dt_max=float(params.dt_max),
        source_positions=SOURCES, source_strengths=[1.,1.,1.], source_budget=budget,
        discrete_energy_rates=weighted['discrete_energy_rates'],
        source_budget_reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        compile_seconds=compile_seconds, compiled_memory=planned_memory(compiled),
        worker_peak_host_rss_bytes=int(rss if sys.platform == 'darwin' else rss*1024),
        warmup=warmup, trials=trials)
    (args.output_dir/'result.json').write_text(json.dumps(result, indent=2)+'\n')


def launch(args, mode):
    directory = args.output_dir/mode
    directory.mkdir()
    command = [sys.executable, '-m', 'experiments.wind_bubble.benchmark_tracking_workflow',
        '--worker', '--mode', mode, '--num-cells', str(args.num_cells),
        '--num-analysis', str(args.num_analysis), '--t-end', str(args.t_end),
        '--repeats', str(args.repeats), '--output-dir', str(directory)]
    if args.require_gpu:
        command.append('--require-gpu')
    env = os.environ.copy(); env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    smi = shutil.which('nvidia-smi'); samples = []; errors = 0
    started = time.perf_counter()
    with (directory/'worker.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                if smi:
                    try:
                        query = subprocess.check_output([smi, '--query-compute-apps=pid,used_gpu_memory',
                            '--format=csv,noheader,nounits'], text=True, timeout=5)
                        amount = parse_gpu_rows(query, child.pid)
                        if amount is not None:
                            samples.append((time.perf_counter()-started, amount))
                    except (OSError, subprocess.SubprocessError):
                        errors += 1
                try:
                    child.wait(timeout=args.sample_interval)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
            raise
        if child.returncode:
            raise RuntimeError(f'{mode} worker failed; inspect {directory / "worker.log"}')
    result = json.loads((directory/'result.json').read_text())
    result.update(process_wall_seconds=time.perf_counter()-started,
        sampled_peak_gpu_process_mib=max((r[1] for r in samples), default=None),
        gpu_memory_samples=len(samples), gpu_query_errors=errors,
        sample_interval_requested_seconds=args.sample_interval)
    with (directory/'gpu_memory_samples.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['elapsed_seconds','this_pid_memory_mib']); writer.writerows(samples)
    (directory/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def check_run_consistency(results):
    """Require comparable experiments before computing a performance ratio."""
    reference = results['diagnostics']
    settings = ('source', 'hostname', 'backend', 'device_kinds', 'cuda_visible_devices', 'preallocate',
                'python', 'jax', 'numpy', 'num_cells', 'num_analysis', 't_end', 'requested_times',
                'solver_dt_max', 'source_positions', 'source_strengths', 'discrete_energy_rates',
                'source_budget_reference_sha256')
    baseline = results['diagnostics']['trials'][0]
    for result in results.values():
        if any(result[key] != reference[key] for key in settings):
            raise AssertionError('Benchmark settings, hardware, environment or source versions differ.')
        if len(result['trials']) != len(reference['trials']):
            raise AssertionError('Require the same number of measured repetitions in both modes.')
        for trial in [result['warmup']]+result['trials']:
            if (trial['iterations'] != baseline['iterations'] or trial['final_state_sha256'] != baseline['final_state_sha256']):
                raise AssertionError('Warmup/trial final states or iteration counts differ between modes.')
            if trial['events'] != reference['num_analysis'] or not math.isclose(trial['final_time'], reference['t_end'], rel_tol=0., abs_tol=1e-6):
                raise AssertionError('Trial analysis-event count or final time differs from the declared setup.')


def verify(directory, results):
    from types import SimpleNamespace
    from experiments.wind_bubble.run_two_bubbles import compare_outputs
    from experiments.wind_bubble.streaming_products import compare_products, check_batch_reference
    check_run_consistency(results)
    def receiver(path):
        with (path/'history.csv').open() as stream:
            rows = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
        # Compare the primary products; tracking is intentionally absent in one mode.
        return SimpleNamespace(directory=path, rows=rows, tracking=None)
    repeats = len(results['tracked']['trials'])
    for index in range(repeats):
        a = directory/'diagnostics'/f'trial_{index:02d}'/'products'
        b = directory/'tracked'/f'trial_{index:02d}'/'products'
        assert compare_outputs(receiver(a), receiver(b))['passed']
        if index:
            assert compare_products(directory/'tracked/trial_00/products/tracking', b/'tracking')['streaming_tracking_match']
    first = directory/'tracked/trial_00/products'
    batch = check_batch_reference(first, first/'tracking')
    return dict(passed=True, compared_trials=repeats, primary_products_match=True,
        all_final_state_hashes_exact=True, warmup_final_states_also_checked=True,
        solver_iterations_exact=True, repeated_tracking_products_match=True,
        scalar_rtol=1e-6, scalar_atol=1e-8, **batch)


def summarize(results):
    baseline = statistics.median(t['total_workflow_seconds'] for t in results['diagnostics']['trials'])
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError('Require positive finite measured baseline times.')
    rows = []
    for mode in MODES:
        result = results[mode]; trials = result['trials']
        median = statistics.median(t['total_workflow_seconds'] for t in trials)
        rows.append(dict(mode=mode, repeats=len(trials), median_workflow_seconds=median,
            min_workflow_seconds=min(t['total_workflow_seconds'] for t in trials),
            max_workflow_seconds=max(t['total_workflow_seconds'] for t in trials),
            added_seconds_vs_diagnostics=median-baseline, overhead_vs_diagnostics_percent=100*(median/baseline-1),
            median_construction_seconds=statistics.median(t['construction_seconds'] for t in trials),
            median_integration_seconds=statistics.median(t['integration_seconds'] for t in trials),
            median_finalization_seconds=statistics.median(t['finalization_seconds'] for t in trials),
            compile_seconds=result['compile_seconds'], process_wall_seconds=result['process_wall_seconds'],
            sampled_peak_gpu_process_mib=result['sampled_peak_gpu_process_mib'],
            worker_peak_host_rss_bytes=result['worker_peak_host_rss_bytes'],
            median_primary_output_bytes=statistics.median(t['primary_output_bytes'] for t in trials),
            median_tracking_output_bytes=statistics.median(t['tracking_output_bytes'] for t in trials),
            median_output_payload_bytes=statistics.median(t['output_payload_bytes'] for t in trials)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-cells', type=int, choices=(32,64), default=64)
    parser.add_argument('--num-analysis', type=int, default=17)
    parser.add_argument('--t-end', type=float, default=.16)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--sample-interval', type=float, default=.2)
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--mode', choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.num_analysis < 3 or args.repeats < 1 or not math.isfinite(args.t_end) or args.t_end <= 0
            or not math.isfinite(args.sample_interval) or args.sample_interval <= 0):
        parser.error('Require >=3 events, >=1 repeat, positive finite end time and sample interval.')
    args.output_dir = args.output_dir.resolve()
    if args.worker:
        if args.mode is None:
            parser.error('--worker requires --mode')
        worker(args); return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    payload = dict(status='running', controller_pid=os.getpid(), source=source_info(), mode_order=list(MODES))
    path = args.output_dir/'benchmark.json'
    path.write_text(json.dumps(payload, indent=2)+'\n')
    try:
        results = {}
        for mode in MODES:
            payload['current_mode'] = mode; path.write_text(json.dumps(payload, indent=2)+'\n')
            print(f'Benchmarking {mode}: fresh process, compile, one warmup, {args.repeats} measured runs...', flush=True)
            results[mode] = launch(args, mode)
        if any(result['source'] != payload['source'] for result in results.values()) or source_info() != payload['source']:
            raise AssertionError('Benchmark sources changed during execution.')
        print('Verifying primary outputs, repeated tracking and batch reference (outside timings)...', flush=True)
        verification = verify(args.output_dir, results)
        rows = summarize(results)
        with (args.output_dir/'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        payload.update(status='complete', modes=results, verification=verification, summary=rows,
            timing_scope='Writer construction + synchronized integration/callback work + terminal close. Compilation, warmup, plots, size scans and verification excluded.',
            memory_scope='Whole-worker host RSS and sampled exact-PID GPU allocation peak, including compilation/warmup and caches. Sampled GPU peak is a lower bound.',
            output_scope='Primary CSV/catalog/histogram/slice files plus tracking files and its interpretation metadata. Worker logs and benchmark reports excluded.',
            limitations='Fixed mode order on a shared machine; cached filesystem, no fsync; one case/resolution/cadence, no general scaling claim.')
        path.write_text(json.dumps(payload, indent=2)+'\n')
        for row in rows:
            print(f"{row['mode']}: median={row['median_workflow_seconds']:.3f}s; output={row['median_output_payload_bytes']} bytes; sampled GPU peak={row['sampled_peak_gpu_process_mib']}", flush=True)
        print(f'Results: {args.output_dir}', flush=True)
    except BaseException as exc:
        payload.update(status='failed', error=str(exc)); path.write_text(json.dumps(payload, indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
