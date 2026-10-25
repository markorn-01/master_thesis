"""Compare scheduled solver-only, online, and stored-snapshot/offline workflows.

The controller uses only the standard library until data-product verification.
Each mode runs in its own child process. No solver or finder physics is changed.
See ONLINE_BENCHMARK.md for timing/memory definitions and limitations.
"""
from __future__ import annotations

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

MODES = ('solver_only', 'online', 'stored_offline')
ROOT = Path(__file__).resolve().parents[2]


def parse_gpu_rows(output, pid):
    """Sum this PID's allocations across GPU rows, excluding every other PID."""
    amounts = []
    for row in csv.reader(output.splitlines()):
        if len(row) < 2:
            continue
        try:
            if int(row[0].strip()) == pid:
                value = float(row[1].strip())
                if math.isfinite(value) and value >= 0:
                    amounts.append(value)
        except ValueError:
            continue
    return sum(amounts) if amounts else None


def planned_memory(executable):
    """Compiler-planned bytes; distinct from observed process GPU memory."""
    stats = executable.memory_analysis()
    if stats is None:
        return None
    fields = {name: int(getattr(stats, name)) for name in
              ('temp_size_in_bytes', 'argument_size_in_bytes', 'output_size_in_bytes', 'alias_size_in_bytes')}
    fields['planned_total_bytes'] = (fields['temp_size_in_bytes'] + fields['argument_size_in_bytes'] +
                                     fields['output_size_in_bytes'] - fields['alias_size_in_bytes'])
    return fields


def source_info():
    paths = ['experiments/wind_bubble/benchmark_online_analysis.py',
             'experiments/wind_bubble/run_online_bubble.py',
             'experiments/shock/analyze_state.py', 'astronomix/time_stepping/time_integration.py']
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(commit=commit, sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})


def worker(args):
    import gc
    import numpy as np
    import jax
    import jax.numpy as jnp
    from astronomix.data_classes.simulation_helper_data import get_helper_data, _helper_data_requirements
    from astronomix.time_stepping.time_integration import (
        _prepare_padded_state, _build_initial_loop_state, _integrate_core,
    )
    from astronomix.time_stepping._utils import _unpad
    from experiments.wind_bubble.run_single_bubble import build_problem
    from experiments.wind_bubble.run_online_bubble import DiagnosticWriter, diagnostic_payload, make_online_callback

    if args.require_gpu and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU required: JAX selected a different backend.')
    if jax.default_backend() == 'gpu' and len(jax.devices()) != 1:
        raise RuntimeError('Select one GPU with CUDA_VISIBLE_DEVICES before benchmarking.')
    setup = argparse.Namespace(num_cells=args.num_cells, num_snapshots=args.num_analysis,
                               t_end=args.t_end, num_injection_cells=args.num_cells//16, plasma_beta=None)
    state, base, params, registry, helper = build_problem(setup)
    base = base._replace(use_specific_snapshot_timepoints=True, print_elapsed_time=False,
                         progress_bar=False, donate_state=False)
    params = params._replace(snapshot_timepoints=jnp.linspace(0., args.t_end, args.num_analysis),
                             dt_max=args.t_end/args.num_analysis)
    config = base._replace(return_snapshots=args.mode == 'stored_offline',
                           activate_snapshot_callback=args.mode != 'stored_offline')
    analysis_config = base._replace(return_snapshots=False, activate_snapshot_callback=False)
    helper_pad = get_helper_data(config, padded=True, requirements=_helper_data_requirements(config))
    centers = np.asarray(helper.geometric_centers)
    dx = float(base.grid_spacing)

    # Stable host receiver permits the same compiled callback to write a new
    # trial directory each time; all callbacks finish before the sink changes.
    class Sink:
        writer = None
        def receive(self, *payload):
            if self.writer is None:
                raise RuntimeError('No active benchmark writer')
            self.writer.receive(*payload)
    sink = Sink()
    if args.mode == 'online':
        callback = make_online_callback(sink.receive, config, analysis_config, helper)
    elif args.mode == 'solver_only':
        def callback(time, padded, registry):
            pass  # Same scheduled timestep boundaries, no diagnostic computation.
    else:
        callback = None

    def integration(initial, dynamic_params, dynamic_helper):
        padded = _prepare_padded_state(initial, config, dynamic_params, registry)
        loop = _build_initial_loop_state(padded, config, dynamic_params)
        end, loop, store, count = _integrate_core(config, dynamic_params, registry,
                                                  dynamic_helper, callback, loop, initial.shape)
        return _unpad(loop.primitive_state, config), store, end, count

    # Parameters/helpers remain dynamic inputs, as in the public integrator.
    jax.block_until_ready((state, params, helper_pad, helper))
    start = time.perf_counter()
    compiled = jax.jit(integration).lower(state, params, helper_pad).compile()
    compile_integration = time.perf_counter()-start
    analysis_compiled = None
    compile_analysis = 0.
    if args.mode == 'stored_offline':
        def analysis(states, index, dynamic_helper):
            physical = jax.lax.dynamic_index_in_dim(states, index, axis=0, keepdims=False)
            return physical, diagnostic_payload(physical, analysis_config, registry, dynamic_helper)
        shapes = jax.ShapeDtypeStruct((args.num_analysis,)+state.shape, state.dtype)
        start = time.perf_counter()
        analysis_compiled = jax.jit(analysis).lower(shapes, np.int32(0), helper).compile()
        compile_analysis = time.perf_counter()-start

    def execute(directory):
        directory.mkdir(parents=True, exist_ok=False)
        writer = None
        if args.mode != 'solver_only':
            writer = DiagnosticWriter(directory/'compact', centers, dx, .0625)
        sink.writer = writer
        try:
            start = time.perf_counter()
            final, store, end, iterations = compiled(state, params, helper_pad)
            jax.block_until_ready((final, store, end, iterations))
            jax.effects_barrier()
            integration_seconds = time.perf_counter()-start
            analysis_seconds = 0.
            snapshot_seconds = 0.
            if args.mode == 'stored_offline':
                raw_dir = directory/'snapshots'
                raw_dir.mkdir()
                times = np.asarray(store.time_points)
                np.testing.assert_allclose(times, np.asarray(params.snapshot_timepoints), rtol=0, atol=1e-6)
                for index, at_time in enumerate(times):
                    start = time.perf_counter()
                    physical, payload = analysis_compiled(store.states, np.int32(index), helper)
                    jax.block_until_ready((physical, payload))
                    writer.receive(at_time, *[np.asarray(x) for x in payload])
                    analysis_seconds += time.perf_counter()-start
                    start = time.perf_counter()
                    # Deliberate storage baseline: raw uncompressed snapshots,
                    # with transfer/write charged separately from shock analysis.
                    np.save(raw_dir/f'snapshot_{index:04d}.npy', np.asarray(physical), allow_pickle=False)
                    snapshot_seconds += time.perf_counter()-start
            if writer is not None:
                writer.close()
                np.testing.assert_allclose([r['time'] for r in writer.rows],
                                           np.asarray(params.snapshot_timepoints), rtol=0, atol=1e-6)
            # Readback for validation is outside the timed workflow in all modes.
            final_np = np.asarray(final)
            if not np.isfinite(final_np).all():
                raise AssertionError('Non-finite final state')
            np.testing.assert_allclose(float(end), args.t_end, rtol=0, atol=1e-6)
            fingerprints = {str(i): dict(min=float(x.min()), max=float(x.max()),
                                          mean=float(x.mean(dtype=np.float64))) for i, x in enumerate(final_np)}
            def size(path):
                return sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.exists() else 0
            return dict(integration_seconds=integration_seconds, offline_analysis_seconds=analysis_seconds,
                        snapshot_transfer_write_seconds=snapshot_seconds,
                        total_workflow_seconds=integration_seconds+analysis_seconds+snapshot_seconds,
                        compact_bytes=size(directory/'compact'), snapshot_file_bytes=size(directory/'snapshots'),
                        output_payload_bytes=size(directory/'compact')+size(directory/'snapshots'),
                        snapshot_array_payload_bytes=(int(store.states.size*store.states.dtype.itemsize)
                                                      if store is not None and store.states is not None else 0),
                        final_time=float(end), iterations=int(iterations), final_fingerprint=fingerprints,
                        final_state_sha256=hashlib.sha256(final_np.tobytes()).hexdigest())
        finally:
            jax.effects_barrier()
            if writer is not None:
                writer.close()
            sink.writer = None

    # The warm-up uses the full workflow, including file writes. Only its own
    # temporary directory is removed; measured trials and user outputs persist.
    with tempfile.TemporaryDirectory(prefix='warmup-', dir=args.output_dir) as tmp:
        warmup = execute(Path(tmp)/'trial')
    trials = []
    for index in range(args.repeats):
        gc.collect()
        record = execute(args.output_dir/f'trial_{index:02d}')
        trials.append(record)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = dict(status='complete', mode=args.mode, source=source_info(),
                  python=platform.python_version(), jax=jax.__version__, numpy=np.__version__,
                  backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
                  device_kinds=[d.device_kind for d in jax.devices()], hostname=platform.node(),
                  platform=platform.platform(), num_cells=args.num_cells, num_analysis=args.num_analysis,
                  t_end=args.t_end, injection_radius=.0625,
                  requested_times=np.asarray(params.snapshot_timepoints).tolist(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  preallocate=os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),
                  compile_integration_seconds=compile_integration, compile_analysis_seconds=compile_analysis,
                  compiled_integration_memory=planned_memory(compiled),
                  compiled_offline_analysis_memory=planned_memory(analysis_compiled) if analysis_compiled else None,
                  worker_peak_host_rss_bytes=int(rss if sys.platform == 'darwin' else rss*1024),
                  warmup=warmup, trials=trials)
    (args.output_dir/'result.json').write_text(json.dumps(result, indent=2)+'\n')


def launch_worker(args, mode):
    directory = args.output_dir/mode
    directory.mkdir()
    command = [sys.executable, '-m', 'experiments.wind_bubble.benchmark_online_analysis',
               '--worker', '--mode', mode, '--num-cells', str(args.num_cells),
               '--num-analysis', str(args.num_analysis), '--t-end', str(args.t_end),
               '--repeats', str(args.repeats), '--output-dir', str(directory.resolve())]
    if args.require_gpu:
        command.append('--require-gpu')
    env = os.environ.copy()
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    smi = shutil.which('nvidia-smi')
    samples = []
    query_errors = 0
    start = time.perf_counter()
    with (directory/'worker.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                if smi:
                    try:
                        output = subprocess.check_output([smi, '--query-compute-apps=pid,used_gpu_memory',
                                                          '--format=csv,noheader,nounits'], text=True, timeout=5)
                        memory = parse_gpu_rows(output, child.pid)
                        if memory is not None:
                            samples.append((time.perf_counter()-start, memory))
                    except (OSError, subprocess.SubprocessError):
                        query_errors += 1
                try:
                    child.wait(timeout=args.sample_interval)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            # Stop only the worker created by this controller, never another
            # user's GPU process, if the benchmark is interrupted.
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
        if child.returncode:
            raise RuntimeError(f'{mode} failed; see {directory / "worker.log"}')
    result = json.loads((directory/'result.json').read_text())
    result['process_wall_seconds'] = time.perf_counter()-start
    result['sampled_peak_gpu_process_mib'] = max((x[1] for x in samples), default=None)
    result['gpu_memory_samples'] = len(samples)
    result['gpu_memory_query_errors'] = query_errors
    result['gpu_sample_interval_requested_seconds'] = args.sample_interval
    with (directory/'gpu_memory_samples.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['elapsed_seconds', 'this_pid_memory_mib'])
        writer.writerows(samples)
    (directory/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def compare_outputs(directory, repeats):
    """Data-product check after all child processes exit; no parent JAX context."""
    import numpy as np
    for index in range(repeats):
        a = directory/'online'/f'trial_{index:02d}'/'compact'
        b = directory/'stored_offline'/f'trial_{index:02d}'/'compact'
        with (a/'history.csv').open() as x, (b/'history.csv').open() as y:
            online, offline = list(csv.DictReader(x)), list(csv.DictReader(y))
        if len(online) != len(offline):
            raise AssertionError('Analysis event counts differ')
        for x, y in zip(online, offline):
            if x.keys() != y.keys():
                raise AssertionError('Diagnostic schemas differ')
            for key in x:
                np.testing.assert_allclose(float(x[key]), float(y[key]), rtol=1e-6, atol=1e-8,
                                           equal_nan=True, err_msg=key)
        if len(list(a.glob('histograms_*.npz'))) != len(online):
            raise AssertionError('Missing online histograms')
        if len(list(b.glob('histograms_*.npz'))) != len(offline):
            raise AssertionError('Missing offline histograms')
        for event in range(len(online)):
            with np.load(a/f'histograms_{event:04d}.npz') as x, np.load(b/f'histograms_{event:04d}.npz') as y:
                for key in ('mach_counts', 'density_jump_counts', 'mach_edges', 'density_jump_edges'):
                    np.testing.assert_array_equal(x[key], y[key])
    return dict(passed=True, compared_trials=repeats, histogram_counts_exact=True,
                scalar_rtol=1e-6, scalar_atol=1e-8)


def summarize(results):
    baseline = statistics.median(t['total_workflow_seconds'] for t in results['solver_only']['trials'])
    rows = []
    for mode, result in results.items():
        trials = result['trials']
        total = statistics.median(t['total_workflow_seconds'] for t in trials)
        rows.append(dict(mode=mode, repeats=len(trials), median_workflow_seconds=total,
                         min_workflow_seconds=min(t['total_workflow_seconds'] for t in trials),
                         max_workflow_seconds=max(t['total_workflow_seconds'] for t in trials),
                         overhead_vs_scheduled_solver_percent=100*(total/baseline-1),
                         median_integration_seconds=statistics.median(t['integration_seconds'] for t in trials),
                         median_offline_analysis_seconds=statistics.median(t['offline_analysis_seconds'] for t in trials),
                         median_snapshot_transfer_write_seconds=statistics.median(t['snapshot_transfer_write_seconds'] for t in trials),
                         compile_seconds=result['compile_integration_seconds']+result['compile_analysis_seconds'],
                         sampled_peak_gpu_process_mib=result['sampled_peak_gpu_process_mib'],
                         worker_peak_host_rss_bytes=result['worker_peak_host_rss_bytes'],
                         median_output_payload_bytes=statistics.median(t['output_payload_bytes'] for t in trials),
                         median_compact_bytes=statistics.median(t['compact_bytes'] for t in trials),
                         median_snapshot_file_bytes=statistics.median(t['snapshot_file_bytes'] for t in trials)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-cells', type=int, default=64)
    parser.add_argument('--num-analysis', type=int, default=20)
    parser.add_argument('--t-end', type=float, default=.2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--sample-interval', type=float, default=.2)
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/online_benchmark_n064'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--mode', choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.num_cells < 16 or args.num_cells % 16 or args.num_cells > 64 or args.num_analysis < 3
            or args.repeats < 1 or not math.isfinite(args.t_end) or args.t_end <= 0
            or not math.isfinite(args.sample_interval) or args.sample_interval <= 0):
        parser.error('Require grid multiple of 16 in [16,64], >=3 analysis times, >=1 repeat and finite positive times/interval.')
    if args.worker:
        if args.mode is None:
            parser.error('--worker requires --mode')
        worker(args)
        return
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    for mode in MODES:
        print(f'Benchmarking {mode} in a fresh process...', flush=True)
        results[mode] = launch_worker(args, mode)
    # A performance comparison is only meaningful when integration schedules
    # and final-state summaries remain consistent across modes and repetitions.
    reference = results['solver_only']['trials'][0]
    for result in results.values():
        for trial in result['trials']:
            if trial['iterations'] != reference['iterations']:
                raise AssertionError('Solver iteration counts differ across benchmark modes')
            for i, values in trial['final_fingerprint'].items():
                for key, value in values.items():
                    if not math.isclose(value, reference['final_fingerprint'][i][key], rel_tol=1e-6, abs_tol=1e-8):
                        raise AssertionError('Final-state summaries differ across benchmark modes')
    verification = compare_outputs(args.output_dir, args.repeats)
    verification['all_final_state_hashes_exact'] = all(
        trial['final_state_sha256'] == reference['final_state_sha256']
        for result in results.values() for trial in result['trials'])
    rows = summarize(results)
    with (args.output_dir/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = dict(status='complete', source=source_info(), modes=results, verification=verification,
                   memory_note='GPU metric is a sampled whole-worker PID allocation peak, including compilation, warmup and allocator caches; it is a lower bound, not a live-buffer measurement.',
                   baseline='Scheduled solver-only; same event boundaries, no diagnostics or output data.',
                   offline_storage='Uncompressed .npy gas snapshots plus matching primary compact diagnostics.',
                   timing='Compilation and warmup excluded from trial timings; execution synchronized including callbacks; plotting and metadata/final validation readback excluded.')
    (args.output_dir/'benchmark.json').write_text(json.dumps(payload, indent=2)+'\n')
    for row in rows:
        print(f"{row['mode']}: median={row['median_workflow_seconds']:.3f}s; "
              f"payload={row['median_output_payload_bytes']} bytes; "
              f"sampled GPU peak={row['sampled_peak_gpu_process_mib']}", flush=True)
    print(f'Results: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
