"""Profile saved-catalog tracking and compare one optimization with a Git baseline.

No hydrodynamics, JAX, source changes, or detector changes. Timing includes
tracking-product writing on the local filesystem, but excludes input loading,
correctness comparisons, profiling, and plot rendering. The historical matcher
is loaded from the explicitly selected revision; all other code is shared.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import cProfile
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import pstats
import subprocess
import sys
import tempfile
import time
import types
from unittest.mock import patch

import numpy as np
import scipy

import experiments.astrolink.streaming_history as history
from experiments.astrolink.expanded_event_tracking import expanded_tracking
from experiments.wind_bubble.streaming_products import (
    StreamingProductWriter, compare_tree, tracking_source_files,
)


CATALOG_KEYS = ('positions', 'normals', 'time', 'grid_spacing', 'mach', 'mach_valid',
                'source_positions', 'upstream_inside_any_source', 'downstream_inside_any_source')
FLOAT_CSV_FIELDS = {'time', 'mach_p16', 'median_mach', 'mach_p84'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def product_payload(directory):
    """Compare decoded products, excluding ZIP/GZIP container timestamps."""
    result = {}
    for path in sorted(Path(directory).iterdir()):
        if path.suffix == '.npz':
            with np.load(path) as data:
                result[path.name] = {
                    key: dict(dtype=str(data[key].dtype), shape=list(data[key].shape),
                              values=data[key].tolist()) for key in data.files}
        elif path.name.endswith('.json.gz'):
            with gzip.open(path, 'rt') as stream:
                result[path.name] = json.load(stream)
        elif path.suffix == '.json':
            result[path.name] = json.loads(path.read_text())
        elif path.suffix == '.csv':
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                for key in FLOAT_CSV_FIELDS:
                    row[key] = float(row[key]) if row[key] else None
            result[path.name] = rows
        else:
            raise ValueError(f'Unexpected tracking product: {path.name}')
    return result


def payload_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def compare_payloads(actual, expected, *, exact=True):
    if exact:
        if actual != expected:
            raise AssertionError('Tracking records, partitions or statistics changed.')
    else:
        compare_tree(actual, expected)


def load_reference(revision, destination):
    """Freeze only the old matchers; keep the replay/writer/statistics shared."""
    resolved = subprocess.check_output(['git', 'rev-parse', '--verify', revision+'^{commit}'], text=True).strip()
    destination.mkdir()
    modules, fingerprints = {}, {}
    for name in ('event_tracking', 'expanded_event_tracking'):
        source = subprocess.check_output(['git', 'show', f'{resolved}:experiments/astrolink/{name}.py'])
        path = destination/(name+'.py')
        path.write_bytes(source)
        fingerprints[str(path)] = sha256(path)
        module = types.ModuleType(f'_tracking_reference_{name}')
        module.__file__ = str(path)
        if name == 'expanded_event_tracking':
            with patch.dict(sys.modules, {'experiments.astrolink.event_tracking': modules['event_tracking']}):
                exec(compile(source, str(path), 'exec'), module.__dict__)
        else:
            exec(compile(source, str(path), 'exec'), module.__dict__)
        modules[name] = module
    return modules['expanded_event_tracking'].expanded_tracking, resolved, fingerprints


def load_inputs(root):
    path = root/'metadata.json'
    metadata = json.loads(path.read_text())
    if metadata['status'] != 'complete':
        raise ValueError('Require a completed input run.')
    hashes = {str(path): sha256(path)}
    catalogs = []
    for index, actual in enumerate(metadata['actual_times']):
        path = root/'online'/f'surfaces_{index:04d}.npz'
        hashes[str(path)] = sha256(path)
        with np.load(path) as data:
            catalog = {key: data[key].copy() for key in CATALOG_KEYS}
        if abs(float(catalog['time'])-actual) > 1e-7:
            raise ValueError('Catalog time differs from recorded actual time.')
        # Timed replays cannot mutate input geometry or properties silently.
        for value in catalog.values():
            value.flags.writeable = False
        catalogs.append(catalog)
    if len(catalogs) < 3:
        raise ValueError('Require at least three saved analysis events.')
    return catalogs, hashes


@contextmanager
def phases():
    """Nonoverlapping top-level tracker phases; writing is the receive remainder."""
    totals, calls = defaultdict(float), defaultdict(int)
    original = {}
    names = dict(owned_catalog='catalog_copy_validation', connectivity_partition='spatial_grouping',
                 expanded_tracking='adjacent_association', group_statistics='geometry_statistics',
                 statistics='table_statistics', assign_segments='segment_assignment',
                 interval_events='event_assembly', deepcopy='result_copy')

    def wrap(function, phase):
        def timed(*args, **kwargs):
            label = phase
            if phase == 'delayed_support':
                label = 'delayed_motion' if kwargs.get('predict', True) else 'delayed_static'
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                totals[label] += time.perf_counter()-start
                calls[label] += 1
        return timed

    original_push = history.StreamingShockHistory.push
    latest = {}

    def push(self, catalog):
        start = time.perf_counter()
        delta = original_push(self, catalog)
        totals['tracker_total'] += time.perf_counter()-start
        latest.update(surface_samples=len(catalog['positions']), groups=len(delta['nodes']),
                      spatial_edges=len(delta['partition']['edges']),
                      candidate_events=sum(len(r['result']['candidates']) for r in delta['adjacent_records']),
                      high_arity_hypotheses=sum(len(r['result']['higher_arity_audit']) for r in delta['adjacent_records']),
                      accepted_events=sum(len(r['result']['events']) for r in delta['adjacent_records']),
                      delayed_records=len(delta['support_records']))
        return delta

    try:
        for name, label in {**names, 'delayed_support': 'delayed_support'}.items():
            original[name] = getattr(history, name)
            setattr(history, name, wrap(original[name], label))
        history.StreamingShockHistory.push = push
        yield totals, calls, latest
    finally:
        history.StreamingShockHistory.push = original_push
        for name, function in original.items():
            setattr(history, name, function)


def replay(catalogs, matcher, *, instrument=False):
    """Temporary products are decoded/checked after the execution timer stops."""
    rows = []
    with tempfile.TemporaryDirectory(prefix='thesis-tracking-replay-') as folder, \
            patch.object(history, 'expanded_tracking', matcher):
        directory = Path(folder)/'tracking'

        def execute(phase_data=None):
            start = time.perf_counter()
            writer = StreamingProductWriter(directory)
            try:
                for index, catalog in enumerate(catalogs):
                    before = dict(phase_data[0]) if phase_data else {}
                    event_start = time.perf_counter()
                    writer.receive(catalog)
                    elapsed = time.perf_counter()-event_start
                    if phase_data:
                        totals, _, latest = phase_data
                        delta = {key: value-before.get(key, 0.) for key, value in totals.items()}
                        tracker = delta.pop('tracker_total')
                        delta['other_tracker_work'] = max(0., tracker-sum(delta.values()))
                        delta['product_writing_bookkeeping'] = max(0., elapsed-tracker)
                        rows.append(dict(event=index, time=float(catalog['time']), **latest,
                                         receive_seconds=elapsed, **delta))
                writer.close()
            except Exception:
                writer.close(success=False)
                raise
            return time.perf_counter()-start

        if instrument:
            with phases() as phase_data:
                elapsed = execute(phase_data)
                calls = dict(phase_data[1])
        else:
            elapsed = execute()
            calls = {}
        payload = product_payload(directory)
    return elapsed, payload, rows, calls


def check_synthetic(reference):
    """Full old/new result equality, including scores, ties and high-arity audits."""
    from experiments.astrolink.freeze_multiframe_baseline import synthetic_cases
    from experiments.wind_bubble.check_five_fragment_events import new_cases
    count = 0
    for case in synthetic_cases():
        for a, b in zip(case['catalogs'], case['catalogs'][1:]):
            for gate in case['gates']:
                for cap in (3, 4, 5):
                    args = [a[k] for k in ('positions', 'normals', 'labels')]
                    args += [b[k] for k in ('positions', 'normals', 'labels')]
                    if expanded_tracking(*args, gate, max_partners=cap) != reference(*args, gate, max_partners=cap):
                        raise AssertionError(f'Synthetic result changed: {case["name"]}')
                    count += 1
    for case in new_cases():
        args = [case[k] for k in ('positions_a', 'normals_a', 'labels_a', 'positions_b', 'normals_b', 'labels_b')]
        for gate in (.04, .07):
            for cap in (3, 4, 5):
                if expanded_tracking(*args, gate, max_partners=cap) != reference(*args, gate, max_partners=cap):
                    raise AssertionError(f'High-arity result changed: {case["name"]}')
                count += 1
    return count


def plot_summary(path, phases_rows, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['adjacent_association', 'spatial_grouping', 'delayed_static', 'delayed_motion',
              'geometry_statistics', 'table_statistics', 'product_writing_bookkeeping', 'other_tracker_work']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#d27828', '#087d8c']
    for mode, color in zip(('baseline', 'optimized'), colors):
        data = [r for r in phases_rows if r['mode'] == mode]
        values = [sum(r.get(label, 0.) for r in data) for label in labels]
        offset = -.18 if mode == 'baseline' else .18
        axes[0].barh(np.arange(len(labels))+offset, values, height=.34, label=mode, color=color)
    axes[0].set_yticks(np.arange(len(labels)), [s.replace('_', ' ') for s in labels])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Seconds, separate instrumented replay')
    axes[0].legend()
    axes[1].bar([r['mode'] for r in summary], [r['median_seconds'] for r in summary], color=colors)
    for i, row in enumerate(summary):
        median = row['median_seconds']
        axes[1].errorbar(i, median, yerr=[[median-row['min_seconds']], [row['max_seconds']-median]],
                         fmt='none', color='black', capsize=5)
        axes[1].annotate(f'{median:.3f} s', (i, row['max_seconds']), xytext=(0, 8),
                         textcoords='offset points', ha='center')
    axes[1].set_ylabel('Replay execution time (s)')
    axes[1].set_ylim(0, max(r['max_seconds'] for r in summary)*1.2)
    for ax in axes:
        ax.grid(axis='x' if ax is axes[0] else 'y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Saved three-bubble catalogs: reuse point support in expanded event matching')
    fig.text(.02, .02, 'CPU replay only; hydro, detection, GPU transfer and rendering excluded. Input catalogs preloaded.\n'
             f'Right: median and min–max of {summary[0]["repeats"]} unprofiled trials after one warmup per mode; tracking-file writes included.', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(root, output, revision, repeats=3):
    if repeats < 3:
        raise ValueError('Require at least three measured repetitions.')
    if output.exists():
        raise ValueError('Use a fresh output directory.')
    catalogs, input_hashes = load_inputs(root)
    sources = tracking_source_files()+[Path(__file__)]
    source_hashes = {str(p): sha256(p) for p in sources}
    output.mkdir(parents=True)
    reference, revision, reference_hashes = load_reference(revision, output/'reference_sources')
    matchers = dict(baseline=reference, optimized=expanded_tracking)
    expected = None
    for mode, matcher in matchers.items():
        print(f'{mode}: one untimed warmup...', flush=True)
        _, payload, _, _ = replay(catalogs, matcher)
        if expected is None:
            expected = payload
        else:
            compare_payloads(payload, expected)
    saved = root/'online/tracking'
    if not saved.is_dir():
        raise ValueError('Require the original saved tracking products for independent archive agreement.')
    for path in saved.iterdir():
        input_hashes[str(path)] = sha256(path)
    compare_payloads(expected, product_payload(saved), exact=False)
    trials = []
    for number in range(repeats):
        order = ('baseline', 'optimized') if number % 2 == 0 else ('optimized', 'baseline')
        for mode in order:
            elapsed, payload, _, _ = replay(catalogs, matchers[mode])
            compare_payloads(payload, expected)
            digest = payload_digest(payload)
            trials.append(dict(trial=number, mode=mode, execution_order=len(trials), seconds=elapsed,
                               products_sha256=digest))
            print(f'{mode} trial {number+1}: {elapsed:.3f} s; full products match', flush=True)
    phase_rows, function_rows, phase_calls = [], [], {}
    for mode, matcher in matchers.items():
        _, payload, rows, calls = replay(catalogs, matcher, instrument=True)
        compare_payloads(payload, expected)
        phase_rows.extend(dict(mode=mode, **row) for row in rows)
        phase_calls[mode] = calls
        profiler = cProfile.Profile()
        profiler.enable()
        _, payload, _, _ = replay(catalogs, matcher)
        profiler.disable()
        compare_payloads(payload, expected)
        stats = pstats.Stats(profiler)
        for (filename, line, name), (primitive, total, own, cumulative, _) in stats.stats.items():
            function_rows.append(dict(mode=mode, file=filename, line=line, function=name,
                                      primitive_calls=primitive, total_calls=total,
                                      own_seconds=own, cumulative_seconds=cumulative))
    synthetic = check_synthetic(reference)
    summary = []
    for mode in matchers:
        values = [r['seconds'] for r in trials if r['mode'] == mode]
        summary.append(dict(mode=mode, repeats=repeats, median_seconds=float(np.median(values)),
                            min_seconds=min(values), max_seconds=max(values)))
    base, optimized = [r['median_seconds'] for r in summary]
    for mode in matchers:
        with gzip.open(output/f'{mode}_products.json.gz', 'wt') as stream:
            json.dump(expected, stream, allow_nan=False)
    write_csv(output/'trials.csv', trials)
    write_csv(output/'summary.csv', summary)
    # Empty early events have no association keys yet; use the union of columns.
    fields = list(dict.fromkeys(key for row in phase_rows for key in row))
    write_csv(output/'event_phases.csv', [{key: row.get(key, 0) for key in fields} for row in phase_rows])
    write_csv(output/'function_profile.csv', sorted(function_rows, key=lambda r: (r['mode'], -r['cumulative_seconds'])))
    plot_summary(output/'tracking_replay_comparison.png', phase_rows, summary)
    for path, digest in {**input_hashes, **source_hashes, **reference_hashes}.items():
        if sha256(path) != digest:
            raise AssertionError(f'Input/source changed during comparison: {path}')
    report = dict(passed=True, input_directory=str(root), events=len(catalogs),
                  group_observations=expected['metadata.json']['group_observations'],
                  reference_revision=revision, warmup_runs_per_mode=1, measured_repetitions=repeats,
                  exact_products_all_trials=True, original_archive_matches=True,
                  original_archive_scalar_rtol=1e-6, original_archive_scalar_atol=1e-8,
                  synthetic_full_result_comparisons=synthetic,
                  median_speedup=base/optimized, median_time_reduction_percent=100*(1-optimized/base),
                  phase_calls=phase_calls, summary=summary,
                  runtime_scope='CPU saved-catalog replay, writer construction/receive/close included. '
                                'Catalog loading, checks, profiling, hydro, detector, GPU transfers and rendering excluded.',
                  timing_limitations='Same process, shared allocator/filesystem caches, alternating trial order, '
                                     'writes closed/flushed without fsync. Separate profiled runs are not speedup timings.',
                  output_scope='Decoded tracking metadata, all partitions, full adjacent/delayed records, '
                               'group statistics/CSV and terminal pending records. Container timestamps excluded.',
                  science_scope='Unchanged geometric associations, not physical ancestry or turbulent-box validation. '
                                'Local replay speedup does not establish end-to-end GPU speedup.',
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                   platform=platform.platform(), machine=platform.machine(), processor=platform.processor()),
                  input_sha256=input_hashes, source_sha256=source_hashes,
                  reference_sha256=reference_hashes,
                  output_sha256={str(p): sha256(p) for p in output.iterdir() if p.is_file()})
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('passed', 'events', 'group_observations',
          'synthetic_full_result_comparisons', 'median_speedup', 'median_time_reduction_percent')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-revision', required=True,
                        help='Git revision before the support-reuse optimization.')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    run(args.directory, args.output_dir, args.reference_revision, args.repeats)
