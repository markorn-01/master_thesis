"""Compare fresh-process incremental replays with frozen offline references."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.astrolink.freeze_multiframe_baseline import verify_manifest, write_csv


def load_json_gzip(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def check_statistics(rows, catalog, labels):
    """Independent sorted-value quantiles, source flags, counts and joins."""
    accounted = np.zeros(len(labels), int)
    for row in rows:
        indices = np.flatnonzero(labels == row['group']); accounted[indices] += 1
        assert len(indices) == row['surface_samples']
        assert hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest() == row['original_rows_sha256']
        assert hashlib.sha256(catalog['positions'][indices].tobytes()).hexdigest() == row['original_positions_sha256']
        values = sorted(float(m) for m, valid in zip(catalog['mach'][indices], catalog['mach_valid'][indices])
                        if valid and np.isfinite(m) and m > 0)
        assert row['valid_mach_samples'] == len(values)
        assert row['invalid_mach_samples'] == len(indices)-len(values)
        for key, q in [('mach_p16', .16), ('median_mach', .5), ('mach_p84', .84)]:
            if values:
                x = q*(len(values)-1); low, high = math.floor(x), math.ceil(x)
                expected = values[low]+(x-low)*(values[high]-values[low])
                np.testing.assert_allclose(row[key], expected, rtol=1e-7, atol=1e-6)
            else:
                assert row[key] is None
        for side in ('upstream', 'downstream'):
            assert row[side+'_source_overlaps'] == int(catalog[side+'_inside_any_source'][indices].sum())
    np.testing.assert_array_equal(accounted, (labels >= 0).astype(int))


def check_case(spec, directory, delayed_reference, hashes):
    folder = directory/spec['name']
    worker = verify_manifest(folder/'verification.json', hashes)
    assert Path(worker['input_directory']) == Path(spec['root'])
    assert worker['reference_histories_loaded'] is False and worker['events'] == spec['events']
    assert worker['max_retained_geometry_catalogs'] <= 2
    assert worker['max_processing_geometry_catalogs'] <= 3
    assert worker['final_geometry_bytes'] == 0
    archive_path = Path(spec['archive']); graph_path = Path(spec['history'])/'graph.json'
    adjacent = load_json_gzip(archive_path)
    graph = json.loads(graph_path.read_text())
    adjacent_lookup = {(r['parent_event'], r['gate_reference_cells']): r['result'] for r in adjacent}
    delayed_lookup = {(r['method'], r['parent_event'], r['gate_reference_cells']): r['result']
                      for r in delayed_reference if r['case'] == spec['name']}
    all_nodes, all_events, all_stats, emitted = [], [], [], []
    max_candidates = 0
    for index in range(spec['events']):
        delta = load_json_gzip(folder/f'event_{index:04d}.json.gz')
        assert delta['event'] == index and delta['time'] == spec['actual_times'][index]
        expected_nodes = [n for n in graph['nodes'] if n['event'] == index]
        assert delta['nodes'] == expected_nodes, f'Node/segment/statistic mismatch: {spec["name"]} event {index}'
        with np.load(folder/f'groups_{index:04d}.npz') as actual, \
                np.load(Path(spec['partitions'])/f'groups_{index:04d}.npz') as expected:
            for key in ('labels', 'raw_labels', 'component_sizes'):
                assert actual[key].dtype == expected[key].dtype
                np.testing.assert_array_equal(actual[key], expected[key])
            labels = expected['labels'].copy()
        source = Path(spec['root'])/'online'/f'surfaces_{index:04d}.npz'
        with np.load(source) as catalog:
            check_statistics(delta['group_statistics'], catalog, labels)
            assert delta['accounting']['detected_samples'] == len(labels)
            assert delta['accounting']['retained_samples'] == int((labels >= 0).sum())
            assert delta['accounting']['unretained_samples'] == int((labels < 0).sum())
        if index == 0:
            assert not delta['adjacent_records'] and not delta['tracking_events']
        else:
            assert len(delta['adjacent_records']) == 2
        for record in delta['adjacent_records']:
            assert record['child_event'] == index and record['parent_event'] == index-1
            assert record['result'] == adjacent_lookup[index-1, record['gate_reference_cells']], \
                f'Full adjacent result mismatch: {spec["name"]} {index-1}'
            max_candidates = max(max_candidates, len(record['result']['candidates']))
        assert all(e['child_event'] == index for e in delta['tracking_events'])
        if index < 2:
            assert not delta['support_records']
        else:
            assert len(delta['support_records']) == 4
        for record in delta['support_records']:
            assert record['parent_event'] == index-2 and record['child_event'] == index-1
            assert record['available_event'] == index and record['available_time'] == delta['time']
            key = record['method'], record['parent_event'], record['gate_reference_cells']
            assert record['result'] == delayed_lookup[key], f'Delayed result mismatch: {spec["name"]} {key}'
            emitted.append(key)
        all_nodes.extend(delta['nodes']); all_events.extend(delta['tracking_events']); all_stats.extend(delta['group_statistics'])
    final = json.loads((folder/'finish.json').read_text())
    assert final['segment_count'] == graph['segment_count']
    assert all_nodes == graph['nodes'] and all_events == graph['events']
    assert final['footprint']['retained_geometry_catalogs'] == 0
    assert final['footprint']['retained_adjacent_intervals'] == 0
    assert len(final['pending_terminal_support']) == 4
    for record in final['pending_terminal_support']:
        key = record['method'], record['parent_event'], record['gate_reference_cells']
        assert record['parent_event'] == spec['events']-2
        assert record['available_event'] is None and record['available_time'] is None
        assert record['result'] == delayed_lookup[key]
        assert record['result']['window_status'] == 'pending_future'
        emitted.append(key)
    assert len(emitted) == len(set(emitted)) == 4*(spec['events']-1)
    assert set(emitted) == set(delayed_lookup)
    if spec['name'] == 'three_n064':
        table_path = Path('outputs/three_bubble_group_history_table_v1/group_history.json')
        hashes[str(table_path)] = hashlib.sha256(table_path.read_bytes()).hexdigest()
        assert all_stats == json.loads(table_path.read_text())
    return dict(case=spec['name'], events=spec['events'], group_observations=len(all_nodes),
                graph_event_explanations=len(all_events), segments=graph['segment_count'],
                exact_partition_events=spec['events'], exact_adjacent_results=2*(spec['events']-1),
                exact_delayed_results=len(emitted), group_statistics_exact=True,
                availability_times_correct=True, pending_terminal_records=4,
                max_adjacent_candidates=max_candidates,
                compute_seconds=worker['compute_seconds'],
                max_retained_geometry_catalogs=worker['max_retained_geometry_catalogs'],
                max_processing_geometry_catalogs=worker['max_processing_geometry_catalogs'],
                max_retained_geometry_bytes=worker['max_retained_geometry_bytes'],
                max_processing_geometry_bytes=worker['max_processing_geometry_bytes'],
                worker_peak_host_rss_bytes=worker['worker_peak_host_rss_bytes'])


def plot_memory(directory, cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for case, color in zip(cases, ['#21868a', '#c07832']):
        with (directory/case['case']/'resource_history.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        x = [int(r['event']) for r in rows]
        axes[0].step(x, [int(r['retained_geometry_catalogs']) for r in rows], where='post',
                     label=case['case'], color=color, linewidth=2)
        axes[1].plot(x, [int(r['retained_geometry_bytes'])/2**20 for r in rows],
                     label=case['case'], color=color)
    axes[0].axhline(3, color='#888888', ls=':', label='Maximum during processing')
    axes[0].set(ylabel='Geometry catalogs', ylim=(0, 3.5), yticks=[0, 1, 2, 3], title='Retained between arrivals: at most two')
    axes[1].set(ylabel='Owned geometry arrays (MiB)', title='Payload follows the current surface population')
    for ax in axes:
        ax.set_xlabel('Received event index'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Incremental tracking: bounded recent-event geometry buffer')
    fig.text(.06, .025, 'Geometry payload excludes hypotheses, caller input, output deltas and temporary arrays. It is not process RSS.\n'
             'Three catalogs are used while processing A–B–C; finish releases the buffer. Increasing surface populations can increase byte use.', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .94)); fig.savefig(directory/'buffer_check.png', dpi=150); plt.close(fig)


def run(directory, selected):
    hashes = {}
    baseline = Path('outputs/multiframe_tracking_baseline_v1')
    comparison = Path('outputs/multiframe_support_comparison_v1')
    verify_manifest(baseline/'verification.json', hashes)
    verify_manifest(comparison/'verification.json', hashes)
    with gzip.open(comparison/'real_results.json.gz', 'rt') as stream:
        delayed = json.load(stream)
    specs = json.loads((baseline/'real_cases.json').read_text())
    cases = []
    for spec in specs:
        if spec['name'] not in selected:
            continue
        case = check_case(spec, directory, delayed, hashes)
        cases.append(case)
        print(json.dumps(case, indent=2), flush=True)
    if len(cases) != len(selected):
        raise ValueError('Unknown or repeated replay case.')
    write_csv(directory/'summary.csv', cases)
    plot_memory(directory, cases)
    for path, expected in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    sources = [Path(__file__), Path('experiments/astrolink/streaming_history.py'),
               Path('experiments/astrolink/replay_streaming_history.py')]
    report = dict(passed=True, cases=cases, validated_cases=sorted(selected),
                  grouping_and_event_parameters_unchanged=True, all_emitted_outputs_match_offline=True,
                  future_support_not_emitted_early=True, component_retains_no_full_history=True,
                  original_results_changed=False, integrated_into_hydro_callback=False,
                  physical_ancestry_established=False,
                  runtime_scope='One fresh-process correctness replay per case, no warmup/repeats or speedup claim.',
                  memory_scope='Two retained geometry catalogs; three while processing. Byte payload is not total memory; recorded process RSS is a post-push high-water including imports and output.',
                  input_sha256=hashes,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  output_sha256={str(directory/name): hashlib.sha256((directory/name).read_bytes()).hexdigest()
                                 for name in ['summary.csv', 'buffer_check.png']})
    (directory/'verification.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('outputs/streaming_history_replay_v1'))
    parser.add_argument('--case', action='append', choices=['three_n064', 'unequal_two_n128'])
    args = parser.parse_args(); run(args.directory, args.case or ['three_n064', 'unequal_two_n128'])
