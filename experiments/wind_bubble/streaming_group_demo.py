"""Join callback tracking products to the existing checked three-bubble display.

An explicit saved-catalog replay can exercise the adapter before a GPU run.
Replay previews are labeled separately from callback-generated demonstrations.
No grouping, matching, state sampling or mesh fitting is performed here.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.astrolink.freeze_multiframe_baseline import check_graph, verify_manifest, write_csv
from experiments.wind_bubble.build_group_history_table import conditional_rows, statistics, visible_relationships
from experiments.wind_bubble.streaming_products import GROUP_FIELDS, compare_tree, load_event


def check_statistic_rows(recomputed, recorded):
    """Verify local reconstruction while retaining the callback's saved values.

    NumPy versions can round float32 percentile interpolation differently.
    Fingerprints and categorical/count fields remain exact; floating summaries
    use the existing online/offline tolerance, not Python dictionary equality.
    """
    if len(recomputed) != len(recorded):
        raise ValueError('Callback group statistics differ from their original samples.')
    for a, b in zip(recomputed, recorded):
        for key in ('original_rows_sha256', 'original_positions_sha256'):
            if a[key] != b[key]:
                raise ValueError('Callback group statistics differ from their original samples: fingerprint mismatch.')
    try:
        compare_tree(recomputed, recorded, 'recomputed_group_statistics')
    except AssertionError as exc:
        raise ValueError('Callback group statistics differ from their original samples.') from exc


def build_tables(root, output, replay_tracking=None, *, expected_sources=3):
    if output.exists():
        raise ValueError('Use a fresh table directory.')
    hashes = {}

    def read_json(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    metadata = read_json(root/'metadata.json')
    if metadata['status'] != 'complete' or len(metadata.get('source_positions', [])) != expected_sources:
        raise ValueError(f'Require a completed {expected_sources}-source simulation.')
    times = metadata['actual_times']
    if len(times) < 3 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError('Require at least three increasing actual event times.')
    if replay_tracking is None:
        if not metadata.get('temporal_tracking'):
            raise ValueError('Callback tracking is absent; a saved replay must be selected explicitly.')
        tracking = root/'online/tracking'
        tracked = read_json(tracking/'metadata.json')
        if tracked['status'] != 'complete' or tracked['events'] != len(times):
            raise ValueError('Callback tracking is incomplete or differs from the simulation schedule.')
        origin = 'hydrodynamic_host_callback'
        if metadata.get('verify_offline'):
            verification = read_json(root/'equivalence.json')
            for key in ('passed', 'streaming_tracking_match', 'tracking_batch_reference_match', 'final_gas_state_matches'):
                if verification.get(key) is not True:
                    raise ValueError(f'Online/offline verification is missing or failed: {key}')
    else:
        tracking = replay_tracking
        tracked = verify_manifest(tracking/'verification.json', hashes)
        if Path(tracked['input_directory']).resolve() != root.resolve() or tracked['events'] != len(times):
            raise ValueError('Replay references a different physical sequence.')
        origin = 'saved_catalog_replay_preview'

    rows, accounting, nodes, events, adjacent, delayed, groups = [], [], [], [], [], [], []
    for index, time in enumerate(times):
        source = root/'online'/f'surfaces_{index:04d}.npz'
        partition = tracking/f'groups_{index:04d}.npz'
        event_path = tracking/f'event_{index:04d}.json.gz'
        for path in (source, partition, event_path):
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(source) as saved:
            catalog = {key: saved[key].copy() for key in saved.files}
        with np.load(partition) as saved:
            labels = saved['labels'].copy()
        record = load_event(tracking, index)
        if record['event'] != index or record['time'] != time or float(catalog['time']) != time:
            raise ValueError('Catalog, callback record and actual event time differ.')
        current_rows, counts = statistics(catalog, labels, record['nodes'], index)
        check_statistic_rows(current_rows, record['group_statistics'])
        if counts != record['accounting']:
            raise ValueError('Callback group statistics differ from their original samples.')
        # Draw the recorded values; local recomputation is validation only.
        current_rows = record['group_statistics']
        if len(record['adjacent_records']) != (2 if index else 0):
            raise ValueError('Require both adjacent gates at each populated interval.')
        for edge in record['adjacent_records']:
            if (edge['parent_event'], edge['child_event']) != (index-1, index):
                raise ValueError('Adjacent results refer to the wrong interval.')
        if len(record['support_records']) != (4 if index >= 2 else 0):
            raise ValueError('Require both support variants and gates when the third event arrives.')
        expected_channels = {(m, g) for m in ('delayed_static_family', 'delayed_normal_motion_family') for g in (1.5, 2.5)}
        if record['support_records'] and {(r['method'], r['gate_reference_cells']) for r in record['support_records']} != expected_channels:
            raise ValueError('Missing or repeated delayed-support channel.')
        for support in record['support_records']:
            if (support['parent_event'], support['child_event'], support['available_event']) != (index-2, index-1, index):
                raise ValueError('Delayed support must become available at C, not at A or B.')
            if support['available_time'] != time:
                raise ValueError('Delayed-support availability time differs from its callback.')
        rows.extend(current_rows); accounting.append(counts); nodes.extend(record['nodes'])
        events.extend(record['tracking_events']); adjacent.extend(record['adjacent_records'])
        delayed.extend(record['support_records'])
        groups.append(sorted(map(int, np.unique(labels[labels >= 0]))))
    final = read_json(tracking/'finish.json')
    if (final['events_received'] != len(times) or final['tracking_event_count'] != len(events)
            or final['footprint']['retained_geometry_catalogs'] != 0):
        raise ValueError('Tracking close summary differs from its saved records.')
    terminal = final['pending_terminal_support']
    if len(terminal) != 4 or any(r['available_event'] is not None or r['available_time'] is not None
                               or r['result']['window_status'] != 'pending_future' for r in terminal):
        raise ValueError('The terminal interval must remain pending without a future event.')
    graph = dict(nodes=nodes, events=events, segment_count=final['segment_count'])
    check_graph(graph, groups, times, adjacent)
    support = conditional_rows(delayed, graph, times)
    relationships = [dict(edge, available_event=edge['child_event'], available_time=times[edge['child_event']])
                     for edge in events]
    visibility = []
    for index, time in enumerate(times):
        existing, conditional = visible_relationships(relationships, support, index)
        visibility.append(dict(event=index, time=time,
            visible_node_ids=[r['node_id'] for r in rows if r['event'] <= index],
            visible_event_ids=[r['event_id'] for r in existing],
            visible_support_ids=[r['support_id'] for r in conditional]))
    if origin == 'hydrodynamic_host_callback':
        path = tracking/'group_history.csv'
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as stream:
            actual = list(csv.DictReader(stream))
        expected = [{key: '' if row[key] is None else str(row[key]) for key in GROUP_FIELDS} for row in rows]
        if actual != expected or tracked['group_observations'] != len(rows):
            raise ValueError('Callback CSV does not reproduce its verified event rows.')

    output.mkdir(parents=True)
    write_csv(output/'group_history.csv', rows)
    write_csv(output/'event_accounting.csv', accounting)
    for name, data in (('group_history', rows), ('event_accounting', accounting),
                       ('tracking_events', relationships), ('conditional_support', support),
                       ('visibility', visibility), ('graph', graph), ('pending_terminal_support', terminal)):
        (output/(name+'.json')).write_text(json.dumps(data, indent=2)+'\n')
    for path, expected_hash in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Input changed during table conversion.')
    sources = [Path(__file__), Path('experiments/wind_bubble/build_group_history_table.py'),
               Path('experiments/astrolink/freeze_multiframe_baseline.py')]
    report = dict(passed=True, root=str(root), tracking_directory=str(tracking), origin=origin,
                  backend=metadata.get('backend'), events=len(times), group_observations=len(rows),
                  segments=final['segment_count'], additional_conditional_relationships=len(support),
                  samples_counted_once=True, original_sample_fingerprints_exact=True,
                  statistics_match_within_tolerance=True, scalar_rtol=1e-6, scalar_atol=1e-8,
                  recorded_callback_statistics_preserved_exact=True,
                  accepted_segment_assignments_exact=True, future_visibility_respected=True,
                  conditional_support_not_in_statistics=True, physical_ancestry_established=False,
                  input_sha256=hashes,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  output_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()})
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def run(root, meshes, output, fps=1., replay_tracking=None):
    if output.exists() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Use a fresh output directory and a positive finite FPS.')
    metadata = json.loads((root/'metadata.json').read_text())
    if metadata['num_cells'] != 64 or len(metadata['actual_times']) != 17:
        raise ValueError('This checked demonstration requires the established 64-cubed, 17-event sequence.')
    output.mkdir(parents=True)
    tables = build_tables(root, output/'tables', replay_tracking)
    # This frozen renderer checks every mesh's original arrays, group join,
    # segment label and time against the newly supplied tables/catalogs.
    from experiments.wind_bubble.animate_group_history import run as animate
    animate(output/'tables', meshes, output/'animation', fps)
    animation = json.loads((output/'animation/verification.json').read_text())
    report = dict(passed=True, origin=tables['origin'], backend=tables['backend'],
                  events=tables['events'], group_observations=tables['group_observations'],
                  decoded_frames=animation['decoded_frames'], playback_seconds=animation['playback_seconds'],
                  additional_conditional_relationships=tables['additional_conditional_relationships'],
                  first_conditional_event=animation['conditional_first_visible_event'],
                  mesh_geometry_changed=False, physical_ancestry_established=False,
                  input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (output/'tables/verification.json', output/'animation/verification.json')})
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--meshes', type=Path, default=Path('outputs/three_bubble_adaptive_sequence_final_n064'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=1.)
    parser.add_argument('--replay-tracking', type=Path,
                        help='Explicit preview from an archived incremental saved-catalog replay, not a new callback run.')
    args = parser.parse_args()
    run(args.directory, args.meshes, args.output_dir, args.fps, args.replay_tracking)
