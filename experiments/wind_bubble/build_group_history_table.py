"""Verified per-group statistics and causal relationship availability for display."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.astrolink.freeze_multiframe_baseline import check_graph, verify_manifest, write_csv


def statistics(data, labels, nodes, event):
    """Every retained sample belongs to exactly one row; unretained stay separate."""
    labels = np.asarray(labels)
    count = len(data['positions'])
    if labels.shape != (count,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Require one integer group label per original sample.')
    groups = sorted(map(int, np.unique(labels[labels >= 0])))
    if len(nodes) != len(groups) or sorted(n['group'] for n in nodes) != groups:
        raise ValueError('History must contain each retained group exactly once.')
    if len({n['segment_id'] for n in nodes}) != len(nodes):
        raise ValueError('Different groups at one event cannot share a continuation segment.')
    for key in ('mach', 'mach_valid', 'upstream_inside_any_source', 'downstream_inside_any_source'):
        if np.asarray(data[key]).shape != (count,):
            raise ValueError('Sample-property length mismatch.')
    rows = []
    visited = np.zeros(count, int)
    actual = float(data['time'])
    for node in sorted(nodes, key=lambda n: n['group']):
        group = node['group']; indices = np.flatnonzero(labels == group)
        if node['node_id'] != f'e{event:04d}:g{group}' or node['event'] != event or abs(node['time']-actual) > 1e-7:
            raise ValueError('Node identity/event/time differs from its catalog.')
        if len(indices) != node['surface_samples']:
            raise ValueError('History/catalog group population differs.')
        visited[indices] += 1
        good = data['mach_valid'][indices] & np.isfinite(data['mach'][indices]) & (data['mach'][indices] > 0)
        mach = data['mach'][indices][good]
        median = float(np.median(mach)) if len(mach) else None
        upstream = int(data['upstream_inside_any_source'][indices].sum())
        downstream = int(data['downstream_inside_any_source'][indices].sum())
        if (median != node['median_mach'] or len(mach) != node['valid_mach_samples'] or
                upstream != node['upstream_inside_source'] or downstream != node['downstream_inside_source']):
            raise ValueError('Recomputed statistics differ from the validated history.')
        rows.append(dict(node_id=node['node_id'], event=event, time=actual, group=group,
                         segment_id=node['segment_id'], previous_group=node['previous_group'],
                         inherited=node['inherited'], surface_samples=len(indices),
                         valid_mach_samples=len(mach), invalid_mach_samples=len(indices)-len(mach),
                         mach_p16=float(np.percentile(mach, 16)) if len(mach) else None,
                         median_mach=median, mach_p84=float(np.percentile(mach, 84)) if len(mach) else None,
                         upstream_source_overlaps=upstream, downstream_source_overlaps=downstream,
                         original_rows_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                         original_positions_sha256=hashlib.sha256(data['positions'][indices].tobytes()).hexdigest()))
    if not np.array_equal(visited, (labels >= 0).astype(int)):
        raise ValueError('A sample was omitted or counted in more than one retained group.')
    accounting = dict(event=event, time=actual, detected_samples=count,
                      retained_samples=int((labels >= 0).sum()), unretained_samples=int((labels < 0).sum()),
                      retained_groups=len(nodes),
                      all_upstream_source_overlaps=int(data['upstream_inside_any_source'].sum()),
                      all_downstream_source_overlaps=int(data['downstream_inside_any_source'].sum()))
    return rows, accounting


def visible_relationships(events, support, now):
    """Visibility is decided by availability, never the earlier parent time."""
    return ([r for r in events if r['available_event'] <= now],
            [r for r in support if r['available_event'] <= now])


def conditional_rows(records, graph, times):
    nodes = {n['node_id']: n for n in graph['nodes']}
    additions = {}
    for record in records:
        for link in record['result']['conditional_support']:
            if not link['already_in_adjacent']:
                key = record['parent_event'], link['parent_group'], link['child_group']
                additions.setdefault(key, []).append((record, link))
    rows = []
    for (event, parent, child), evidence in sorted(additions.items()):
        if event+2 >= len(times):
            raise ValueError('Conditional support cannot invent a future observation.')
        parent_id = f'e{event:04d}:g{parent}'; child_id = f'e{event+1:04d}:g{child}'
        if parent_id not in nodes or child_id not in nodes:
            raise ValueError('Conditional endpoint is not an existing group observation.')
        coverage, future_ids, selections = {}, set(), set()
        for record, link in evidence:
            if link['conflicting_adjacent_targets']:
                raise ValueError('Conflicting support requires a separate display policy.')
            selections.add((record['method'], record['gate_reference_cells']))
            family = next(f for f in record['result']['analysis']['families'] if f['child_group'] == child)
            if not family['coverage_pass']:
                raise ValueError('A failed family cannot become a conditional display edge.')
            future_ids.add(f'e{event+2:04d}:g{family["later_group"]}')
            coverage[str(record['gate_reference_cells'])] = family['child_union_coverage']
        expected = {(m, g) for m in ('delayed_static_family', 'delayed_normal_motion_family') for g in (1.5, 2.5)}
        if selections != expected or len(future_ids) != 1:
            raise ValueError('This checked display requires agreement of both methods and gates.')
        future_id = next(iter(future_ids))
        if not any(e['status'] == 'shared' and e['kind'] == 'continuation' and
                   e['parents'] == [child_id] and e['children'] == [future_id] for e in graph['events']):
            raise ValueError('Conditional support has no accepted future seed.')
        rows.append(dict(support_id=f's{len(rows):04d}', parent_node=parent_id, child_node=child_id,
                         future_node=future_id, parent_event=event, child_event=event+1,
                         available_event=event+2, parent_time=times[event], child_time=times[event+1],
                         available_time=times[event+2], evidence='conditional_geometric_support',
                         narrow_union_coverage=coverage['1.5'], wide_union_coverage=coverage['2.5'],
                         inherited_id_assigned=False))
    return rows


def build(root, history, comparison, output):
    if output.exists():
        raise ValueError('Use a fresh group-table directory.')
    hashes = {}
    verify_manifest(history/'verification.json', hashes)
    verify_manifest(comparison/'verification.json', hashes)
    paths = [root/'metadata.json', history/'graph.json', history/'tracking_results.json.gz',
             comparison/'real_results.json.gz']
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if str(path) in hashes and hashes[str(path)] != digest:
            raise ValueError('History/comparison references a different input.')
        hashes[str(path)] = digest
    meta = json.loads(paths[0].read_text()); graph = json.loads(paths[1].read_text())
    with gzip.open(paths[2], 'rt') as stream:
        adjacent = json.load(stream)
    with gzip.open(paths[3], 'rt') as stream:
        delayed = [r for r in json.load(stream) if r['case'] == 'three_n064']
    times = meta['actual_times']
    if meta['status'] != 'complete' or len(times) != 17:
        raise ValueError('Require the checked complete three-bubble sequence.')
    rows, events, groups = [], [], []
    for i, time in enumerate(times):
        source = root/'online'/f'surfaces_{i:04d}.npz'; partition = history/f'groups_{i:04d}.npz'
        for path in (source, partition):
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if str(path) in hashes and hashes[str(path)] != actual:
                raise ValueError('Catalog or partition differs from validated input.')
            hashes[str(path)] = actual
        with np.load(source) as saved, np.load(partition) as labels:
            data = {k: saved[k].copy() for k in saved.files}
            labels = labels['labels'].copy()
        if abs(float(data['time'])-time) > 1e-7:
            raise ValueError('Requested history event has a different actual time.')
        nodes = [n for n in graph['nodes'] if n['event'] == i]
        result, accounting = statistics(data, labels, nodes, i)
        rows.extend(result); events.append(accounting)
        groups.append(sorted(map(int, np.unique(labels[labels >= 0]))))
    check_graph(graph, groups, times, adjacent)
    relationships = [dict(e, available_event=e['child_event'], available_time=times[e['child_event']])
                     for e in graph['events']]
    support = conditional_rows(delayed, graph, times)
    visibility = []
    for i in range(len(times)):
        existing, conditional = visible_relationships(relationships, support, i)
        visibility.append(dict(event=i, time=times[i],
                               visible_node_ids=[r['node_id'] for r in rows if r['event'] <= i],
                               visible_event_ids=[r['event_id'] for r in existing],
                               visible_support_ids=[r['support_id'] for r in conditional]))
    output.mkdir(parents=True)
    write_csv(output/'group_history.csv', rows)
    write_csv(output/'event_accounting.csv', events)
    write_csv(output/'conditional_support.csv', support)
    for name, value in [('group_history.json', rows), ('event_accounting.json', events),
                        ('tracking_events.json', relationships), ('conditional_support.json', support),
                        ('visibility.json', visibility)]:
        (output/name).write_text(json.dumps(value, indent=2)+'\n')
    # JSON readback must preserve missing values, IDs, counts and availability.
    assert json.loads((output/'group_history.json').read_text()) == rows
    assert json.loads((output/'conditional_support.json').read_text()) == support
    for path, expected in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    report = dict(passed=True, root=str(root), history=str(history), comparison=str(comparison),
                  events=len(times), group_observations=len(rows), segments=graph['segment_count'],
                  additional_conditional_relationships=len(support), samples_counted_once=True,
                  recorded_actual_times=True, conditional_support_not_in_statistics=True,
                  accepted_segment_assignments_exact=True, future_visibility_respected=True,
                  input_sha256=hashes,
                  source_sha256={str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  output_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()})
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('passed', 'events', 'group_observations', 'segments', 'additional_conditional_relationships')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/three_bubbles_iwr_compgpu7_n064'))
    parser.add_argument('--history', type=Path, default=Path('outputs/three_bubble_history_expanded_n064'))
    parser.add_argument('--comparison', type=Path, default=Path('outputs/multiframe_support_comparison_v1'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); build(args.root, args.history, args.comparison, args.output_dir)
