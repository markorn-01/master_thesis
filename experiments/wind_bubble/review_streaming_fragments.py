"""Read-only review of conditional fragment support from saved callback records.

Shows original A/B/C surface samples and preserves accepted/conditional roles.
No regrouping, new matching, new ancestry, or hydrodynamic evolution is added.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.astrolink.audit_delayed_support import check_window
from experiments.wind_bubble.review_delayed_support import evidence_roles, group_rows, render
from experiments.wind_bubble.streaming_products import load_event


CHANNELS = {(method, gate) for method in ('delayed_static_family', 'delayed_normal_motion_family')
            for gate in (1.5, 2.5)}


def family_evidence(records, child, accepted):
    """Require agreement of both saved variants/gates; never promote a link."""
    if len(records) != 4 or {(r['method'], r['gate_reference_cells']) for r in records} != CHANNELS:
        raise ValueError('Require four distinct delayed-support channels.')
    families, additions, coverage = [], [], {}
    for record in records:
        result = record['result']
        if (result['window_status'] != 'evaluated' or result['physical_ancestry_established']
                or result['inherited_id_assigned'] or result['conflicts']):
            raise ValueError('Require evaluated, nonconflicting conditional evidence without inherited IDs.')
        selected = [f for f in result['analysis']['families'] if f['child_group'] == child]
        if len(selected) != 1 or not selected[0]['coverage_pass']:
            raise ValueError('A displayed child must have one passing conditional family.')
        family = selected[0]
        members = set(family['parent_groups'])
        if len(members) != len(family['parent_groups']):
            raise ValueError('Repeated family parent.')
        links = [link for link in result['conditional_support'] if link['child_group'] == child]
        if len(links) != len(members) or {link['parent_group'] for link in links} != members:
            raise ValueError('Family and conditional-link memberships disagree.')
        for link in links:
            if (link['already_in_adjacent'] != (link['parent_group'] in accepted)
                    or link['conflicting_adjacent_targets'] or link['physical_ancestry_established']):
                raise ValueError('Conditional evidence has an incorrect accepted/ancestry role.')
        additional = {link['parent_group'] for link in links if not link['already_in_adjacent']}
        evidence_roles(members, accepted, additional)
        families.append((tuple(sorted(members)), family['later_group']))
        additions.append(tuple(sorted(additional)))
        coverage[f'{record["method"]}:{record["gate_reference_cells"]}'] = family['child_union_coverage']
    if len(set(families)) != 1 or len(set(additions)) != 1:
        raise ValueError('Family membership differs between saved methods or distances.')
    parents, later = families[0]
    return dict(parents=list(parents), later=later, additional=list(additions[0]),
                roles=evidence_roles(parents, accepted, additions[0]), coverage=coverage)


def load_review(root, parent_event):
    hashes = {}

    def remember(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    remember(root/'metadata.json')
    remember(root/'equivalence.json')
    meta = json.loads((root/'metadata.json').read_text())
    verification = json.loads((root/'equivalence.json').read_text())
    if meta['status'] != 'complete' or not meta.get('temporal_tracking'):
        raise ValueError('Require completed callback tracking.')
    for key in ('passed', 'streaming_tracking_match', 'tracking_batch_reference_match', 'final_gas_state_matches'):
        if verification.get(key) is not True:
            raise ValueError(f'Missing or failed GPU verification: {key}')
    times = meta['actual_times']
    if parent_event < 0 or parent_event+2 >= len(times):
        raise ValueError('A review needs three actual observations.')
    frames, records = [], []
    for event in range(parent_event, parent_event+3):
        folder = root/'online'
        for path in (folder/f'surfaces_{event:04d}.npz', folder/'tracking'/f'groups_{event:04d}.npz',
                     folder/'tracking'/f'event_{event:04d}.json.gz'):
            remember(path)
        with np.load(folder/f'surfaces_{event:04d}.npz') as data:
            physical = {key: data[key].copy() for key in data.files}
        with np.load(folder/'tracking'/f'groups_{event:04d}.npz') as data:
            labels = data['labels'].copy()
        record = load_event(folder/'tracking', event)
        if record['event'] != event or record['time'] != times[event] or float(physical['time']) != times[event]:
            raise ValueError('Catalog, record and actual event times disagree.')
        if labels.shape != (len(physical['positions']),):
            raise ValueError('Partition and physical catalog differ.')
        np.testing.assert_array_equal(physical['source_positions'], meta['source_positions'])
        for node in record['nodes']:
            indices = group_rows(physical['positions'], labels, node['group'], node['surface_samples'])
            if int(physical['upstream_inside_any_source'][indices].sum()) != node['upstream_inside_source']:
                raise ValueError('Source-overlap flags differ from saved node statistics.')
            if int(physical['downstream_inside_any_source'][indices].sum()) != node['downstream_inside_source']:
                raise ValueError('Source-overlap flags differ from saved node statistics.')
        frames.append(dict(event=event, time=times[event], physical=physical, labels=labels))
        records.append(record)
    a, b, c = records
    support = c['support_records']
    if any((r['parent_event'], r['child_event'], r['available_event'], r['available_time']) !=
           (parent_event, parent_event+1, parent_event+2, times[parent_event+2]) for r in support):
        raise ValueError('Future evidence is assigned to the wrong observation/time.')
    current = {r['gate_reference_cells']: r['result'] for r in b['adjacent_records']}
    if set(current) != {1.5, 2.5} or len(b['adjacent_records']) != 2:
        raise ValueError('Require both current matching gates.')
    future = [r['result'] for r in c['adjacent_records']]
    if len(future) != 2 or {r['gate_reference_cells'] for r in c['adjacent_records']} != {1.5, 2.5}:
        raise ValueError('Require both future matching gates.')
    geometry = [dict(positions=f['physical']['positions'], normals=f['physical']['normals'], labels=f['labels']) for f in frames]
    audits = []
    for record in support:
        audits.append(dict(method=record['method'], gate=record['gate_reference_cells'],
                           **check_window(geometry, times[parent_event:parent_event+3],
                                          current[record['gate_reference_cells']], future, record['result'])))
    reviews = []
    for event in b['tracking_events']:
        if event['status'] != 'shared' or event['kind'] != 'combination' or len(event['children']) != 1:
            continue
        child = int(event['children'][0].split(':g')[1])
        accepted = {int(name.split(':g')[1]) for name in event['parents']}
        for gate, result in current.items():
            expected = [e for e in result['events'] if e['kind'] == 'combination'
                        and e['children'] == [child] and set(e['parents']) == accepted]
            if len(expected) != 1:
                raise ValueError('Shared combination differs from the per-gate accepted events.')
        family = family_evidence(support, child, accepted)
        if not family['additional']:
            continue
        if not any(e['status'] == 'shared' and e['kind'] == 'continuation'
                   and e['parents'] == [f'e{parent_event+1:04d}:g{child}']
                   and e['children'] == [f'e{parent_event+2:04d}:g{family["later"]}']
                   for e in c['tracking_events']):
            raise ValueError('The family lacks a saved shared future continuation.')
        selected_frames = []
        for frame, record, groups in zip(frames, records, (family['parents'], [child], [family['later']])):
            observations = []
            for group in groups:
                node = next(n for n in record['nodes'] if n['group'] == group)
                indices = group_rows(frame['physical']['positions'], frame['labels'], group, node['surface_samples'])
                observations.append(dict(group=group, rows=indices, samples=len(indices),
                    upstream=node['upstream_inside_source'], downstream=node['downstream_inside_source']))
            selected_frames.append(dict(frame, groups=observations))
        points = np.concatenate([f['physical']['positions'][g['rows']] for f in selected_frames for g in f['groups']])
        pad = 2*float(frames[0]['physical']['grid_spacing'])
        coverage = {str(gate): family['coverage'][f'delayed_normal_motion_family:{gate}'] for gate in (1.5, 2.5)}
        reviews.append(dict(case=f'{len(meta["source_positions"])}_sources_child{child}',
            parent_event=parent_event, child=child, later=family['later'], roles=family['roles'],
            frames=selected_frames, low=points.min(axis=0)-pad, high=points.max(axis=0)+pad,
            coverage=coverage, all_channel_coverage=family['coverage']))
    if not reviews:
        raise ValueError('No shared combinations with consensus additional support in this window.')
    return reviews, hashes, audits, meta


def run(root, output, parent_event=13):
    if output.exists():
        raise ValueError('Use a fresh review directory.')
    reviews, hashes, audits, metadata = load_review(root, parent_event)
    output.mkdir(parents=True)
    summaries, assignments, joined = [], [], []
    for review in reviews:
        image = render(review, output, title=f'{len(metadata["source_positions"])} bubbles, '
                       f'{metadata["num_cells"]}³ | B:g{review["child"]}')
        arrays = {}
        counts = []
        for i, frame in enumerate(review['frames']):
            counts.append(sum(g['samples'] for g in frame['groups']))
            indices = np.sort(np.concatenate([g['rows'] for g in frame['groups']]))
            arrays[f'frame{i}_original_rows'] = indices
            arrays[f'frame{i}_positions'] = frame['physical']['positions'][indices]
            arrays[f'frame{i}_groups'] = frame['labels'][indices]
            for group in frame['groups']:
                joined.append(dict(event=frame['event'], group=group['group'], samples=group['samples'],
                    original_rows_sha256=hashlib.sha256(group['rows'].astype('<i8').tobytes()).hexdigest(),
                    original_positions_sha256=hashlib.sha256(frame['physical']['positions'][group['rows']].tobytes()).hexdigest(),
                    upstream_source_overlaps=group['upstream'], downstream_source_overlaps=group['downstream']))
        np.savez_compressed(output/f'family_child{review["child"]}.npz', **arrays)
        for group, role in review['roles'].items():
            assignments.append(dict(parent_group=group, child_group=review['child'], later_group=review['later'],
                                    role=role, available_event=parent_event+1 if role == 'accepted' else parent_event+2,
                                    physical_ancestry_established=False))
        summaries.append(dict(child_group=review['child'], later_group=review['later'],
            accepted_parent_groups=[g for g, role in review['roles'].items() if role == 'accepted'],
            additional_conditional_groups=[g for g, role in review['roles'].items() if role == 'conditional'],
            samples_A_B_C=counts, coverage=review['all_channel_coverage'], image=image))
    if len({row['parent_group'] for row in assignments}) != len(assignments):
        raise ValueError('A parent fragment was displayed in multiple consensus families.')
    with (output/'assignments.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(assignments[0]))
        writer.writeheader(); writer.writerows(assignments)
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Input changed during read-only review.')
    sources = [Path(__file__), Path('experiments/wind_bubble/review_delayed_support.py'),
               Path('experiments/astrolink/audit_delayed_support.py')]
    report = dict(passed=True, root=str(root), parent_event=parent_event,
        actual_times=metadata['actual_times'][parent_event:parent_event+3], families=summaries,
        accepted_fragments=sum(row['role']=='accepted' for row in assignments),
        additional_conditional_fragments=sum(row['role']=='conditional' for row in assignments),
        dense_geometry_audits=audits, original_sample_joins=joined,
        positions_changed=False, tracking_decisions_changed=False, inherited_ids_added=False,
        conditional_support_available_only_at_C=True, physical_ancestry_established=False,
        scope='Read-only geometric evidence review. Coverage is not a probability or complete ancestry. '
              'Existing maximum-five accepted events remain unchanged.',
        input_sha256=hashes, source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        output_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()})
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','actual_times','accepted_fragments','additional_conditional_fragments')},indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--parent-event', type=int, default=13)
    args = parser.parse_args()
    run(args.directory, args.output_dir, args.parent_event)
