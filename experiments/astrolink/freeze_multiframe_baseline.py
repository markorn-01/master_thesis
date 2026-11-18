"""Freeze a bounded comparison before integrating delayed group association.

Reuses the current maximum-five adjacent tracker. Future samples and stipulated
truth are archived for evaluation, but never passed into the baseline matcher.
No tracking algorithm, physical sample or previous result is modified.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from experiments.astrolink.check_tracklet_fragments import cases as three_state_cases
from experiments.astrolink.controlled_tracking_cases import tracking_cases, evaluate_links
from experiments.astrolink.expanded_event_tracking import expanded_tracking
from experiments.astrolink.event_history import assemble_history
from experiments.wind_bubble.apply_event_tracking import check_candidates, check_selection_dp


def summarize(result, truth=None):
    row = dict(accepted_links=sum(bool(r['accepted']) for r in result['links']),
               accepted_events=len(result['events']),
               ambiguous_parents=sum(r['status'] == 'ambiguous' for r in result['parents']),
               unmatched_parents=sum(r['status'] == 'unmatched' for r in result['parents']),
               ambiguous_children=sum(r['status'] == 'ambiguous' for r in result['children']),
               unmatched_children=sum(r['status'] == 'unmatched' for r in result['children']))
    if truth is not None:
        metrics = evaluate_links(result, truth)
        row.update(false_links=len(metrics['false_links']),
                   missed_links=len(metrics['missed_links']),
                   correct_links=len(metrics['correct_links']),
                   exact_correspondence=metrics['exact_correspondence'])
    return row


def adjacent_baseline(catalogs, gate):
    """The baseline deliberately reads only the first two observations."""
    a, b = catalogs[:2]
    return expanded_tracking(a['positions'], a['normals'], a['labels'],
                             b['positions'], b['normals'], b['labels'], gate)


def check_graph(graph, groups, times, records):
    rebuilt = assemble_history(groups, times, records)
    if graph['events'] != rebuilt['events'] or graph['segment_count'] != rebuilt['segment_count']:
        raise ValueError('Archived selections differ from their saved graph.')
    expected = [{k: node[k] for k in rebuilt['nodes'][0]} for node in graph['nodes']]
    if expected != rebuilt['nodes']:
        raise ValueError('Saved group/segment observations differ from archived selections.')


def synthetic_cases():
    result = []
    for case in tracking_cases():
        catalogs = [{key: case[f'{key}_{suffix}'].copy()
                     for key in ('positions', 'normals', 'labels')} for suffix in ('a', 'b')]
        result.append(dict(name=case['name'], suite='adjacent_regression', catalogs=catalogs,
                           times=[0., 1.], gates=[.10, .20, .35], truth=case['expected_links'],
                           meaning=case['meaning'], identifiable=True))
    for case in three_state_cases():
        result.append(dict(name=case['name'], suite='three_state_development',
                           catalogs=[case[k] for k in ('a', 'b', 'c')], times=[0., 1., 2.],
                           gates=[.05, .07, .09], truth=case['truth'], meaning=case['meaning'],
                           identifiable=not case['name'].startswith('unobservable_ancestry'),
                           expected_future_continuations_evaluation_only=case['continuations']))
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def verify_manifest(path, hashes):
    report = json.loads(path.read_text())
    if report.get('passed') is not True:
        raise ValueError(f'Unverified reference: {path}')
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for section in ('input_sha256', 'source_sha256', 'output_sha256'):
        for name, expected in report.get(section, {}).items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Changed reference: {name}')
            hashes[name] = expected
    return report


def protocol():
    return dict(
        baseline='Frozen maximum-five adjacent event tracker; no future observations.',
        candidate_scope='One-event-latency conditional fragment-support layer, separate from accepted ancestry events.',
        methods_to_compare=['adjacent_max5', 'delayed_static_family', 'delayed_normal_motion_family'],
        unchanged=dict(max_partners=5, normal_angle_degrees=60, coverage=.5,
                       exclusive_fraction=.1, distance_weight=.25, tie_tolerance=1e-8,
                       family_coverage=.5, family_score_margin=.1),
        real_gates_domain_units=[1.5/64, 2.5/64],
        evidence_channels=['accepted_adjacent_events', 'conditional_family_support', 'unresolved'],
        forbidden_shortcuts=['No truth/source-position input to association.',
                             'Derive future continuation seeds from the frozen adjacent tracker; expected synthetic continuations are evaluation-only.',
                             'No physical estimator, detection or spatial-group changes.',
                             'No mesh geometry input; original sparse samples only.',
                             'No promotion of family support to inherited physical identity.',
                             'No invented future observation at the final saved event.'],
        evaluation=dict(
            synthetic='Report false/missed links, exact configurations and ambiguous/unmatched groups by suite and identifiability; known development cases, not held-out validation.',
            hydro='Report changed links, added support, unresolved groups, source-overlap flags and distance sensitivity. No physical false-link rate is available.',
            runtime='Paired reruns on the same machine: one untimed warmup and three timed repeats per method, excluding input/output and independent verification. Separate adjacent computation from support-layer cost; prior archive times are not a performance baseline.',
            latency='At most one future event; final transition stays provisional if that event is unavailable.'),
        stopping_rule=[
            'Evaluate only the two delayed variants with the existing frozen settings across this complete suite.',
            'A reduction in missed synthetic links counts as an ancestry improvement only if false links do not increase on identifiable cases; report each case, not only totals.',
            'Evaluate conditional support separately: supported-family coverage can improve without resolving ancestry. Identical-observation controls must remain explicitly nonidentifiable.',
            'Require preservation of input arrays and all unchanged accepted events; any conflict is reported, never silently replaced.',
            'Record paired runtime and bounded candidate counts; make no many-bubble scaling claim.',
            'If neither delayed variant improves the declared task, retain the adjacent baseline and document that outcome; do not start automatic threshold tuning.'],
        new_algorithm_implemented=False,
        research_scope='A bounded geometric tracking experiment, not physical ancestry recovery or a complete turbulent-box tracker.')


def run(output):
    if output.exists():
        raise ValueError('Use a fresh output directory.')
    hashes = {}
    references = [
        dict(name='unequal_two_n128', root=Path('outputs/two_bubbles_detection_extension_iwr_n128'),
             history=Path('outputs/expanded_full_sequence_validation_n128'),
             partitions=Path('outputs/two_bubbles_detection_extension_review_n128'), events=21,
             focus_parent_events=[9, 10, 19]),
        dict(name='three_n064', root=Path('outputs/three_bubbles_iwr_compgpu7_n064'),
             history=Path('outputs/three_bubble_history_expanded_n064'),
             partitions=Path('outputs/three_bubble_history_expanded_n064'), events=17,
             focus_parent_events=[6, 7, 13])]
    for spec in references:
        verify_manifest(spec['history']/'verification.json', hashes)
        verify_manifest(spec['partitions']/'verification.json', hashes)
    output.mkdir(parents=True)
    (output/'protocol.json').write_text(json.dumps(protocol(), indent=2)+'\n')
    rows, results, descriptions = [], [], []
    for case in synthetic_cases():
        name = case['suite']+'__'+case['name']
        arrays = {f'{key}_{index}': value for index, data in enumerate(case['catalogs'])
                  for key, value in data.items()}
        np.savez_compressed(output/(name+'.npz'), **arrays)
        descriptions.append({k: v for k, v in case.items() if k != 'catalogs'})
        a, b = case['catalogs'][:2]
        for gate in case['gates']:
            start = time.perf_counter()
            result = adjacent_baseline(case['catalogs'], gate)
            elapsed = time.perf_counter()-start
            audit = check_candidates(a, b, a['labels'], b['labels'], result)
            check_selection_dp(result)
            shuffled = [{k: v[::-1].copy() for k, v in data.items()} for data in (a, b)]
            x, y = shuffled
            other = expanded_tracking(x['positions'], x['normals'], x['labels'],
                                      y['positions'], y['normals'], y['labels'], gate)
            if evaluate_links(result, [])['accepted_links'] != evaluate_links(other, [])['accepted_links']:
                raise AssertionError('Baseline link set depends on row order.')
            rows.append(dict(suite=case['suite'], case=case['name'], gate=gate,
                             identifiable=case['identifiable'], **summarize(result, case['truth']),
                             observed_seconds_unbenchmarked=elapsed))
            results.append(dict(suite=case['suite'], case=case['name'], gate=gate,
                                result=result, independent_geometry=audit, independent_dp=True,
                                reversed_row_links_exact=True))
    (output/'synthetic_cases.json').write_text(json.dumps(descriptions, indent=2)+'\n')
    write_csv(output/'synthetic_baseline.csv', rows)
    with gzip.open(output/'synthetic_baseline.json.gz', 'wt') as stream:
        json.dump(results, stream)
    real_rows, real_cases = [], []
    for spec in references:
        archive = spec['history']/'tracking_results.json.gz'
        graph_path = spec['history']/'graph.json'
        for path in (archive, graph_path):
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(archive, 'rt') as stream:
            records = json.load(stream)
        graph = json.loads(graph_path.read_text())
        assert len(records) == 2*(spec['events']-1)
        meta_path = spec['root']/'metadata.json'
        hashes[str(meta_path)] = hashlib.sha256(meta_path.read_bytes()).hexdigest()
        meta = json.loads(meta_path.read_text())
        groups = []
        for event in range(spec['events']):
            source = spec['root']/'online'/f'surfaces_{event:04d}.npz'
            partition = spec['partitions']/f'groups_{event:04d}.npz'
            for path in (source, partition):
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if str(path) in hashes and hashes[str(path)] != actual:
                    raise ValueError(f'Changed catalog/partition: {path}')
                hashes[str(path)] = actual
            with np.load(source) as data, np.load(partition) as group:
                labels = group['labels']
                groups.append(list(map(int, np.unique(labels[labels >= 0]))))
                assert labels.shape == (len(data['positions']),)
                np.testing.assert_allclose(float(data['time']), meta['actual_times'][event], rtol=0, atol=1e-7)
                for node in (n for n in graph['nodes'] if n['event'] == event):
                    assert int((labels == node['group']).sum()) == node['surface_samples']
        assert sorted((r['parent_event'], r['gate_reference_cells']) for r in records) == [
            (i, g) for i in range(spec['events']-1) for g in (1.5, 2.5)]
        check_graph(graph, groups, meta['actual_times'], records)
        for record in records:
            real_rows.append(dict(case=spec['name'], parent_event=record['parent_event'],
                                  gate_reference_cells=record['gate_reference_cells'],
                                  **summarize(record['result']), physical_truth_available=False))
        real_cases.append(dict(name=spec['name'], root=str(spec['root']), history=str(spec['history']),
                               partitions=str(spec['partitions']), archive=str(archive),
                               events=spec['events'], configurations=len(records),
                               focus_parent_events=spec['focus_parent_events'],
                               terminal_parent_event=spec['events']-2,
                               terminal_future_event_available=False,
                               actual_times=meta['actual_times'],
                               accepted_event_counts=dict(Counter(e['status']+'_'+e['kind'] for e in graph['events']))))
    write_csv(output/'real_baseline.csv', real_rows)
    (output/'real_cases.json').write_text(json.dumps(real_cases, indent=2)+'\n')
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Input changed during freeze: {path}')
    sources = [Path(__file__)]+[Path('experiments/astrolink')/(name+'.py') for name in
               ('event_tracking', 'expanded_event_tracking', 'event_history', 'tracklet_fragments',
                'controlled_tracking_cases', 'check_tracklet_fragments')]+[
                    Path('experiments/wind_bubble/apply_event_tracking.py')]
    aggregate = []
    for suite in ('adjacent_regression', 'three_state_development'):
        selected = [r for r in rows if r['suite'] == suite]
        aggregate.append(dict(suite=suite, configurations=len(selected),
                              false_links=sum(r['false_links'] for r in selected),
                              missed_links=sum(r['missed_links'] for r in selected),
                              exact_configurations=sum(r['exact_correspondence'] for r in selected)))
    report = dict(passed=True, synthetic_configurations=len(rows), synthetic_aggregate=aggregate,
                  independent_synthetic_checks=len(results), reversed_row_checks=len(results),
                  real_sequences=len(references), real_interval_gate_records=len(real_rows),
                  graphs_reassembled_exact=len(references),
                  real_results_reused_without_rerunning=True, physical_ancestry_established=False,
                  new_tracker_implemented=False, runtime_benchmarked=False,
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   scipy=scipy.__version__, platform=platform.platform()),
                  input_sha256=hashes,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  output_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in output.iterdir() if p.is_file()})
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('passed', 'synthetic_aggregate', 'real_sequences',
                                           'real_interval_gate_records', 'new_tracker_implemented')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args().output_dir)
