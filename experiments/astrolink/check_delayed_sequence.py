"""Evaluate two frozen delayed support variants against the fixed checkpoint."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from experiments.astrolink.audit_delayed_support import check_window
from experiments.astrolink.controlled_tracking_cases import evaluate_links
from experiments.astrolink.delayed_support import delayed_support
from experiments.astrolink.freeze_multiframe_baseline import (
    adjacent_baseline, check_graph, summarize, verify_manifest, write_csv)


METHODS = [('delayed_static_family', False), ('delayed_normal_motion_family', True)]


def pairs(result):
    return {(r['parent_group'], r['child_group']) for r in result['conditional_support']}


def signature(result):
    """Compare categorical decisions, not incidental floating summation order."""
    analysis = result['analysis']
    return dict(status=result['window_status'], seeds=result['future_continuations'],
                support=sorted(pairs(result)), conflicts=result['conflicts'],
                unresolved=result['unresolved'],
                assignments=[(r['parent_group'], r['child_group'], r['status'])
                             for r in analysis['assignments']] if analysis else [])


def evidence_summary(result, current, truth=None):
    base = {(r['parent_group'], r['child_group']) for r in current['links'] if r['accepted']}
    support = pairs(result)
    added = support-base
    row = dict(window_status=result['window_status'], accepted_events_unchanged=True,
               baseline_accepted_links=len(base), conditional_support_links=len(support),
               additional_support_links=len(added), conflicts=len(result['conflicts']),
               adjacent_unresolved_parents=sum(r['status'] != 'assigned' for r in current['parents']),
               conditional_only_parents=sum(r['support_status'] == 'conditional_support_only' for r in result['unresolved']),
               seed_count=len(result['future_continuations']),
               compared_parent_family_pairs=(len(result['analysis']['assignments'])*len(result['analysis']['models']))
               if result['analysis'] else 0)
    if truth is not None:
        baseline = summarize(current, truth)
        hypothetical = evaluate_links(dict(links=[dict(parent_group=p, child_group=c, accepted=True)
                                                 for p, c in sorted(base|support)]), truth)
        row.update(baseline_false_links=baseline['false_links'], baseline_missed_links=baseline['missed_links'],
                   hypothetical_union_false_links=len(hypothetical['false_links']),
                   hypothetical_union_missed_links=len(hypothetical['missed_links']),
                   hypothetical_union_exact=hypothetical['exact_correspondence'])
    return row


def support_sequence(catalogs, times, records, predict):
    results = []
    lookup = {(r['parent_event'], r['gate_reference_cells']): r['result'] for r in records}
    for index in range(len(catalogs)-1):
        future = [lookup[index+1, g] for g in (1.5, 2.5)] if index+2 < len(catalogs) else None
        for gate in (1.5, 2.5):
            result = delayed_support(catalogs[index:index+3], times[index:index+3],
                                     lookup[index, gate], future, predict=predict)
            results.append(dict(parent_event=index, gate_reference_cells=gate, result=result))
    return results


def compute_adjacent(catalogs):
    return [dict(parent_event=i, child_event=i+1, gate_reference_cells=g,
                 result=adjacent_baseline(catalogs[i:i+2], g/64))
            for i in range(len(catalogs)-1) for g in (1.5, 2.5)]


def run(baseline, output):
    if output.exists():
        raise ValueError('Use a fresh comparison directory.')
    hashes = {}
    frozen = verify_manifest(baseline/'verification.json', hashes)
    protocol = json.loads((baseline/'protocol.json').read_text())
    assert protocol['methods_to_compare'] == ['adjacent_max5']+[m for m, _ in METHODS]
    output.mkdir(parents=True)
    rows, details, audit_count, shuffle_count = [], [], 0, 0
    descriptions = json.loads((baseline/'synthetic_cases.json').read_text())
    with gzip.open(baseline/'synthetic_baseline.json.gz', 'rt') as stream:
        frozen_records = json.load(stream)
    for case in descriptions:
        path = baseline/(case['suite']+'__'+case['name']+'.npz')
        with np.load(path) as arrays:
            catalogs = [{k: arrays[f'{k}_{i}'].copy() for k in ('positions', 'normals', 'labels')}
                        for i in range(len(case['times']))]
        for gate in case['gates']:
            current = next(r['result'] for r in frozen_records
                           if (r['suite'], r['case'], r['gate']) == (case['suite'], case['name'], gate))
            future = [adjacent_baseline(catalogs[1:], gate)] if len(catalogs) == 3 else None
            reversed_catalogs = [{k: v[::-1].copy() for k, v in c.items()} for c in catalogs]
            reversed_current = adjacent_baseline(reversed_catalogs, gate)
            reversed_future = [adjacent_baseline(reversed_catalogs[1:], gate)] if len(catalogs) == 3 else None
            for method, predict in METHODS:
                result = delayed_support(catalogs, case['times'], current, future, predict=predict)
                audit = check_window(catalogs, case['times'], current, future, result)
                audit_count += 1
                other = delayed_support(reversed_catalogs, case['times'], reversed_current,
                                        reversed_future, predict=predict)
                assert signature(result) == signature(other)
                shuffle_count += 1
                rows.append(dict(suite=case['suite'], case=case['name'], gate=gate, method=method,
                                 identifiable=case['identifiable'], **evidence_summary(result, current, case['truth'])))
                details.append(dict(suite=case['suite'], case=case['name'], gate=gate,
                                    method=method, result=result, independent_audit=audit))
    write_csv(output/'synthetic_comparison.csv', rows)
    with gzip.open(output/'synthetic_results.json.gz', 'wt') as stream:
        json.dump(details, stream)
    print('Synthetic comparison: 96 audited method/configuration results.', flush=True)
    real_cases = json.loads((baseline/'real_cases.json').read_text())
    real_rows, family_rows, link_rows, real_details, runtime_rows, distance_rows = [], [], [], [], [], []
    for spec in real_cases:
        print('Loading '+spec['name'], flush=True)
        with gzip.open(spec['archive'], 'rt') as stream:
            archived = json.load(stream)
        graph = json.loads((Path(spec['history'])/'graph.json').read_text())
        catalogs = []
        nodes = {(n['event'], n['group']): n for n in graph['nodes']}
        for i in range(spec['events']):
            with np.load(Path(spec['root'])/'online'/f'surfaces_{i:04d}.npz') as data, \
                    np.load(Path(spec['partitions'])/f'groups_{i:04d}.npz') as groups:
                catalogs.append(dict(positions=data['positions'].copy(), normals=data['normals'].copy(),
                                     labels=groups['labels'].copy()))
        check_graph(graph, [list(map(int, np.unique(c['labels'][c['labels'] >= 0]))) for c in catalogs],
                    spec['actual_times'], archived)
        results_by_method = {}
        for method, predict in METHODS:
            saved = support_sequence(catalogs, spec['actual_times'], archived, predict)
            lookup = {(r['parent_event'], r['gate_reference_cells']): r['result'] for r in archived}
            for record in saved:
                i, g, result = record['parent_event'], record['gate_reference_cells'], record['result']
                future = [lookup[i+1, gate] for gate in (1.5, 2.5)] if i+2 < len(catalogs) else None
                audit = check_window(catalogs[i:i+3], spec['actual_times'][i:i+3], lookup[i, g], future, result)
                audit_count += 1
                row = dict(case=spec['name'], parent_event=i, time=spec['actual_times'][i],
                           gate_reference_cells=g, method=method, **evidence_summary(result, lookup[i, g]),
                           physical_truth_available=False)
                support_parents = {r['parent_group'] for r in result['conditional_support']}
                row.update(support_parent_upstream_source_samples=sum(nodes[i, p]['upstream_inside_source'] for p in support_parents),
                           support_parent_downstream_source_samples=sum(nodes[i, p]['downstream_inside_source'] for p in support_parents))
                real_rows.append(row)
                for link in result['conditional_support']:
                    n = nodes[i, link['parent_group']]
                    link_rows.append(dict(case=spec['name'], method=method, parent_event=i,
                                          gate_reference_cells=g, **link,
                                          parent_samples=n['surface_samples'],
                                          upstream_source_overlaps=n['upstream_inside_source'],
                                          downstream_source_overlaps=n['downstream_inside_source']))
                if result['analysis']:
                    for family in result['analysis']['families']:
                        family_rows.append(dict(case=spec['name'], method=method, parent_event=i,
                                                gate_reference_cells=g, **family))
                real_details.append(dict(case=spec['name'], method=method, **record, independent_audit=audit))
            results_by_method[method] = saved
            print(f'{spec["name"]}: {method} audited across {len(saved)} interval/gate results.', flush=True)
        for i in range(spec['events']-1):
            for method, _ in METHODS:
                narrow, wide = [r['result'] for r in results_by_method[method] if r['parent_event'] == i]
                distance_rows.append(dict(case=spec['name'], method=method, parent_event=i,
                                          support_sets_identical=pairs(narrow) == pairs(wide),
                                          shared_support_links=len(pairs(narrow)&pairs(wide)),
                                          narrow_only=len(pairs(narrow)-pairs(wide)),
                                          wide_only=len(pairs(wide)-pairs(narrow))))
        # Paired CPU execution: same loaded inputs; file I/O and independent
        # checks are outside each timer. Warmup and three repeats are explicit.
        print(spec['name']+': one untimed warmup.', flush=True)
        warm = compute_adjacent(catalogs)
        assert [r['result'] for r in warm] == [r['result'] for r in archived]
        for _, predict in METHODS:
            support_sequence(catalogs, spec['actual_times'], warm, predict)
        for trial in range(3):
            start = time.perf_counter(); measured = compute_adjacent(catalogs)
            adjacent_seconds = time.perf_counter()-start
            assert [r['result'] for r in measured] == [r['result'] for r in archived]
            runtime_rows.append(dict(case=spec['name'], trial=trial, method='adjacent_max5',
                                     adjacent_seconds=adjacent_seconds, support_seconds=0.,
                                     combined_seconds=adjacent_seconds,
                                     max_adjacent_candidates=max(len(r['result']['candidates']) for r in measured)))
            order = METHODS if trial % 2 == 0 else METHODS[::-1]
            for method, predict in order:
                start = time.perf_counter()
                measured_support = support_sequence(catalogs, spec['actual_times'], measured, predict)
                support_seconds = time.perf_counter()-start
                assert measured_support == results_by_method[method]
                runtime_rows.append(dict(case=spec['name'], trial=trial, method=method,
                                         adjacent_seconds=adjacent_seconds, support_seconds=support_seconds,
                                         combined_seconds=adjacent_seconds+support_seconds,
                                         max_adjacent_candidates=max(len(r['result']['candidates']) for r in measured)))
            print(f'{spec["name"]}: measured repeat {trial+1}/3 complete ({adjacent_seconds:.3f}s adjacent).', flush=True)
    for filename, table in [('real_comparison.csv', real_rows), ('families.csv', family_rows),
                            ('support_links.csv', link_rows), ('runtime_trials.csv', runtime_rows),
                            ('distance_comparison.csv', distance_rows)]:
        write_csv(output/filename, table)
    with gzip.open(output/'real_results.json.gz', 'wt') as stream:
        json.dump(real_details, stream)
    runtime_summary = []
    for case in real_cases:
        for method in ['adjacent_max5']+[m for m, _ in METHODS]:
            selected = [r for r in runtime_rows if r['case'] == case['name'] and r['method'] == method]
            runtime_summary.append(dict(case=case['name'], method=method, repeats=len(selected),
                                        median_adjacent_seconds=float(np.median([r['adjacent_seconds'] for r in selected])),
                                        median_support_seconds=float(np.median([r['support_seconds'] for r in selected])),
                                        median_combined_seconds=float(np.median([r['combined_seconds'] for r in selected])),
                                        minimum_combined_seconds=min(r['combined_seconds'] for r in selected),
                                        maximum_combined_seconds=max(r['combined_seconds'] for r in selected)))
    write_csv(output/'runtime_summary.csv', runtime_summary)
    synthetic_summary = []
    for suite in ('adjacent_regression', 'three_state_development'):
        for identifiable in (True, False):
            for method, _ in METHODS:
                selected = [r for r in rows if (r['suite'], r['identifiable'], r['method']) == (suite, identifiable, method)]
                if not selected:
                    continue
                synthetic_summary.append(dict(suite=suite, identifiable=identifiable, method=method,
                                              configurations=len(selected),
                                              **{key: sum(r[key] for r in selected) for key in
                                                 ('baseline_false_links', 'baseline_missed_links',
                                                  'hypothetical_union_false_links', 'hypothetical_union_missed_links',
                                                  'additional_support_links')}))
    write_csv(output/'synthetic_summary.csv', synthetic_summary)
    real_summary = []
    for case in real_cases:
        for method, _ in METHODS:
            selected = [r for r in real_rows if (r['case'], r['method']) == (case['name'], method)]
            real_summary.append(dict(case=case['name'], method=method, interval_gate_results=len(selected),
                                     **{key: sum(r[key] for r in selected) for key in
                                        ('additional_support_links', 'conflicts', 'conditional_only_parents')},
                                     statuses=dict(Counter(r['window_status'] for r in selected))))
    plot_summary(rows, real_rows, output)
    for path, expected in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    sources = [Path(__file__)]+[Path('experiments/astrolink')/(name+'.py') for name in
               ('delayed_support', 'audit_delayed_support', 'tracklet_fragments',
                'event_tracking', 'expanded_event_tracking', 'freeze_multiframe_baseline')]
    report = dict(passed=True, frozen_baseline=str(baseline), synthetic_method_configurations=len(rows),
                  real_method_interval_gate_results=len(real_rows), independent_audits=audit_count,
                  synthetic_reversed_row_checks=shuffle_count, synthetic_summary=synthetic_summary,
                  real_summary=real_summary, runtime_summary=runtime_summary,
                  accepted_adjacent_events_exact=True, physical_arrays_unchanged=True,
                  physical_ancestry_established=False, inherited_ids_added=False,
                  future_seed_truth_used=False, maximum_future_event_latency=1,
                  terminal_windows_pending=True, candidate_settings_tuned=False,
                  cpu_runtime_scope='Common paired adjacent stage plus each support layer; one warmup and three repeats; no GPU scaling claim.',
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
                  input_sha256=hashes,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  output_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('passed', 'synthetic_summary', 'real_summary', 'runtime_summary')}, indent=2))


def plot_summary(synthetic, real, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    selected = [r for r in synthetic if r['suite'] == 'three_state_development' and r['identifiable']]
    labels = ['Adjacent', 'Static support', 'Motion support']
    for ax, metric, title in zip(axes[0], ['false_links', 'missed_links'],
                                  ['Hypothetical false links', 'Hypothetical missed links']):
        values = [sum(r['baseline_'+metric] for r in selected if r['method'] == METHODS[0][0])]
        values += [sum(r['hypothetical_union_'+metric] for r in selected if r['method'] == m) for m, _ in METHODS]
        bars = ax.bar(labels, values, color=['#89929b', '#d48b37', '#21868a'])
        ax.bar_label(bars); ax.set(title=title, ylabel='Sum over 15 identifiable development configurations')
        ax.set_ylim(0, max(values+[1])*1.3)
    for ax, name in zip(axes[1], dict.fromkeys(r['case'] for r in real)):
        for (method, _), color in zip(METHODS, ['#d48b37', '#21868a']):
            selected = [r for r in real if r['case'] == name and r['method'] == method]
            x = sorted({r['time'] for r in selected})
            values = [sum(r['additional_support_links'] for r in selected if r['time'] == t) for t in x]
            ax.plot(x, values, marker='o', markersize=3, label=method.replace('delayed_', '').replace('_', ' '), color=color)
        ax.set(title=name.replace('_', ' '), xlabel='Parent simulation time', ylabel='Additional support links (two gates summed)')
        ax.legend(fontsize=8)
    fig.suptitle('Delayed fragment support beside unchanged adjacent tracking')
    fig.text(.04, .025, 'Top: diagnostic only—what happens if conditional support is mistaken for ancestry. Identical-observation controls are reported separately.\n'
             'Bottom: added geometric evidence, not proven ancestors or unique shocks. Final support windows remain pending; no accepted event is changed.', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .95)); fig.savefig(output/'support_comparison.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('outputs/multiframe_tracking_baseline_v1'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); run(args.baseline, args.output_dir)
