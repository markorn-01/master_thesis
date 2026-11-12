"""Apply the fixed tracker at 64³ and compare descriptive histories with 128³."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.event_tracking import event_tracking
from experiments.astrolink.event_history import assemble_history
from experiments.wind_bubble.apply_event_tracking import check_candidates, check_selection_dp, event_key
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write


def group_statistics(data, labels, nodes):
    """Describe geometry relative to nearest source, without assigning ancestry."""
    positions = data['positions']; normals = data['normals'].astype(float)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    radial = positions[:, None, :] - data['source_positions'][None, :, :]
    distances = np.linalg.norm(radial, axis=2)
    nearest = np.argmin(distances, axis=1)
    vector = radial[np.arange(len(positions)), nearest]
    radius = distances[np.arange(len(positions)), nearest]
    alignment = np.divide(np.einsum('ij,ij->i', vector, normals), radius,
                          out=np.full(len(radius), np.nan), where=radius > 0)
    result = []
    if {n['group'] for n in nodes} != set(map(int, np.unique(labels[labels >= 0]))):
        raise ValueError('Graph/catalog groups differ.')
    for node in nodes:
        mask = labels == node['group']
        good = mask & data['mach_valid'] & np.isfinite(data['mach']) & (data['mach'] > 0)
        a = alignment[mask]; a = a[np.isfinite(a)]
        result.append(dict(node, surface_samples=int(mask.sum()),
            valid_mach_samples=int(good.sum()),
            median_mach=float(np.median(data['mach'][good])) if good.any() else None,
            median_nearest_source_radius=float(np.median(radius[mask])),
            median_nearest_source_alignment=float(np.median(a)) if len(a) else None))
    return result


def checked_hashes(report):
    for key in ['input_sha256', 'source_sha256']:
        for path, digest in report[key].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Validated source/input changed: {path}')


def run(comparison, controls, high_history, output):
    meta_path = comparison/'metadata.json'; meta = json.loads(meta_path.read_text())
    control_path = controls/'verification.json'; control = json.loads(control_path.read_text())
    # Match the tested tracker exactly. No changes to coverage, arity or distances.
    for path, digest in control['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Controlled source changed: {path}')
    high_report_path = high_history/'verification.json'
    checked_hashes(json.loads(high_report_path.read_text()))
    high_graph_path = high_history/'graph.json'; high_graph = json.loads(high_graph_path.read_text())
    roots = {64: Path(meta['reference_directory']), 128: Path(meta['target_directory'])}
    metadata = {n: json.loads((root/'metadata.json').read_text()) for n, root in roots.items()}
    fields = ['source_positions', 'injection_radius', 'ambient_density', 'ambient_pressure',
              'gamma', 'wind_terminal_velocity', 't_end', 'domain', 'units',
              'discrete_mass_rates', 'discrete_energy_rates']
    for key in fields:
        if metadata[64][key] != metadata[128][key]: raise ValueError(f'Setups differ: {key}')
    times = metadata[64]['actual_times']
    np.testing.assert_allclose(times, metadata[128]['actual_times'], atol=1e-7, rtol=0)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [meta_path, control_path, high_report_path, high_graph_path]}
    catalogs, labels = {}, {}
    for n, root in roots.items():
        catalogs[n], labels[n] = [], []
        path = root/'metadata.json'; hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for index, time in enumerate(times):
            source = root/'online'/f'surfaces_{index:04d}.npz'
            partition = Path(meta['comparison_directory'])/f'groups_n{n:03d}_event{index:04d}.npz'
            for path in [source, partition]: hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(source) as data:
                catalogs[n].append({key:data[key].copy() for key in
                    ['positions', 'normals', 'mach', 'mach_valid', 'source_positions', 'time']})
            with np.load(partition) as data: labels[n].append(data['native_cells_a60_labels'].copy())
            np.testing.assert_allclose(float(catalogs[n][-1]['time']), time, atol=1e-7, rtol=0)
    output.mkdir(parents=True, exist_ok=False)
    records, checks, agreement = [], [], []
    for index in range(len(times)-1):
        a, b = catalogs[64][index:index+2]; selected = []
        for gate in [1.5, 2.5]:
            result = event_tracking(a['positions'], a['normals'], labels[64][index],
                b['positions'], b['normals'], labels[64][index+1], gate/64)
            for key, value in control['parameters'].items(): assert result['parameters'][key] == value
            check = check_candidates(a, b, labels[64][index], labels[64][index+1], result)
            check_selection_dp(result)
            checks.append(dict(parent_event=index, gate_reference_cells=gate, independent_dp_check=True, **check))
            records.append(dict(parent_event=index, child_event=index+1,
                parent_time=times[index], child_time=times[index+1], gate_reference_cells=gate, result=result))
            selected.append({event_key(e) for e in result['events']})
            print(f'64³ {index:02d}→{index+1:02d} gate {gate}: '
                  f'{dict(Counter(e[0] for e in selected[-1]))}', flush=True)
        agreement.append(dict(parent_event=index, event_sets_identical=selected[0] == selected[1]))
    groups = [sorted(map(int, np.unique(l[l >= 0]))) for l in labels[64]]
    low_graph = assemble_history(groups, times, records)
    graphs = {64:low_graph, 128:high_graph}; statistics, summaries, description = [], [], []
    for n, graph in graphs.items():
        for index, time in enumerate(times):
            nodes = [node for node in graph['nodes'] if node['event'] == index]
            rows = group_statistics(catalogs[n][index], labels[n][index], nodes)
            statistics.extend(dict(num_cells=n, **row) for row in rows)
            largest = max(rows, key=lambda r:r['surface_samples']) if rows else None
            inward = [r for r in rows if r['median_nearest_source_alignment'] is not None
                       and r['median_nearest_source_alignment'] < 0]
            summaries.append(dict(num_cells=n, event=index, time=time, retained_groups=len(rows),
                retained_samples=sum(r['surface_samples'] for r in rows),
                largest_group_fraction=largest['surface_samples']/sum(r['surface_samples'] for r in rows) if rows else None,
                inward_oriented_groups=len(inward), largest_group_median_mach=largest['median_mach'] if largest else None))
        branches = [e for e in graph['events'] if e['status'] == 'shared' and e['kind'] != 'continuation']
        first_inward = next((r['time'] for r in summaries if r['num_cells'] == n and r['inward_oriented_groups']), None)
        description.append(dict(num_cells=n, observations=len(graph['nodes']), segments=graph['segment_count'],
            events=dict(Counter(f"{e['status']}_{e['kind']}" for e in graph['events'])),
            shared_branch_intervals=[dict(kind=e['kind'], parent_time=times[e['parent_event']],
                child_time=times[e['child_event']]) for e in branches],
            first_inward_oriented_group_time=first_inward))
    with gzip.open(output/'n064_results.json.gz', 'wt') as stream: json.dump(records, stream)
    selected_events = [dict(parent_event=r['parent_event'], child_event=r['child_event'],
        gate_reference_cells=r['gate_reference_cells'], kind=e['kind'],
        parents=','.join(map(str,e['parents'])), children=','.join(map(str,e['children'])),
        exclusion_gap=r['result']['selection']['exclusion_gaps'][str(e['candidate_index'])])
        for r in records for e in r['result']['events']]
    csv_write(output/'n064_selected_events.csv', selected_events)
    (output/'n064_graph.json').write_text(json.dumps(low_graph, indent=2)+'\n')
    csv_write(output/'group_statistics.csv', statistics); csv_write(output/'event_summary.csv', summaries)
    csv_write(output/'n064_distance_agreement.csv', agreement)
    render(graphs, times, summaries, output)
    # Saved JSON contains all low-resolution observations and exactly the shared events.
    saved = json.loads((output/'n064_graph.json').read_text())
    assert len(saved['nodes']) == sum(len(g) for g in groups)
    for index in range(len(times)-1):
        results = [r['result'] for r in records if r['parent_event'] == index]
        common = {event_key(e) for e in results[0]['events']} & {event_key(e) for e in results[1]['events']}
        assert sum(e['status'] == 'shared' and e['parent_event'] == index for e in saved['events']) == len(common)
    for path, digest in hashes.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    report = dict(resolutions=description, setup_fields_matched=fields, checks=checks,
        n064_identical_event_sets_across_distances=sum(r['event_sets_identical'] for r in agreement),
        fixed_tracking_parameters=control['parameters'], gates_code_units=[1.5/64, 2.5/64],
        grouping='native radius 1.5 cells; signed-normal threshold 60 degrees; minimum 20 samples',
        grouping_physical_scale_identical=False, physical_convergence_established=False,
        physical_ancestry_truth_available=False, input_sha256=hashes,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path('experiments/astrolink/event_tracking.py'), Path('experiments/astrolink/event_history.py'),
             Path('experiments/wind_bubble/apply_event_tracking.py')]})
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(description, indent=2))


def render(graphs, times, rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for n, color in [(64, '#23818d'), (128, '#ca6c29')]:
        selected = [r for r in rows if r['num_cells'] == n]
        for ax, key, title in zip(axes.flat,
                ['retained_groups', 'inward_oriented_groups', 'largest_group_fraction', 'largest_group_median_mach'],
                ['Retained group count', 'Groups with median inward orientation', 'Largest group / retained samples', 'Median Mach in the largest group']):
            ax.plot(times, [np.nan if r[key] is None else r[key] for r in selected], 'o-', color=color, label=f'{n}³', ms=4)
            ax.set_title(title); ax.grid(alpha=.2); ax.legend()
    for ax in axes[1]: ax.set_xlabel('Simulation time (code units)')
    fig.suptitle('Fixed event tracker: resolution sensitivity of detected group histories')
    fig.text(.05, .02, 'Inward orientation: median normal dot nearest-source radial direction < 0 (descriptive, not an ancestry label).\n'
             'Same physical matching distances; grouping radius remains 1.5 native cells. Differences include detection and grouping sensitivity.', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .95)); fig.savefig(output/'resolution_summary.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(13, 11), sharex=True)
    for ax, (n, graph) in zip(axes, graphs.items()):
        lookup = {node['node_id']:node for node in graph['nodes']}
        for e in graph['events']:
            parents = [lookup[p] for p in e['parents']]; children = [lookup[c] for c in e['children']]
            x = (parents[0]['time']+children[0]['time'])/2
            y = np.mean([node['segment_id'] for node in parents+children])
            for node in parents+children:
                ax.plot([node['time'], x], [node['segment_id'], y],
                    ls='-' if e['status'] == 'shared' else '--',
                    color='#c65b30' if e['status'] != 'shared' else '#8754ae' if e['kind'] != 'continuation' else '#23818d')
        ax.scatter([node['time'] for node in graph['nodes']], [node['segment_id'] for node in graph['nodes']], s=22, c='#213d51', zorder=3)
        ax.set(title=f'{n}³: {len(graph["nodes"])} group observations; {graph["segment_count"]} continuation segments',
               ylabel='Local segment label', ylim=(graph['segment_count']-.3, -.8))
        ax.set_yticks(range(graph['segment_count'])); ax.grid(axis='x', alpha=.2)
    axes[-1].set_xlabel('Simulation time (code units)')
    fig.suptitle('Shared continuations and branches; distance-specific events dashed')
    fig.text(.05, .025, 'Labels are independent between resolutions. Purple: shared branch; green: shared continuation; dashed orange: distance-specific alternative.\n'
             'Starts/ends are observations, not proven physical births/deaths. Event agreement is not physical convergence.', fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .96)); fig.savefig(output/'resolution_histories.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, default=Path('outputs/two_bubbles_temporal_comparison_n064_n128'))
    parser.add_argument('--controls', type=Path, default=Path('outputs/event_tracking_controls_solver_validation'))
    parser.add_argument('--high-history', type=Path, default=Path('outputs/two_bubbles_event_history_v1_n128'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); run(args.comparison, args.controls, args.high_history, args.output_dir)
