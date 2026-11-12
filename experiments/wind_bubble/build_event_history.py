"""Build a complete group-event graph from the checked two-distance archive."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.event_history import assemble_history
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write


def render(graph, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    nodes = graph['nodes']; lookup = {n['node_id']: n for n in nodes}
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = {'continuation': '#188477', 'combination': '#8754ae', 'separation': '#d37c18'}
    for event in graph['events']:
        parents = [lookup[p] for p in event['parents']]; children = [lookup[c] for c in event['children']]
        shared = event['status'] == 'shared'
        color = colors[event['kind']] if shared else '#c65b30'
        middle = (parents[0]['time'] + children[0]['time']) / 2
        level = np.mean([n['segment_id'] for n in parents + children])
        for n in parents + children:
            ax.plot([n['time'], middle], [n['segment_id'], level], color=color,
                    linestyle='-' if shared else '--', lw=1.6, alpha=.85, zorder=1)
        if event['kind'] != 'continuation':
            ax.scatter([middle], [level], marker='D', c=color, s=25, zorder=3)
    for node in nodes:
        ax.scatter(node['time'], node['segment_id'], s=48, c='#213d51', zorder=4)
        ax.annotate(str(node['group']), (node['time'], node['segment_id']),
                    xytext=(0, 6), textcoords='offset points', ha='center', fontsize=7)
    ax.axvspan(.10, .11, color='#f1cbb8', alpha=.3, zorder=0)
    ax.set(xlabel='Simulation time (code units)', ylabel='Continuation segment (graph labels)',
           title='Automatic two-bubble group history: all detected groups')
    ax.set_yticks(range(graph['segment_count']), [f'G{i:02d}' for i in range(graph['segment_count'])])
    ax.set_xticks(np.arange(0, .161, .01)); ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
    ax.legend(handles=[Line2D([], [], color=colors['continuation'], label='Shared continuation'),
                       Line2D([], [], color=colors['combination'], label='Shared combination'),
                       Line2D([], [], color='#c65b30', linestyle='--', label='Wide-distance-only combination')],
              loc='lower left', bbox_to_anchor=(0, 1.035), ncol=3, fontsize=9)
    fig.text(.07, .025, 'Node labels are catalog group numbers. Solid events agree at both tested distances. Dashed events are alternatives.\n'
             'Shaded interval: distance-sensitive parent subsets. Segment starts/ends do not prove physical births/deaths; G labels are not shock IDs.', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .96)); fig.savefig(output/'event_history.png', dpi=150); plt.close(fig)
    # Show each longer segment separately: no interpolation through branches or gaps.
    segments = [[n for n in nodes if n['segment_id'] == i] for i in range(graph['segment_count'])]
    selected = [s for s in segments if len(s) >= 3]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for rows in selected:
        for ax, key in zip(axes, ['median_mach', 'surface_samples']):
            ax.plot([r['time'] for r in rows], [r[key] for r in rows], marker='o', ms=3,
                    label=f"G{rows[0]['segment_id']:02d}")
    axes[0].set(ylabel='Median valid Mach', title='Statistics within continuation segments (at least three observations)')
    axes[1].set(ylabel='Surface sample count', xlabel='Simulation time (code units)')
    for ax in axes: ax.grid(alpha=.2); ax.legend(ncol=5, fontsize=8)
    fig.text(.08, .02, 'Count-weighted statistics; counts are not areas. Separate curves remain separate across combinations and uncertain intervals.', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(output/'segment_statistics.png', dpi=150); plt.close(fig)


def run(directory, comparison, output):
    verification = directory/'verification.json'; archive = directory/'results.json.gz'
    report = json.loads(verification.read_text())
    # Refuse to assemble from catalogs or tracker sources changed since validation.
    for key in ['input_sha256', 'source_sha256']:
        for path, digest in report[key].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Validated input/source changed: {path}')
    meta_path = comparison/'metadata.json'; meta = json.loads(meta_path.read_text())
    root = Path(meta['target_directory']); partitions = Path(meta['comparison_directory'])
    with gzip.open(archive, 'rt') as stream: records = json.load(stream)
    times = json.loads((root/'metadata.json').read_text())['actual_times']
    groups, statistics = [], {}
    hashes = {}
    for path in [verification, archive, meta_path, root/'metadata.json']:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for index, time in enumerate(times):
        source = root/'online'/f'surfaces_{index:04d}.npz'
        partition = partitions/f'groups_n128_event{index:04d}.npz'
        for path in [source, partition]: hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(source) as data, np.load(partition) as labels_file:
            labels = labels_file['native_cells_a60_labels']
            np.testing.assert_allclose(float(data['time']), time, atol=1e-7, rtol=0)
            groups.append(sorted(map(int, np.unique(labels[labels >= 0]))))
            for group in groups[-1]:
                mask = labels == group
                valid = mask & data['mach_valid'] & np.isfinite(data['mach']) & (data['mach'] > 0)
                statistics[index, group] = dict(surface_samples=int(mask.sum()), valid_mach_samples=int(valid.sum()),
                    median_mach=float(np.median(data['mach'][valid])) if valid.any() else None)
    graph = assemble_history(groups, times, records)
    for node in graph['nodes']: node.update(statistics[node['event'], node['group']])
    output.mkdir(parents=True, exist_ok=False)
    (output/'graph.json').write_text(json.dumps(graph, indent=2, allow_nan=False)+'\n')
    csv_write(output/'group_histories.csv', graph['nodes'])
    # Independent readback: every observation is present; edges advance exactly one event.
    saved = json.loads((output/'graph.json').read_text()); lookup = {n['node_id']: n for n in saved['nodes']}
    assert len(lookup) == sum(map(len, groups))
    for e in saved['events']:
        assert all(lookup[p]['event']+1 == lookup[c]['event'] for p in e['parents'] for c in e['children'])
        if e['status'] == 'shared' and e['kind'] == 'continuation':
            assert lookup[e['parents'][0]]['segment_id'] == lookup[e['children'][0]]['segment_id']
        else:
            assert not {lookup[p]['segment_id'] for p in e['parents']} & {lookup[c]['segment_id'] for c in e['children']}
    render(saved, output)
    for path, digest in hashes.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    counts = Counter((e['status'], e['kind']) for e in graph['events'])
    summary = dict(saved_events=len(times), group_observations=len(lookup), continuation_segments=graph['segment_count'],
                   event_counts={f'{status}_{kind}': value for (status, kind), value in sorted(counts.items())},
                   input_sha256=hashes, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [Path(__file__), Path('experiments/astrolink/event_history.py')]},
                   physical_identity_established=False, original_tracking_results_changed=False)
    (output/'verification.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if not k.endswith('sha256')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('outputs/two_bubbles_event_tracking_v1_n128'))
    parser.add_argument('--comparison', type=Path, default=Path('outputs/two_bubbles_temporal_comparison_n064_n128'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); run(args.directory, args.comparison, args.output_dir)
