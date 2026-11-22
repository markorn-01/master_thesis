"""Review two saved 64-cubed grouping transitions without rerunning tracking."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.wind_bubble.animate_event_history import segment_rows
from experiments.wind_bubble.streaming_products import compare_products, load_event


def run(root, output):
    if output.exists():
        raise ValueError('Use a fresh review directory.')
    metadata = json.loads((root/'metadata.json').read_text())
    if metadata['status'] != 'complete' or metadata['num_cells'] != 64 or len(metadata['actual_times']) != 17:
        raise ValueError('This bounded review requires the completed 64-cubed, 17-event run.')
    comparison = compare_products(root/'online/tracking', root/'offline_reference/tracking')
    hashes = {}

    def remember(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    for path in (root/'metadata.json', root/'equivalence.json',
                 root/'online/tracking/metadata.json', root/'online/tracking/finish.json'):
        remember(path)
    frames, nodes, events, records = [], [], [], []
    for index, time in enumerate(metadata['actual_times']):
        catalog_path = root/'online'/f'surfaces_{index:04d}.npz'
        groups_path = root/'online/tracking'/f'groups_{index:04d}.npz'
        record_path = root/'online/tracking'/f'event_{index:04d}.json.gz'
        for path in (catalog_path, groups_path, record_path):
            remember(path)
        record = load_event(root/'online/tracking', index)
        with np.load(catalog_path) as catalog, np.load(groups_path) as groups:
            positions, sources = catalog['positions'].copy(), catalog['source_positions'].copy()
            labels = groups['labels'].copy()
            np.testing.assert_allclose(float(catalog['time']), time, rtol=0, atol=1e-7)
        np.testing.assert_allclose(record['time'], time, rtol=0, atol=1e-7)
        segments = segment_rows(labels, record['nodes'])
        if sum(n['surface_samples'] for n in record['nodes']) != int((labels >= 0).sum()):
            raise ValueError('Group counts do not account for the retained source samples.')
        frames.append(dict(time=time, positions=positions, sources=sources, segments=segments,
                           nodes=record['nodes'], unretained_samples=int((labels < 0).sum())))
        nodes.extend(record['nodes']); events.extend(record['tracking_events']); records.append(record)
    lookup = {node['node_id']: node for node in nodes}
    combinations = [e for e in events if e['kind'] == 'combination' and e['status'] == 'shared']
    transitions = []
    for event in combinations:
        transitions.append(dict(parent_event=event['parent_event'], child_event=event['child_event'],
            parent_segments=[lookup[n]['segment_id'] for n in event['parents']],
            child_segments=[lookup[n]['segment_id'] for n in event['children']],
            parent_counts=[lookup[n]['surface_samples'] for n in event['parents']],
            child_counts=[lookup[n]['surface_samples'] for n in event['children']]))
    # Captions below describe this exact saved case. Refuse silently different data.
    expected = [(7, 8, [0, 1], [2]), (13, 14, list(range(3, 8)), [13]),
                (13, 14, list(range(8, 13)), [14])]
    if [(e['parent_event'], e['child_event'], e['parent_segments'], e['child_segments'])
            for e in transitions] != expected:
        raise ValueError('Transition structure differs from this bounded review.')
    output.mkdir(parents=True)
    displayed = render(frames, nodes, events, output)
    for name, expected_hash in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Input changed during rendering.')
    counts = Counter((e['status'], e['kind']) for e in events)
    report = dict(passed=True, source_directory=str(root), events=len(frames),
                  group_observations=len(nodes), geometric_event_counts={f'{s}_{k}': n for (s,k),n in counts.items()},
                  comparison=comparison, transitions=transitions, displayed_frames=displayed,
                  view='completed saved history, full surfaces then inner-group zoom',
                  positions_moved=False, tracking_decisions_changed=False,
                  conditional_support_drawn=False, physical_ancestry_established=False,
                  input_sha256=hashes, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__), Path('experiments/wind_bubble/animate_event_history.py'),
                              Path('experiments/wind_bubble/streaming_products.py'))},
                  output_sha256={str(output/'tracking_review.png'): hashlib.sha256((output/'tracking_review.png').read_bytes()).hexdigest()})
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','events','group_observations','geometric_event_counts')}, indent=2))


def render(frames, nodes, events, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    colors = list(plt.get_cmap('tab20').colors)
    colors[0], colors[1], colors[2] = '#2466a8', '#25805b', '#7b8c9c'
    fig = plt.figure(figsize=(15.5, 12.5), dpi=130)
    grid = fig.add_gridspec(3, 3, left=.05, right=.96, bottom=.145, top=.88,
                          height_ratios=[1, 1, 1.25], width_ratios=[1, 1, .83], hspace=.30, wspace=.18)
    fig.suptitle('Online tracking: what the saved group relationships describe', x=.05, ha='left', y=.97, fontsize=19)
    fig.text(.05, .925, 'Two equal wind sources | 64\u00b3 grid | 17 analysis events | G labels and colors match the history below', fontsize=11)
    displayed = []
    for row, indices in enumerate(((7, 8), (13, 14))):
        for column, index in enumerate(indices):
            frame = frames[index]; p = frame['positions']; segments = frame['segments']
            ax = fig.add_subplot(grid[row, column], projection='3d')
            lower = np.array([.12, .26, .26]) if row == 0 else np.array([.24, .40, .40])
            upper = np.array([.88, .74, .74]) if row == 0 else np.array([.76, .60, .60])
            shown = ((p >= lower) & (p <= upper)).all(axis=1)
            if row == 1:
                shown &= segments != 2
            audit = dict(event=index, time=frame['time'], original_samples=len(p),
                         displayed_samples=int(shown.sum()), displayed_group_counts={})
            unretained = shown & (segments < 0)
            ax.scatter(*p[unretained].T, c='#bbbbbb', s=5, depthshade=False)
            for node in frame['nodes']:
                sid = node['segment_id']; selected = shown & (segments == sid)
                audit['displayed_group_counts'][str(sid)] = int(selected.sum())
                ax.scatter(*p[selected].T, c=[colors[sid]], s=5 if row == 0 else 15,
                           alpha=.72 if row == 0 else .9, depthshade=False, linewidths=0)
            ax.scatter(*frame['sources'].T, marker='*', c='black', s=80, depthshade=False)
            ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]),
                   xlabel='x', ylabel='y', zlabel='z', box_aspect=upper-lower)
            ax.view_init(elev=25, azim=-65)
            ax.tick_params(labelsize=7, pad=0)
            ax.set_xticks([.2, .5, .8] if row == 0 else [.3, .5, .7])
            for setter in (ax.set_yticks, ax.set_zticks):
                setter([.3, .5, .7] if row == 0 else [.4, .5, .6])
            titles = {7:'A: t = 0.07 | two groups', 8:'B: t = 0.08 | one group',
                      13:'C: t = 0.13 | ten small groups', 14:'D: t = 0.14 | two larger groups'}
            ax.set_title(titles[index], loc='left', fontsize=12, pad=10)
            if row == 1:
                ax.text2D(.02, -.05, 'Inner groups only; G02 omitted; spatial zoom', transform=ax.transAxes, fontsize=9, color='#555555')
            displayed.append(audit)
        text = fig.add_subplot(grid[row, 2]); text.axis('off')
        if row == 0:
            text.text(0, .86, 'First grouping transition', fontsize=13, weight='bold')
            text.text(0, .65, 'G00 + G01  →  G02', fontsize=15)
            text.text(0, .46, 'At t = 0.07:\n1,344 samples in each group.\n\nAt t = 0.08:\n3,096 samples in one group.', fontsize=11, va='top', linespacing=1.5)
        else:
            text.text(0, .90, 'Later fragment combinations', fontsize=13, weight='bold')
            text.text(0, .74, 'G03–G07  →  G13\nG08–G12  →  G14', fontsize=14, linespacing=1.6, va='top')
            text.text(0, .38, 'Ten groups: 20 samples each.\nTwo later groups: 152 each.\n\nG02 continues separately\nthrough this transition.', fontsize=11, va='top', linespacing=1.5)

    ax = fig.add_subplot(grid[2, :]); lookup = {n['node_id']: n for n in nodes}
    for event in events:
        endpoints = [lookup[n] for n in event['parents']+event['children']]
        middle = np.mean([endpoints[0]['time'], endpoints[-1]['time']])
        level = np.mean([n['segment_id'] for n in endpoints])
        for node in endpoints:
            ax.plot([node['time'], middle], [node['segment_id'], level],
                    color=colors[node['segment_id']], lw=1.5, alpha=.8,
                    ls='-' if event['status'] == 'shared' else ':')
    for node in nodes:
        ax.scatter(node['time'], node['segment_id'], color=colors[node['segment_id']], s=34, zorder=4)
    for left, right in ((.07, .08), (.13, .14)):
        ax.axvspan(left, right, color='#cedde8', alpha=.28, zorder=0)
    ax.set(title='Complete saved history | dots: group observations; joins: accepted geometric associations',
           xlabel='Simulation time (code units)', xlim=(-.003, .164), ylim=(14.7, -.7))
    ax.set_yticks(range(15), [f'G{s:02d}' for s in range(15)], fontsize=9)
    ax.set_xticks(np.arange(0, .161, .01)); ax.grid(axis='x', alpha=.22)
    fig.text(.05, .075, 'Stars mark wind sources. Positions are the original refined surface samples; this review adds no smoothing.\n'
             'G labels describe geometric continuation segments. A combination starts a new label; physical ancestry is not established.\n'
             'Sample counts are detections at each time and need not sum across times. Conditional support is not drawn in this overview.\n'
             'Light-gray points have no retained group label.',
             fontsize=9, linespacing=1.5, va='top')
    target = output/'tracking_review.png'
    fig.savefig(target, dpi=150)
    plt.close(fig)
    with Image.open(target) as image:
        image.load()
    return displayed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('outputs/online_streaming_iwr_compgpu14_n064'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.directory, args.output_dir)
