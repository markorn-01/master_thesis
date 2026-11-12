"""Show fixed group-history labels on original saved 3D shock samples."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def segment_rows(labels, nodes):
    """Join catalog groups to history nodes; unretained samples stay -1."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError('Group labels must be one-dimensional.')
    groups = {int(x) for x in labels if x >= 0}
    if len({n['group'] for n in nodes}) != len(nodes) or {n['group'] for n in nodes} != groups:
        raise ValueError('History must represent each retained group exactly once.')
    segments = np.full(len(labels), -1, dtype=int)
    for node in nodes:
        mask = labels == node['group']
        if mask.sum() != node['surface_samples'] or node['segment_id'] < 0:
            raise ValueError('History and catalog group populations differ.')
        segments[mask] = node['segment_id']
    return segments


def cutaway_mask(positions, plane=.5):
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError('Require finite Nx3 positions.')
    return positions[:, 2] <= plane


def run(history, comparison, output, fps=2):
    if fps <= 0:
        raise ValueError('FPS must be positive.')
    graph_path = history/'graph.json'; validation = history/'verification.json'
    graph = json.loads(graph_path.read_text()); report = json.loads(validation.read_text())
    for key in ['input_sha256', 'source_sha256']:
        for path, digest in report[key].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Validated history input/source changed: {path}')
    meta_path = comparison/'metadata.json'; meta = json.loads(meta_path.read_text())
    root = Path(meta['target_directory']); partitions = Path(meta['comparison_directory'])
    times = json.loads((root/'metadata.json').read_text())['actual_times']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [graph_path, validation, meta_path]}
    frames, audit = [], []
    for index, time in enumerate(times):
        source = root/'online'/f'surfaces_{index:04d}.npz'
        partition = partitions/f'groups_n128_event{index:04d}.npz'
        for path in [source, partition]: hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(source) as data, np.load(partition) as grouped:
            positions = data['positions'].copy(); sources = data['source_positions'].copy()
            labels = grouped['native_cells_a60_labels'].copy()
            np.testing.assert_allclose(float(data['time']), time, atol=1e-7, rtol=0)
        nodes = [n for n in graph['nodes'] if n['event'] == index]
        if any(abs(n['time']-time) > 1e-7 for n in nodes) or len(positions) != len(labels):
            raise ValueError('Catalog/history event mismatch.')
        segments = segment_rows(labels, nodes); cut = cutaway_mask(positions)
        for node in nodes:
            # Independent group-index readback before using the per-point segment array.
            indices = np.flatnonzero(labels == node['group'])
            np.testing.assert_array_equal(np.flatnonzero(segments == node['segment_id']), indices)
            audit.append(dict(event=index, group=node['group'], segment_id=node['segment_id'],
                              full_points=len(indices), cutaway_points=int(cut[indices].sum()),
                              positions_sha256=hashlib.sha256(positions[indices].tobytes()).hexdigest()))
        frames.append(dict(event=index, time=float(time), positions=positions, sources=sources,
                           segments=segments, cutaway=cut, nodes=nodes))
    assert sum(len(f['nodes']) for f in frames) == len(graph['nodes'])
    output.mkdir(parents=True, exist_ok=False)
    render(graph, frames, output, fps)
    from PIL import Image
    with Image.open(output/'event_history_surfaces.gif') as image:
        decoded = image.n_frames; size = list(image.size)
        durations = []
        for i in range(decoded):
            image.seek(i); image.load(); durations.append(image.info.get('duration', 0))
    assert decoded == len(frames)
    for path, digest in hashes.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    result = dict(events=len(frames), decoded_frames=decoded, frame_size=size, fps=fps,
                  playback_seconds=sum(durations)/1000, group_observations=len(audit), groups=audit,
                  all_samples_shown_in_full_view=True, cutaway_plane='z <= 0.5',
                  unretained_samples_color='gray', positions_moved=False,
                  tracking_decisions_changed=False, physical_identity_established=False,
                  input_sha256=hashes, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['groups', 'input_sha256', 'source_sha256']}, indent=2))


def render(graph, frames, output, fps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import to_hex
    colors = [to_hex(c) for c in plt.get_cmap('tab20').colors] + ['#303030']
    # Emphasize the long histories; mute the outer shell so inner groups stand out.
    for segment, color in {0:'#2369ad', 1:'#2e8b57', 2:'#93a5b3', 17:'#c54838', 18:'#7e43ad'}.items():
        colors[segment] = color
    if graph['segment_count'] > len(colors):
        raise ValueError('This reviewed display supports at most 21 distinct segment colors.')
    lookup = {n['node_id']: n for n in graph['nodes']}
    fig = plt.figure(figsize=(18, 8))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], left=.01, right=.985,
                            bottom=.22, top=.83, wspace=.12)
    views = [fig.add_subplot(grid[i], projection='3d') for i in [0, 1]]
    ax = fig.add_subplot(grid[2])
    for event in graph['events']:
        parents = [lookup[p] for p in event['parents']]; children = [lookup[c] for c in event['children']]
        middle = (parents[0]['time']+children[0]['time'])/2
        level = np.mean([n['segment_id'] for n in parents+children])
        alternative = event['status'] != 'shared'
        for node in parents+children:
            ax.plot([node['time'], middle], [node['segment_id'], level],
                    color='#c65b30' if alternative else colors[node['segment_id']],
                    ls='--' if alternative else '-', lw=1, alpha=.7)
    for node in graph['nodes']:
        ax.scatter(node['time'], node['segment_id'], color=colors[node['segment_id']], s=15)
    ax.axvspan(.10, .11, color='#e9a382', alpha=.16)
    cursor = ax.axvline(0, color='black', lw=1.2)
    current = ax.scatter([], [], facecolors='none', edgecolors='black', s=80, zorder=5)
    ax.set(ylim=(graph['segment_count']-.4, -.6), xlim=(-.003, .163),
           xlabel='Simulation time', title='Same G labels in the history graph')
    ax.set_yticks(range(graph['segment_count']), [f'G{i:02d}' for i in range(graph['segment_count'])], fontsize=8)
    ax.set_xticks([0, .04, .08, .12, .16]); ax.grid(axis='x', alpha=.15)
    title = fig.suptitle('', fontsize=17, y=.96)
    note = fig.text(.025, .16, '', fontsize=10, va='top', family='monospace')
    fig.text(.5, .885, 'Fixed segment colors, camera and bounds | stars: wind sources | gray: samples outside retained groups',
             ha='center', fontsize=11)
    fig.text(.025, .035, 'Original refined positions; no smoothing or point trajectories. Cutaway hides z > 0.5 only for display.\n'
             'G labels follow shared continuations. Combinations start a new label; dashed links are distance-specific alternatives, not established ancestry.', fontsize=10)

    def update(index):
        frame = frames[index]; p = frame['positions']; segments = frame['segments']
        for view, cut in zip(views, [np.ones(len(p), dtype=bool), frame['cutaway']]):
            view.clear()
            gray = cut & (segments < 0)
            if gray.any(): view.scatter(*p[gray].T, c='#bcbcbc', s=2, alpha=.3, depthshade=False, linewidths=0)
            for node in frame['nodes']:
                selected = cut & (segments == node['segment_id'])
                view.scatter(*p[selected].T, c=colors[node['segment_id']], s=3, alpha=.8,
                             depthshade=False, linewidths=0)
            view.scatter(*frame['sources'].T, marker='*', c='black', s=40, depthshade=False)
            view.set(xlim=(.1, .9), ylim=(.1, .9), zlim=(.1, .9), xlabel='x', ylabel='y', zlabel='z', box_aspect=(1, 1, 1))
            view.view_init(elev=25, azim=-65)
        views[0].set_title('Full detected surfaces')
        views[1].set_title('Cutaway: inspect inner groups (z <= 0.5)')
        cursor.set_xdata([frame['time']]*2)
        current.set_offsets(np.array([[frame['time'], n['segment_id']] for n in frame['nodes']]).reshape(-1, 2))
        title.set_text(f"Two-bubble group histories on shock surfaces | t = {frame['time']:.2f}")
        entries = [f"G{n['segment_id']:02d} (group {n['group']:2d}): {n['surface_samples']:5,d}" for n in frame['nodes']]
        lines = ['   |   '.join(entries[i:i+5]) for i in range(0, len(entries), 5)]
        note.set_text('Current groups and full-view sample counts:\n'+'\n'.join(lines) if entries else 'No detected groups at this time.')
        return []

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=False)
    animation.save(output/'event_history_surfaces.gif', writer=PillowWriter(fps=fps), dpi=95)
    for index in [8, 11]:
        update(index); fig.savefig(output/f'event_{index:04d}.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, default=Path('outputs/two_bubbles_event_history_v1_n128'))
    parser.add_argument('--comparison', type=Path, default=Path('outputs/two_bubbles_temporal_comparison_n064_n128'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=2)
    args = parser.parse_args(); run(args.history, args.comparison, args.output_dir, args.fps)
