"""Animate saved surface samples beside two verified geometric histories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.wind_bubble.animate_event_history import cutaway_mask, segment_rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_report(path, hashes):
    report = json.loads(path.read_text())
    if report.get('passed') is not True:
        raise ValueError(f'Require a passed verification: {path}')
    hashes[str(path)] = digest(path)
    for section in ['input_sha256', 'source_sha256']:
        for name, expected in report[section].items():
            if digest(name) != expected:
                raise ValueError(f'Validated input/source changed: {name}')
            hashes[name] = expected
    return report


def paired_segments(labels, old_nodes, new_nodes):
    """Reject a misleading common color map when segment assignments differ."""
    old = segment_rows(labels, old_nodes)
    new = segment_rows(labels, new_nodes)
    if not np.array_equal(old, new):
        raise ValueError('The two histories assign different segment colors; use separate views.')
    return new


def load_frames(baseline, candidate, comparison, plane):
    hashes = {}
    for directory in [baseline, candidate]:
        checked_report(directory/'verification.json', hashes)
    report = checked_report(comparison/'verification.json', hashes)
    graphs = [json.loads((d/'graph.json').read_text()) for d in [baseline, candidate]]
    metas = [json.loads((d/'metadata.json').read_text()) for d in [baseline, candidate]]
    if any(m['status'] != 'complete' for m in metas):
        raise ValueError('Require completed tracking histories.')
    if Path(metas[0]['input_directory']).resolve() != Path(metas[1]['input_directory']).resolve():
        raise ValueError('Histories must use the same simulation.')
    # The comparison must refer to these exact graphs, not another pair of modes.
    for directory in [baseline, candidate]:
        for name in ['graph.json', 'metadata.json', 'verification.json']:
            path = directory/name
            if report['input_sha256'].get(str(path)) != digest(path):
                raise ValueError(f'Comparison does not verify this history: {path}')
    root = Path(metas[0]['input_directory'])
    metadata_path = root/'metadata.json'
    hashes[str(metadata_path)] = digest(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    times = np.asarray(metadata['actual_times'])
    if len(times) < 2 or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError('Require finite increasing event times.')
    if any(m['events'] != len(times) for m in metas):
        raise ValueError('History and simulation event counts differ.')
    changes_path = comparison/'changed_events.json'
    hashes[str(changes_path)] = digest(changes_path)
    changes = json.loads(changes_path.read_text())
    frames = []
    for index, time in enumerate(times):
        source = root/'online'/f'surfaces_{index:04d}.npz'
        group_paths = [d/f'groups_{index:04d}.npz' for d in [baseline, candidate]]
        for path in [source, *group_paths]:
            if str(path) in hashes and digest(path) != hashes[str(path)]:
                raise ValueError(f'Catalog/partition changed: {path}')
            hashes[str(path)] = digest(path)
        with np.load(source) as data:
            positions = data['positions'].copy()
            sources = data['source_positions'].copy()
            if not np.isclose(float(data['time']), time, atol=1e-7, rtol=0):
                raise ValueError('Catalog and recorded event time differ.')
            overlaps = {side: int(data[f'{side}_inside_any_source'].sum())
                        for side in ['upstream', 'downstream']}
        with np.load(group_paths[0]) as old, np.load(group_paths[1]) as new:
            np.testing.assert_array_equal(old['labels'], new['labels'])
            labels = new['labels'].copy()
        if len(labels) != len(positions):
            raise ValueError('Catalog and partition populations differ.')
        nodes = [[n for n in g['nodes'] if n['event'] == index] for g in graphs]
        if any(abs(n['time']-time) > 1e-7 for ns in nodes for n in ns):
            raise ValueError('History and catalog times differ.')
        segments = paired_segments(labels, *nodes)
        cut = cutaway_mask(positions, plane)
        # Check the display join against original group indices independently.
        groups = []
        for node in nodes[1]:
            selected = np.flatnonzero(labels == node['group'])
            np.testing.assert_array_equal(selected, np.flatnonzero(segments == node['segment_id']))
            groups.append(dict(group=node['group'], segment=node['segment_id'], samples=len(selected),
                               cutaway_samples=int(cut[selected].sum()),
                               positions_sha256=hashlib.sha256(positions[selected].tobytes()).hexdigest()))
        frames.append(dict(event=index, time=float(time), positions=positions, sources=sources,
                           segments=segments, cutaway=cut, nodes=nodes[1], overlaps=overlaps,
                           groups=groups))
    if any(sum(len(f['nodes']) for f in frames) != len(g['nodes']) for g in graphs):
        raise ValueError('History contains unmatched observations.')
    return graphs, frames, changes, metas, hashes


def render(graphs, frames, changes, metas, output, fps, plane):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D

    segment_count = max(g['segment_count'] for g in graphs)
    if segment_count > 20:
        raise ValueError('This color display supports at most 20 continuation segments.')
    colors = list(plt.get_cmap('tab20').colors)
    # Muted blue-gray for large outward groups; inner fragments keep distinct colors.
    outward = {n['segment_id'] for n in graphs[1]['nodes']
               if n['median_nearest_source_alignment'] > .8}
    for segment in outward:
        colors[segment] = '#718c9e'
    # Preserve distinct initial outer groups in the graph and scene.
    for offset, segment in enumerate(sorted(outward)):
        colors[segment] = ['#526d82', '#7d9c9f', '#93a8be', '#67778f', '#a0b0b8'][offset % 5]
    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], left=.055, right=.97,
                           bottom=.17, top=.89, wspace=.22, hspace=.22)
    views = [fig.add_subplot(grid[0, i], projection='3d') for i in [0, 1]]
    axes = [fig.add_subplot(grid[1, i]) for i in [0, 1]]
    cursors, rings = [], []
    times = np.array([f['time'] for f in frames]); pad = np.ptp(times)*.025
    all_positions = np.concatenate([f['positions'] for f in frames]+[frames[0]['sources']])
    low = all_positions.min(axis=0)-.04
    high = all_positions.max(axis=0)+.04
    for ax, graph, meta in zip(axes, graphs, metas):
        lookup = {n['node_id']: n for n in graph['nodes']}
        for event in graph['events']:
            nodes = [lookup[n] for n in event['parents']+event['children']]
            middle = np.mean([lookup[event['parents'][0]]['time'], lookup[event['children'][0]]['time']])
            level = np.mean([n['segment_id'] for n in nodes])
            alternative = event['status'] != 'shared'
            color = '#be651e' if alternative else '#8e438f' if event['kind'] != 'continuation' else '#426c78'
            for node in nodes:
                ax.plot([node['time'], middle], [node['segment_id'], level], color=color,
                        ls='--' if alternative else '-', lw=1.4)
        for node in graph['nodes']:
            ax.scatter(node['time'], node['segment_id'], s=21, color=colors[node['segment_id']], zorder=3)
        for start, end in sorted({(c['parent_time'], c['child_time']) for c in changes}):
            ax.axvspan(start, end, color='#e4b841', alpha=.17)
        ax.set(xlim=(times[0]-pad, times[-1]+pad), ylim=(segment_count-.5, -.8),
               xlabel='Simulation time', title=f'Maximum {meta["settings"]["max_partners"]} partners')
        ax.set_yticks(range(segment_count), [f'G{s:02d}' for s in range(segment_count)], fontsize=8)
        ax.grid(axis='x', alpha=.2)
        cursors.append(ax.axvline(times[0], c='black', lw=1))
        rings.append(ax.scatter([], [], s=95, facecolors='none', edgecolors='black', zorder=5))
    title = fig.suptitle('', y=.97, fontsize=17)
    fig.text(.5, .925, 'Same original surface positions and group colors in both histories | stars: wind sources',
             ha='center', fontsize=11)
    note = fig.text(.055, .12, '', va='top', fontsize=10)
    handles = [Line2D([], [], c='#426c78', label='Shared continuation'),
               Line2D([], [], c='#8e438f', label='Shared combination'),
               Line2D([], [], c='#be651e', ls='--', label='Distance-dependent alternative'),
               Line2D([], [], marker='o', ls='', c='#bcbcbc', label='Outside retained groups')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .045), ncol=4, frameon=False)
    fig.text(.055, .025, 'Graphs show the full saved interval; black cursors/rings mark the current event. Yellow shading: changed association.\n'
             'G labels identify geometric continuation segments, not gas particles. Cutaway is display only; no smoothing or point trajectories.', fontsize=9)
    change_notes = []; seen = set()
    for change in changes:
        if change['parent_event'] in seen:
            continue
        seen.add(change['parent_event'])
        old = [len(e['parents']) for e in change['removed'] if e['kind'] == 'combination']
        new = [len(e['parents']) for e in change['added'] if e['kind'] == 'combination']
        change_notes.append(f'{change["parent_time"]:.2f} to {change["child_time"]:.2f}: '
                            f'combination parent counts {old} become {new}.')
    alternatives = sum(e['status'] != 'shared' for e in graphs[1]['events'])
    change_note = ' '.join(change_notes)+f' Distance-dependent alternatives: {alternatives}.'

    def update(index):
        frame = frames[index]; p = frame['positions']; segments = frame['segments']
        for ax, mask, name in zip(views, [np.ones(len(p), bool), frame['cutaway']],
                                  ['Full detected surfaces', f'Cutaway: z <= {plane:g}']):
            ax.clear()
            for segment in np.unique(segments):
                selected = mask & (segments == segment)
                ax.scatter(*p[selected].T, s=3 if segment in outward else 10,
                           color=colors[segment] if segment >= 0 else '#bcbcbc',
                           alpha=.6 if segment in outward or segment < 0 else 1,
                           depthshade=False, linewidths=0)
            ax.scatter(*frame['sources'].T, c='black', marker='*', s=65, depthshade=False)
            for source_index, source in enumerate(frame['sources']):
                ax.text(*source, f' S{source_index+1}', fontsize=8)
            ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
                   xlabel='x', ylabel='y', zlabel='z', box_aspect=high-low, title=name)
            ax.view_init(elev=25, azim=-35)
        for cursor, ring in zip(cursors, rings):
            cursor.set_xdata([frame['time']]*2)
            ring.set_offsets(np.array([[frame['time'], n['segment_id']] for n in frame['nodes']]).reshape(-1, 2))
        title.set_text(f'{len(frame["sources"])}-source shock surfaces and geometric histories | t = {frame["time"]:.2f}')
        note.set_text(f'{len(p):,} samples; {len(frame["nodes"])} retained groups; {(segments < 0).sum():,} samples outside retained groups. '
                      f'Source-overlap flags: upstream {frame["overlaps"]["upstream"]}, downstream {frame["overlaps"]["downstream"]}.\n'
                      +change_note)

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=False)
    animation.save(output/'tracked_surfaces.gif', writer=PillowWriter(fps=fps), dpi=95)
    selected = {len(frames)-1}
    for c in changes:
        selected.update([c['parent_event'], c['parent_event']+1])
    for index in sorted(selected):
        update(index)
        fig.savefig(output/f'event_{index:04d}.png', dpi=140)
    plt.close(fig)
    render_details(graphs, frames, changes, colors, output)


def render_details(graphs, frames, changes, colors, output):
    """Zoom into changed combinations, using a fixed box in both states."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    seen = set()
    for change in changes:
        index = change['parent_event']
        if index in seen:
            continue
        seen.add(index)
        for added in change['added']:
            if added['kind'] != 'combination':
                continue
            parent, child = frames[index:index+2]
            pnodes = [n for n in parent['nodes'] if n['group'] in added['parents']]
            cnodes = [n for n in child['nodes'] if n['group'] in added['children']]
            involved = [parent['positions'][np.isin(parent['segments'], [n['segment_id'] for n in pnodes])],
                        child['positions'][np.isin(child['segments'], [n['segment_id'] for n in cnodes])]]
            points = np.concatenate(involved)
            lo, hi = points.min(axis=0), points.max(axis=0)
            margin = np.maximum((hi-lo)*.15, .005)
            lo -= margin; hi += margin
            fig = plt.figure(figsize=(13, 7))
            axes = [fig.add_subplot(1, 2, i+1, projection='3d') for i in range(2)]
            handles = {}
            for ax, frame in zip(axes, [parent, child]):
                p = frame['positions']
                inside = ((p >= lo) & (p <= hi)).all(axis=1)
                for segment in np.unique(frame['segments'][inside]):
                    mask = inside & (frame['segments'] == segment)
                    color = colors[segment] if segment >= 0 else '#bcbcbc'
                    ax.scatter(*p[mask].T, c=[color], s=30, depthshade=False)
                    if segment >= 0:
                        label = f'G{segment:02d}'
                        handles[label] = Line2D([], [], marker='o', ls='', c=color, label=label)
                        ax.text(*p[mask].mean(axis=0), ' '+label, fontsize=10)
                stars = frame['sources'][((frame['sources'] >= lo) & (frame['sources'] <= hi)).all(axis=1)]
                if len(stars): ax.scatter(*stars.T, c='black', marker='*', s=100)
                ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
                       box_aspect=hi-lo, xlabel='x', ylabel='y', zlabel='z', title=f't = {frame["time"]:.2f}')
                ax.view_init(elev=25, azim=-35)
            old_groups = [e['parents'] for e in change['removed'] if e['kind'] == 'combination']
            old_segments = [[n['segment_id'] for n in parent['nodes'] if n['group'] in groups] for groups in old_groups]
            fig.suptitle('Changed association: original surface samples in a fixed zoom', fontsize=15)
            fig.legend(handles=[handles[k] for k in sorted(handles)], loc='lower center',
                       bbox_to_anchor=(.5, .11), ncol=8, frameon=False)
            fig.text(.06, .065, f'Baseline parent segments: {old_segments}. Expanded: '
                     f'{[n["segment_id"] for n in pnodes]} -> {[n["segment_id"] for n in cnodes]}.\n'
                     'Other nearby groups remain visible; their inclusion is not implied by proximity. '
                     'These points are samples, not tracked gas particles.', fontsize=10)
            fig.subplots_adjust(left=.03, right=.96, top=.91, bottom=.23, wspace=.1)
            fig.savefig(output/f'changed_interval_{index:04d}_detail.png', dpi=140)
            plt.close(fig)


def run(baseline, candidate, comparison, output, fps=1, plane=.5):
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(plane):
        raise ValueError('Require positive finite FPS and a finite cutaway plane.')
    if output.exists():
        raise ValueError('Use a fresh output directory.')
    graphs, frames, changes, metas, hashes = load_frames(baseline, candidate, comparison, plane)
    output.mkdir(parents=True)
    render(graphs, frames, changes, metas, output, fps, plane)
    from PIL import Image
    durations = []
    with Image.open(output/'tracked_surfaces.gif') as image:
        decoded = image.n_frames; size = list(image.size)
        for index in range(decoded):
            image.seek(index); image.load(); durations.append(image.info.get('duration', 0))
    if decoded != len(frames):
        raise ValueError('GIF frame count differs from saved events.')
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f'Input changed during rendering: {path}')
    report = dict(passed=True, events=len(frames), decoded_frames=decoded, frame_size=size, fps=fps,
                  playback_seconds=sum(durations)/1000, cutaway_plane_z=plane,
                  all_samples_in_full_view=True, positions_moved=False, tracking_decisions_changed=False,
                  same_segment_colors_verified=True, physical_ancestry_established=False,
                  frames=[{k: f[k] for k in ['event', 'time', 'groups', 'overlaps']} for f in frames],
                  input_sha256=hashes, source_sha256={str(p): digest(p) for p in
                      [Path(__file__), Path('experiments/wind_bubble/animate_event_history.py')]})
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['frames', 'input_sha256', 'source_sha256']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline', 'candidate', 'comparison', 'output-dir']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--fps', type=float, default=1)
    parser.add_argument('--cutaway-z', type=float, default=.5)
    args = parser.parse_args()
    run(args.baseline, args.candidate, args.comparison, args.output_dir, args.fps, args.cutaway_z)
