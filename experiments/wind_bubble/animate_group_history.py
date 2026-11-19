"""Synchronize checked smooth surfaces, causal history and original group statistics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.astrolink.freeze_multiframe_baseline import verify_manifest
from experiments.wind_bubble.animate_adaptive_sequence import load_sequence, verify_encoded_frames
from experiments.wind_bubble.build_group_history_table import visible_relationships


def load_inputs(tables, meshes):
    hashes = {}; report = verify_manifest(tables/'verification.json', hashes)
    rows = json.loads((tables/'group_history.json').read_text())
    accounting = json.loads((tables/'event_accounting.json').read_text())
    events = json.loads((tables/'tracking_events.json').read_text())
    support = json.loads((tables/'conditional_support.json').read_text())
    visibility = json.loads((tables/'visibility.json').read_text())
    frames, mesh_hashes, mesh_report = load_sequence(meshes, Path(report['root']))
    for path, expected in mesh_hashes.items():
        if path in hashes and hashes[path] != expected:
            raise ValueError('Tables and meshes refer to different physical inputs.')
        hashes[path] = expected
    if len(frames) != report['events'] or mesh_report['events_requiring_group_fallback']:
        raise ValueError('Require the complete approved mesh sequence for this demonstration.')
    for i, frame in enumerate(frames):
        mesh = frame['mesh']; current = [r for r in rows if r['event'] == i]
        if accounting[i]['event'] != frame['event'] or accounting[i]['time'] != float(mesh['time']):
            raise ValueError('Mesh/table time mismatch.')
        if sum(r['surface_samples'] for r in current) != accounting[i]['retained_samples']:
            raise ValueError('Per-group counts do not match event accounting.')
        if len(mesh['original_positions']) != accounting[i]['detected_samples']:
            raise ValueError('Mesh/catalog sample counts differ.')
        for row in current:
            selected = mesh['original_labels'] == row['group']
            if (int(selected.sum()) != row['surface_samples'] or
                    hashlib.sha256(mesh['original_positions'][selected].tobytes()).hexdigest() != row['original_positions_sha256']):
                raise ValueError('Mesh/table original-sample join differs.')
            np.testing.assert_array_equal(mesh['original_segments'][selected], row['segment_id'])
        existing, conditional = visible_relationships(events, support, i)
        if ([r['event_id'] for r in existing] != visibility[i]['visible_event_ids'] or
                [r['support_id'] for r in conditional] != visibility[i]['visible_support_ids'] or
                [r['node_id'] for r in rows if r['event'] <= i] != visibility[i]['visible_node_ids']):
            raise ValueError('Causal visibility differs from the verified table.')
    return frames, rows, accounting, events, support, visibility, hashes, report


def render(frames, rows, accounting, events, support, output, fps, segment_count):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    palette = list(plt.get_cmap('tab20').colors)
    if segment_count > len(palette):
        raise ValueError('The demonstration exceeds its distinct segment palette.')
    times = np.array([float(f['mesh']['time']) for f in frames])
    lookup = {r['node_id']: r for r in rows}
    points = np.concatenate([f['mesh']['original_positions'] for f in frames]+[frames[0]['mesh']['sources']])
    low, high = points.min(axis=0)-.04, points.max(axis=0)+.04
    fig = plt.figure(figsize=(16, 11), dpi=95)
    grid = fig.add_gridspec(2, 2, left=.065, right=.965, bottom=.27, top=.86, hspace=.38, wspace=.25)
    view = fig.add_subplot(grid[0, 0], projection='3d')
    graph_ax = fig.add_subplot(grid[0, 1])
    mach_ax = fig.add_subplot(grid[1, 0]); count_ax = fig.add_subplot(grid[1, 1])
    title = fig.suptitle('', y=.975, fontsize=17)
    subtitle = fig.text(.5, .925, '', ha='center', fontsize=11)
    note = fig.text(.065, .115, '', va='top', fontsize=10)
    fig.text(.065, .035, 'Colors identify geometric continuation segments, not physical ownership. Statistics use original detected samples; meshes are display only.\n'
             'Only observations available at the current event are shown. Combinations start new segments; conditional support does not join statistic curves.\n'
             'Source-overlap flags remain visible. Sample counts are not areas; Mach bands show sample spread, not confidence intervals.', fontsize=9)
    headers = [None]*len(frames)
    legend = None

    def update(index):
        nonlocal legend
        frame = frames[index]; mesh = frame['mesh']; now = times[index]
        current = [r for r in rows if r['event'] == index]
        prefix = [r for r in rows if r['event'] <= index]
        visible_events, visible_support = visible_relationships(events, support, index)
        for ax in (view, graph_ax, mach_ax, count_ax):
            ax.clear()
        v, faces, groups = mesh['vertices'], mesh['faces'], mesh['vertex_groups']
        shown = faces[(v[faces, 2] <= .5).all(axis=1)]
        mapping = {r['group']: r['segment_id'] for r in current}
        if len(shown):
            colors = np.array([palette[mapping[int(g)]] for g in groups[shown[:, 0]]])
            view.add_collection3d(Poly3DCollection(v[shown], facecolors=colors, edgecolors=colors, linewidths=0,
                shade=True, lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45)))
        p = mesh['original_positions']
        fallback = (p[:, 2] <= .5) & ~np.isin(mesh['original_labels'], mesh['approved_groups'])
        view.scatter(*p[fallback].T, color='#858585', s=4, depthshade=False)
        view.scatter(*mesh['sources'].T, color='black', marker='*', s=45, depthshade=False)
        view.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
                 xlabel='x', ylabel='y', zlabel='z', box_aspect=high-low,
                 title='Current checked surfaces | z ≤ 0.5 cutaway')
        view.view_init(25, -35); view.tick_params(labelsize=8)
        for edge in visible_events:
            nodes = [lookup[n] for n in edge['parents']+edge['children']]
            center_x = np.mean([lookup[edge['parents'][0]]['time'], lookup[edge['children'][0]]['time']])
            center_y = np.mean([n['segment_id'] for n in nodes])
            color = '#497b7b'
            style = '-'
            if edge['status'] != 'shared':
                color, style = '#909090', ':'
            for node in nodes:
                graph_ax.plot([node['time'], center_x], [node['segment_id'], center_y], color=color, ls=style, lw=1.3)
        for edge in visible_support:
            a, b = lookup[edge['parent_node']], lookup[edge['child_node']]
            graph_ax.plot([a['time'], b['time']], [a['segment_id'], b['segment_id']],
                          color='#e38a24', ls='--', lw=2.5, zorder=4)
        for row in prefix:
            graph_ax.scatter(row['time'], row['segment_id'], c=[palette[row['segment_id']]], s=19, zorder=5)
        if current:
            graph_ax.scatter([r['time'] for r in current], [r['segment_id'] for r in current],
                             facecolors='none', edgecolors='black', s=65, zorder=6)
        graph_ax.axvline(now, color='black', lw=.8)
        graph_ax.set(xlim=(times[0]-.003, times[-1]+.003), ylim=(segment_count-.5, -.7),
                     title='Observed group history | no future links')
        graph_ax.set_yticks(range(segment_count), [f'G{s:02d}' for s in range(segment_count)], fontsize=7)
        graph_ax.grid(axis='x', alpha=.2)
        graph_ax.legend(handles=[Line2D([], [], color='#497b7b', label='Accepted shared event'),
                                Line2D([], [], color='#909090', ls=':', label='Distance-specific alternative'),
                                Line2D([], [], color='#e38a24', ls='--', label='Conditional support')],
                        fontsize=7, loc='upper left', bbox_to_anchor=(0, -.16),
                        ncol=3, frameon=False, borderaxespad=0)
        for segment in sorted({r['segment_id'] for r in prefix}):
            observed = {r['event']: r for r in prefix if r['segment_id'] == segment}
            x = times[:index+1]
            med = [observed[i]['median_mach'] if i in observed else np.nan for i in range(index+1)]
            lower = [observed[i]['mach_p16'] if i in observed else np.nan for i in range(index+1)]
            upper = [observed[i]['mach_p84'] if i in observed else np.nan for i in range(index+1)]
            counts = [observed[i]['surface_samples'] if i in observed else np.nan for i in range(index+1)]
            color = palette[segment]
            mach_ax.plot(x, med, 'o-', color=color, markersize=3, linewidth=1.3)
            mach_ax.fill_between(x, lower, upper, color=color, alpha=.12)
            count_ax.plot(x, counts, 'o-', color=color, markersize=3, linewidth=1.3)
        mach_ax.set(xlim=(times[0]-.003, times[-1]+.003), ylim=(0, 17),
                    xlabel='Simulation time', ylabel='Mach number', title='Group Mach | median and sample p16–p84')
        count_ax.set(xlim=(times[0]-.003, times[-1]+.003), ylim=(10, 10000), yscale='log',
                     xlabel='Simulation time', ylabel='Detected samples (log scale)', title='Group sample counts | original samples only')
        for ax in (mach_ax, count_ax):
            ax.axvline(now, color='black', lw=.8); ax.grid(alpha=.2); ax.tick_params(labelsize=8)
        if not current:
            mach_ax.text(.5, .5, 'No detected groups', transform=mach_ax.transAxes, ha='center', color='#666666')
        if legend:
            legend.remove()
        handles = [Line2D([], [], color=palette[r['segment_id']], marker='o', label=f'Current G{r["segment_id"]:02d}')
                   for r in sorted(current, key=lambda r: r['segment_id'])]
        legend = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .175), ncol=4,
                            frameon=False, fontsize=9) if handles else None
        account = accounting[index]
        note.set_text(f'Detected samples: {account["detected_samples"]:,} | retained groups: {len(current)} | '
                      f'outside groups: {account["unretained_samples"]:,}\n'
                      f'All-sample source-overlap flags: upstream {account["all_upstream_source_overlaps"]:,}, '
                      f'downstream {account["all_downstream_source_overlaps"]:,}. Conditional relationships visible: {len(visible_support)}.')
        title.set_text(f'Three wind bubbles | simulation t = {now:.2f} | event {index:02d}')
        subtitle.set_text(f'Checked surface display + verified group statistics | conditional links available: {len(visible_support)}')
        fig.canvas.draw()
        headers[index] = np.asarray(fig.canvas.buffer_rgba())[:90, :, :3].copy()

    movie = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=False)
    movie.save(output/'three_bubble_group_evolution.gif', writer=PillowWriter(fps=fps), dpi=95)
    encoded = verify_encoded_frames(output/'three_bubble_group_evolution.gif', headers)
    for index in (0, 8, 13, 14, 15, 16):
        update(index); fig.savefig(output/f'event_{index:04d}.png', dpi=140)
    plt.close(fig)
    return encoded, dict(bounds_low=low.tolist(), bounds_high=high.tolist(), camera=[25, -35], cutaway_z=.5)


def run(tables, meshes, output, fps):
    if output.exists() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Use a fresh output directory and positive finite FPS.')
    frames, rows, accounting, events, support, visibility, hashes, table_report = load_inputs(tables, meshes)
    output.mkdir(parents=True)
    encoded, view = render(frames, rows, accounting, events, support, output, fps, table_report['segments'])
    from PIL import Image
    for path in output.glob('*.png'):
        with Image.open(path) as image:
            image.load()
    for path, expected in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    report = dict(passed=True, **encoded, fps=fps, events=len(frames),
                  group_observations=len(rows), view=view, future_measurements_shown=False,
                  conditional_first_visible_event=min((e['available_event'] for e in support), default=None),
                  per_event_visibility=visibility, physical_samples_changed=False,
                  accepted_tracking_changed=False, mesh_geometry_changed=False,
                  conditional_support_in_statistics=False, physical_ancestry_established=False,
                  input_sha256=hashes,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [Path(__file__), Path('experiments/wind_bubble/animate_adaptive_sequence.py'),
                                  Path('experiments/wind_bubble/build_group_history_table.py')]},
                  output_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()})
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('passed', 'events', 'decoded_frames', 'playback_seconds', 'conditional_first_visible_event')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', type=Path, default=Path('outputs/three_bubble_group_history_table_v1'))
    parser.add_argument('--meshes', type=Path, default=Path('outputs/three_bubble_adaptive_sequence_final_n064'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=1.)
    args = parser.parse_args(); run(args.tables, args.meshes, args.output_dir, args.fps)
