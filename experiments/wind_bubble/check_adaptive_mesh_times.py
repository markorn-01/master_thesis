"""Apply the frozen adaptive display to three saved states around a transition."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.wind_bubble.adaptive_surface_display import refine_display, distances_to_mesh
from experiments.wind_bubble.animate_event_history import segment_rows
from experiments.wind_bubble.finalize_surface_display import remove_defects
from experiments.wind_bubble.implicit_surface_display import DisplaySettings, implicit_samples
from experiments.wind_bubble.research_surface_display import measure, write_metrics


def verify_report(path, hashes):
    report = json.loads(path.read_text())
    if report.get('passed') is not True: raise ValueError(f'Unverified input: {path}')
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for section in ['input_sha256', 'source_sha256', 'output_sha256']:
        for name, expected in report.get(section, {}).items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Changed validated input/source: {name}')
            hashes[name] = expected
    return report


def display_faces(faces, groups, group_checks, metrics):
    """Keep approved groups only; unsafe global geometry falls back to points."""
    f = np.asarray(faces, int).reshape(-1, 3); groups = np.asarray(groups)
    if len(f) and (np.ptp(groups[f], axis=1) != 0).any():
        raise ValueError('Mesh crosses supplied groups.')
    safe = not any(metrics[key] for key in ['self_intersecting', 'nonmanifold_vertices',
                                           'nonmanifold_edges', 'inconsistent_shared_edge_winding'])
    approved = [r['group'] for r in group_checks if r['accuracy_passed']] if safe else []
    return f[np.isin(groups[f[:, 0]], approved)], approved


def load_event(root, history, graph, index, hashes):
    source = root/'online'/f'surfaces_{index:04d}.npz'
    partition = history/f'groups_{index:04d}.npz'
    for path in [source, partition]:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if str(path) in hashes and hashes[str(path)] != actual: raise ValueError(f'Changed event input: {path}')
        hashes[str(path)] = actual
    with np.load(source) as data: physical = {key: data[key].copy() for key in data.files}
    with np.load(partition) as data: labels = data['labels'].copy()
    nodes = [n for n in graph['nodes'] if n['event'] == index]
    if labels.shape != (len(physical['positions']),): raise ValueError('Partition/catalog length mismatch.')
    segments = segment_rows(labels, nodes)
    actual_time = float(physical['time'])
    if any(abs(n['time']-actual_time) > 1e-7 for n in nodes): raise ValueError('History/catalog time mismatch.')
    return physical, labels, nodes, segments


def group_accuracy(physical, labels, nodes, v, f, vertex_groups, tolerance=.25):
    rows = []
    for node in nodes:
        group = node['group']; selected = labels == group
        faces = f[vertex_groups[f[:, 0]] == group]
        distance = distances_to_mesh(physical['positions'][selected], v, faces)/float(physical['grid_spacing'])
        fraction = float((distance <= tolerance).mean())
        rows.append(dict(group=group, segment=node['segment_id'], samples=int(selected.sum()), faces=len(faces),
                         fraction_within_quarter_cell=fraction, accuracy_passed=fraction >= .95,
                         sample_to_own_group_mesh_p95_cells=float(np.percentile(distance, 95)) if len(faces) else None,
                         upstream_source_overlaps=int(physical['upstream_inside_any_source'][selected].sum()),
                         downstream_source_overlaps=int(physical['downstream_inside_any_source'][selected].sum())))
    return rows


def render(frames, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    palette = list(plt.get_cmap('tab20').colors)
    p = np.concatenate([frame['physical']['positions'] for frame in frames])
    low = p.min(axis=0)-.04; high = p.max(axis=0)+.04

    def draw(ax, frame, mesh):
        physical = frame['physical']; points = physical['positions']; labels = frame['labels']
        cut = points[:, 2] <= .5
        if mesh:
            v = frame['vertices']; f = frame['faces']; vg = frame['vertex_groups']
            shown = f[(v[f, 2] <= .5).all(axis=1)]
            mapping = {n['group']: n['segment_id'] for n in frame['nodes']}
            if len(shown):
                color = np.array([palette[mapping[int(g)] % len(palette)] for g in vg[shown[:, 0]]])
                ax.add_collection3d(Poly3DCollection(v[shown], facecolors=color, edgecolors=color, linewidths=0,
                    shade=True, lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45)))
            fallback = cut & ~np.isin(labels, frame['approved_groups'])
            ax.scatter(*points[fallback].T, c='#858585', s=6, depthshade=False)
        else:
            for segment in np.unique(frame['segments']):
                selected = cut & (frame['segments'] == segment)
                ax.scatter(*points[selected].T, c=[palette[segment % len(palette)] if segment >= 0 else '#858585'],
                           s=3 if segment == 4 else 9, depthshade=False, linewidths=0)
        ax.scatter(*physical['source_positions'].T, c='black', marker='*', s=45, depthshade=False)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
               box_aspect=high-low, xlabel='x', ylabel='y', zlabel='z')
        ax.view_init(elev=25, azim=-35)
        ax.set_title(f't = {float(physical["time"]):.2f} | '+('Adaptive display' if mesh else 'Original surface samples'), fontsize=11)

    fig = plt.figure(figsize=(12, 4.2*len(frames)+1.8))
    for row, frame in enumerate(frames):
        for column in range(2):
            ax = fig.add_subplot(len(frames), 2, row*2+column+1, projection='3d')
            draw(ax, frame, bool(column))
        count = len(frame['approved_groups'])
        fig.text(.51, .865-row*(.76/len(frames)),
                 f'{count}/{len(frame["nodes"])} groups meet the frozen display checks', fontsize=10)
    fig.suptitle('Three-state check of the frozen adaptive display', fontsize=17, y=.98)
    fig.text(.5, .945, 'Same camera, bounds and continuation-segment colors | display cutaway: z <= 0.5', ha='center', fontsize=11)
    fig.text(.04, .025, 'Stars: wind sources. Gray: points outside approved display groups; failed groups are never silently meshed.\n'
             'Each state is reconstructed independently. No temporal smoothing or point trajectories; these three states do not validate animation flicker.\n'
             'Geometry remains an approximation to detections. Physical arrays, group partitions and temporal links are unchanged.', fontsize=10)
    fig.subplots_adjust(left=.02, right=.97, top=.90, bottom=.12, wspace=.04, hspace=.16)
    fig.savefig(output/'three_time_comparison.png', dpi=130); plt.close(fig)

    fig = plt.figure(figsize=(12, 7)); axes = [fig.add_subplot(1, 2, i+1, projection='3d') for i in range(2)]
    heading = fig.suptitle('', fontsize=16, y=.97)
    info = fig.text(.04, .10, '', fontsize=10)
    fig.text(.04, .035, 'Three saved states only; independent fits with frozen settings. z <= 0.5 display cutaway.\n'
             'Gray: original points outside approved groups. No physical or tracking changes.', fontsize=10)
    fig.subplots_adjust(left=.01, right=.98, top=.88, bottom=.18, wspace=.03)

    def update(index):
        frame = frames[index]
        for column, ax in enumerate(axes): ax.clear(); draw(ax, frame, bool(column))
        heading.set_text(f'Frozen adaptive surface display | saved state {index+1}/{len(frames)}')
        failed = [r['group'] for r in frame['group_checks'] if not r['accuracy_passed']]
        info.set_text(f'Event {frame["event"]}: {len(frame["physical"]["positions"]):,} original samples; '
                      f'{len(frame["approved_groups"])}/{len(frame["nodes"])} approved groups. '
                      f'Groups below the fixed accuracy target: {failed}.')
    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000, blit=False)
    animation.save(output/'three_state_preview.gif', writer=PillowWriter(fps=1), dpi=95)
    plt.close(fig)


def run(root, history, reference, output, events):
    if len(events) < 2 or any(b <= a for a, b in zip(events, events[1:])) or events[0] < 0:
        raise ValueError('Use at least two strictly increasing nonnegative event indices.')
    if output.exists(): raise ValueError('Use a fresh output directory.')
    hashes = {}; prior = verify_report(reference/'verification.json', hashes)
    if prior.get('conditional_display_eligible') is not True: raise ValueError('Reference candidate is not conditionally eligible.')
    verify_report(history/'verification.json', hashes)
    graph_path = history/'graph.json'; hashes[str(graph_path)] = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    graph = json.loads(graph_path.read_text()); settings = DisplaySettings()
    output.mkdir(parents=True)
    frames, summaries, checks = [], [], []
    for index in events:
        physical, labels, nodes, segments = load_event(root, history, graph, index, hashes)
        case = dict(positions=physical['positions'], normals=physical['normals'], labels=labels,
                    dx=float(physical['grid_spacing']))
        started = time.perf_counter()
        print(f'Event {index}: constructing {len(nodes)} groups at t={float(physical["time"]):.2f}', flush=True)
        v, f, vg, construction = implicit_samples(case['positions'], case['normals'], labels, case['dx'], settings)
        v, f, vg, cleanup = remove_defects(v, f, vg)
        v, candidates, vg, refinement = refine_display(case['positions'], case['normals'], labels, case['dx'], v, f, vg, settings)
        metrics = measure(case, v, candidates, vg)
        rows = group_accuracy(physical, labels, nodes, v, candidates, vg)
        for row, recorded in zip(rows, refinement['groups']):
            if row['group'] != recorded['group'] or row['accuracy_passed'] != recorded['passed']:
                raise ValueError('Independent per-group accuracy readback differs from refinement.')
        f, approved = display_faces(candidates, vg, rows, metrics)
        arrays = dict(vertices=v, candidate_faces=candidates, faces=f, vertex_groups=vg,
                      original_labels=labels, original_segments=segments, approved_groups=np.asarray(approved, int),
                      time=physical['time'], grid_spacing=physical['grid_spacing'])
        arrays.update({'original_'+key: physical[key] for key in ['positions', 'normals', 'mach',
                        'upstream_inside_any_source', 'downstream_inside_any_source']})
        path = output/f'mesh_{index:04d}.npz'; np.savez_compressed(path, **arrays)
        with np.load(path) as saved:
            for key, expected in arrays.items(): np.testing.assert_array_equal(saved[key], expected)
        row = dict(event=index, time=float(physical['time']), samples=len(labels), retained_groups=len(nodes),
                   approved_groups=len(approved), all_groups_display_eligible=len(approved) == len(nodes),
                   excluded_from_grouping=int((labels < 0).sum()), displayed_mesh_faces=len(f),
                   fallback_points=int((~np.isin(labels, approved)).sum()),
                   upstream_source_overlaps=int(physical['upstream_inside_any_source'].sum()),
                   downstream_source_overlaps=int(physical['downstream_inside_any_source'].sum()), **metrics)
        summaries.append(row)
        checks.append(dict(event=index, groups=rows, construction=construction, initial_cleanup=cleanup,
                           refinement=refinement, construction_and_readback_seconds=time.perf_counter()-started))
        frames.append(dict(event=index, physical=physical, labels=labels, nodes=nodes, segments=segments,
                           vertices=v, faces=f, vertex_groups=vg, approved_groups=approved, group_checks=rows))
        print(json.dumps(row), flush=True)
    write_metrics(output/'summary.csv', summaries)
    write_metrics(output/'group_accuracy.csv', [dict(event=r['event'], **g) for r in checks for g in r['groups']])
    (output/'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    render(frames, output)
    from PIL import Image
    with Image.open(output/'three_state_preview.gif') as image:
        if image.n_frames != len(events): raise ValueError('GIF frame count differs from saved states.')
        for index in range(image.n_frames): image.seek(index); image.load()
    with Image.open(output/'three_time_comparison.png') as image: image.load()
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected: raise ValueError(f'Changed input: {path}')
    sources = [Path(__file__)]+[Path('experiments/wind_bubble')/(name+'.py') for name in
        ['adaptive_surface_display', 'implicit_surface_display', 'finalize_surface_display', 'research_surface_display', 'animate_event_history']]
    result = dict(passed=True, events=events, all_selected_events_display_eligible=all(r['all_groups_display_eligible'] for r in summaries),
                  settings=asdict(settings), target_fraction=.95, tolerance_cells=.25, max_refinements=2,
                  frozen_algorithm_unchanged=True, original_physical_arrays_exact=True,
                  grouping_changed=False, tracking_changed=False, temporal_smoothing=False,
                  temporal_flicker_validated=False, physical_topology_established=False,
                  decoded_gif_frames=len(events), gif_fps=1,
                  input_sha256=hashes, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  output_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    (output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['passed', 'events', 'all_selected_events_display_eligible']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/three_bubbles_iwr_compgpu7_n064'))
    parser.add_argument('--history', type=Path, default=Path('outputs/three_bubble_history_expanded_n064'))
    parser.add_argument('--reference', type=Path, default=Path('outputs/three_bubble_surface_display_adaptive_n064'))
    parser.add_argument('--events', type=int, nargs='+', default=[12, 13, 14])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.history, args.reference, args.output_dir, args.events)
