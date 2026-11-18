"""Final combined-mesh safety pass after independent adaptive refinement.

Only events failing a combined geometry check are repaired. Existing support,
accuracy and geometry criteria stay fixed; unsuccessful repairs keep points.
Previously checked artifacts are never overwritten.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.wind_bubble.check_adaptive_mesh_times import verify_report, load_event, group_accuracy, display_faces
from experiments.wind_bubble.finalize_surface_display import remove_defects
from experiments.wind_bubble.orient_display_mesh import orient_faces
from experiments.wind_bubble.research_surface_display import measure, write_metrics


def triangle_keys(v, f, groups):
    """Geometry/group keys remain comparable after unused-vertex compaction."""
    return [(int(groups[face[0]]), *sorted(tuple(point) for point in v[face])) for face in f]


def intersection_audit(v, f, groups):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    pairs = np.asarray(mesh.get_self_intersecting_triangles(), int).reshape(-1, 2)
    return dict(intersection_pairs=len(pairs), involved_faces=len(np.unique(pairs)),
                cross_group_pairs=int((groups[f[pairs[:, 0], 0]] != groups[f[pairs[:, 1], 0]]).sum()) if len(pairs) else 0,
                face_pairs=pairs.tolist(), face_groups=groups[f[pairs, 0]].tolist())


def plot_repair(output, event, old, new, removed, physical):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    old_v, old_f, old_g = old; v, f, g = new
    involved = old_v[old_f[removed]].reshape(-1, 3)
    dx = float(physical['grid_spacing']); low = involved.min(axis=0)-dx; high = involved.max(axis=0)+dx
    palette = plt.get_cmap('tab10')
    fig = plt.figure(figsize=(16, 6.5))
    panels = [(old_v, old_f, old_g, 'Original candidate near contact'),
              (old_v, old_f[removed], old_g, 'Only removed intersecting faces'),
              (v, f, g, 'Final combined cleanup')]
    for i, (points, faces, groups, title) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        center = points[faces].mean(axis=1)
        local = faces[((center >= low) & (center <= high)).all(axis=1)]
        if len(local):
            colors = np.tile([.85, .2, .16, 1.], (len(local), 1)) if i == 1 else palette(groups[local[:, 0]] % 10)
            ax.add_collection3d(Poly3DCollection(points[local], facecolors=colors, edgecolors='#444444', linewidths=.3, alpha=.8))
        original = physical['positions']; selected = ((original >= low) & (original <= high)).all(axis=1)
        ax.scatter(*original[selected].T, color='black', s=10, depthshade=False)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
               xlabel='x', ylabel='y', zlabel='z', box_aspect=high-low, title=title)
        ax.view_init(elev=22, azim=-35)
    fig.suptitle(f'Final display cleanup at event {event}, simulation t = {float(physical["time"]):.2f}', fontsize=16, y=.98)
    fig.text(.035, .04, f'{len(removed)} faces removed; no retained vertex moves and no new triangle is added. Colors identify supplied geometric groups.\n'
             'Black points are original detected samples. The resulting gap is retained; this operation does not infer a physical merger or change diagnostics.', fontsize=10)
    fig.subplots_adjust(left=.02, right=.98, top=.88, bottom=.20, wspace=.12)
    fig.savefig(output/f'repair_{event:04d}.png', dpi=150); plt.close(fig)


def run(directory, output, root, history):
    if output.exists(): raise ValueError('Use a fresh final-sequence directory.')
    hashes = {}; previous = verify_report(directory/'verification.json', hashes)
    verify_report(history/'verification.json', hashes)
    graph_path = history/'graph.json'; hashes[str(graph_path)] = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    graph = json.loads(graph_path.read_text())
    metadata = json.loads((directory/'metadata.json').read_text())
    if metadata['status'] != 'complete': raise ValueError('Input sequence is incomplete.')
    output.mkdir(parents=True)
    summaries, checks, repaired, attempted, unchanged = [], [], [], [], []
    changed_meshes = []
    for event in previous['events']:
        old_path = directory/f'mesh_{event:04d}.npz'; old_check_path = directory/f'event_{event:04d}.json'
        for path in [old_path, old_check_path]: hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        check = json.loads(old_check_path.read_text()); summary = dict(check['summary'])
        geometry_failed = any(summary[key] for key in ['self_intersecting', 'nonmanifold_vertices',
                                                      'nonmanifold_edges', 'inconsistent_shared_edge_winding'])
        if not geometry_failed:
            shutil.copy2(old_path, output/old_path.name)
            shutil.copy2(old_check_path, output/old_check_path.name)
            unchanged.append(event)
            if hashlib.sha256((output/old_path.name).read_bytes()).hexdigest() != hashes[str(old_path)]:
                raise ValueError('Unchanged event copy differs.')
        else:
            attempted.append(event)
            physical, labels, nodes, segments = load_event(root, history, graph, event, hashes)
            with np.load(old_path) as data: arrays = {key: data[key].copy() for key in data.files}
            for key in ['positions', 'normals', 'mach', 'upstream_inside_any_source', 'downstream_inside_any_source']:
                np.testing.assert_array_equal(arrays['original_'+key], physical[key])
            np.testing.assert_array_equal(arrays['original_labels'], labels)
            np.testing.assert_array_equal(arrays['original_segments'], segments)
            old_v, old_f, old_g = arrays['vertices'], arrays['candidate_faces'], arrays['vertex_groups']
            audit = intersection_audit(old_v, old_f, old_g)
            v, candidate, g, cleanup = remove_defects(old_v, old_f, old_g)
            candidate, orientation = orient_faces(v, candidate)
            before = Counter(triangle_keys(old_v, old_f, old_g)); remaining = Counter(triangle_keys(v, candidate, g))
            if remaining-before: raise ValueError('Final cleanup added or moved triangle geometry.')
            removal = []
            for index, key in enumerate(triangle_keys(old_v, old_f, old_g)):
                if remaining[key]: remaining[key] -= 1
                else: removal.append(index)
            if len(removal) != cleanup['removed_faces']: raise ValueError('Removed-face accounting differs.')
            metrics = measure(dict(positions=physical['positions'], normals=physical['normals'], labels=labels,
                                   dx=float(physical['grid_spacing'])), v, candidate, g)
            groups = group_accuracy(physical, labels, nodes, v, candidate, g)
            faces, approved = display_faces(candidate, g, groups, metrics)
            accepted = len(approved) == len(nodes) and bool(nodes)
            arrays.update(vertices=v, candidate_faces=candidate, faces=faces, vertex_groups=g,
                          approved_groups=np.asarray(approved, int))
            np.savez_compressed(output/old_path.name, **arrays)
            with np.load(output/old_path.name) as saved:
                for key, expected in arrays.items(): np.testing.assert_array_equal(saved[key], expected)
            summary.update(status='mesh_approved' if accepted else 'geometry_fallback',
                approved_groups=len(approved), all_retained_groups_approved=accepted,
                displayed_mesh_faces=len(faces), fallback_points=int((~np.isin(labels, approved)).sum()),
                post_refinement_removed_faces=len(removal), post_refinement_orientation_changes=orientation['changed_faces'], **metrics)
            final_check = dict(intersection_audit=audit, cleanup=cleanup, orientation=orientation,
                               no_new_or_moved_triangles=True, group_checks=groups,
                               previous_status=check['summary']['status'], accepted=accepted)
            check['summary'] = summary; check['details'] = dict(check['details'], groups=groups, final_combined_cleanup=final_check)
            check['mesh_sha256'] = hashlib.sha256((output/old_path.name).read_bytes()).hexdigest()
            check['final_cleanup_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            (output/old_check_path.name).write_text(json.dumps(check, indent=2)+'\n')
            (output/f'repair_{event:04d}.json').write_text(json.dumps(final_check, indent=2)+'\n')
            if removal: plot_repair(output, event, (old_v, old_f, old_g), (v, candidate, g), removal, physical)
            if accepted: repaired.append(event)
            changed_meshes.append(event)
            print(json.dumps(dict(event=event, before_pairs=audit['intersection_pairs'],
                removed_faces=len(removal), all_groups_approved=accepted,
                group_agreement=[row['fraction_within_quarter_cell'] for row in groups])), flush=True)
        summaries.append(summary); checks.append(check)
    write_metrics(output/'summary.csv', summaries)
    group_rows = [dict(event=c['summary']['event'], time=c['summary']['time'], **g) for c in checks for g in c['details']['groups']]
    if group_rows: write_metrics(output/'group_accuracy.csv', group_rows)
    metadata.update(parent_sequence=str(directory), final_combined_cleanup=True)
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected: raise ValueError('Input changed during final cleanup.')
    sources = [Path(__file__)]+[Path('experiments/wind_bubble')/(name+'.py') for name in
        ['check_adaptive_mesh_times', 'finalize_surface_display', 'orient_display_mesh', 'research_surface_display']]
    report = dict(passed=True, events=previous['events'], full_saved_sequence=previous['full_saved_sequence'],
        no_detection_events=previous['no_detection_events'],
        events_requiring_group_fallback=[r['event'] for r in summaries if r['status'].endswith('fallback')],
        all_events_with_retained_groups_approved=all(r['all_retained_groups_approved'] for r in summaries if r['retained_groups']),
        reused_events=previous['reused_events'], cleanup_attempted_events=attempted, repaired_events=repaired,
        unchanged_event_meshes_exact=unchanged, final_combined_cleanup_added=True,
        fitting_support_accuracy_thresholds_unchanged=True, no_new_or_moved_triangle_geometry=True,
        physical_arrays_exact=True, grouping_changed=False, tracking_changed=False,
        temporal_smoothing=False, temporal_flicker_validated=False, physical_topology_established=False,
        input_sha256=hashes, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        output_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['passed', 'repaired_events', 'events_requiring_group_fallback']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('outputs/three_bubble_adaptive_sequence_n064'))
    parser.add_argument('--root', type=Path, default=Path('outputs/three_bubbles_iwr_compgpu7_n064'))
    parser.add_argument('--history', type=Path, default=Path('outputs/three_bubble_history_expanded_n064'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.directory, args.output_dir, args.root, args.history)
