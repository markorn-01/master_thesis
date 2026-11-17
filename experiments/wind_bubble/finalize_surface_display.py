"""Conservatively remove display-mesh defects and inspect remaining accuracy.

No hole filling, coordinate welding, smoothing, or source attribution occurs.
All triangles touching detected defects are removed, with explicit accounting.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.wind_bubble.mesh_benchmark_cases import cases
from experiments.wind_bubble.research_surface_display import measure, compact, write_metrics, NAMES


def render(output, displays, filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    methods = ['points', 'ball_saved', 'imls_saved', 'imls_fitted', 'poisson_supported']
    fig = plt.figure(figsize=(21, 4.3*len(displays)+1.3))
    palette = plt.get_cmap('tab20')
    for row, (case, meshes) in enumerate(displays):
        p = case['positions']; center = (p.min(axis=0)+p.max(axis=0))/2
        radius = max(np.ptp(p, axis=0).max()/2, case['dx'])*1.08
        cut = case.get('cutaway_z')
        for col, method in enumerate(methods):
            ax = fig.add_subplot(len(displays), len(methods), row*len(methods)+col+1, projection='3d')
            if method == 'points':
                keep = p[:, 2] <= cut if cut is not None else np.ones(len(p), bool)
                ax.scatter(*p[keep].T, c=palette(case['labels'][keep]%20), s=2, depthshade=False)
            else:
                v, f, groups, _, _ = meshes[method]
                if cut is not None: f = f[(v[f, 2] <= cut).all(axis=1)]
                if len(f):
                    color = palette(groups[f[:, 0]]%20)
                    ax.add_collection3d(Poly3DCollection(v[f], facecolors=color, edgecolors=color, linewidths=0,
                        shade=True, lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45)))
            ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
                   zlim=(center[2]-radius, center[2]+radius), box_aspect=(1, 1, 1))
            ax.view_init(elev=50 if 'hole' in case['name'] else 25, azim=-35)
            ax.set_title(NAMES[method]+'\n'+case['name'].replace('_', ' '), fontsize=10, pad=4)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle('Shock-surface display comparison after conservative defect removal', fontsize=17, y=.975)
    fig.text(.5, .94, 'Same input samples and supplied groups; new implicit vertices are display approximations', ha='center', fontsize=12)
    fig.text(.025, .02, 'Colors identify supplied geometric groups. Lower bubble row: display cutaway z <= 0.5. '
             'No physical measurement or tracking decision is changed.\n'
             'Known holes and nearby sheets are evaluated separately. Smooth appearance does not establish correct physical topology.', fontsize=11)
    fig.subplots_adjust(left=.005, right=.995, top=.87, bottom=.09, hspace=.20, wspace=.005)
    fig.savefig(output/filename, dpi=130); plt.close(fig)


def remove_defects(vertices, faces, groups, max_passes=8):
    import open3d as o3d
    v = np.asarray(vertices, float); f = np.asarray(faces, int).reshape(-1, 3).copy()
    reports = []
    for _ in range(max_passes):
        if not len(f):
            break
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
        bad_vertices = np.asarray(mesh.get_non_manifold_vertices(), int)
        pairs = np.asarray(mesh.get_self_intersecting_triangles(), int).reshape(-1, 2)
        bad = np.isin(f, bad_vertices).any(axis=1)
        if len(pairs):
            bad[np.unique(pairs)] = True
        reports.append(dict(nonmanifold_vertices=len(bad_vertices), intersection_pairs=len(pairs),
                            removed_faces=int(bad.sum())))
        if not bad.any():
            return (*compact(v, f, groups), dict(passed=True, passes=reports,
                    removed_faces=sum(r['removed_faces'] for r in reports)))
        f = f[~bad]
    if not len(f):
        return (np.empty((0, 3)), np.empty((0, 3), int), np.empty(0, int),
                dict(passed=True, passes=reports, removed_faces=sum(r['removed_faces'] for r in reports)))
    raise ValueError('Display defects remain after the bounded removal passes.')


def run(research, output, root, history, event):
    verified = json.loads((research/'verification.json').read_text())
    if verified.get('passed') is not True:
        raise ValueError('Require a complete research comparison.')
    hashes = {}
    for section in ['input_sha256', 'source_sha256']:
        for path, expected in verified[section].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Research input/source changed: {path}')
    output.mkdir(parents=True, exist_ok=False)
    surface = root/'online'/f'surfaces_{event:04d}.npz'
    partition = history/f'groups_{event:04d}.npz'
    for path in [surface, partition, research/'verification.json', research/'summary.csv']:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(surface) as source, np.load(partition) as partition_data:
        physical = {key: source[key].copy() for key in source.files}
        bubble = dict(name=f'three_source_event_{event:04d}', positions=source['positions'].copy(),
                      normals=source['normals'].copy(), labels=partition_data['labels'].copy(), dx=float(source['grid_spacing']))
    inputs = cases()+[bubble]; rows, reports, displays = [], [], []
    for case in inputs:
        meshes = {}
        for method in ['ball_saved', 'imls_saved', 'imls_fitted', 'poisson_supported']:
            filename = f'bubble_{method}.npz' if case is bubble else f'{case["name"]}_{method}.npz'
            path = research/filename; hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as data:
                old_v, old_f, old_groups = [data[k] for k in ['vertices', 'faces', 'vertex_groups']]
            v, f, groups, report = remove_defects(old_v, old_f, old_groups)
            metrics = measure(case, v, f, groups)
            rows.append(dict(case=case['name'], method=method, removed_faces=report['removed_faces'], **metrics))
            reports.append(dict(case=case['name'], method=method, **report))
            # Verify that cleanup only removes faces, preserving retained
            # triangle coordinates exactly (including their winding).
            keys = {tuple(old_v[face].ravel()) for face in old_f}
            if any(tuple(v[face].ravel()) not in keys for face in f):
                raise AssertionError('Cleanup changed or added a triangle.')
            arrays = dict(vertices=v, faces=f, vertex_groups=groups)
            if case is bubble:
                arrays.update(original_positions=physical['positions'], original_normals=physical['normals'],
                              original_mach=physical['mach'], original_labels=bubble['labels'], time=physical['time'])
            np.savez_compressed(output/filename, **arrays)
            with np.load(output/filename) as readback:
                for key, expected in arrays.items():
                    np.testing.assert_array_equal(readback[key], expected)
            meshes[method] = (v, f, groups, report, 0.)
        if case is bubble:
            render(output, [(case, meshes), (dict(case, name='same_event_cutaway', cutaway_z=.5), meshes)], 'bubble_comparison.png')
        else:
            displays.append((case, meshes))
    render(output, displays, 'control_comparison.png')
    write_metrics(output/'summary.csv', rows)
    (output/'cleanup.json').write_text(json.dumps(reports, indent=2)+'\n')
    # This is a conditional display eligibility check. The unresolved same-group
    # two-sheet counterexample is explicitly reported and prevents a universal
    # reconstruction claim, even if other controls pass.
    eligibility = []
    for method in ['ball_saved', 'imls_saved', 'imls_fitted', 'poisson_supported']:
        tested = [r for r in rows if r['method'] == method and r['case'] != 'near_sheets_same_group_counterexample']
        failures = []
        for row in tested:
            if row['self_intersecting'] or row['nonmanifold_vertices'] or row['nonmanifold_edges'] or row['inconsistent_shared_edge_winding']:
                failures.append(row['case']+': topology')
            if row.get('sampled_hole_intrusion_faces', 0):
                failures.append(row['case']+': known hole intrusion')
            if row['case'] == 'near_sheets_separate_groups' and row['off_true_sheet_probe_fraction'] > .01:
                failures.append(row['case']+': displaced separate sheets')
            if not row['faces'] or row['sample_fraction_within_quarter_cell'] < .95:
                failures.append(row['case']+': incomplete quarter-cell sample agreement')
        eligibility.append(dict(method=method, conditional_display_eligible=not failures, failures=failures,
                                universal_surface_reconstruction_validated=False))
    (output/'eligibility.json').write_text(json.dumps(eligibility, indent=2)+'\n')
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Input changed: {path}')
    sources = [Path(__file__), Path('experiments/wind_bubble/research_surface_display.py')]
    report = dict(passed=True, meshes_checked=len(rows), new_faces_added=False, coordinates_changed_by_cleanup=False,
                  physical_arrays_changed=False, tracking_changed=False,
                  universal_solution_established=False, input_sha256=hashes,
                  output_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(eligibility=eligibility, meshes_checked=len(rows)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('outputs/three_bubbles_iwr_compgpu7_n064'))
    parser.add_argument('--history', type=Path, default=Path('outputs/three_bubble_history_expanded_n064'))
    parser.add_argument('--event-index', type=int, default=16)
    args = parser.parse_args()
    run(args.research, args.output_dir, args.root, args.history, args.event_index)
