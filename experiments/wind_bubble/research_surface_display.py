"""Bounded reconstruction comparison: controls followed by one saved event.

No simulation, physical estimator, production default, or tracking choice is
changed. MLS/Poisson vertices are display approximations, saved separately.
"""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from experiments.wind_bubble.ball_pivot_mesh import ball_pivot_samples
from experiments.wind_bubble.compare_patch_normals import plane_normals
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write
from experiments.wind_bubble.implicit_surface_display import DisplaySettings, implicit_samples, trim_supported
from experiments.wind_bubble.mesh_benchmark_cases import cases, evaluate_mesh
from experiments.wind_bubble.surface_mesh import mesh_topology
from experiments.wind_bubble.verify_temporal_mesh import face_components


METHODS = ['ball_saved', 'imls_saved', 'imls_fitted', 'poisson_supported', 'poisson_raw']
NAMES = dict(ball_saved='Guarded ball pivoting', imls_saved='Local implicit: saved normals',
             imls_fitted='Local implicit: display-fit normals', poisson_supported='Poisson + support trimming',
             poisson_raw='Poisson without support trimming', points='Original samples')


def write_metrics(path, rows):
    """Controls have extra truth-only columns; preserve their union explicitly."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def poisson_samples(p, n, labels, dx, settings):
    import open3d as o3d
    vertices, faces, groups, raw_faces, reports = [], [], [], [], []
    offset = 0
    for group in np.unique(labels[labels >= 0]):
        rows = np.flatnonzero(labels == group)
        if len(rows) < 6:
            continue
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(p[rows])
        cloud.normals = o3d.utility.Vector3dVector(n[rows]/np.linalg.norm(n[rows], axis=1)[:, None])
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=6, scale=1.1, linear_fit=True, n_threads=1)
        v = np.asarray(mesh.vertices).copy(); raw = np.asarray(mesh.triangles).copy()
        f, support = trim_supported(v, raw, p[rows], n[rows], dx, settings)
        vertices.append(v); faces.append(f+offset); raw_faces.append(raw+offset)
        groups.append(np.full(len(v), group, int)); offset += len(v)
        reports.append(dict(group=int(group), **support))
    return (np.concatenate(vertices) if vertices else np.empty((0, 3)),
            np.concatenate(faces) if faces else np.empty((0, 3), int),
            np.concatenate(groups) if groups else np.empty(0, int),
            np.concatenate(raw_faces) if raw_faces else np.empty((0, 3), int), reports)


def build_all(case, settings):
    p, n, labels, dx = [case[k] for k in ['positions', 'normals', 'labels', 'dx']]
    start = time.perf_counter()
    f, report, _ = ball_pivot_samples(p, n, labels, dx, radius_factors=(1.,))
    results = {'ball_saved': (p.copy(), f, labels.copy(), report, time.perf_counter()-start)}
    for name in ['imls_saved', 'imls_fitted']:
        start = time.perf_counter(); used = n
        fitted = None
        if name == 'imls_fitted':
            used, fitted, _, _ = plane_normals(p, n, labels, 2*dx)
        v, f, groups, report = implicit_samples(p, used, labels, dx, settings)
        if fitted is not None:
            report['invalid_plane_fits'] = int(((labels >= 0) & ~fitted).sum())
        report['display_normals'] = name
        results[name] = (v, f, groups, report, time.perf_counter()-start)
    start = time.perf_counter()
    v, f, groups, raw, report = poisson_samples(p, n, labels, dx, settings)
    elapsed = time.perf_counter()-start
    results['poisson_supported'] = (v, f, groups, dict(groups=report, depth=6, linear_fit=True), elapsed)
    results['poisson_raw'] = (v, raw, groups, dict(groups=report, depth=6, linear_fit=True), elapsed)
    return results


def compact(v, f, groups):
    v = np.asarray(v, dtype=float)
    used, inverse = np.unique(f, return_inverse=True)
    return v[used], inverse.reshape(-1, 3), groups[used]


def measure(case, vertices, faces, groups):
    import open3d as o3d
    v, f, groups = compact(vertices, faces, groups)
    p = case['positions']; dx = case['dx']; retained = case['labels'] >= 0
    if len(f):
        if (np.ptp(groups[f], axis=1) != 0).any():
            raise AssertionError('A triangle crosses supplied groups.')
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        distances = scene.compute_distance(o3d.core.Tensor(p[retained].astype(np.float32))).numpy()/dx
        d, nearest = cKDTree(p).query(v)
        tri = v[f]; cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        area = np.linalg.norm(cross, axis=1).sum()/2
        lengths = np.linalg.norm(tri-np.roll(tri, 1, axis=1), axis=2); semiperimeter = lengths.sum(axis=1)/2
        heron = np.sqrt(np.maximum(0, semiperimeter*np.prod(semiperimeter[:, None]-lengths, axis=1))).sum()
        np.testing.assert_allclose(area, heron, atol=1e-8, rtol=1e-8)
        topology = mesh_topology(f)
        edges = Counter(tuple(sorted((int(a), int(b)))) for face in f for a, b in zip(face, np.roll(face, -1)))
        boundary_length = sum(np.linalg.norm(v[a]-v[b]) for (a, b), count in edges.items() if count == 1)
        extra = dict(components=face_components(f)['edge_connected_components'], boundary_length_cells=boundary_length/dx,
                     nonmanifold_vertices=len(mesh.get_non_manifold_vertices()), self_intersecting=bool(mesh.is_self_intersecting()),
                     sample_to_mesh_p50_cells=float(np.median(distances)), sample_to_mesh_p95_cells=float(np.percentile(distances, 95)),
                     sample_fraction_within_quarter_cell=float((distances <= .25).mean()),
                     display_vertex_to_sample_max_cells=float(d.max()/dx), mesh_area=float(area),
                     area_fraction=float(area/case['reference_area']) if case.get('reference_area') else None)
    else:
        nearest = np.empty(0, int); topology = mesh_topology(f)
        extra = dict(components=0, boundary_length_cells=0., nonmanifold_vertices=0, self_intersecting=False,
                     sample_to_mesh_p50_cells=None, sample_to_mesh_p95_cells=None,
                     sample_fraction_within_quarter_cell=0., display_vertex_to_sample_max_cells=None,
                     mesh_area=0., area_fraction=None)
    metrics = dict(vertices=len(v), faces=len(f), **topology, **extra)
    if 'truth_sheet' in case:
        proxy = dict(case, positions=v, truth_sheet=case['truth_sheet'][nearest])
        control = evaluate_mesh(proxy, f)
        metrics.update(sampled_hole_intrusion_faces=control['sampled_hole_intrusion_faces'],
                       cross_nearest_truth_sheet_faces=control['cross_stipulated_sheet_faces'])
        if 'near_sheets' in case['name']:
            # Nearest-sample labels can miss a false middle sheet. Check actual
            # distances to the two stipulated planes at vertices and centroids.
            probes = np.concatenate([v, v[f].mean(axis=1)])
            off = np.minimum(np.abs(probes[:, 2]), np.abs(probes[:, 2]-.1)) > .02 if len(probes) else np.array([], bool)
            metrics['off_true_sheet_probe_fraction'] = float(off.mean()) if len(off) else 0.
    return metrics


def render(output, cases_and_meshes, filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    columns = ['points', 'ball_saved', 'imls_saved', 'imls_fitted', 'poisson_supported']
    fig = plt.figure(figsize=(21, 3.6*len(cases_and_meshes)))
    palette = plt.get_cmap('tab20')
    for row, (case, meshes) in enumerate(cases_and_meshes):
        p = case['positions']; lo = p.min(axis=0); hi = p.max(axis=0)
        center = (lo+hi)/2; radius = max(np.ptp(p, axis=0).max()/2, case['dx'])*1.08
        cut = case.get('cutaway_z')
        for col, method in enumerate(columns):
            ax = fig.add_subplot(len(cases_and_meshes), len(columns), row*len(columns)+col+1, projection='3d')
            if method == 'points':
                keep = p[:, 2] <= cut if cut is not None else np.ones(len(p), bool)
                ax.scatter(*p[keep].T, c=palette(case['labels'][keep]%20), s=2, depthshade=False)
            else:
                v, f, groups, _, _ = meshes[method]
                if cut is not None:
                    f = f[(v[f, 2] <= cut).all(axis=1)]
                if len(f):
                    color = palette(groups[f[:, 0]]%20)
                    # Flat lighting is an honest mesh view: no vertex locations
                    # are moved to manufacture a smooth silhouette.
                    ax.add_collection3d(Poly3DCollection(v[f], facecolors=color, edgecolors=color, linewidths=0,
                                                       shade=True, lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45)))
            ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
                   zlim=(center[2]-radius, center[2]+radius), box_aspect=(1, 1, 1))
            ax.view_init(elev=50 if 'hole' in case['name'] else 25, azim=-35)
            ax.set_title(case['name'].replace('_', ' ')+'\n'+NAMES[method], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle('Same input samples: direct triangles versus support-limited smooth approximations', fontsize=15, y=.99)
    fig.text(.025, .012, 'Colors: supplied geometric groups. Implicit/Poisson outputs use new display vertices; original physical samples and diagnostics are unchanged.\n'
             'Holes and two-sheet controls test invented geometry. A smooth appearance alone is not a reconstruction-accuracy result.', fontsize=10)
    fig.subplots_adjust(left=.005, right=.995, top=.94, bottom=.065, hspace=.18, wspace=.005)
    fig.savefig(output/filename, dpi=130); plt.close(fig)


def run(output, root, history, index):
    from experiments.wind_bubble.animate_tracked_surfaces import checked_report
    settings = DisplaySettings(); hashes = {}
    checked_report(history/'verification.json', hashes)
    surface = root/'online'/f'surfaces_{index:04d}.npz'; partition = history/f'groups_{index:04d}.npz'
    for path in [surface, partition, history/'graph.json', root/'metadata.json']:
        if str(path) in hashes and hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path)]:
            raise ValueError(f'Changed input: {path}')
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(surface) as data:
        original = {key: data[key].copy() for key in data.files}
    with np.load(partition) as data:
        labels = data['labels'].copy()
    output.mkdir(parents=True, exist_ok=False)
    rows, reports, control_displays = [], [], []
    inputs = cases()
    for case in inputs:
        meshes = build_all(case, settings); control_displays.append((case, meshes))
        for method, (v, f, groups, report, elapsed) in meshes.items():
            metrics = measure(case, v, f, groups)
            rows.append(dict(case=case['name'], method=method, seconds=elapsed, **metrics))
            reports.append(dict(case=case['name'], method=method, construction=report))
            np.savez_compressed(output/f'{case["name"]}_{method}.npz', vertices=v, faces=f, vertex_groups=groups)
            print(case['name'], method, metrics, flush=True)
    render(output, control_displays, 'control_comparison.png')
    bubble = dict(name=f'three_source_event_{index:04d}', positions=original['positions'], normals=original['normals'],
                  labels=labels, dx=float(original['grid_spacing']))
    print('Building all bubble display candidates...', flush=True)
    meshes = build_all(bubble, settings)
    for method, (v, f, groups, report, elapsed) in meshes.items():
        metrics = measure(bubble, v, f, groups)
        rows.append(dict(case=bubble['name'], method=method, seconds=elapsed, **metrics))
        reports.append(dict(case=bubble['name'], method=method, construction=report))
        # Nearest-sample attribution is display-only. Store indices and distance,
        # rather than pretending new vertices have measured shock properties.
        d, nearest = cKDTree(bubble['positions']).query(v)
        np.savez_compressed(output/f'bubble_{method}.npz', vertices=v, faces=f, vertex_groups=groups,
                            original_positions=original['positions'], original_normals=original['normals'],
                            original_mach=original['mach'], original_labels=labels,
                            nearest_sample_index=nearest, nearest_sample_distance=d, time=original['time'])
        print(bubble['name'], method, metrics, flush=True)
    render(output, [(bubble, meshes), (dict(bubble, name='same_event_cutaway', cutaway_z=.5), meshes)], 'bubble_comparison.png')
    write_metrics(output/'summary.csv', rows)
    (output/'construction.json').write_text(json.dumps(reports, indent=2)+'\n')
    # Read back physical arrays and all generated mesh arrays before reporting.
    for method, (v, f, groups, _, _) in meshes.items():
        with np.load(output/f'bubble_{method}.npz') as saved:
            for key in ['positions', 'normals', 'mach']:
                np.testing.assert_array_equal(saved['original_'+key], original[key])
            np.testing.assert_array_equal(saved['original_labels'], labels)
            for key, expected in [('vertices', v), ('faces', f), ('vertex_groups', groups)]:
                np.testing.assert_array_equal(saved[key], expected)
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Input changed: {path}')
    sources = [Path(__file__)]+[Path('experiments/wind_bubble')/(name+'.py') for name in
        ['implicit_surface_display', 'ball_pivot_mesh', 'surface_mesh', 'compare_patch_normals', 'mesh_benchmark_cases', 'verify_temporal_mesh']]
    result = dict(passed=True, controlled_meshes=len(inputs)*len(METHODS), bubble_meshes=len(METHODS),
                  settings=settings.__dict__, source_hashes_unchanged=True, original_physical_arrays_exact=True,
                  new_vertices_are_display_only=True, production_default_changed=False, tracking_changed=False,
                  universal_or_physical_surface_solution_established=False,
                  versions={name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'open3d', 'scikit-image']},
                  input_sha256=hashes, source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    (output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if not key.endswith('sha256')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('outputs/three_bubbles_iwr_compgpu7_n064'))
    parser.add_argument('--history', type=Path, default=Path('outputs/three_bubble_history_expanded_n064'))
    parser.add_argument('--event-index', type=int, default=16)
    args = parser.parse_args()
    run(args.output_dir, args.root, args.history, args.event_index)
