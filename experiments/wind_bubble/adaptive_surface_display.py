"""Resolve extraction-grid boundary loss without relaxing support limits."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.wind_bubble.implicit_surface_display import DisplaySettings, implicit_samples
from experiments.wind_bubble.finalize_surface_display import remove_defects, render
from experiments.wind_bubble.mesh_benchmark_cases import cases
from experiments.wind_bubble.research_surface_display import measure, write_metrics


def distances_to_mesh(points, vertices, faces):
    import open3d as o3d
    if not len(faces): return np.full(len(points), np.inf)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(vertices, float)),
                                    o3d.utility.Vector3iVector(np.asarray(faces, int)))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(np.asarray(points, np.float32))).numpy()


def refine_display(p, n, labels, dx, vertices, faces, groups,
                   settings=DisplaySettings(), target_fraction=.95, tolerance_cells=.25, max_refinements=2):
    """Measure same-group sample error and halve grid steps only when needed.

    The implicit function and every support threshold stay fixed. Inadequate
    groups keep an explicit failure flag after the bounded refinements; this
    does not increase smoothing radius or invent an association.
    """
    if not 0 < target_fraction <= 1 or tolerance_cells <= 0 or max_refinements < 0:
        raise ValueError('Invalid display accuracy target.')
    result_v, result_f, result_g, reports = [], [], [], []
    offset = 0
    for group in np.unique(labels[labels >= 0]):
        ids = np.flatnonzero(labels == group)
        f = faces[groups[faces[:, 0]] == group]
        v = vertices.copy(); g = groups.copy(); attempts = []
        for level in range(max_refinements+1):
            distance = distances_to_mesh(p[ids], v, f)/dx
            fraction = float((distance <= tolerance_cells).mean())
            attempts.append(dict(refinement=level, step_cells=settings.step_cells/(2**level),
                                 fraction_within_tolerance=fraction, faces=len(f)))
            if fraction >= target_fraction or level == max_refinements:
                break
            local = replace(settings, step_cells=settings.step_cells/(2**(level+1)))
            v, f, g, _ = implicit_samples(p[ids], n[ids], np.full(len(ids), group), dx, local)
            v, f, g, cleanup = remove_defects(v, f, g)
            attempts[-1]['next_cleanup_removed_faces'] = cleanup['removed_faces']
        if len(f):
            used, inverse = np.unique(f, return_inverse=True)
            result_v.append(v[used]); result_f.append(inverse.reshape(-1, 3)+offset)
            result_g.append(np.full(len(used), group)); offset += len(used)
        reports.append(dict(group=int(group), passed=fraction >= target_fraction, attempts=attempts))
    return (np.concatenate(result_v) if result_v else np.empty((0, 3)),
            np.concatenate(result_f) if result_f else np.empty((0, 3), int),
            np.concatenate(result_g) if result_g else np.empty(0, int),
            dict(groups=reports, all_group_accuracy_targets_met=all(r['passed'] for r in reports),
                 tolerance_cells=tolerance_cells, target_fraction=target_fraction,
                 max_refinements=max_refinements, support_settings_unchanged=True))


def run(checked, output, root, history, event):
    from experiments.wind_bubble.research_surface_display import NAMES
    NAMES['imls_saved'] = 'Adaptive implicit: saved normals'
    report = json.loads((checked/'verification.json').read_text())
    if report.get('passed') is not True: raise ValueError('Require checked display inputs.')
    for section in ['input_sha256', 'source_sha256', 'output_sha256']:
        for path, expected in report[section].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Checked display input changed: {path}')
    output.mkdir(parents=True, exist_ok=False)
    hashes = {str(checked/'verification.json'): hashlib.sha256((checked/'verification.json').read_bytes()).hexdigest()}
    surface = root/'online'/f'surfaces_{event:04d}.npz'
    partition = history/f'groups_{event:04d}.npz'
    for path in [surface, partition]: hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(surface) as source, np.load(partition) as partition_data:
        physical = {key: source[key].copy() for key in source.files}
        bubble = dict(name=f'three_source_event_{event:04d}', positions=source['positions'].copy(),
                      normals=source['normals'].copy(), labels=partition_data['labels'].copy(), dx=float(source['grid_spacing']))
    rows, records, displays = [], [], []
    for case in cases()+[bubble]:
        path = checked/('bubble_imls_saved.npz' if case is bubble else f'{case["name"]}_imls_saved.npz')
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as mesh:
            v, f, groups = [mesh[k].copy() for k in ['vertices', 'faces', 'vertex_groups']]
        v, f, groups, record = refine_display(case['positions'], case['normals'], case['labels'], case['dx'], v, f, groups)
        metrics = measure(case, v, f, groups)
        rows.append(dict(case=case['name'], method='adaptive_imls_saved', **metrics))
        records.append(dict(case=case['name'], **record))
        name = 'bubble_adaptive_imls.npz' if case is bubble else f'{case["name"]}_adaptive_imls.npz'
        arrays = dict(vertices=v, faces=f, vertex_groups=groups)
        if case is bubble:
            arrays.update(original_positions=physical['positions'], original_normals=physical['normals'],
                          original_mach=physical['mach'], original_labels=bubble['labels'], time=physical['time'])
        np.savez_compressed(output/name, **arrays)
        with np.load(output/name) as saved:
            for key, expected in arrays.items(): np.testing.assert_array_equal(saved[key], expected)
        print(case['name'], metrics, record, flush=True)
        # Use the existing comparison layout: the third column now shows the
        # adaptive candidate, while the fourth preserves the earlier fit variant.
        meshes = {}
        for method in ['ball_saved', 'imls_saved', 'imls_fitted', 'poisson_supported']:
            filename = f'bubble_{method}.npz' if case is bubble else f'{case["name"]}_{method}.npz'
            original_path = checked/filename
            hashes[str(original_path)] = hashlib.sha256(original_path.read_bytes()).hexdigest()
            with np.load(original_path) as original:
                meshes[method] = (*[original[k].copy() for k in ['vertices', 'faces', 'vertex_groups']], {}, 0.)
        meshes['imls_saved'] = (v, f, groups, record, 0.)
        if case is bubble:
            render(output, [(case, meshes), (dict(case, name='same_event_cutaway', cutaway_z=.5), meshes)], 'bubble_comparison.png')
        else: displays.append((case, meshes))
    render(output, displays, 'control_comparison.png')
    write_metrics(output/'summary.csv', rows)
    (output/'refinements.json').write_text(json.dumps(records, indent=2)+'\n')
    failures = []
    for row, record in zip(rows, records):
        if row['case'] == 'near_sheets_same_group_counterexample': continue
        if not record['all_group_accuracy_targets_met']: failures.append(row['case']+': per-group accuracy')
        if row['nonmanifold_vertices'] or row['nonmanifold_edges'] or row['inconsistent_shared_edge_winding'] or row['self_intersecting']:
            failures.append(row['case']+': topology')
        if row.get('sampled_hole_intrusion_faces', 0): failures.append(row['case']+': hole intrusion')
        if row['case'] == 'near_sheets_separate_groups' and row['off_true_sheet_probe_fraction'] > .01:
            failures.append(row['case']+': displaced separate sheets')
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected: raise ValueError('Changed input.')
    sources = [Path(__file__)]+[Path('experiments/wind_bubble')/(name+'.py') for name in
        ['implicit_surface_display', 'finalize_surface_display', 'research_surface_display']]
    verified = dict(passed=True, meshes_checked=len(rows), conditional_display_eligible=not failures, failures=failures,
                    universal_solution_established=False, input_sha256=hashes,
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    physical_arrays_changed=False, production_default_changed=False, tracking_changed=False,
                    output_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    (output/'verification.json').write_text(json.dumps(verified, indent=2)+'\n')
    print(json.dumps(dict(conditional_display_eligible=not failures, failures=failures), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checked', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('outputs/three_bubbles_iwr_compgpu7_n064'))
    parser.add_argument('--history', type=Path, default=Path('outputs/three_bubble_history_expanded_n064'))
    parser.add_argument('--event-index', type=int, default=16)
    args = parser.parse_args()
    run(args.checked, args.output_dir, args.root, args.history, args.event_index)
