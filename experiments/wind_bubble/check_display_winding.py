"""Recheck saved time-check candidates with an orientation-only display fix."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.wind_bubble.check_adaptive_mesh_times import (
    verify_report, load_event, group_accuracy, display_faces, render,
)
from experiments.wind_bubble.orient_display_mesh import orient_faces
from experiments.wind_bubble.research_surface_display import measure, write_metrics


def run(previous, output, root, history, reference):
    if output.exists(): raise ValueError('Use a fresh output directory.')
    hashes = {}; prior = verify_report(previous/'verification.json', hashes)
    verify_report(history/'verification.json', hashes)
    verify_report(reference/'verification.json', hashes)
    graph_path = history/'graph.json'; hashes[str(graph_path)] = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    graph = json.loads(graph_path.read_text())
    # A previously passing static mesh must be a bitwise no-op under this fix.
    reference_mesh = reference/'bubble_adaptive_imls.npz'
    hashes[str(reference_mesh)] = hashlib.sha256(reference_mesh.read_bytes()).hexdigest()
    with np.load(reference_mesh) as saved:
        reference_faces, reference_check = orient_faces(saved['vertices'], saved['faces'])
        np.testing.assert_array_equal(reference_faces, saved['faces'])
    output.mkdir(parents=True)
    frames, summary, checks = [], [], []
    for index in prior['events']:
        physical, labels, nodes, segments = load_event(root, history, graph, index, hashes)
        path = previous/f'mesh_{index:04d}.npz'
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as saved: arrays = {key: saved[key].copy() for key in saved.files}
        for key in ['positions', 'normals', 'mach', 'upstream_inside_any_source', 'downstream_inside_any_source']:
            np.testing.assert_array_equal(arrays['original_'+key], physical[key])
        np.testing.assert_array_equal(arrays['original_labels'], labels)
        np.testing.assert_array_equal(arrays['original_segments'], segments)
        v, vg = arrays['vertices'], arrays['vertex_groups']
        candidates, orientation = orient_faces(v, arrays['candidate_faces'])
        np.testing.assert_array_equal(np.sort(candidates, axis=1), np.sort(arrays['candidate_faces'], axis=1))
        case = dict(positions=physical['positions'], normals=physical['normals'], labels=labels,
                    dx=float(physical['grid_spacing']))
        metrics = measure(case, v, candidates, vg)
        rows = group_accuracy(physical, labels, nodes, v, candidates, vg)
        f, approved = display_faces(candidates, vg, rows, metrics)
        arrays.update(candidate_faces=candidates, faces=f, approved_groups=np.asarray(approved, int))
        np.savez_compressed(output/f'mesh_{index:04d}.npz', **arrays)
        with np.load(output/f'mesh_{index:04d}.npz') as saved:
            for key, expected in arrays.items(): np.testing.assert_array_equal(saved[key], expected)
        summary.append(dict(event=index, time=float(physical['time']), samples=len(labels),
            retained_groups=len(nodes), approved_groups=len(approved), all_groups_display_eligible=len(approved) == len(nodes),
            orientation_changed_faces=orientation['changed_faces'],
            original_inconsistent_edges=orientation['before']['inconsistent_shared_edge_winding'],
            excluded_from_grouping=int((labels < 0).sum()), displayed_mesh_faces=len(f),
            fallback_points=int((~np.isin(labels, approved)).sum()),
            upstream_source_overlaps=int(physical['upstream_inside_any_source'].sum()),
            downstream_source_overlaps=int(physical['downstream_inside_any_source'].sum()), **metrics))
        checks.append(dict(event=index, orientation=orientation, groups=rows))
        frames.append(dict(event=index, physical=physical, labels=labels, nodes=nodes, segments=segments,
                           vertices=v, faces=f, vertex_groups=vg, approved_groups=approved, group_checks=rows))
        print(json.dumps(summary[-1]), flush=True)
    write_metrics(output/'summary.csv', summary)
    write_metrics(output/'group_accuracy.csv', [dict(event=c['event'], **g) for c in checks for g in c['groups']])
    (output/'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    render(frames, output)
    from PIL import Image
    with Image.open(output/'three_state_preview.gif') as image:
        if image.n_frames != len(frames): raise ValueError('GIF event count differs.')
        for index in range(image.n_frames): image.seek(index); image.load()
    with Image.open(output/'three_time_comparison.png') as image: image.load()
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected: raise ValueError(f'Changed input: {path}')
    sources = [Path(__file__)]+[Path('experiments/wind_bubble')/(name+'.py') for name in
        ['orient_display_mesh', 'check_adaptive_mesh_times', 'research_surface_display']]
    verified = dict(passed=True, events=prior['events'],
        all_selected_events_display_eligible=all(r['all_groups_display_eligible'] for r in summary),
        frozen_fitting_settings_unchanged=True, orientation_stage_added=True,
        triangle_geometry_exact=True, vertices_moved=False, faces_added_or_removed=False,
        physical_arrays_exact=True, grouping_changed=False, tracking_changed=False,
        reference_static_faces_exact=True, reference_check=reference_check,
        temporal_flicker_validated=False, physical_topology_established=False,
        decoded_gif_frames=len(frames), gif_fps=1, input_sha256=hashes,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        output_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    (output/'verification.json').write_text(json.dumps(verified, indent=2)+'\n')
    print(json.dumps({key: verified[key] for key in ['passed', 'events', 'all_selected_events_display_eligible']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('outputs/three_bubbles_iwr_compgpu7_n064'))
    parser.add_argument('--history', type=Path, default=Path('outputs/three_bubble_history_expanded_n064'))
    parser.add_argument('--reference', type=Path, default=Path('outputs/three_bubble_surface_display_adaptive_n064'))
    args = parser.parse_args()
    run(args.previous, args.output_dir, args.root, args.history, args.reference)
