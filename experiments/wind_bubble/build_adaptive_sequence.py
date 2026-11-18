"""Build checked display meshes for every saved event, with resumable output."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.wind_bubble.adaptive_surface_display import refine_display
from experiments.wind_bubble.check_adaptive_mesh_times import verify_report, load_event, group_accuracy, display_faces
from experiments.wind_bubble.finalize_surface_display import remove_defects
from experiments.wind_bubble.implicit_surface_display import DisplaySettings, implicit_samples
from experiments.wind_bubble.orient_display_mesh import orient_faces
from experiments.wind_bubble.research_surface_display import measure, write_metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_event(physical, labels, nodes, cached=None):
    """Run the checked fit/cleanup/refinement/orientation, or verify saved geometry."""
    settings = DisplaySettings()
    p, n, dx = physical['positions'], physical['normals'], float(physical['grid_spacing'])
    details = dict(reused_checked_mesh=cached is not None)
    failure = None
    if cached is not None:
        for key in ['positions', 'normals', 'mach']:
            np.testing.assert_array_equal(cached['original_'+key], physical[key])
        np.testing.assert_array_equal(cached['original_labels'], labels)
        v, vg = cached['vertices'].copy(), cached['vertex_groups'].copy()
        candidate = cached.get('candidate_faces', cached['faces']).copy()
        original_candidate = candidate.copy()
    else:
        try:
            v, candidate, vg, details['construction'] = implicit_samples(p, n, labels, dx, settings)
            v, candidate, vg, details['cleanup'] = remove_defects(v, candidate, vg)
            v, candidate, vg, details['refinement'] = refine_display(p, n, labels, dx, v, candidate, vg, settings)
        except ValueError as error:
            # Keep known resource/geometry failures visible; programming errors
            # and invalid input are never converted into successful fits.
            if not any(message in str(error) for message in ['display limit exceeded', 'Display defects remain']):
                raise
            failure = str(error)
            v, candidate, vg = np.empty((0, 3)), np.empty((0, 3), int), np.empty(0, int)
    before_orientation = candidate.copy()
    try:
        candidate, details['orientation'] = orient_faces(v, candidate)
    except ValueError as error:
        if not any(message in str(error) for message in ['requires an orientable manifold', 'Native orientation failed',
                                                        'Inconsistent winding remains']):
            raise
        failure = str(error)
        details['orientation'] = dict(passed=False, changed_faces=0, error=failure)
    np.testing.assert_array_equal(np.sort(candidate, axis=1), np.sort(before_orientation, axis=1))
    if cached is not None:
        # The reviewed passing frames must be exact regressions, including winding.
        np.testing.assert_array_equal(candidate, original_candidate)
    case = dict(positions=p, normals=n, labels=labels, dx=dx)
    metrics = measure(case, v, candidate, vg)
    groups = group_accuracy(physical, labels, nodes, v, candidate, vg)
    approved_faces, approved = display_faces(candidate, vg, groups, metrics)
    if failure is not None:
        approved_faces, approved = np.empty((0, 3), int), []
    if not len(p): status = 'no_detections'
    elif not nodes: status = 'no_retained_groups'
    elif failure: status = 'construction_fallback'
    elif len(approved) == len(nodes): status = 'mesh_approved'
    elif any(metrics[k] for k in ['self_intersecting', 'nonmanifold_vertices', 'nonmanifold_edges', 'inconsistent_shared_edge_winding']):
        status = 'geometry_fallback'
    else: status = 'partial_accuracy_fallback'
    details.update(groups=groups, failure=failure)
    return v, candidate, approved_faces, vg, approved, status, metrics, details


def run(root, history, reference, time_reference, output, events=None, resume=False):
    hashes = {}; static_report = verify_report(reference/'verification.json', hashes)
    temporal_report = verify_report(time_reference/'verification.json', hashes)
    if not static_report['conditional_display_eligible'] or not temporal_report['all_selected_events_display_eligible']:
        raise ValueError('Require passing static and three-state references.')
    verify_report(history/'verification.json', hashes)
    graph_path = history/'graph.json'; metadata_path = root/'metadata.json'
    for path in [graph_path, metadata_path]: hashes[str(path)] = digest(path)
    graph = json.loads(graph_path.read_text()); simulation = json.loads(metadata_path.read_text())
    count = len(simulation['actual_times'])
    events = list(range(count)) if events is None else events
    if not events or min(events) < 0 or max(events) >= count or any(b <= a for a, b in zip(events, events[1:])):
        raise ValueError('Use increasing saved-event indices.')
    sources = [Path(__file__)]+[Path('experiments/wind_bubble')/(name+'.py') for name in
        ['adaptive_surface_display', 'check_adaptive_mesh_times', 'implicit_surface_display',
         'finalize_surface_display', 'orient_display_mesh', 'research_surface_display', 'animate_event_history']]
    source_hashes = {str(path): digest(path) for path in sources}
    config = dict(root=str(root), history=str(history), reference=str(reference), time_reference=str(time_reference),
                  events=events, settings=asdict(DisplaySettings()), target_fraction=.95, tolerance_cells=.25, max_refinements=2)
    manifest_path = output/'metadata.json'
    if output.exists():
        if not resume: raise ValueError('Use a fresh directory or explicitly resume this sequence.')
        manifest = json.loads(manifest_path.read_text())
        if manifest['config'] != config or manifest['source_sha256'] != source_hashes or manifest['input_sha256'] != hashes:
            raise ValueError('Cannot resume after configuration, code or reference changes.')
    else:
        output.mkdir(parents=True)
        manifest = dict(status='running', config=config, source_sha256=source_hashes, input_sha256=dict(hashes), completed_events=[])
    manifest.pop('error', None)
    manifest.update(status='running'); manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    summaries = []; cache_events = set(temporal_report['events']); checks = []
    try:
        for event in events:
            physical, labels, nodes, segments = load_event(root, history, graph, event, hashes)
            if abs(float(physical['time'])-simulation['actual_times'][event]) > 1e-7:
                raise ValueError('Simulation metadata/catalog time mismatch.')
            check_path = output/f'event_{event:04d}.json'; mesh_path = output/f'mesh_{event:04d}.npz'
            if event in manifest['completed_events']:
                check = json.loads(check_path.read_text())
                if digest(mesh_path) != check['mesh_sha256'] or check['source_sha256'] != source_hashes:
                    raise ValueError('Changed completed event cache.')
                for path, expected in check['event_input_sha256'].items():
                    if digest(path) != expected: raise ValueError('Changed completed event input.')
                summaries.append(check['summary']); checks.append(check)
                print(f'Resumed checked event {event}', flush=True)
                continue
            print(f'Event {event}/{count-1}: t={float(physical["time"]):.2f}; {len(labels)} samples; {len(nodes)} groups', flush=True)
            started = time.perf_counter(); cached = None; cache_path = None
            if event in cache_events: cache_path = time_reference/f'mesh_{event:04d}.npz'
            elif event == count-1 and count == 17:
                # The static reference is the saved final event; verify its time.
                cache_path = reference/'bubble_adaptive_imls.npz'
            if cache_path is not None:
                hashes[str(cache_path)] = digest(cache_path)
                with np.load(cache_path) as data: cached = {key: data[key].copy() for key in data.files}
                if float(cached['time']) != float(physical['time']): raise ValueError('Cached reference time differs.')
            v, candidate, faces, vg, approved, status, metrics, details = build_event(physical, labels, nodes, cached)
            arrays = dict(vertices=v, candidate_faces=candidate, faces=faces, vertex_groups=vg,
                          original_labels=labels, original_segments=segments, approved_groups=np.asarray(approved, int),
                          time=physical['time'], grid_spacing=physical['grid_spacing'])
            arrays.update({'original_'+key: physical[key] for key in ['positions', 'normals', 'mach',
                           'upstream_inside_any_source', 'downstream_inside_any_source']})
            np.savez_compressed(mesh_path, **arrays)
            with np.load(mesh_path) as saved:
                for key, expected in arrays.items(): np.testing.assert_array_equal(saved[key], expected)
            summary = dict(event=event, time=float(physical['time']), status=status, samples=len(labels),
                retained_groups=len(nodes), approved_groups=len(approved),
                all_retained_groups_approved=len(approved) == len(nodes) if nodes else None,
                excluded_from_grouping=int((labels < 0).sum()), displayed_mesh_faces=len(faces),
                fallback_points=int((~np.isin(labels, approved)).sum()),
                upstream_source_overlaps=int(physical['upstream_inside_any_source'].sum()),
                downstream_source_overlaps=int(physical['downstream_inside_any_source'].sum()),
                reused_checked_mesh=cached is not None, orientation_changes_this_run=details['orientation']['changed_faces'], **metrics)
            event_inputs = {str(root/'online'/f'surfaces_{event:04d}.npz'): digest(root/'online'/f'surfaces_{event:04d}.npz'),
                            str(history/f'groups_{event:04d}.npz'): digest(history/f'groups_{event:04d}.npz')}
            if cache_path is not None: event_inputs[str(cache_path)] = digest(cache_path)
            check = dict(summary=summary, details=details, mesh_sha256=digest(mesh_path), source_sha256=source_hashes,
                         event_input_sha256=event_inputs, construction_and_checks_seconds=time.perf_counter()-started)
            check_path.write_text(json.dumps(check, indent=2)+'\n')
            summaries.append(summary); checks.append(check)
            manifest['completed_events'].append(event)
            manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
            print(json.dumps(summary), flush=True)
        write_metrics(output/'summary.csv', summaries)
        group_rows = [dict(event=c['summary']['event'], time=c['summary']['time'], **g) for c in checks for g in c['details']['groups']]
        if group_rows: write_metrics(output/'group_accuracy.csv', group_rows)
        for path, expected in {**hashes, **source_hashes}.items():
            if digest(path) != expected: raise ValueError(f'Input/source changed during sequence: {path}')
        manifest.update(status='complete'); manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        report = dict(passed=True, events=events, full_saved_sequence=events == list(range(count)),
            no_detection_events=[r['event'] for r in summaries if r['status'] == 'no_detections'],
            events_requiring_group_fallback=[r['event'] for r in summaries if r['status'].endswith('fallback')],
            all_events_with_retained_groups_approved=all(r['all_retained_groups_approved'] for r in summaries if r['retained_groups']),
            reused_events=[r['event'] for r in summaries if r['reused_checked_mesh']],
            settings=asdict(DisplaySettings()), physical_arrays_exact=True, original_groups_exact=True,
            original_tracking_unchanged=True, frozen_fit_and_orientation_unchanged=True,
            temporal_smoothing=False, temporal_flicker_validated=False, physical_topology_established=False,
            input_sha256=hashes, source_sha256=source_hashes,
            output_sha256={str(path): digest(path) for path in output.iterdir() if path.is_file() and path.name != 'verification.json'})
        (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: report[k] for k in ['passed', 'events', 'events_requiring_group_fallback', 'reused_events']}, indent=2))
    except Exception as error:
        manifest.update(status='failed', error=str(error)); manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/three_bubbles_iwr_compgpu7_n064'))
    parser.add_argument('--history', type=Path, default=Path('outputs/three_bubble_history_expanded_n064'))
    parser.add_argument('--reference', type=Path, default=Path('outputs/three_bubble_surface_display_adaptive_n064'))
    parser.add_argument('--time-reference', type=Path, default=Path('outputs/three_bubble_adaptive_mesh_oriented_n064'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--events', type=int, nargs='+')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.root, args.history, args.reference, args.time_reference, args.output_dir, args.events, args.resume)
