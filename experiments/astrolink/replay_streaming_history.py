"""Fresh-process replay of sparse catalogs, with no reference histories loaded."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import scipy

from experiments.astrolink.streaming_history import StreamingShockHistory
from experiments.astrolink.freeze_multiframe_baseline import write_csv


def run(root, output):
    if output.exists():
        raise ValueError('Use a fresh replay directory.')
    start = time.perf_counter()
    metadata_path = root/'metadata.json'
    meta = json.loads(metadata_path.read_text())
    if meta['status'] != 'complete':
        raise ValueError('Require completed saved surface catalogs.')
    hashes = {str(metadata_path): hashlib.sha256(metadata_path.read_bytes()).hexdigest()}
    output.mkdir(parents=True)
    tracker = StreamingShockHistory()
    metrics = []
    keys = ('positions', 'normals', 'time', 'grid_spacing', 'mach', 'mach_valid',
            'source_positions', 'upstream_inside_any_source', 'downstream_inside_any_source')
    for index, expected_time in enumerate(meta['actual_times']):
        path = root/'online'/f'surfaces_{index:04d}.npz'
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as source:
            catalog = {key: source[key].copy() for key in keys}
        if abs(float(catalog['time'])-expected_time) > 1e-7:
            raise ValueError('Actual catalog time differs from the saved schedule.')
        before = time.perf_counter(); delta = tracker.push(catalog)
        elapsed = time.perf_counter()-before
        partition = delta.pop('partition')
        np.savez_compressed(output/f'groups_{index:04d}.npz', **{
            k: partition[k] for k in ('labels', 'raw_labels', 'component_sizes')})
        with gzip.open(output/f'event_{index:04d}.json.gz', 'wt') as stream:
            json.dump(delta, stream)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metrics.append(dict(event=index, time=delta['time'], compute_seconds=elapsed,
                            **delta['footprint'],
                            worker_peak_host_rss_bytes=int(rss if sys.platform == 'darwin' else rss*1024)))
        print(f'event {index:02d}: {elapsed:.3f}s compute; retained catalogs={delta["footprint"]["retained_geometry_catalogs"]}', flush=True)
        del delta, partition, catalog
    before = time.perf_counter(); final = tracker.finish()
    finish_seconds = time.perf_counter()-before
    (output/'finish.json').write_text(json.dumps(final, indent=2)+'\n')
    write_csv(output/'resource_history.csv', metrics)
    workflow_seconds = time.perf_counter()-start
    for path, expected in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    sources = [Path(__file__)]+[Path('experiments/astrolink')/(name+'.py') for name in
        ('streaming_history', 'surface_history', 'shock_connectivity', 'event_tracking',
         'expanded_event_tracking', 'delayed_support', 'tracklet_fragments', 'continuation_segments')]+[
        Path('experiments/wind_bubble')/(name+'.py') for name in
        ('build_group_history_table', 'compare_event_history_resolutions')]
    report = dict(passed=True, input_directory=str(root), settings=asdict(tracker.settings),
                  events=len(metrics), segments=final['segment_count'],
                  compute_seconds=sum(r['compute_seconds'] for r in metrics)+finish_seconds,
                  finish_seconds=finish_seconds, workflow_seconds_excluding_imports=workflow_seconds,
                  max_retained_geometry_catalogs=max((r['retained_geometry_catalogs'] for r in metrics), default=0),
                  max_retained_geometry_bytes=max((r['retained_geometry_bytes'] for r in metrics), default=0),
                  max_processing_geometry_catalogs=final['footprint']['max_processing_geometry_catalogs'],
                  max_processing_geometry_bytes=final['footprint']['max_processing_geometry_bytes'],
                  worker_peak_host_rss_bytes=max((r['worker_peak_host_rss_bytes'] for r in metrics), default=0),
                  final_geometry_bytes=final['footprint']['retained_geometry_bytes'],
                  reference_histories_loaded=False, repetitions=1, warmup_runs=0,
                  memory_scope='Geometry nbytes exclude Python hypotheses, current caller input, output deltas and temporary arrays; RSS is whole-worker high-water including imports and output.',
                  runtime_scope='One correctness replay: push/finish computation timed; catalog I/O, output writing and validation excluded. Not a comparative benchmark.',
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
                  input_sha256=hashes,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  output_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()})
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('passed', 'events', 'compute_seconds', 'max_retained_geometry_catalogs', 'max_processing_geometry_catalogs')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); run(args.directory, args.output_dir)
