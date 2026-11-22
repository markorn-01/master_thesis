"""Optional host-callback products and independent batch verification.

No JAX dependency: the collision callback supplies its current sparse catalog.
The writer does not read those catalogs from disk or accumulate their history.
Stored catalogs are read only by explicitly requested offline verification.
"""
import csv
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.astrolink.streaming_history import StreamingShockHistory, MODES


GROUP_FIELDS = ('node_id', 'event', 'time', 'group', 'segment_id', 'previous_group', 'inherited',
                'surface_samples', 'valid_mach_samples', 'invalid_mach_samples', 'mach_p16',
                'median_mach', 'mach_p84', 'upstream_source_overlaps', 'downstream_source_overlaps',
                'original_rows_sha256', 'original_positions_sha256')


def tracking_source_files():
    return [Path(__file__)]+[Path('experiments/astrolink')/(name+'.py') for name in
        ('streaming_history', 'surface_history', 'shock_connectivity', 'event_tracking', 'event_history',
         'expanded_event_tracking', 'continuation_segments', 'continuation_status',
         'delayed_support', 'tracklet_fragments')]+[Path('experiments/wind_bubble')/(name+'.py')
        for name in ('build_group_history_table', 'compare_event_history_resolutions')]


class StreamingProductWriter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.tracker = StreamingShockHistory()
        self.events = 0; self.groups = 0; self.closed = False; self.failed = False
        self.stream = (self.directory/'group_history.csv').open('x', newline='')
        self.csv = csv.DictWriter(self.stream, fieldnames=GROUP_FIELDS); self.csv.writeheader()
        self.metadata = dict(status='running', mode='incremental_maximum_five_with_delayed_support',
                             events=0, group_observations=0, catalog_input='current host callback catalog',
                             input_catalogs_read_from_disk=False, future_event_latency=1,
                             settings=asdict(self.tracker.settings),
                             physical_ancestry_established=False)
        self._write_metadata()

    def _write_metadata(self):
        (self.directory/'metadata.json').write_text(json.dumps(self.metadata, indent=2)+'\n')

    def receive(self, catalog):
        if self.closed or self.failed:
            raise ValueError('Tracking writer is closed or failed.')
        try:
            delta = self.tracker.push(catalog)
            partition = delta.pop('partition')
            np.savez_compressed(self.directory/f'groups_{self.events:04d}.npz', **{
                k: partition[k] for k in ('labels', 'raw_labels', 'component_sizes')})
            with gzip.open(self.directory/f'event_{self.events:04d}.json.gz', 'wt') as stream:
                json.dump(delta, stream)
            self.csv.writerows(delta['group_statistics']); self.stream.flush()
            self.events += 1; self.groups += len(delta['nodes'])
            self.metadata.update(events=self.events, group_observations=self.groups,
                                 footprint=delta['footprint'])
            self._write_metadata()
        except Exception:
            self.failed = True
            raise

    def close(self, *, success=True):
        if self.closed:
            return
        try:
            if success and not self.failed:
                final = self.tracker.finish()
                (self.directory/'finish.json').write_text(json.dumps(final, indent=2)+'\n')
                self.metadata.update(status='complete', footprint=final['footprint'],
                                     segment_count=final['segment_count'])
            else:
                self.metadata['status'] = 'failed'
        except Exception:
            self.metadata['status'] = 'failed'
            raise
        finally:
            self.tracker = None  # Also release working geometry on failed runs.
            self.stream.close(); self.closed = True
            self._write_metadata()


def load_event(directory, index):
    with gzip.open(Path(directory)/f'event_{index:04d}.json.gz', 'rt') as stream:
        return json.load(stream)


def compare_tree(a, b, path='root'):
    """Categorical fields are exact; floating diagnostics use established tolerances."""
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError(f'{path}: different keys/types')
        for key in a:
            # Validate fingerprints against each side's actual source catalog
            # separately: a physical coordinate difference within the declared
            # float tolerance need not have the same cryptographic fingerprint.
            if key != 'original_positions_sha256':
                compare_tree(a[key], b[key], path+'.'+key)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise AssertionError(f'{path}: different list lengths/types')
        for index, (x, y) in enumerate(zip(a, b)):
            compare_tree(x, y, f'{path}[{index}]')
    elif isinstance(a, float):
        if not isinstance(b, (float, int)) or isinstance(b, bool):
            raise AssertionError(f'{path}: different scalar types')
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True, err_msg=path)
    elif type(a) is not type(b) or a != b:
        raise AssertionError(f'{path}: categorical mismatch: {a!r} != {b!r}')


def validate_fingerprints(directory, index, record):
    with np.load(Path(directory)/f'groups_{index:04d}.npz') as groups, \
            np.load(Path(directory).parent/f'surfaces_{index:04d}.npz') as catalog:
        labels = groups['labels']
        for row in record['group_statistics']:
            ids = np.flatnonzero(labels == row['group'])
            assert row['original_rows_sha256'] == hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()
            assert row['original_positions_sha256'] == hashlib.sha256(catalog['positions'][ids].tobytes()).hexdigest()


def compare_products(online, offline):
    online, offline = Path(online), Path(offline)
    meta = [json.loads((d/'metadata.json').read_text()) for d in (online, offline)]
    if any(m['status'] != 'complete' for m in meta):
        raise AssertionError('Require completed tracking writers.')
    compare_tree(*meta)
    exact = True
    expected_csv = [[], []]
    for i in range(meta[0]['events']):
        a, b = load_event(online, i), load_event(offline, i)
        validate_fingerprints(online, i, a); validate_fingerprints(offline, i, b)
        compare_tree(a, b, f'event_{i}')
        for side, record in enumerate((a, b)):
            expected_csv[side].extend({key: '' if row[key] is None else str(row[key]) for key in GROUP_FIELDS}
                                      for row in record['group_statistics'])
        exact = exact and a == b
        with np.load(online/f'groups_{i:04d}.npz') as x, np.load(offline/f'groups_{i:04d}.npz') as y:
            assert x.files == y.files
            for key in x.files:
                np.testing.assert_array_equal(x[key], y[key])
    a, b = [json.loads((d/'finish.json').read_text()) for d in (online, offline)]
    compare_tree(a, b, 'finish')
    exact = exact and a == b
    with (online/'group_history.csv').open() as x, (offline/'group_history.csv').open() as y:
        for side, rows in enumerate([list(csv.DictReader(x)), list(csv.DictReader(y))]):
            assert len(rows) == meta[0]['group_observations']
            assert rows == expected_csv[side], 'Tracking CSV does not reproduce its own event records.'
    return dict(streaming_tracking_match=True, compared_tracking_events=meta[0]['events'],
                compared_group_observations=meta[0]['group_observations'],
                tracking_full_records_exact=exact)


def check_batch_reference(catalog_directory, tracking_directory):
    """Independent whole-sequence execution from stored reference catalogs.

    This intentionally retains all reference catalogs only for a small explicit
    verification run. It does not replay the incremental tracker as its oracle.
    """
    from experiments.astrolink.surface_history import HistorySettings, analyze_surface_history
    from experiments.astrolink.delayed_support import delayed_support
    from experiments.wind_bubble.compare_event_history_resolutions import group_statistics
    from experiments.wind_bubble.build_group_history_table import statistics
    root, tracked = Path(catalog_directory), Path(tracking_directory)
    meta = json.loads((tracked/'metadata.json').read_text())
    catalogs = []
    for i in range(meta['events']):
        with np.load(root/f'surfaces_{i:04d}.npz') as data:
            catalogs.append({k: data[k].copy() for k in ('positions', 'normals', 'time', 'grid_spacing',
                'mach', 'mach_valid', 'source_positions', 'upstream_inside_any_source', 'downstream_inside_any_source')})
    times = [float(c['time']) for c in catalogs]
    batch = analyze_surface_history(catalogs, times, HistorySettings(max_partners=5))
    lookup = {(r['parent_event'], r['gate_reference_cells']): r['result'] for r in batch['records']}
    delayed_count = 0
    for i, (catalog, partition) in enumerate(zip(catalogs, batch['partitions'])):
        record = load_event(tracked, i)
        with np.load(tracked/f'groups_{i:04d}.npz') as saved:
            for key in saved.files:
                np.testing.assert_array_equal(saved[key], partition[key])
        nodes = [n for n in batch['graph']['nodes'] if n['event'] == i]
        nodes = group_statistics(catalog, partition['labels'], nodes)
        for node in nodes:
            selected = partition['labels'] == node['group']
            node.update(upstream_inside_source=int(catalog['upstream_inside_any_source'][selected].sum()),
                        downstream_inside_source=int(catalog['downstream_inside_any_source'][selected].sum()))
        compare_tree(record['nodes'], nodes, f'batch_nodes_{i}')
        rows, accounting = statistics(catalog, partition['labels'], nodes, i)
        compare_tree(record['group_statistics'], rows)
        compare_tree(record['accounting'], accounting)
        compare_tree(record['tracking_events'], [e for e in batch['graph']['events'] if e['child_event'] == i])
        for adjacent in record['adjacent_records']:
            compare_tree(adjacent['result'], lookup[i-1, adjacent['gate_reference_cells']])
        assert len(record['adjacent_records']) == (2 if i else 0)
        assert len(record['support_records']) == (4 if i >= 2 else 0)
        for support in record['support_records']:
            assert support['available_event'] == i and support['parent_event'] == i-2
            assert support['available_time'] == times[i]
            future = [lookup[i-1, g] for g in (1.5, 2.5)]
            expected = delayed_support([dict(c, labels=p['labels']) for c, p in
                zip(catalogs[i-2:i+1], batch['partitions'][i-2:i+1])], times[i-2:i+1],
                lookup[i-2, support['gate_reference_cells']], future,
                predict=dict(MODES)[support['method']])
            compare_tree(support['result'], expected); delayed_count += 1
    final = json.loads((tracked/'finish.json').read_text())
    assert final['events_received'] == len(catalogs)
    assert final['segment_count'] == batch['graph']['segment_count']
    assert final['tracking_event_count'] == len(batch['graph']['events'])
    assert len(final['pending_terminal_support']) == 4
    for support in final['pending_terminal_support']:
        assert support['available_event'] is None and support['available_time'] is None
        assert support['parent_event'] == len(catalogs)-2
        expected = delayed_support([dict(c, labels=p['labels']) for c, p in
            zip(catalogs[-2:], batch['partitions'][-2:])], times[-2:],
            lookup[len(catalogs)-2, support['gate_reference_cells']], predict=dict(MODES)[support['method']])
        compare_tree(support['result'], expected); delayed_count += 1
    return dict(tracking_batch_reference_match=True, batch_reference_events=len(catalogs),
                batch_reference_adjacent_results=len(batch['records']),
                batch_reference_delayed_results=delayed_count)
