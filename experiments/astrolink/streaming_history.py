"""Incremental grouping, event history and one-event-delayed fragment support.

Consumes sparse physical catalogs, not hydro states. It retains two owned
geometry catalogs between pushes, uses a third during a push, and emits
completed records for a caller to write rather than accumulating a history.
The algorithms/thresholds are the existing frozen maximum-five prototype.
"""
from copy import deepcopy

import numpy as np

from experiments.astrolink.continuation_segments import assign_segments
from experiments.astrolink.delayed_support import delayed_support
from experiments.astrolink.expanded_event_tracking import expanded_tracking
from experiments.astrolink.shock_connectivity import connectivity_partition
from experiments.astrolink.surface_history import HistorySettings
from experiments.wind_bubble.build_group_history_table import statistics
from experiments.wind_bubble.compare_event_history_resolutions import group_statistics


MODES = [('delayed_static_family', False), ('delayed_normal_motion_family', True)]


def owned_catalog(source):
    """Copy only required arrays so views cannot retain a caller's full state."""
    required = ('positions', 'normals', 'mach', 'mach_valid', 'source_positions',
                'upstream_inside_any_source', 'downstream_inside_any_source')
    data = {key: np.array(source[key], copy=True, order='C') for key in required}
    data['time'] = float(source['time']); data['grid_spacing'] = float(source['grid_spacing'])
    p, n = data['positions'], data['normals']
    if p.ndim != 2 or p.shape[1:] != (3,) or n.shape != p.shape:
        raise ValueError('Require matching Nx3 positions/normals.')
    if not np.isfinite(p).all() or not np.isfinite(n).all() or (np.linalg.norm(n, axis=1) <= 0).any():
        raise ValueError('Require finite positions and nonzero finite normals.')
    if not np.isfinite(data['time']) or not np.isfinite(data['grid_spacing']) or data['grid_spacing'] <= 0:
        raise ValueError('Require finite time and positive finite grid spacing.')
    sources = data['source_positions']
    if sources.ndim != 2 or sources.shape[1:] != (3,) or not len(sources) or not np.isfinite(sources).all():
        raise ValueError('Require finite source positions for the existing descriptive statistics.')
    for key in ('mach', 'mach_valid', 'upstream_inside_any_source', 'downstream_inside_any_source'):
        if data[key].shape != (len(p),):
            raise ValueError('Require one physical property/flag per sparse sample.')
    for key in ('mach_valid', 'upstream_inside_any_source', 'downstream_inside_any_source'):
        if data[key].dtype != np.bool_:
            raise ValueError('Validity/source-overlap flags must be Boolean arrays.')
    return data


def interval_events(previous_event, selections, event_counter):
    """Same sorted hypotheses, sharing rule and event IDs as assemble_history."""
    hypotheses = {}; sets = []
    for record in selections:
        selected = set()
        for event in record['result']['events']:
            key = event['kind'], tuple(sorted(event['parents'])), tuple(sorted(event['children']))
            selected.add(key)
            hypotheses.setdefault(key, []).append(record['gate_reference_cells'])
        sets.append(selected)
    shared = sets[0] & sets[1]
    pairs = [(p[0], c[0]) for kind, p, c in sorted(shared) if kind == 'continuation']
    events = []
    for (kind, parents, children), gates in sorted(hypotheses.items()):
        events.append(dict(event_id=f'h{event_counter+len(events):04d}', parent_event=previous_event,
                           child_event=previous_event+1, kind=kind,
                           parents=[f'e{previous_event:04d}:g{p}' for p in parents],
                           children=[f'e{previous_event+1:04d}:g{c}' for c in children],
                           status='shared' if (kind, parents, children) in shared else 'distance_specific',
                           selected_at_gates=sorted(gates)))
    return pairs, events


class StreamingShockHistory:
    """push() emits current observations and any newly available past support.

    A caller owns the returned delta and must write/discard it to bound its
    own memory. No complete node/event/output list is stored by this class.
    finish() emits pending terminal windows and releases retained geometry;
    no future observation or inherited identity is fabricated.
    """
    def __init__(self, settings=None):
        self.settings = HistorySettings(max_partners=5) if settings is None else settings
        if self.settings.max_partners != 5:
            raise ValueError('Delayed support currently requires the frozen maximum-five model.')
        self._recent = []
        self._previous_results = []
        self._mapping = {}
        self._next_segment = 0
        self._next_event = 0
        self._count = 0
        self._spacing = None
        self._sources = None
        self._closed = False
        self._max_processing_catalogs = 0
        self._max_processing_geometry_bytes = 0

    def footprint(self):
        """Owned geometry bytes exclude Python metadata, outputs and temporaries."""
        return dict(events_received=self._count, closed=self._closed,
                    retained_geometry_catalogs=len(self._recent),
                    retained_geometry_bytes=sum(a.nbytes for frame in self._recent for a in frame['geometry'].values()),
                    retained_adjacent_intervals=int(bool(self._previous_results)),
                    retained_group_mapping_entries=len(self._mapping),
                    max_processing_geometry_catalogs=self._max_processing_catalogs,
                    max_processing_geometry_bytes=self._max_processing_geometry_bytes)

    def push(self, catalog):
        if self._closed:
            raise ValueError('Tracker is closed.')
        data = owned_catalog(catalog); time = data['time']; spacing = data['grid_spacing']
        if self._recent and time <= self._recent[-1]['time']:
            raise ValueError('Events must arrive in strictly increasing actual time order.')
        if self._spacing is not None and (spacing != self._spacing or not np.array_equal(data['source_positions'], self._sources)):
            raise ValueError('This replay requires fixed grid spacing and source positions.')
        partition = connectivity_partition(data['positions'], data['normals'], spacing,
            radius_cells=self.settings.grouping_radius_cells, max_angle_degrees=self.settings.normal_angle,
            min_display_size=self.settings.min_group_samples)
        geometry = dict(positions=data['positions'], normals=data['normals'], labels=partition['labels'])
        current = dict(time=time, geometry=geometry)
        index = self._count
        new_results, events, pairs = [], [], []
        if self._recent:
            a = self._recent[-1]['geometry']
            for number, gate in enumerate(self.settings.association_gates):
                result = expanded_tracking(a['positions'], a['normals'], a['labels'],
                                           geometry['positions'], geometry['normals'], geometry['labels'], gate)
                new_results.append(dict(parent_event=index-1, child_event=index, gate_index=number,
                                        gate_reference_cells=gate*64, result=result))
            pairs, events = interval_events(index-1, new_results, self._next_event)
        groups = sorted(map(int, np.unique(geometry['labels'][geometry['labels'] >= 0])))
        mapping, nodes, next_segment = assign_segments(self._mapping, groups, pairs, self._next_segment)
        nodes = [dict(node_id=f'e{index:04d}:g{row["group"]}', event=index, time=time, **row) for row in nodes]
        nodes = group_statistics(data, partition['labels'], nodes)
        for node in nodes:
            mask = partition['labels'] == node['group']
            node.update(upstream_inside_source=int(data['upstream_inside_any_source'][mask].sum()),
                        downstream_inside_source=int(data['downstream_inside_any_source'][mask].sum()))
        stats, accounting = statistics(data, partition['labels'], nodes, index)
        support_records = []
        processing = self._recent+[current]
        if len(processing) == 3:
            catalogs = [r['geometry'] for r in processing]; times = [r['time'] for r in processing]
            future = [r['result'] for r in new_results]
            for record in self._previous_results:
                for method, predict in MODES:
                    result = delayed_support(catalogs, times, record['result'], future, predict=predict)
                    support_records.append(dict(method=method, parent_event=index-2, child_event=index-1,
                                                available_event=index, available_time=time,
                                                gate_reference_cells=record['gate_reference_cells'], result=result))
        # Commit only after all algorithms/statistics succeed. Returned labels
        # and adjacent hypotheses are detached from the retained working state.
        emitted_partition = {key: value.copy() for key, value in partition.items()}
        emitted_results = deepcopy(new_results)
        self._max_processing_catalogs = max(self._max_processing_catalogs, len(processing))
        self._max_processing_geometry_bytes = max(self._max_processing_geometry_bytes,
            sum(a.nbytes for frame in processing for a in frame['geometry'].values()))
        self._recent = processing[-2:]
        self._previous_results = new_results
        self._mapping = mapping
        self._next_segment = next_segment
        self._next_event += len(events)
        self._count += 1
        self._spacing = spacing
        self._sources = data['source_positions'].copy()
        return dict(event=index, time=time, partition=emitted_partition, nodes=nodes,
                    tracking_events=events, adjacent_records=emitted_results, support_records=support_records,
                    group_statistics=stats, accounting=accounting, footprint=self.footprint())

    def finish(self):
        if self._closed:
            raise ValueError('Tracker is closed.')
        pending = []
        if len(self._recent) == 2:
            catalogs = [r['geometry'] for r in self._recent]; times = [r['time'] for r in self._recent]
            for record in self._previous_results:
                for method, predict in MODES:
                    result = delayed_support(catalogs, times, record['result'], predict=predict)
                    pending.append(dict(method=method, parent_event=self._count-2, child_event=self._count-1,
                                        available_event=None, available_time=None,
                                        gate_reference_cells=record['gate_reference_cells'], result=result))
        self._recent = []; self._previous_results = []; self._mapping = {}; self._sources = None
        self._closed = True
        return dict(events_received=self._count, segment_count=self._next_segment,
                    tracking_event_count=self._next_event, pending_terminal_support=pending,
                    footprint=self.footprint(), physical_ancestry_established=False)
