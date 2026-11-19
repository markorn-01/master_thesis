"""One-event-delayed geometric support beside immutable adjacent events.

Reuses the existing fragment-family model. Accepted future continuations are
derived from adjacent matching, never supplied as physical truth. Conditional
support neither changes the event graph nor assigns inherited identities.
"""
from copy import deepcopy

import numpy as np

from experiments.astrolink.tracklet_fragments import attach_fragments, validate_catalog


def continuations(result):
    return {(e['parents'][0], e['children'][0]) for e in result['events']
            if e['kind'] == 'continuation' and len(e['parents']) == len(e['children']) == 1}


def prepare_window(catalogs, current, future_results):
    """Keep established continuations out of fragment reassociation.

    For real two-gate comparisons, future seeds must be accepted at both gates.
    Synthetic comparisons have one gate and use that gate's accepted seeds.
    Combination parents remain available as family context, so an omitted
    fragment can add support without altering the accepted combination.
    """
    seeds = sorted(set.intersection(*(continuations(r) for r in future_results)))
    protected = continuations(current)
    protected_a = {p for p, _ in protected}
    protected_b = {c for _, c in protected}
    seeds = [(b, c) for b, c in seeds if b not in protected_b]
    selections = [
        (catalogs[0]['labels'] >= 0) & ~np.isin(catalogs[0]['labels'], list(protected_a)),
        np.isin(catalogs[1]['labels'], [b for b, c in seeds]),
        np.isin(catalogs[2]['labels'], [c for b, c in seeds])]
    selected = [{key: np.asarray(data[key])[mask].copy() for key in ('positions', 'normals', 'labels')}
                for data, mask in zip(catalogs, selections)]
    return selected, seeds, sorted(protected)


def delayed_support(catalogs, times, current, future_results=None, *, predict=True):
    """Return evidence channels; all future context is at most one event later.

    A two-state input has no future event and returns pending support. Future
    records may contain one gate (synthetic) or two gates (real consensus).
    The current accepted event list is copied exactly and never optimized again.
    """
    if len(catalogs) not in (2, 3) or len(times) != len(catalogs):
        raise ValueError('Require two or three catalogs with matching times.')
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError('Require finite, strictly increasing observation times.')
    for data in catalogs:
        validate_catalog(data)
    p = current['parameters']
    fixed = dict(coverage=.5, exclusive_fraction=.1, distance_weight=.25,
                 normal_angle=60., tie_tolerance=1e-8, max_partners=5)
    if any(p.get(k) != value for k, value in fixed.items()):
        raise ValueError('Require the frozen maximum-five adjacent settings.')
    accepted_pairs = {(link['parent_group'], link['child_group'])
                      for link in current['links'] if link['accepted']}
    result = dict(accepted_adjacent_events=deepcopy(current['events']),
                  baseline_parameters=deepcopy(p), window_status='pending_future',
                  future_seed_policy='intersection_of_accepted_continuations',
                  future_continuations=[], protected_continuations=[],
                  conditional_support=[], conflicts=[], analysis=None,
                  unresolved=[], prediction_enabled=bool(predict),
                  future_event_latency=1, physical_ancestry_established=False,
                  inherited_id_assigned=False)
    if len(catalogs) == 2:
        if future_results is not None:
            raise ValueError('Future results require a third observation.')
    else:
        if not future_results:
            raise ValueError('Derive future adjacent results before delayed support.')
        for future in future_results:
            if any(future['parameters'].get(k) != value for k, value in fixed.items()):
                raise ValueError('Future seeds must use the frozen adjacent model.')
        selected, seeds, protected = prepare_window(catalogs, current, future_results)
        result['future_continuations'] = [list(pair) for pair in seeds]
        result['protected_continuations'] = [list(pair) for pair in protected]
        a, b, c = selected
        if not seeds:
            result['window_status'] = 'no_unprotected_future_seed'
        elif not len(a['labels']):
            result['window_status'] = 'no_unprotected_parent'
        else:
            parents = np.unique(a['labels'])
            if len(parents)*len(seeds) > 5000:
                raise ValueError('Too many fragment/family comparisons for this bounded experiment.')
            raw = attach_fragments(a, b, c, seeds, times, p['gate'], predict=predict)
            result['window_status'] = 'evaluated'
            result['analysis'] = {key: deepcopy(raw[key]) for key in ('models', 'assignments', 'parameters')}
            result['analysis']['families'] = [
                {**{k: v for k, v in family.items() if k != 'accepted'},
                 'coverage_pass': family['accepted']} for family in raw['families']]
            for link in raw['links']:
                parent, child = link['parent_group'], link['child_group']
                other_targets = sorted(c for p, c in accepted_pairs if p == parent and c != child)
                row = dict(parent_group=parent, child_group=child,
                           evidence='conditional_geometric_support',
                           already_in_adjacent=(parent, child) in accepted_pairs,
                           conflicting_adjacent_targets=other_targets,
                           physical_ancestry_established=False)
                result['conditional_support'].append(row)
                if other_targets:
                    result['conflicts'].append(deepcopy(row))
    support_parents = {r['parent_group'] for r in result['conditional_support']}
    for parent in current['parents']:
        if parent['status'] == 'assigned':
            continue
        result['unresolved'].append(dict(
            parent_group=parent['parent_group'], adjacent_status=parent['status'],
            support_status='conditional_support_only' if parent['parent_group'] in support_parents
            else result['window_status'] if result['window_status'] != 'evaluated' else 'no_accepted_family',
            physical_ancestry_established=False))
    return result
