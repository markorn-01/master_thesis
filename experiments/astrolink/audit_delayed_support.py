"""Independent blocked dense audit of the existing fragment-family equations."""
import numpy as np
from scipy.spatial.distance import cdist

from experiments.astrolink.delayed_support import prepare_window


def dense_support(pa, na, pb, nb, gate, angle, block=128):
    """All-pairs oracle, bounded in memory rather than a second KD tree.

    A normal-compatible nearest distance/index for every A row and the union
    of compatible B rows suffice for the family equations. No row matching
    between observations is assumed. Ties choose the lowest B row index.
    """
    # Normals are normalized once by the caller. Repeated normalization of
    # float32 simulation normals would define a slightly different oracle.
    nearest = np.full(len(pa), np.inf)
    indices = np.full(len(pa), -1, int)
    footprint = np.zeros(len(pb), bool)
    if not len(pb):
        return nearest, indices, footprint
    for start in range(0, len(pa), block):
        stop = min(start+block, len(pa))
        distance = cdist(pa[start:stop], pb)
        distance[(na[start:stop]@nb.T) < np.cos(np.deg2rad(angle))] = np.inf
        j = np.argmin(distance, axis=1)
        nearest[start:stop] = distance[np.arange(stop-start), j]
        indices[start:stop] = j
        footprint |= (distance <= gate).any(axis=0)
    return nearest, indices, footprint


def check_window(catalogs, times, current, future, result):
    assert result['accepted_adjacent_events'] == current['events']
    assert result['physical_ancestry_established'] is False
    assert result['inherited_id_assigned'] is False
    if result['analysis'] is None:
        assert not result['conditional_support'] and not result['conflicts']
        if len(catalogs) == 2:
            assert result['window_status'] == 'pending_future'
        else:
            selected, seeds, _ = prepare_window(catalogs, current, future)
            assert not seeds or not len(selected[0]['labels'])
        return dict(dense_models=0, dense_parent_family_pairs=0)
    (a, b, c), seeds, protected = prepare_window(catalogs, current, future)
    a, b, c = [dict(data, positions=data['positions'].astype(float),
                    normals=data['normals']/np.linalg.norm(data['normals'], axis=1)[:, None])
               for data in (a, b, c)]
    assert result['future_continuations'] == [list(pair) for pair in seeds]
    assert result['protected_continuations'] == [list(pair) for pair in protected]
    raw = result['analysis']; p = raw['parameters']; gate = p['gate']
    ratio = (times[1]-times[0])/(times[2]-times[1])
    models = []; scores = {}; footprints = {}; checks = 0
    for child, later in seeds:
        bi = b['labels'] == child; ci = c['labels'] == later
        bp, bn = b['positions'][bi], b['normals'][bi]
        cp, cn = c['positions'][ci], c['normals'][ci]
        distance, nearest, footprint = dense_support(bp, bn, cp, cn, gate, p['normal_angle'])
        good = distance <= gate
        step = float(np.median(np.einsum('ij,ij->i', cp[nearest[good]]-bp[good], bn[good]))) if good.any() else 0.
        usable = min(good.mean(), footprint.mean()) >= p['coverage']
        models.append(dict(child_group=child, later_group=later, usable=bool(usable),
                           future_parent_coverage=float(good.mean()),
                           future_child_coverage=float(footprint.mean()),
                           median_normal_displacement=step, backward_time_ratio=float(ratio)))
        if not usable:
            continue
        predicted = bp-ratio*step*bn if result['prediction_enabled'] else bp
        for parent in sorted(map(int, np.unique(a['labels']))):
            pi = a['labels'] == parent
            distance, _, footprint = dense_support(a['positions'][pi], a['normals'][pi],
                                                    predicted, bn, gate, p['normal_angle'])
            good = distance <= gate; fraction = good.mean()
            footprints[parent, child] = footprint
            checks += 1
            if fraction >= p['coverage']:
                score = fraction*(1-p['distance_weight']*distance[good].mean()/gate)
                scores.setdefault(parent, []).append((float(score), child, float(fraction)))
    assert len(models) == len(raw['models'])
    for expected, actual in zip(models, raw['models']):
        for key, value in expected.items():
            if isinstance(value, float):
                np.testing.assert_allclose(value, actual[key], rtol=1e-11, atol=1e-12)
            else:
                assert value == actual[key]
    assignments = []
    for parent in sorted(map(int, np.unique(a['labels']))):
        ranked = sorted(scores.get(parent, []), key=lambda r: (-r[0], r[1]))
        row = dict(parent_group=parent, child_group=None, status='no_support', score_margin=None)
        if ranked:
            score, child, fraction = ranked[0]
            gap = score-(ranked[1][0] if len(ranked) > 1 else 0.)
            row.update(score_margin=gap, best_coverage=fraction)
            row['status'] = 'provisional_support' if gap >= p['margin'] else 'ambiguous_family'
            if gap >= p['margin']:
                row['child_group'] = child
        assignments.append(row)
    families = []
    for model in models:
        child = model['child_group']
        chosen = [r for r in assignments if r['child_group'] == child]
        if not chosen:
            continue
        union = np.logical_or.reduce([footprints[r['parent_group'], child] for r in chosen])
        coverage = union.mean(); passed = coverage >= p['coverage']
        for row in chosen:
            row['status'] = 'conditional_support' if passed else 'insufficient_family_coverage'
        families.append(dict(child_group=child, later_group=model['later_group'],
                             parent_groups=[r['parent_group'] for r in chosen],
                             child_union_coverage=float(coverage), coverage_pass=bool(passed),
                             physical_ancestry_established=False))
    assert families == raw['families']
    assert len(assignments) == len(raw['assignments'])
    for expected, actual in zip(assignments, raw['assignments']):
        assert expected.keys() == actual.keys()
        for key, value in expected.items():
            if isinstance(value, float):
                np.testing.assert_allclose(value, actual[key], rtol=1e-11, atol=1e-12)
            else:
                assert value == actual[key]
    expected_pairs = {(r['parent_group'], r['child_group']) for r in assignments if r['status'] == 'conditional_support'}
    assert expected_pairs == {(r['parent_group'], r['child_group']) for r in result['conditional_support']}
    accepted = {(r['parent_group'], r['child_group']) for r in current['links'] if r['accepted']}
    for row in result['conditional_support']:
        assert row['already_in_adjacent'] == ((row['parent_group'], row['child_group']) in accepted)
        assert row['conflicting_adjacent_targets'] == sorted(c for a, c in accepted if a == row['parent_group'] and c != row['child_group'])
    assert result['conflicts'] == [r for r in result['conditional_support'] if r['conflicting_adjacent_targets']]
    return dict(dense_models=len(models), dense_parent_family_pairs=checks)
