"""Delayed geometric fragment support conditioned on a later continuation.

Original bounded adaptation of delayed decisions/track-segment linking, not
Reid's probabilistic MHT or Jaqaman's LAP implementation. No physical ancestry
or material point correspondence is inferred.
"""
import numpy as np
from scipy.spatial import cKDTree


def compatible_pairs(pa, na, pb, nb, gate, angle=60.):
    """Independent surface samples, gated by distance and signed normals."""
    pairs = cKDTree(pa).sparse_distance_matrix(cKDTree(pb), gate, output_type='coo_matrix')
    keep = np.einsum('ij,ij->i', na[pairs.row], nb[pairs.col]) >= np.cos(np.deg2rad(angle))
    return pairs.row[keep], pairs.col[keep], pairs.data[keep]


def validate_catalog(catalog):
    p, n, labels = [np.asarray(catalog[k]) for k in ['positions', 'normals', 'labels']]
    if p.ndim != 2 or p.shape[1] != 3 or n.shape != p.shape or labels.shape != (len(p),):
        raise ValueError('Require Nx3 positions/normals and N group labels.')
    if not np.isfinite(p).all() or not np.isfinite(n).all() or (np.linalg.norm(n, axis=1) <= 0).any():
        raise ValueError('Require finite geometry and nonzero normals.')
    return p.astype(float), n/np.linalg.norm(n, axis=1)[:, None], labels


def attach_fragments(a, b, c, continuations, times, gate, *, predict=True,
                     coverage=.5, margin=.1, normal_angle=60., distance_weight=.25):
    """Attach any number of fragments to distinct future-conditioned families.

    Continuations are supplied stable B->C pairs. Median normal displacement
    between B and C estimates a single normal speed for each surface. Predict B
    backward to A, then assign each A fragment to a uniquely preferred B family.
    A family is reported only if the fragments jointly cover >= coverage of the
    predicted B samples. Overlapping fragments are not exclusive ancestry claims.
    """
    if (not np.isfinite(gate) or gate <= 0 or len(times) != 3 or
            not np.isfinite(times).all() or not times[0] < times[1] < times[2] or
            not 0 < coverage <= 1 or not 0 < margin < 1 or not 0 < normal_angle < 90 or
            not 0 <= distance_weight < 1):
        raise ValueError('Invalid window, gate or thresholds.')
    pa, na, la = validate_catalog(a); pb, nb, lb = validate_catalog(b); pc, nc, lc = validate_catalog(c)
    continuations = list(continuations)
    if len({p for p, q in continuations}) != len(continuations) or len({q for p, q in continuations}) != len(continuations):
        raise ValueError('Future continuations must be one-to-one.')
    models, support = [], {}
    ratio = (times[1]-times[0])/(times[2]-times[1])
    for child, later in sorted(continuations):
        bi = np.flatnonzero(lb == child); ci = np.flatnonzero(lc == later)
        if not len(bi) or not len(ci): raise ValueError('Unknown continuation group.')
        i, j, distances = compatible_pairs(pb[bi], nb[bi], pc[ci], nc[ci], gate, normal_angle)
        best = {}
        for x, y, d in zip(i, j, distances):
            if int(x) not in best or d < best[int(x)][1]: best[int(x)] = (int(y), float(d))
        step = float(np.median([np.dot(pc[ci[y]]-pb[bi[x]], nb[bi[x]]) for x, (y, _) in best.items()])) if best else 0.
        coverage_b = len(set(map(int, i)))/len(bi); coverage_c = len(set(map(int, j)))/len(ci)
        usable = min(coverage_b, coverage_c) >= coverage
        models.append(dict(child_group=int(child), later_group=int(later), usable=usable,
            future_parent_coverage=coverage_b, future_child_coverage=coverage_c,
            median_normal_displacement=step, backward_time_ratio=float(ratio)))
        if not usable: continue
        predicted = pb[bi] - (ratio*step*nb[bi] if predict else 0)
        i, j, distances = compatible_pairs(pa, na, predicted, nb[bi], gate, normal_angle)
        for parent in sorted(map(int, np.unique(la[la >= 0]))):
            keep = la[i] == parent; pi, cj, d = i[keep], j[keep], distances[keep]
            per_point = {}
            for row, distance in zip(pi, d):
                per_point[int(row)] = min(per_point.get(int(row), np.inf), float(distance))
            fraction = len(per_point)/int((la == parent).sum())
            cost = float(np.mean(list(per_point.values())))/gate if per_point else 0.
            support[parent, int(child)] = dict(coverage=fraction,
                score=fraction*(1-distance_weight*cost), child_rows=set(map(int, cj)), child_size=len(bi))
    assignments = []
    for parent in sorted(map(int, np.unique(la[la >= 0]))):
        ranked = sorted([(value['score'], child, value) for (p, child), value in support.items()
                         if p == parent and value['coverage'] >= coverage], key=lambda x:(-x[0], x[1]))
        row = dict(parent_group=parent, child_group=None, status='no_support', score_margin=None)
        if ranked:
            best, child, value = ranked[0]; gap = best-(ranked[1][0] if len(ranked) > 1 else 0.)
            row.update(score_margin=float(gap), best_coverage=value['coverage'])
            if gap >= margin: row.update(child_group=child, status='provisional_support')
            else: row['status'] = 'ambiguous_family'
        assignments.append(row)
    families = []
    for model in models:
        child = model['child_group']; selected = [r for r in assignments if r['child_group'] == child]
        if not selected: continue
        union = set().union(*(support[r['parent_group'], child]['child_rows'] for r in selected))
        fraction = len(union)/support[selected[0]['parent_group'], child]['child_size']
        accepted = fraction >= coverage
        for r in selected: r['status'] = 'conditional_support' if accepted else 'insufficient_family_coverage'
        families.append(dict(child_group=child, later_group=model['later_group'],
            parent_groups=[r['parent_group'] for r in selected], child_union_coverage=fraction,
            accepted=accepted, physical_ancestry_established=False))
    links = [dict(parent_group=r['parent_group'], child_group=r['child_group'], accepted=True)
             for r in assignments if r['status'] == 'conditional_support']
    return dict(models=models, assignments=assignments, families=families, links=links,
        physical_ancestry_established=False, delayed_decision=True, prediction_enabled=predict,
        parameters=dict(coverage=coverage, margin=margin, normal_angle=normal_angle,
                        distance_weight=distance_weight, gate=gate))
