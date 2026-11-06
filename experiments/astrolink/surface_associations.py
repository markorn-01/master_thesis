"""Geometric group associations between fresh surface catalogs, not point IDs."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def association_scores(positions_a, normals_a, labels_a, positions_b, normals_b,
                       labels_b, max_distance, max_angle_degrees=60.):
    """Count unique covered points in each direction for every group pair.

    A point is covered when at least one point of the other group is within
    max_distance and has a compatible signed normal. No one-to-one assignment,
    source origin, persistent row ID, velocity prediction or gap bridging.
    Negative labels are excluded. All eligible groups are returned, including
    pairs with zero support. Coverage is by sample count, not surface area.
    """
    pa, pb = np.asarray(positions_a, float), np.asarray(positions_b, float)
    na, nb = np.asarray(normals_a, float), np.asarray(normals_b, float)
    la, lb = np.asarray(labels_a), np.asarray(labels_b)
    for p, n, labels in [(pa, na, la), (pb, nb, lb)]:
        if p.ndim != 2 or p.shape[1] != 3 or n.shape != p.shape or labels.shape != (len(p),):
            raise ValueError('Require positions/normals (N,3) and labels (N,).')
        length = np.linalg.norm(n, axis=1)
        if not (np.isfinite(p).all() and np.isfinite(n).all() and (length > 0).all()):
            raise ValueError('Require finite geometry and nonzero normals.')
    if not np.isfinite(max_distance) or max_distance <= 0 or not 0 < max_angle_degrees < 90:
        raise ValueError('Invalid association gates.')
    ids_a, sizes_a = np.unique(la[la >= 0], return_counts=True)
    ids_b, sizes_b = np.unique(lb[lb >= 0], return_counts=True)
    if not len(ids_a) or not len(ids_b): return []
    na = na / np.linalg.norm(na, axis=1)[:, None]
    nb = nb / np.linalg.norm(nb, axis=1)[:, None]
    covered_a, covered_b = {}, {}
    rows_b = np.flatnonzero(lb >= 0)
    tree = cKDTree(pb[rows_b])
    cutoff = np.cos(np.deg2rad(max_angle_degrees))
    for i in np.flatnonzero(la >= 0):
        neighbors = rows_b[np.asarray(tree.query_ball_point(pa[i], max_distance), dtype=int)]
        neighbors = neighbors[(nb[neighbors] @ na[i]) >= cutoff]
        for group_b in np.unique(lb[neighbors]):
            key = (int(la[i]), int(group_b))
            covered_a.setdefault(key, set()).add(int(i))
            covered_b.setdefault(key, set()).update(map(int, neighbors[lb[neighbors] == group_b]))
    rows = []
    for group_a, size_a in zip(ids_a, sizes_a):
        for group_b, size_b in zip(ids_b, sizes_b):
            key = (int(group_a), int(group_b))
            count_a, count_b = len(covered_a.get(key, ())), len(covered_b.get(key, ()))
            rows.append(dict(parent_group=int(group_a), child_group=int(group_b),
                             parent_samples=int(size_a), child_samples=int(size_b),
                             covered_parent_samples=count_a, covered_child_samples=count_b,
                             parent_coverage=count_a / int(size_a), child_coverage=count_b / int(size_b)))
    return rows


def accepted_links(scores, min_coverage):
    """Require the declared minimum coverage independently in both directions."""
    if not np.isfinite(min_coverage) or not 0 < min_coverage <= 1:
        raise ValueError('Coverage threshold must be in (0,1].')
    return [row for row in scores if row['parent_coverage'] >= min_coverage
            and row['child_coverage'] >= min_coverage]
