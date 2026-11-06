"""Simple signed-normal proximity graph, not a general physical shock tracker."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def connectivity_partition(positions, normals, dx, radius_cells=1.5,
                           max_angle_degrees=30., min_display_size=20):
    """Retain every component; gray small ones only in the display partition.

    Edges require distance <= radius_cells*dx and signed normal dot product
    >= cos(max_angle_degrees). Opposite normals do not become equivalent.
    No source positions, Mach thresholds, curvature or temporal prediction.
    """
    positions, normals = np.asarray(positions,dtype=float), np.asarray(normals,dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or normals.shape != positions.shape:
        raise ValueError('Require matching (N,3) positions and normals.')
    if (not np.isfinite(dx) or dx <= 0 or not np.isfinite(radius_cells) or radius_cells <= 0
            or not np.isfinite(max_angle_degrees) or not 0 < max_angle_degrees < 90
            or int(min_display_size) != min_display_size or min_display_size < 1):
        raise ValueError('Invalid connectivity thresholds.')
    lengths = np.linalg.norm(normals,axis=1)
    good = (np.isfinite(positions).all(axis=1) & np.isfinite(normals).all(axis=1)
            & np.isfinite(lengths) & (lengths > 0))
    rows = np.flatnonzero(good)
    raw = np.full(len(positions),-2,dtype=np.int32)
    labels = raw.copy()
    if not len(rows):
        return dict(labels=labels,raw_labels=raw,eligible_rows=rows,
                    edges=np.empty((0,2),dtype=np.int32),component_sizes=np.array([],dtype=int))
    p,n = positions[good],normals[good]/lengths[good,None]
    pairs = cKDTree(p).query_pairs(radius_cells*dx,eps=0,output_type='ndarray')
    if len(pairs):
        dots = np.einsum('ij,ij->i',n[pairs[:,0]],n[pairs[:,1]])
        pairs = pairs[dots >= np.cos(np.deg2rad(max_angle_degrees))]
    ii = np.concatenate((pairs[:,0],pairs[:,1]))
    jj = np.concatenate((pairs[:,1],pairs[:,0]))
    graph = coo_matrix((np.ones(len(ii),dtype=np.uint8),(ii,jj)),shape=(len(rows),len(rows))).tocsr()
    count, local = connected_components(graph,directed=False)
    sizes = np.bincount(local,minlength=count)
    raw[rows] = local
    labels[rows] = np.where(sizes[local] >= min_display_size,local,-1)
    return dict(labels=labels,raw_labels=raw,eligible_rows=rows,
                edges=rows[pairs].astype(np.int32),component_sizes=sizes)
