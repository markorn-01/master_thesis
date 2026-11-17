"""Experimental, support-limited implicit MLS surfaces for display only.

This is a local oriented-plane blend, not an implementation of Alexa's full
MLS projection or a guarantee of reconstructing the true physical surface.
All newly generated vertices are separate from the diagnostic samples.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class DisplaySettings:
    step_cells: float = .5
    support_cells: float = 2.
    kernel_cells: float = .8
    max_nearest_cells: float = .85
    max_plane_offset_cells: float = .30
    min_neighbors: int = 3
    normal_angle: float = 60.
    require_local_hull: bool = True
    max_grid_nodes: int = 3_000_000

    def __post_init__(self):
        for key in ['step_cells', 'support_cells', 'kernel_cells',
                    'max_nearest_cells', 'max_plane_offset_cells']:
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError('Display lengths must be positive and finite.')
        if not 0 < self.normal_angle < 90 or self.min_neighbors < 3 or self.max_grid_nodes < 8:
            raise ValueError('Invalid neighborhood or grid settings.')


def geometry(positions, normals, labels, dx):
    p = np.asarray(positions, float); n = np.asarray(normals, float)
    labels = np.asarray(labels)
    if p.ndim != 2 or p.shape[1:] != (3,) or n.shape != p.shape or labels.shape != (len(p),):
        raise ValueError('Require matching Nx3 geometry and N labels.')
    if not np.isfinite(p).all() or not np.isfinite(n).all() or (np.linalg.norm(n, axis=1) == 0).any():
        raise ValueError('Require finite geometry and nonzero normals.')
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError('Require positive finite grid spacing.')
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Group labels must be integers.')
    return p, n/np.linalg.norm(n, axis=1)[:, None], labels


def plane_field(queries, p, n, dx, settings=DisplaySettings()):
    """Blend same-group oriented tangent planes in compact spatial support.

    Neighbors must agree with the nearest sample's signed normal. Unsupported
    queries return NaN, never a fabricated outside/inside sign.
    """
    queries = np.asarray(queries, float)
    if queries.ndim != 2 or queries.shape[1:] != (3,) or not np.isfinite(queries).all():
        raise ValueError('Require finite Nx3 query points.')
    tree = cKDTree(p); nearest_distance, nearest = tree.query(queries)
    neighbors = tree.query_ball_point(queries, settings.support_cells*dx)
    values = np.full(len(queries), np.nan); counts = np.zeros(len(queries), int)
    normal_cos = np.cos(np.deg2rad(settings.normal_angle))
    for row, ids in enumerate(neighbors):
        if len(ids) < settings.min_neighbors:
            continue
        ids = np.asarray(ids, int)
        ids = ids[(n[ids] @ n[nearest[row]]) >= normal_cos]
        if len(ids) < settings.min_neighbors:
            continue
        delta = queries[row]-p[ids]; d2 = np.einsum('ij,ij->i', delta, delta)
        t = np.sqrt(d2)/(settings.support_cells*dx)
        weights = np.exp(-d2/(settings.kernel_cells*dx)**2)*np.maximum(0, 1-t)**4*(4*t+1)
        if weights.sum() > 1e-20:
            values[row] = np.sum(weights*np.einsum('ij,ij->i', delta, n[ids]))/weights.sum()
            counts[row] = len(ids)
    return values, counts, nearest_distance, nearest


def trim_supported(vertices, faces, p, n, dx, settings=DisplaySettings()):
    """Sample vertices, edge midpoints and centroid for local support.

    This is an explicit finite sampling check, not a continuous guarantee that
    a triangle has no unsupported interior or self intersection.
    """
    v = np.asarray(vertices, float); f = np.asarray(faces, int).reshape(-1, 3)
    if not len(f):
        return f, dict(proposed_faces=0, unsupported_faces=0, supported_faces=0)
    tri = v[f]
    probes = np.concatenate([tri, (tri+np.roll(tri, 1, axis=1))/2, tri.mean(axis=1)[:, None]], axis=1)
    tree = cKDTree(p)
    distance, nearest = tree.query(probes.reshape(-1, 3))
    delta = probes.reshape(-1, 3)-p[nearest]
    offsets = np.abs(np.einsum('ij,ij->i', delta, n[nearest]))
    supported = ((distance <= settings.max_nearest_cells*dx) &
                 (offsets <= settings.max_plane_offset_cells*dx)).reshape(-1, 7).all(axis=1)
    if settings.require_local_hull:
        supported &= local_hull_support(probes.reshape(-1, 3), p, n, dx, settings).reshape(-1, 7).all(axis=1)
    cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    length = np.linalg.norm(cross, axis=1)
    supported &= length > dx*dx*1e-10
    # Orient using the nearest sample at the centroid, preserving signed normals.
    centroid_nearest = nearest.reshape(-1, 7)[:, -1]
    flip = np.einsum('ij,ij->i', cross, n[centroid_nearest]) < 0
    f = f.copy(); f[flip] = f[flip][:, [0, 2, 1]]
    return f[supported], dict(proposed_faces=len(f), unsupported_faces=int((~supported).sum()),
                             supported_faces=int(supported.sum()))


def local_hull_support(queries, p, n, dx, settings=DisplaySettings()):
    """Require local samples to surround a query in a signed tangent plane.

    A maximum angular gap no greater than pi is the 2D convex-hull membership
    test. This discourages extrapolation across sampling boundaries, but cannot
    identify two unresolved, similarly oriented sheets in one supplied group.
    """
    tree = cKDTree(p); queries = np.asarray(queries)
    distance, nearest = tree.query(queries)
    neighbors = tree.query_ball_point(queries, settings.support_cells*dx)
    result = distance <= dx*1e-7
    normal_cos = np.cos(np.deg2rad(settings.normal_angle))
    for row, ids in enumerate(neighbors):
        if result[row] or len(ids) < settings.min_neighbors:
            continue
        normal = n[nearest[row]]
        ids = np.asarray(ids, int); ids = ids[n[ids] @ normal >= normal_cos]
        if len(ids) < settings.min_neighbors:
            continue
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        u = np.cross(normal, axis); u /= np.linalg.norm(u); v = np.cross(normal, u)
        delta = p[ids]-queries[row]
        xy = np.column_stack([delta@u, delta@v])
        if (np.linalg.norm(xy, axis=1) <= dx*1e-7).any():
            result[row] = True
            continue
        angles = np.sort(np.arctan2(xy[:, 1], xy[:, 0]))
        result[row] = np.diff(np.r_[angles, angles[0]+2*np.pi]).max() <= np.pi+1e-8
    return result


def implicit_samples(positions, normals, labels, dx, settings=DisplaySettings()):
    from skimage.measure import marching_cubes
    p, n, labels = geometry(positions, normals, labels, dx)
    vertices, faces, groups, reports = [], [], [], []
    offset = 0
    for group in np.unique(labels[labels >= 0]):
        ids = np.flatnonzero(labels == group); points = p[ids]; normals = n[ids]
        if len(ids) < settings.min_neighbors:
            continue
        step = settings.step_cells*dx
        origin = np.floor((points.min(axis=0)-dx)/step)*step
        shape = np.ceil((points.max(axis=0)+dx-origin)/step).astype(int)+1
        size = int(np.prod(shape))
        if size > settings.max_grid_nodes:
            raise ValueError(f'Group {group} needs {size} grid nodes; explicit display limit exceeded.')
        # Sparse evaluation band: query only lattice nodes within sample support.
        tree = cKDTree(points)
        values = np.full(size, np.nan)
        for start in range(0, size, 16384):
            flat = np.arange(start, min(size, start+16384))
            queries = origin+np.column_stack(np.unravel_index(flat, tuple(shape)))*step
            distance = tree.query(queries)[0]
            near = distance <= settings.max_nearest_cells*dx+step*np.sqrt(3)
            if near.any():
                value, _, _, _ = plane_field(queries[near], points, normals, dx, settings)
                values[flat[near]] = value
        field = values.reshape(shape)
        # Require all eight corners to be supported. Unknown nodes never get a
        # sign, so marching cubes cannot create an artificial support boundary.
        valid = np.isfinite(field)
        cells = np.ones(tuple(shape-1), bool)
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    cells &= valid[a:a+shape[0]-1, b:b+shape[1]-1, c:c+shape[2]-1]
        mask = np.zeros(tuple(shape), bool); mask[:-1, :-1, :-1] = cells
        safe = np.nan_to_num(field, nan=0.)
        if not mask.any() or not (safe[valid].min() < 0 < safe[valid].max()):
            reports.append(dict(group=int(group), grid_nodes=size, faces=0))
            continue
        try:
            v, f, _, _ = marching_cubes(safe.astype(np.float32), level=0, spacing=(step,)*3,
                                       mask=mask, allow_degenerate=False, gradient_direction='ascent')
        except RuntimeError as error:
            if 'No surface found' not in str(error):
                raise
            reports.append(dict(group=int(group), grid_nodes=size, faces=0))
            continue
        v = v+origin
        # Some marching-cubes masks permit transition cells at their boundary.
        # Independently require the floor cell of each face centroid to be in
        # our all-eight-corners-valid set before evaluating support probes.
        ci = np.floor((v[f].mean(axis=1)-origin)/step).astype(int)
        inside = ((ci >= 0) & (ci < shape-1)).all(axis=1)
        allowed = np.zeros(len(f), bool)
        allowed[inside] = cells[tuple(ci[inside].T)]
        f = f[allowed]
        f, support = trim_supported(v, f, points, normals, dx, settings)
        if not len(f):
            reports.append(dict(group=int(group), grid_nodes=size, faces=0, **support))
            continue
        used, inverse = np.unique(f, return_inverse=True)
        vertices.append(v[used]); faces.append(inverse.reshape(-1, 3)+offset)
        groups.append(np.full(len(used), group, int)); offset += len(used)
        reports.append(dict(group=int(group), grid_nodes=size, faces=len(f), **support))
    return (np.concatenate(vertices) if vertices else np.empty((0, 3)),
            np.concatenate(faces) if faces else np.empty((0, 3), int),
            np.concatenate(groups) if groups else np.empty(0, int),
            dict(method='compact oriented-plane implicit MLS and masked marching cubes',
                 settings=asdict(settings), groups=reports, new_display_vertices=True,
                 physical_samples_modified=False))
