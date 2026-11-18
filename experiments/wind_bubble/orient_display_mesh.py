"""Repair triangle winding without changing the fitted display geometry."""
import numpy as np

from experiments.wind_bubble.surface_mesh import mesh_topology


def orient_faces(vertices, faces):
    """Orient a manifold mesh, minimizing flips per edge-connected component.

    Original vertices, face membership, boundaries and all physical normal
    arrays remain unchanged. Nonorientable/nonmanifold inputs are rejected.
    """
    import open3d as o3d
    v = np.asarray(vertices, float); original = np.asarray(faces, int).reshape(-1, 3)
    before = mesh_topology(original)
    if not len(original) or before['inconsistent_shared_edge_winding'] == 0:
        return original.copy(), dict(passed=True, changed_faces=0, before=before, after=before)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(original))
    if before['nonmanifold_edges'] or len(mesh.get_non_manifold_vertices()) or not mesh.is_orientable():
        raise ValueError('Orientation-only repair requires an orientable manifold mesh.')
    if not mesh.orient_triangles(): raise ValueError('Native orientation failed.')
    np.testing.assert_array_equal(np.asarray(mesh.vertices), v)
    result = np.asarray(mesh.triangles).copy()
    np.testing.assert_array_equal(np.sort(result, axis=1), np.sort(original, axis=1))
    parent = np.arange(len(result)); owners = {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for i, face in enumerate(result):
        for a, b in zip(face, np.roll(face, -1)):
            key = tuple(sorted((int(a), int(b))))
            if key in owners:
                parent[find(i)] = find(owners[key])
            else: owners[key] = i
    components = np.array([find(i) for i in range(len(result))])
    old_normal = np.cross(v[original[:, 1]]-v[original[:, 0]], v[original[:, 2]]-v[original[:, 0]])
    new_normal = np.cross(v[result[:, 1]]-v[result[:, 0]], v[result[:, 2]]-v[result[:, 0]])
    flipped = np.einsum('ij,ij->i', old_normal, new_normal) < 0
    for component in np.unique(components):
        rows = np.flatnonzero(components == component)
        if flipped[rows].sum() > len(rows)/2:
            result[rows] = result[rows][:, [0, 2, 1]]
    after = mesh_topology(result)
    if after['inconsistent_shared_edge_winding']:
        raise ValueError('Inconsistent winding remains.')
    for key in ['edges', 'boundary_edges', 'nonmanifold_edges']:
        if before[key] != after[key]: raise ValueError('Orientation changed mesh connectivity.')
    new_normal = np.cross(v[result[:, 1]]-v[result[:, 0]], v[result[:, 2]]-v[result[:, 0]])
    changed = int((np.einsum('ij,ij->i', old_normal, new_normal) < 0).sum())
    return result, dict(passed=True, changed_faces=changed, before=before, after=after,
                        vertices_moved=False, faces_added_or_removed=False)
