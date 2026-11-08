"""Diagnose one deterministic display-mesh patch without changing reconstruction."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from experiments.wind_bubble.ball_pivot_mesh import guard_faces


def triangle_angles(p, n, faces):
    """Maximum edge departure from endpoint tangent planes and normal spread."""
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n, axis=1)[:, None]
    tangent = np.zeros(len(faces))
    spread = np.zeros(len(faces))
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        edge = p[faces[:, b]] - p[faces[:, a]]
        length = np.linalg.norm(edge, axis=1)
        if (length == 0).any():
            raise ValueError('Angles require distinct triangle vertices.')
        for end in [a, b]:
            dot = np.einsum('ij,ij->i', edge, n[faces[:, end]]) / length
            tangent = np.maximum(tangent, np.degrees(np.arcsin(np.clip(abs(dot), 0, 1))))
        dot = np.einsum('ij,ij->i', n[faces[:, a]], n[faces[:, b]])
        spread = np.maximum(spread, np.degrees(np.arccos(np.clip(dot, -1, 1))))
    return tangent, spread


def run(mesh, output, direction='diagonal'):
    digest = hashlib.sha256(mesh.read_bytes()).hexdigest()
    with np.load(mesh) as a:
        p = a['positions'].astype(float); n = a['normals'].astype(float)
        labels = a['labels']; raw = a['raw_faces']; accepted = a['faces']
        dx = float(a['grid_spacing']); time = float(a['time'])
    # A reproducible location, not a search for the best/worst-looking patch.
    group = 1
    rows = np.flatnonzero(labels == group)
    if not len(rows):
        raise ValueError('This diagnostic requires event 16 inner group 1.')
    view = np.array([0., 1, 0]) if direction == 'axis' else np.array([-1., 1, 1]) / np.sqrt(3)
    anchor = rows[np.argmax(p[rows] @ view)]
    center = p[anchor]
    core = rows[np.linalg.norm(p[rows] - center, axis=1) <= 3 * dx]
    context = rows[np.linalg.norm(p[rows] - center, axis=1) <= 5 * dx]
    patch_faces = raw[(labels[raw[:, 0]] == group) &
                      (np.linalg.norm(p[raw].mean(axis=1) - center, axis=1) <= 3 * dx)]
    kept_keys = {tuple(sorted(f)) for f in accepted}
    tangent, spread = triangle_angles(p, n, patch_faces)
    records = []
    for i, face in enumerate(patch_faces):
        kept = tuple(sorted(face)) in kept_keys
        single, rejected = guard_faces(p, n, labels, face[None], dx)
        reasons = [k for k, v in rejected.items() if v]
        # On this input there must be no duplicate/nonmanifold global rejection.
        if kept != bool(len(single)):
            raise AssertionError('Global topology rejection needs separate attribution.')
        records.append(dict(face_index=i, vertex_0=int(face[0]), vertex_1=int(face[1]),
                            vertex_2=int(face[2]), accepted=kept,
                            first_rejection='accepted' if kept else reasons[0],
                            max_tangent_departure_degrees=float(tangent[i]),
                            max_pair_normal_angle_degrees=float(spread[i])))
    # Reproduce the whole saved acceptance set before interpreting a local patch.
    recomputed, rejection_counts = guard_faces(p, n, labels, raw, dx)
    assert {tuple(sorted(f)) for f in recomputed} == kept_keys
    raw_used = np.unique(raw[labels[raw[:, 0]] == group])
    kept_used = np.unique(accepted[labels[accepted[:, 0]] == group])
    distance = cKDTree(p[rows]).query(p[core], k=2)[0][:, 1] / dx
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'triangles.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    report = dict(time=time, group=group, anchor_row=int(anchor), center=center.tolist(),
                  selection='maximum projection in fixed direction; face centroids within 3 cell widths',
                  anchor_direction=view.tolist(),
                  core_samples=len(core), context_samples=len(context), raw_patch_triangles=len(patch_faces),
                  first_rejection_counts=dict(Counter(r['first_rejection'] for r in records)),
                  core_samples_unused_by_any_raw_group_face=int((~np.isin(core, raw_used)).sum()),
                  core_samples_unused_by_any_accepted_group_face=int((~np.isin(core, kept_used)).sum()),
                  nearest_other_sample_distance_cells_p16_p50_p84=np.percentile(distance, [16, 50, 84]).tolist(),
                  whole_event_rejection_counts=rejection_counts, saved_faces_reproduced=True,
                  input=str(mesh), input_sha256=digest, vertices_moved=False, thresholds_changed=False)
    render(p, n, center, context, patch_faces, records, dx, output, view)
    assert hashlib.sha256(mesh.read_bytes()).hexdigest() == digest
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def render(p, n, center, context, faces, records, dx, output, view):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    u = np.cross(view, [0., 0, 1]); u /= np.linalg.norm(u)
    v = np.cross(view, u)
    xy = np.column_stack(((p-center) @ u, (p-center) @ v)) / dx
    palette = dict(accepted='#087f8c', tangent='#cf6230', normal='#8c54a2',
                   face_normal='#ce4677', minimum_angle='#ab8d20')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for j, ax in enumerate(axes):
        ax.scatter(*xy[context].T, s=7, color='black', zorder=3)
        subset = [i for i, r in enumerate(records) if j == 0 or r['accepted'] == (j == 1)]
        colors = [palette.get(records[i]['first_rejection'], 'gray') for i in subset]
        ax.add_collection(PolyCollection(xy[faces[subset]], facecolors=colors,
                                        edgecolors='.3', linewidths=.35, alpha=.55))
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect='equal', xlabel='Local u offset (cell widths)',
               ylabel='Local v offset (cell widths)', title=['All proposed triangles', 'Accepted triangles', 'Rejected triangles'][j])
    from matplotlib.patches import Patch
    names = sorted(set(r['first_rejection'] for r in records))
    fig.legend(handles=[Patch(color=palette.get(k, 'gray'), label=k.replace('_', ' ')) for k in names],
               loc='lower center', ncol=len(names), bbox_to_anchor=(.5, .065))
    fig.suptitle('Inner-surface patch at t=0.16: original refined positions, fixed reconstruction')
    fig.text(.03, .015, 'View along fixed patch direction; projection can overlap triangles. Black: original samples within 5 cells. Face centers within 3 cells.\n'
             'Colors show first failed guard, not independent causes. No points moved, no smoothing, no physical openings inferred.', fontsize=9)
    fig.tight_layout(rect=(0, .15, 1, .93)); fig.savefig(output / 'patch_triangles.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for accepted, color, label in [(True, '#087f8c', 'Accepted'), (False, '#cf6230', 'Rejected')]:
        subset = [r for r in records if r['accepted'] == accepted]
        for ax, key in zip(axes, ['max_tangent_departure_degrees', 'max_pair_normal_angle_degrees']):
            ax.hist([r[key] for r in subset], bins=np.arange(0, 95, 5), histtype='step', color=color, label=label)
    for ax, limit, title in zip(axes, [20, 45], ['Edge departure from endpoint tangent planes', 'Pairwise saved-normal angle']):
        ax.axvline(limit, color='black', linestyle='--', label=f'Guard: {limit} degrees')
        ax.set(xlabel='Maximum angle per triangle (degrees)', ylabel='Proposed triangles', title=title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / 'patch_angles.png', dpi=140); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, default=Path('outputs/two_bubbles_temporal_mesh_n128/mesh_0016.npz'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--direction', choices=['axis', 'diagonal'], default='diagonal')
    args = parser.parse_args(); run(args.mesh, args.output_dir, args.direction)
