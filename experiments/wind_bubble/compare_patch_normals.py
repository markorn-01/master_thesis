"""Display-only comparison of saved normals and local geometric plane fits."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from experiments.wind_bubble.ball_pivot_mesh import guard_faces
from experiments.wind_bubble.inspect_mesh_patch import triangle_angles


def plane_normals(p, saved, labels, radius):
    """Fit centered covariance planes within each group; sign-align to saved normals.

    Require at least six points and a distinct smallest eigenvalue. Unresolved
    neighborhoods keep the saved direction and are explicitly marked invalid.
    This geometric fit does not distinguish nearby sheets in the same group.
    """
    p = np.asarray(p, float); saved = np.asarray(saved, float)
    labels = np.asarray(labels)
    if p.ndim != 2 or p.shape[1] != 3 or saved.shape != p.shape or labels.shape != (len(p),):
        raise ValueError('Require matching point, normal and group arrays.')
    if not np.isfinite(p).all() or not np.isfinite(saved).all() or (np.linalg.norm(saved, axis=1) == 0).any() or not np.isfinite(radius) or radius <= 0:
        raise ValueError('Require finite geometry, nonzero normals and positive radius.')
    result = saved / np.linalg.norm(saved, axis=1)[:, None]
    valid = np.zeros(len(p), bool); counts = np.zeros(len(p), int)
    eigenvalues = np.full((len(p), 3), np.nan)
    for group in np.unique(labels[labels >= 0]):
        rows = np.flatnonzero(labels == group)
        neighbors = cKDTree(p[rows]).query_ball_point(p[rows], radius)
        for row, local in zip(rows, neighbors):
            counts[row] = len(local)
            if len(local) < 6: continue
            points = p[rows[local]]; centered = points - points.mean(axis=0)
            values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
            eigenvalues[row] = values
            if values[2] <= 0 or values[1] <= 1e-8 * values[2] or values[1]-values[0] <= 1e-8 * values[2]: continue
            normal = vectors[:, 0]
            if normal @ result[row] < 0: normal = -normal
            result[row] = normal; valid[row] = True
    return result, valid, counts, eigenvalues


def run(mesh, output):
    digest = hashlib.sha256(mesh.read_bytes()).hexdigest()
    with np.load(mesh) as a:
        data = {k: a[k].copy() for k in a.files}
    p = data['positions'].astype(float); saved = data['normals'].astype(float)
    labels = data['labels']; raw = data['raw_faces']; dx = float(data['grid_spacing'])
    # Fit the selected inner group only; all other labels are excluded from fits.
    fit_labels = np.where(labels == 1, labels, -1)
    modes = {'saved': saved / np.linalg.norm(saved, axis=1)[:, None]}
    arrays = {}; checks = 0
    for cells in [2, 3]:
        name = f'plane_{cells}cells'
        normals, valid, counts, values = plane_normals(p, saved, fit_labels, cells*dx)
        modes[name] = normals
        for key, value in [('normals', normals), ('valid', valid), ('neighbor_count', counts), ('eigenvalues', values)]:
            arrays[f'{name}_{key}'] = value
        # Independent SVD readback and brute-force neighborhood counts for every fitted point.
        rows = np.flatnonzero(labels == 1)
        for row in rows:
            points = p[rows[np.linalg.norm(p[rows]-p[row], axis=1) <= cells*dx]]
            assert len(points) == counts[row]
            if valid[row]:
                _, _, vh = np.linalg.svd(points-points.mean(axis=0), full_matrices=False)
                np.testing.assert_allclose(abs(vh[-1] @ normals[row]), 1., atol=1e-10)
                checks += 1
    output.mkdir(parents=True, exist_ok=False)
    summary = []; displays = []
    for direction, view in [('axis', np.array([0., 1, 0])), ('diagonal', np.array([-1., 1, 1])/np.sqrt(3))]:
        rows = np.flatnonzero(labels == 1); anchor = rows[np.argmax(p[rows] @ view)]
        center = p[anchor]; core = rows[np.linalg.norm(p[rows]-center, axis=1) <= 3*dx]
        context = rows[np.linalg.norm(p[rows]-center, axis=1) <= 5*dx]
        patch = raw[(labels[raw[:, 0]] == 1) & (np.linalg.norm(p[raw].mean(axis=1)-center, axis=1) <= 3*dx)]
        arrays[f'{direction}_raw_faces'] = patch
        for name, normals in modes.items():
            # Apply guards on the whole original proposal set, then crop for display.
            accepted, rejected = guard_faces(p, normals, labels, raw, dx)
            keys = {tuple(sorted(f)) for f in accepted}
            keep = np.array([tuple(sorted(f)) in keys for f in patch])
            if name == 'saved':
                assert keys == {tuple(sorted(f)) for f in data['faces']}
            tangent, spread = triangle_angles(p, normals, patch)
            deviation = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', modes['saved'][core], normals[core]), -1, 1)))
            record = dict(patch=direction, mode=name, proposals=len(patch), accepted=int(keep.sum()),
                          core_samples=len(core), normal_deviation_p16_p50_p84=np.percentile(deviation,[16,50,84]).tolist(),
                          tangent_departure_p16_p50_p84=np.percentile(tangent,[16,50,84]).tolist(),
                          normal_spread_p16_p50_p84=np.percentile(spread,[16,50,84]).tolist(),
                          core_fit_invalid=0 if name == 'saved' else int((~arrays[f'{name}_valid'][core]).sum()))
            summary.append(record); arrays[f'{direction}_{name}_accepted'] = keep
            displays.append((direction, name, view, center, context, patch[keep]))
    arrays.update(positions=data['positions'], saved_normals=data['normals'], labels=labels)
    np.savez_compressed(output/'normal_comparison.npz', **arrays)
    render(p, dx, displays, output)
    with np.load(output/'normal_comparison.npz') as a:
        np.testing.assert_array_equal(a['positions'], data['positions'])
        np.testing.assert_array_equal(a['saved_normals'], data['normals'])
    assert hashlib.sha256(mesh.read_bytes()).hexdigest() == digest
    report = dict(records=summary, independent_svd_checks=checks, original_arrays_exact=True,
                  input=str(mesh), input_sha256=digest,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  proposal_triangles_fixed=True, guard_thresholds_fixed=True,
                  physical_diagnostics_changed=False, production_method_changed=False,
                  limitations=['Local plane fitting can mix nearby sheets within one group.',
                               'This compares fixed raw triangles; it does not rerun ball pivoting with fitted normals.',
                               'More accepted triangles do not establish correct physical topology.'])
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


def render(p, dx, displays, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, (direction, name, view, center, context, faces) in zip(axes.flat, displays):
        u = np.cross(view,[0.,0,1]); u /= np.linalg.norm(u); v = np.cross(view,u)
        xy = np.column_stack(((p-center)@u, (p-center)@v))/dx
        ax.scatter(*xy[context].T, s=7, c='black', zorder=3)
        ax.add_collection(PolyCollection(xy[faces],facecolors='#087f8c',edgecolors='.3',linewidths=.35,alpha=.55))
        ax.set(xlim=(-4,4),ylim=(-4,4),aspect='equal',xlabel='Local u (cell widths)',ylabel='Local v (cell widths)',
               title=f'{direction} | {name.replace("_", " ")} | {len(faces)} accepted')
    fig.suptitle('Same positions and proposed triangles; compare normals used by display guards')
    fig.text(.03,.02,'Black: original refined samples. Local covariance planes use neighbors within 2 or 3 grid spacings, within the same group.\n'
             'This is a display diagnostic. Physical normals, Mach numbers and heating estimates remain unchanged. More faces do not prove correct topology.',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.95)); fig.subplots_adjust(hspace=.4)
    fig.savefig(output/'normal_comparison.png',dpi=140); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',type=Path,default=Path('outputs/two_bubbles_temporal_mesh_n128/mesh_0016.npz'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args(); run(args.mesh,args.output_dir)
