"""Inspect graph edges joining stricter components; no physical ownership labels."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.shock_connectivity import connectivity_partition


def joining_edges(positions, normals, dx, reference_labels, edges):
    """Return candidate edges between different eligible reference components.

    A 'joining edge' is relative to a declared partition, not a false-edge label.
    Normal offsets measure chord alignment with the endpoint normals: a locally
    tangential edge has values near zero; large values warrant inspection.
    """
    p, n = np.asarray(positions, float), np.asarray(normals, float)
    labels, edges = np.asarray(reference_labels), np.asarray(edges, dtype=int).reshape(-1, 2)
    a, b = edges.T
    keep = (labels[a] >= 0) & (labels[b] >= 0) & (labels[a] != labels[b])
    pairs = edges[keep]
    a, b = pairs.T
    delta = p[b] - p[a]
    distance = np.linalg.norm(delta, axis=1)
    unit = n / np.linalg.norm(n, axis=1)[:, None]
    direction = np.divide(delta, distance[:, None], out=np.zeros_like(delta), where=distance[:, None] > 0)
    return dict(pairs=pairs, distance_cells=distance / dx,
                normal_angle_degrees=np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i', unit[a], unit[b]), -1, 1))),
                normal_chord_alignment_a=np.abs(np.einsum('ij,ij->i', direction, unit[a])),
                normal_chord_alignment_b=np.abs(np.einsum('ij,ij->i', direction, unit[b])))


def write_rows(path, rows, fields=None):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def render(path, p, n, labels, bridges, sources, pressure, density, z, dx, time):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    section = np.abs(p[:, 2] - z) <= 1.5 * dx
    pairs = bridges['pairs']
    show = section[pairs[:, 0]] & section[pairs[:, 1]]
    segments = p[pairs[show], :2]
    groups = np.unique(labels[labels >= 0])
    colors = np.tile([.5, .5, .5, 1.], (len(p), 1))
    for j, group in enumerate(groups):
        colors[labels == group] = plt.get_cmap('tab10')(j % 10)
    for ax, field, title in zip(axes, [pressure, density, None], ['Pressure', 'Density', 'Normals and joining edges']):
        if field is not None:
            im = ax.imshow(field.T, origin='lower', extent=(0, 1, 0, 1), cmap='Greys',
                           norm=LogNorm(float(field.min()), max(float(field.max()), float(field.min()) * 1.01)))
            fig.colorbar(im, ax=ax, shrink=.65, pad=.02)
        ax.scatter(p[section, 0], p[section, 1], c=colors[section], s=8, zorder=3)
        ax.add_collection(LineCollection(segments, colors='#db1675', linewidths=1.1, zorder=4))
        ax.scatter(sources[:, 0], sources[:, 1], marker='*', c='black', s=60, zorder=5)
        ax.set(xlim=(.25, .75), ylim=(.25, .75), xlabel='x', ylabel='y', title=title, aspect='equal')
    rows = np.flatnonzero(section)
    rows = rows[np.abs(p[rows, 0] - .5) < .09]
    axes[2].quiver(p[rows, 0], p[rows, 1], n[rows, 0], n[rows, 1],
                   angles='xy', scale_units='xy', scale=40, width=.004, color='black', zorder=5)
    fig.suptitle(f'Graph connections near collision at t={time:.2f}', fontsize=17)
    fig.text(.04, .04, 'Point colors: raw components at r=1.5 cells, angle=30°. Pink: r=2.5/60° edges joining different reference components.\n'
             f'Gas slice z={z:.5f}; points within ±1.5 cells, lines only if both endpoints are in that slab. '
             'Projection is illustrative; all edge measurements are 3D.\n'
             'A joining edge is not automatically incorrect. Sources are shown only for orientation; colors are not persistent shock IDs.', fontsize=9)
    fig.tight_layout(rect=(.01, .16, .99, .93))
    fig.savefig(path, dpi=140); plt.close(fig)
    return int(show.sum())


def run(directory, output, events):
    output.mkdir(parents=True, exist_ok=False)
    summary, hashes = [], {}
    for event in events:
        paths = [directory / f'surfaces_{event:04d}.npz', directory / f'slice_{event:04d}.npz']
        for path in paths: hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(paths[0]) as a:
            if int(a['schema_version']) != 3: raise ValueError('Require multi-source schema 3.')
            p, n = a['positions'], a['normals']
            dx, time = float(a['grid_spacing']), float(a['time'])
            sources, mach, jump = a['source_positions'], a['mach'], a['density_jump']
        graphs = {(r, angle): connectivity_partition(p, n, dx, r, angle, 20)
                  for r in (1.5, 2.5) for angle in (30., 60.)}
        reference = graphs[1.5, 30.]['raw_labels']
        all_edges = []
        for (r, angle), graph in graphs.items():
            bridges = joining_edges(p, n, dx, reference, graph['edges'])
            row = dict(event_index=event, time=time, radius_cells=r, angle_degrees=angle,
                       reference_components=len(graphs[1.5, 30.]['component_sizes']),
                       raw_components=len(graph['component_sizes']), edges=len(graph['edges']),
                       joining_edges=len(bridges['pairs']))
            for key in ['distance_cells', 'normal_angle_degrees', 'normal_chord_alignment_a', 'normal_chord_alignment_b']:
                values = bridges[key]
                for suffix, q in [('min', 0), ('median', 50), ('max', 100)]:
                    row[f'{key}_{suffix}'] = float(np.percentile(values, q)) if len(values) else None
            summary.append(row)
            for k, (i, j) in enumerate(bridges['pairs']):
                edge = dict(radius_cells=r, angle_degrees=angle, row_a=int(i), row_b=int(j),
                            component_a=int(reference[i]), component_b=int(reference[j]))
                edge.update({key: float(bridges[key][k]) for key in bridges if key != 'pairs'})
                for suffix, idx in [('a', i), ('b', j)]:
                    edge.update({f'{axis}_{suffix}': float(p[idx, dim]) for dim, axis in enumerate('xyz')})
                    edge[f'mach_{suffix}'] = float(mach[idx]); edge[f'density_jump_{suffix}'] = float(jump[idx])
                all_edges.append(edge)
        # Headers remain defined even when there are no joining edges.
        fields = ['radius_cells','angle_degrees','row_a','row_b','component_a','component_b',
                  'distance_cells','normal_angle_degrees','normal_chord_alignment_a','normal_chord_alignment_b',
                  'x_a','y_a','z_a','mach_a','density_jump_a','x_b','y_b','z_b','mach_b','density_jump_b']
        write_rows(output / f'event_{event:04d}_joining_edges.csv', all_edges, fields)
        bridges = joining_edges(p, n, dx, reference, graphs[2.5, 60.]['edges'])
        with np.load(paths[1]) as a:
            np.testing.assert_allclose(float(a['time']), time, atol=1e-7, rtol=0)
            shown = render(output / f'event_{event:04d}_connections.png', p, n, reference, bridges, sources,
                           a['pressure'], a['density'], float(a['z_coordinate']), dx, time)
        print(json.dumps(dict(event=event, time=time, joining_edges=len(bridges['pairs']), displayed_edges=shown,
                              components={f'r{r}_a{angle}': len(g['component_sizes']) for (r,angle),g in graphs.items()})), flush=True)
    write_rows(output / 'summary.csv', summary)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in hashes.items())
    metadata = dict(events=events, input_sha256=hashes, input_hashes_unchanged=True,
                    reference='raw components at r=1.5 cells, signed normal angle <=30 degrees',
                    comparison_radius_cells=[1.5,2.5], comparison_angle_degrees=[30,60],
                    interpretation='relative graph connections, not physical shock truth labels',
                    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                   [Path(__file__),Path('experiments/astrolink/shock_connectivity.py')]})
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--events', nargs='+', type=int, default=[6,8,9,10])
    args = parser.parse_args()
    run(args.directory, args.output_dir, args.events)
