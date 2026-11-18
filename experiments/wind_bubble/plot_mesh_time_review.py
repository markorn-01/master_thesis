"""Readable original-sample versus oriented-display time-check figure."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.wind_bubble.check_adaptive_mesh_times import verify_report


def run(directory, root):
    hashes = {}; report = verify_report(directory/'verification.json', hashes)
    rows = list(csv.DictReader((directory/'summary.csv').open()))
    frames = []
    for index in report['events']:
        mesh_path = directory/f'mesh_{index:04d}.npz'
        source_path = root/'online'/f'surfaces_{index:04d}.npz'
        for path in [mesh_path, source_path]:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if str(path) in hashes and hashes[str(path)] != actual: raise ValueError('Changed input.')
            hashes[str(path)] = actual
        with np.load(mesh_path) as data: frame = {key: data[key].copy() for key in data.files}
        with np.load(source_path) as data:
            np.testing.assert_array_equal(frame['original_positions'], data['positions'])
            frame['sources'] = data['source_positions'].copy()
        frame['row'] = next(row for row in rows if int(row['event']) == index)
        frames.append(frame)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    palette = list(plt.get_cmap('tab20').colors)
    p = np.concatenate([frame['original_positions'] for frame in frames])
    low, high = p.min(axis=0)-.04, p.max(axis=0)+.04
    fig = plt.figure(figsize=(12, 4.3*len(frames)+1.4))
    grid = fig.add_gridspec(len(frames), 2, top=.90, bottom=.115, left=.015, right=.985, hspace=.17, wspace=.02)
    for row, frame in enumerate(frames):
        p = frame['original_positions']; labels = frame['original_labels']; segments = frame['original_segments']
        cut = p[:, 2] <= .5; summary = frame['row']
        for column in [0, 1]:
            ax = fig.add_subplot(grid[row, column], projection='3d')
            if column == 0:
                for segment in np.unique(segments):
                    selected = cut & (segments == segment)
                    ax.scatter(*p[selected].T, color=[palette[segment % 20] if segment >= 0 else '#858585'],
                               s=3 if segment == 4 else 9, depthshade=False, linewidths=0)
                title = f'Original samples | t = {float(frame["time"]):.2f}\n{len(p):,} detected samples'
            else:
                v, f, vg = frame['vertices'], frame['faces'], frame['vertex_groups']
                shown = f[(v[f, 2] <= .5).all(axis=1)]
                segment_for_group = {int(group): int(segments[np.flatnonzero(labels == group)[0]]) for group in np.unique(labels[labels >= 0])}
                if len(shown):
                    color = np.array([palette[segment_for_group[int(group)] % 20] for group in vg[shown[:, 0]]])
                    ax.add_collection3d(Poly3DCollection(v[shown], facecolors=color, edgecolors=color, linewidths=0,
                        shade=True, lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45)))
                fallback = cut & ~np.isin(labels, frame['approved_groups'])
                ax.scatter(*p[fallback].T, color='#858585', s=6, depthshade=False)
                title = f'Oriented adaptive display | t = {float(frame["time"]):.2f}\n'
                title += f'{summary["approved_groups"]}/{summary["retained_groups"]} retained groups approved'
            ax.scatter(*frame['sources'].T, color='black', marker='*', s=45, depthshade=False)
            ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]), box_aspect=high-low,
                   xlabel='x', ylabel='y', zlabel='z', title=title)
            ax.view_init(elev=25, azim=-35)
            ax.title.set_fontsize(11)
    fig.suptitle('Three-state display check: original samples and smooth surfaces', fontsize=17, y=.975)
    fig.text(.5, .94, 'Fixed camera, bounds and continuation-segment colors | display cutaway: z <= 0.5', ha='center', fontsize=11)
    fig.text(.04, .035, 'Black stars: wind sources. Gray points: samples outside approved display groups. Each state is fitted independently.\n'
             'One triangle is reversed at t=0.12 and at t=0.14; no surface vertex or triangle membership changes.\n'
             'Physical measurements, spatial grouping and tracking are unchanged. Three states do not validate temporal flicker or physical topology.', fontsize=10)
    path = directory/'review.png'
    if path.exists(): raise ValueError('Review image already exists.')
    fig.savefig(path, dpi=140); plt.close(fig)
    for name, expected in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected: raise ValueError('Changed input.')
    verification = dict(passed=True, input_sha256=hashes, output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path.with_suffix('.verification.json').write_text(json.dumps(verification, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('outputs/three_bubbles_iwr_compgpu7_n064'))
    args = parser.parse_args()
    run(args.directory, args.root)
