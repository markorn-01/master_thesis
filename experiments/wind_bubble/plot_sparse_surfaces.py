"""Plot saved surface points without loading gas snapshots or initializing JAX."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.wind_bubble.sparse_surfaces import SCHEMA_VERSION


FIELDS = {
    'mach': ('mach_valid', 'Profile-aware Mach number'),
    'thermalization_flux': ('thermalization_flux_valid', 'Local thermalization flux (code units)'),
    'density_jump': ('density_jump_valid', 'Sampled density ratio'),
}


def plot_catalog(catalog_path, output_path, *, field='mach', max_points=100000):
    """Render a point cloud. Display subsampling never changes the saved catalog."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    catalog_path, output_path = Path(catalog_path), Path(output_path)
    if field not in FIELDS or max_points < 1:
        raise ValueError('Choose a supported field and a positive max-points value.')
    if catalog_path.resolve() == output_path.resolve():
        raise ValueError('The plot must not overwrite its source catalog.')
    valid_key, label = FIELDS[field]
    with np.load(catalog_path, allow_pickle=False) as catalog:
        if int(catalog['schema_version']) != SCHEMA_VERSION:
            raise ValueError('Unsupported sparse surface schema version.')
        positions = catalog['positions']
        values = catalog[field]
        valid = catalog[valid_key] & np.isfinite(values) & (values > 0)
        time = float(catalog['time'])
        lengths = catalog['grid_shape'] * float(catalog['grid_spacing'])
    count = len(values)
    if positions.shape != (count, 3) or valid.shape != (count,):
        raise ValueError('Catalog point and property shapes differ.')
    displayed = np.linspace(0, count-1, min(count, max_points), dtype=int)
    selected_positions, selected_valid = positions[displayed], valid[displayed]
    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    if selected_valid.any():
        points = selected_positions[selected_valid]
        artist = ax.scatter(*points.T, c=values[displayed][selected_valid], s=7,
                            cmap='viridis', linewidths=0, depthshade=False)
        fig.colorbar(artist, ax=ax, shrink=.65, pad=.1, label=label)
    if (~selected_valid).any():
        ax.scatter(*selected_positions[~selected_valid].T, c='lightgray', s=7,
                   linewidths=0, depthshade=False, label='Invalid property')
        ax.legend(loc='upper left')
    if count == 0:
        ax.text2D(.1, .9, 'No detected surface samples outside injection', transform=ax.transAxes)
    ax.set(xlim=(0, lengths[0]), ylim=(0, lengths[1]), zlim=(0, lengths[2]),
           xlabel='x (code units)', ylabel='y (code units)', zlabel='z (code units)',
           title=f'Shock surface points at t = {time:.5f}\n{label}')
    ax.set_box_aspect(lengths)
    ax.view_init(elev=24, azim=35)
    fig.supxlabel(f'{count:,} samples outside injection; {int((~valid).sum()):,} invalid {field} values\n'
                  f'{len(displayed):,} points displayed. Point cloud, not a connected mesh.', fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return dict(surface_samples=count, displayed_points=len(displayed), invalid_values=int((~valid).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('catalog', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--field', choices=FIELDS, default='mach')
    parser.add_argument('--max-points', type=int, default=100000,
                        help='Limit displayed points only; the catalog retains every sample.')
    args = parser.parse_args()
    output = args.output or args.catalog.with_name(f'{args.catalog.stem}_{args.field}.png')
    print(plot_catalog(args.catalog, output, field=args.field, max_points=args.max_points))
    print(f'Figure: {output}')


if __name__ == '__main__':
    main()
