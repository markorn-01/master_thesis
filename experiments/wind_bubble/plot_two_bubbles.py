"""Inspect one unlabeled two-source event with physical background slices."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def render(directory, index, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LogNorm

    directory, output = Path(directory), Path(output)
    with np.load(directory/f'surfaces_{index:04d}.npz') as a:
        if int(a['schema_version']) != 3:
            raise ValueError('This viewer requires the multi-source catalog schema.')
        pos, mach, valid = a['positions'], a['mach'], a['mach_valid']
        sources, time, dx = a['source_positions'], float(a['time']), float(a['grid_spacing'])
        known = a['sample_positions_valid']
        if len(pos) and not (a['source_distances'] > float(a['injection_radius'])).all():
            raise ValueError('Catalog contains samples inside a source.')
        quality = dict(unknown_endpoints=int((~known).sum()),
                       upstream_inside_any_source=int(a['upstream_inside_any_source'].sum()),
                       downstream_inside_any_source=int(a['downstream_inside_any_source'].sum()))
        density_jump, rho_valid = a['density_jump'], a['density_jump_valid']
    with np.load(directory/f'histograms_{index:04d}.npz') as h:
        for field, keep, name in [(mach,valid,'mach'), (density_jump,rho_valid,'density_jump')]:
            np.testing.assert_array_equal(np.histogram(field[keep], bins=h[f'{name}_edges'])[0], h[f'{name}_counts'])
        np.testing.assert_allclose(float(h['time']), time, atol=1e-7, rtol=0)
    with np.load(directory/f'slice_{index:04d}.npz') as a:
        density, pressure, z = a['density'], a['pressure'], float(a['z_coordinate'])
        np.testing.assert_allclose(float(a['time']), time, atol=1e-7, rtol=0)
    if not all(np.isfinite(f).all() and (f > 0).all() for f in (density, pressure)):
        raise ValueError('Background fields must be finite and positive.')
    fig = plt.figure(figsize=(13,7))
    grid = fig.add_gridspec(2,2, width_ratios=[1.2,1], left=.04,right=.96,top=.88,bottom=.12,wspace=.18,hspace=.35)
    ax = fig.add_subplot(grid[:,0], projection='3d')
    high = max(1., float(np.ceil(mach[valid].max()))) if valid.any() else 1.
    scatter = ax.scatter(*pos[valid].T,c=mach[valid],cmap='viridis',norm=Normalize(0,high),s=3,alpha=.7,depthshade=False)
    if (~valid).any():
        ax.scatter(*pos[~valid].T,c='0.7',s=3)
    ax.scatter(*sources.T, marker='*',c=['#dc6a22','#c52e62'],s=110,edgecolors='white')
    ax.set(xlim=(0,1),ylim=(0,1),zlim=(0,1),xlabel='x',ylabel='y',zlabel='z',title='Detected surfaces, colored by Mach')
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=24,azim=35)
    fig.colorbar(scatter,ax=ax,shrink=.55,pad=.03,label='Profile-aware Mach number')
    for row,(field,label) in enumerate([(pressure,'Pressure'),(density,'Density')]):
        axis=fig.add_subplot(grid[row,1])
        lo,hi=float(field.min()),float(field.max())
        image=axis.imshow(field.T,origin='lower',extent=(0,1,0,1),norm=LogNorm(lo,max(hi,lo*1.01)),cmap='magma')
        axis.scatter(sources[:,0],sources[:,1],marker='*',c=['#dc6a22','#c52e62'],s=65,edgecolors='white')
        axis.set(xlabel='x',ylabel='y',title=f'{label} slice at z={z:.4f}')
        fig.colorbar(image,ax=axis,pad=.02)
    fig.suptitle(f'Two wind sources: t={time:.3f}',fontsize=17)
    fig.text(.04,.045,f'{len(pos)} surface samples outside both injection spheres. Stars mark sources.\nUnlabeled point cloud; source positions do not assign shock ownership. Code units. Scales are event-specific.',fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output,dpi=140)
    plt.close(fig)
    report=dict(time=time, surface_samples=len(pos), histograms_reconstructed_exact=True, **quality,
                minimum_surface_boundary_clearance=float(np.minimum(pos,1-pos).min()) if len(pos) else None,
                grid_spacing=dx, source_positions=sources.tolist())
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--event-index',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    render(args.directory,args.event_index,args.output)


if __name__ == '__main__':
    main()
