"""Compare saved AstroLink labels in a thin spatial section, without reclustering."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def render(directory, event_index, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes = [('positions','Positions only'),('positions_normals_w1','Positions + normals, w=1'),
             ('positions_normals_w4','Positions + normals, w=4')]
    metadata = json.loads((directory/'metadata.json').read_text())
    # Obtain actual coordinates and dx from the input catalog rather than
    # assuming a simulation resolution or using source labels as clustering input.
    catalog_path = next(Path(p) for p in metadata['input_sha256'] if Path(p).name == f'surfaces_{event_index:04d}.npz')
    with np.load(catalog_path) as s:
        dx,sources = float(s['grid_spacing']),s['source_positions']
    fig,axes = plt.subplots(2,3,figsize=(14,9))
    for row,seed in enumerate([0,42]):
        for column,(mode,title) in enumerate(modes):
            with np.load(directory/f'event_{event_index:04d}_{mode}_order{seed}.npz') as a:
                pos,labels,time = a['positions'],a['deepest_nonroot_labels'],float(a['time'])
            keep = np.abs(pos[:,2]-.5) <= 1.5*dx
            groups = np.unique(labels[labels>=0])
            colors = np.tile([.6,.6,.6,.7],(len(pos),1))
            for k,group in enumerate(groups):
                colors[labels==group] = plt.get_cmap('turbo')((k+.5)/max(len(groups),1))
            ax = axes[row,column]
            ax.scatter(pos[keep,0],pos[keep,1],c=colors[keep],s=12,edgecolors='none')
            ax.scatter(sources[:,0],sources[:,1],marker='*',s=90,c='black')
            assigned=float(np.mean(labels>=0))
            ax.set(xlim=(.05,.95),ylim=(.15,.85),xlabel='x',ylabel='y',
                   title=f'{title}\n{len(groups)} display groups, {assigned:.1%} assigned (full catalog)')
            ax.set_aspect('equal');ax.grid(alpha=.1)
            ax.text(.03,.04,'Original order' if seed==0 else 'Shuffled order (seed 42)',transform=ax.transAxes,fontsize=10)
    fig.suptitle(f'AstroLink grouping at t={time:.2f}: midplane section',fontsize=17)
    fig.text(.04,.04,'Only points with |z - 0.5| <= 1.5 dx are shown. Clustering used every eligible point in 3D.\nGray: no non-root assignment. Colors are local hierarchy labels, not persistent physical identities.',fontsize=10)
    fig.subplots_adjust(left=.06,right=.98,bottom=.14,top=.86,hspace=.55,wspace=.25)
    fig.savefig(output,dpi=130)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--event-index',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.directory,a.event_index,a.output)


if __name__=='__main__':
    main()
