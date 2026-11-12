"""Show conditional inner-fragment families and a display-only surfel option."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.wind_bubble.surfel_display import disk_vertices


def run(directory,output):
    result_path=directory/'real_results.json';check_path=directory/'verification.json'
    report=json.loads(check_path.read_text())
    for key in ['input_sha256','source_sha256']:
        for path,digest in report[key].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError(f'Changed validated input: {path}')
    records=json.loads(result_path.read_text())
    meta=json.loads(Path('outputs/two_bubbles_temporal_comparison_n064_n128/metadata.json').read_text())
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [result_path,check_path]}
    panels=[];audit=[]
    for n in [64,128]:
        record=next(r for r in records if r['num_cells']==n and r['method']=='delayed_family' and r['gate']==1.5/64)
        families=[f for f in record['result']['families'] if f['accepted']]
        root=Path(meta['reference_directory' if n==64 else 'target_directory']);row=[]
        for offset in range(3):
            index=record['first_event']+offset
            source=root/'online'/f'surfaces_{index:04d}.npz'
            partition=Path(meta['comparison_directory'])/f'groups_n{n:03d}_event{index:04d}.npz'
            for path in [source,partition]:hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(source) as data,np.load(partition) as grouping:
                labels=grouping['native_cells_a60_labels'];parts=[]
                for f in families:
                    groups=f['parent_groups'] if offset==0 else [f['child_group'] if offset==1 else f['later_group']]
                    mask=np.isin(labels,groups);p=data['positions'][mask].copy();normals=data['normals'][mask].copy()
                    parts.append(dict(positions=p,normals=normals,groups=groups))
                    audit.append(dict(num_cells=n,event=index,family_child=f['child_group'],groups=groups,
                        surface_samples=len(p),coordinate_sha256=hashlib.sha256(p.tobytes()).hexdigest()))
                row.append(dict(time=float(data['time']),parts=parts,sources=data['source_positions'].copy(),dx=float(data['grid_spacing'])))
        panels.append(row)
    output.mkdir(parents=True,exist_ok=False)
    for mode in ['points','oriented_disks']:render(panels,output,mode)
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    result=dict(panels=audit,input_sha256=hashes,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('experiments/wind_bubble/surfel_display.py')]},
        positions_moved=False,normals_changed=False,connectivity_created=False,physical_ancestry_established=False,
        oriented_disk_radius_in_cells=.45,polygon_vertices_per_disk=10,
        scope='Only conditional inner families at the narrow gate; full family populations, no subsampling.')
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(panels=len(audit),positions_moved=False,connectivity_created=False),indent=2))


def render(panels,output,mode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    colors=['#d16c28','#376da6'];fig=plt.figure(figsize=(15,9))
    for row,frames in enumerate(panels):
        for col,frame in enumerate(frames):
            ax=fig.add_subplot(2,3,3*row+col+1,projection='3d')
            counts=[]
            for color,part in zip(colors,frame['parts']):
                p=part['positions'];counts.append(len(p))
                if mode=='points':ax.scatter(*p.T,c=color,s=5,depthshade=False,linewidths=0)
                else:
                    disks=disk_vertices(p,part['normals'],.45*frame['dx'])
                    ax.add_collection3d(Poly3DCollection(disks,facecolors=color,edgecolors='none',alpha=1))
            ax.scatter(*frame['sources'].T,c='black',s=40,marker='*',depthshade=False)
            ax.set(xlim=(.23,.77),ylim=(.37,.63),zlim=(.37,.63),xlabel='x',ylabel='y',zlabel='z',box_aspect=(2,1,1))
            ax.view_init(25,-65)
            ax.set_title(f'{[64,128][row]}³ | t={frame["time"]:.2f} | samples {counts[0]}, {counts[1]}',fontsize=11)
    fig.suptitle('Conditional inner-surface families: '+('original points' if mode=='points' else 'oriented disks (display only)'),fontsize=16)
    fig.text(.04,.03,'Left: earlier fragments; middle/right: later coherent groups. Colors represent conditional support families, not established physical ancestry.\n'
             'Same original positions, normals and bounds. Disks have radius 0.45 cell widths and no connectivity; no holes are reconstructed or physical areas recomputed.',fontsize=10)
    fig.subplots_adjust(top=.89,bottom=.15,left=.01,right=.99,wspace=.04,hspace=.12)
    fig.savefig(output/f'inner_families_{mode}.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('outputs/tracklet_fragment_support_v1'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();run(args.directory,args.output_dir)
