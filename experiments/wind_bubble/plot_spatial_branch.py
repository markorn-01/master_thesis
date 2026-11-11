"""Show a reviewed candidate branch beside original before/after surface points."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def run(branch,histories,root,partitions,output):
    graph_path=branch/'candidate_graph.json';verification_path=branch/'verification.json'
    graph=json.loads(graph_path.read_text());prior=json.loads(verification_path.read_text())
    # Check the evidence used to construct this branch before reusing it.
    for path,digest in prior['input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    history_path=histories/'group_histories.csv';history=list(csv.DictReader(history_path.open()))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [graph_path,verification_path,history_path]}
    parents=sorted({e['parent_segment'] for e in graph['edges']});children={e['child_segment'] for e in graph['edges']}
    assert len(parents)==2 and len(children)==1
    child=next(iter(children));assert all((e['parent_event'],e['child_event'])==(7,8) for e in graph['edges'])
    display=[];audit=[]
    for event,segments in [(7,parents),(8,[child])]:
        source=root/'online'/f'surfaces_{event:04d}.npz';partition=partitions/f'groups_n128_event{event:04d}.npz'
        for path in [source,partition]:hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(source) as a:data={k:a[k].copy() for k in ['positions','time','source_positions']}
        with np.load(partition) as a:labels=a['native_cells_a60_labels'].copy()
        np.testing.assert_allclose(float(data['time']),event*.01,atol=1e-7,rtol=0)
        selected=[]
        for segment in segments:
            rows=[r for r in history if int(r['event'])==event and int(r['segment_id'])==segment]
            assert len(rows)==1;row=rows[0];group=int(row['group']);indices=np.flatnonzero(labels==group)
            assert len(indices)==int(row['surface_samples'])
            selected.append(dict(segment=segment,group=group,indices=indices))
            audit.append(dict(event=event,time=float(data['time']),segment_id=segment,local_group=group,
                              displayed_points=len(indices),total_group_points=len(indices),
                              coordinates_sha256=hashlib.sha256(data['positions'][indices].tobytes()).hexdigest()))
        # Every retained group is represented exactly once; no hidden subsampling.
        assert {s['group'] for s in selected}==set(map(int,np.unique(labels[labels>=0])))
        assert len(np.unique(np.concatenate([s['indices'] for s in selected])))==int((labels>=0).sum())
        display.append((data,selected))
    output.mkdir(parents=True,exist_ok=False);render(display,parents,child,output)
    from PIL import Image
    with Image.open(output/'spatial_branch.png') as image:
        size=list(image.size);image.verify()
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    (output/'verification.json').write_text(json.dumps(dict(displayed_groups=audit,candidate_edges=graph['edges'],
        image_size=size,input_sha256=hashes,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        all_group_points_displayed=True,positions_moved=False,new_clustering_performed=False,
        tracking_decisions_changed=False,physical_merger_confirmed=False,
        view=dict(camera=[25,-65],bounds=[.1,.9],point_size=3,depthshade=False)),indent=2)+'\n')
    print(json.dumps(dict(groups=audit,image_size=size),indent=2))


def render(display,parents,child,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    colors={parents[0]:'#e07820',parents[1]:'#298c43',child:'#237ab5'}
    fig=plt.figure(figsize=(16,6.5));grid=fig.add_gridspec(1,3,width_ratios=[1.35,.8,1.35])
    for col,(data,selected) in zip([0,2],display):
        ax=fig.add_subplot(grid[0,col],projection='3d');p=data['positions']
        for item in selected:
            points=p[item['indices']]
            ax.scatter(*points.T,s=3,color=colors[item['segment']],depthshade=False,
                       label=f'S{item["segment"]:03d}: {len(points):,} samples')
        ax.set(xlim=(.1,.9),ylim=(.1,.9),zlim=(.1,.9),box_aspect=(1,1,1),xlabel='x',ylabel='y',zlabel='z')
        ax.view_init(elev=25,azim=-65)
        ax.set_title(f't = {float(data["time"]):.2f}: '+('two detected groups' if col==0 else 'one detected group'),fontsize=13)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.04),fontsize=10,frameon=False)
    ax=fig.add_subplot(grid[0,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    locations={parents[0]:(.03,.66),parents[1]:(.03,.20),child:(.65,.43)}
    for segment,(x,y) in locations.items():
        ax.add_patch(FancyBboxPatch((x,y),.29,.15,boxstyle='round,pad=.015',facecolor='white',edgecolor=colors[segment],linewidth=2))
        ax.text(x+.145,y+.075,f'S{segment:03d}',ha='center',va='center',fontsize=13,color=colors[segment])
    for parent in parents:
        x,y=locations[parent]
        ax.annotate('',xy=(.625,.505),xytext=(x+.32,y+.075),
                    arrowprops=dict(arrowstyle='->',linestyle='--',linewidth=2,color=colors[parent]))
    ax.set_title('Candidate predecessors',fontsize=12)
    ax.text(.5,.03,'Schematic links\n(not spatial trajectories)',ha='center',va='top',fontsize=10)
    fig.suptitle('Where the branching history sits in the simulation',fontsize=17)
    fig.text(.025,.055,'Original refined shock-surface points, same camera and spatial bounds; no triangles, smoothing or subsampling.\n'
             'Colors identify the three provisional segments. The two candidate links have support at both tested distances; no confirmed physical merger is inferred.',fontsize=10)
    fig.subplots_adjust(left=.015,right=.98,top=.85,bottom=.24,wspace=.02)
    fig.savefig(output/'spatial_branch.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch',type=Path,default=Path('outputs/two_bubbles_branching_history_n128'))
    parser.add_argument('--histories',type=Path,default=Path('outputs/two_bubbles_provisional_histories_n128'))
    parser.add_argument('--root',type=Path,default=Path('outputs/two_bubbles_budget_matched_iwr_compgpu3_n128'))
    parser.add_argument('--partitions',type=Path,default=Path('outputs/two_bubbles_resolution_comparison_n064_n128'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();run(args.branch,args.histories,args.root,args.partitions,args.output_dir)
