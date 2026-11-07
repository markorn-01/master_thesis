"""Spatial inspection of saved candidate associations, without assigning shock IDs."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.surface_associations import association_scores

CASES=[('n064_fragments',64,'native_cells',13,1.5),
       ('n128_large_groups',128,'native_cells_scaled_count',13,1.5),
       ('n128_fragments_narrow',128,'native_cells',13,1.5),
       ('n128_fragments_wide',128,'native_cells',13,2.5),
       ('n128_redundant_links',128,'native_cells',14,2.5)]


def choose_record(records,n,mode,event,gate):
    found=[r for r in records if r['config']['num_cells']==n and r['config']['mode']==mode
           and r['config']['parent_event']==event and r['config']['gate_reference_cells']==gate
           and r['config']['segmentation_angle']==60]
    if len(found)!=1:raise ValueError('Require exactly one saved configuration.')
    return found[0]


def candidate_edges(result,scores):
    """Include saved eligible links and independently reconstructed competing targets."""
    edges=[]
    for l in result['links']:
        edges.append(dict(parent_group=l['parent_group'],child_group=l['child_group'],
                          kind=('redundant' if l['redundant_support'] else 'accepted') if l['accepted'] else 'insufficient_pool'))
    for p in result['parents']:
        ranked=sorted([x for x in scores if x['parent_group']==p['parent_group']],
                      key=lambda x:(-x['parent_coverage'],x['child_group']))
        if ranked:
            assert ranked[0]['child_group']==p['best_child_group']
            assert ranked[0]['parent_coverage']==p['parent_coverage']
            assert (ranked[1]['parent_coverage'] if len(ranked)>1 else 0.)==p['runner_up_parent_coverage']
        if p['status']=='ambiguous_target':
            for kind,target in zip(['ambiguous_best','ambiguous_runner_up'],ranked[:2]):
                edges.append(dict(parent_group=p['parent_group'],child_group=target['child_group'],
                                  kind=kind,parent_coverage=target['parent_coverage']))
    return edges


def render(path,record,a,b,la,lb):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    c=record['config'];r=record['result']
    scores=association_scores(a['positions'],a['normals'],la,b['positions'],b['normals'],lb,c['gate_code_units'])
    edges=candidate_edges(r,scores)
    fig=plt.figure(figsize=(17,9));gs=fig.add_gridspec(2,3,width_ratios=[1,1,1.15])
    ax_net=fig.add_subplot(gs[:,2])
    colors=[];layouts=[];node_data=[]
    for col,(s,labels,rows,side) in enumerate([(a,la,r['parents'],'parent'),(b,lb,r['children'],'child')]):
        ids=sorted(int(x) for x in np.unique(labels[labels>=0]))
        counts={i:int((labels==i).sum()) for i in ids}
        order=sorted(ids,key=lambda i:(-counts[i],float(s['positions'][labels==i,0].mean()),i))
        palette={i:plt.get_cmap('tab20')(j%20) for j,i in enumerate(order)}
        layout={i:float(y) for i,y in zip(order,np.linspace(.94,.06,len(order)))}
        colors.append(palette);layouts.append(layout)
        node_data.append([dict(group=i,samples=counts[i],network_y=layout[i]) for i in order])
        for row,(axes,slab_axis) in enumerate([((0,1),2),((0,2),1)]):
            ax=fig.add_subplot(gs[row,col]);p=s['positions'];mask=np.abs(p[:,slab_axis]-.5)<=.04
            ax.scatter(p[mask&(labels<0),axes[0]],p[mask&(labels<0),axes[1]],s=9,c='.65',marker='x',linewidths=.6)
            for i in order:
                keep=mask&(labels==i)
                ax.scatter(p[keep,axes[0]],p[keep,axes[1]],s=8,color=palette[i],alpha=.8)
            sources=s['source_positions'];ax.scatter(sources[:,axes[0]],sources[:,axes[1]],marker='*',s=90,c='black')
            ax.set(xlim=(.12,.88),ylim=(.12,.88),aspect='equal',xlabel='x',ylabel='yz'[row],
                   title=f"{'Previous' if col==0 else 'Next'} t={float(s['time']):.2f}: {'xy' if row==0 else 'xz'} slab")
            ax.grid(alpha=.15)
        for i in order:
            item=next(x for x in rows if x[f'{side}_group']==i)
            warning=(item['status']=='ambiguous_target') if side=='parent' else not item['accepted']
            ax_net.scatter(col,layout[i],s=80,color=palette[i],edgecolors='#be3030' if warning else 'black',linewidths=1.8,zorder=4)
            text=f'{"P" if col==0 else "C"}{i} | {counts[i]}'
            ax_net.text(col+(-.035 if col==0 else .035),layout[i],text,ha='right' if col==0 else 'left',va='center',fontsize=9)
    styles=dict(accepted=('#238b45','-',1.5),redundant=('#d87411','--',2),
                ambiguous_best=('#b42d36',':',1.8),ambiguous_runner_up=('#b42d36','--',1.3),insufficient_pool=('.5',':',1))
    for edge in edges:
        p,q=edge['parent_group'],edge['child_group'];color,ls,lw=styles[edge['kind']]
        ax_net.plot([0,1],[layouts[0][p],layouts[1][q]],color=color,ls=ls,lw=lw,zorder=2)
    ax_net.set(xlim=(-.42,1.43),ylim=(0,1.04),title='Saved group-level candidate links\nP/C = local label | sample count')
    ax_net.axis('off')
    handles=[Line2D([0],[0],color=v[0],ls=v[1],label=k.replace('_',' ')) for k,v in styles.items() if any(e['kind']==k for e in edges)]
    if handles:ax_net.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.01),fontsize=9,frameon=False)
    fig.suptitle(f"{c['num_cells']}³: t={c['parent_time']:.2f} → {c['child_time']:.2f} | {c['mode'].replace('_',' ')} | gate={c['gate_code_units']:.5f}",fontsize=15)
    fig.text(.035,.025,'Colors identify event-local groups independently in each column; matching colors do not imply identity. Gray crosses: size-excluded samples.\n'
             'Spatial panels project samples within ±0.04 of the central plane; the graph uses all 3D samples. Stars: sources (display only).\n'
             'Red node border: ambiguous parent or child without accepted support. Red lines show competing best/runner-up targets reconstructed from point neighborhoods.\n'
             'Network layout is schematic; lines are group-level candidates, not sample trajectories. No physical shock identity or merger is established.',fontsize=9)
    fig.subplots_adjust(left=.055,right=.98,top=.88,bottom=.18,wspace=.32,hspace=.36)
    fig.savefig(path,dpi=140);plt.close(fig)
    return dict(config=c,edges=edges,reconstructed_pair_scores=scores,parent_nodes=node_data[0],child_nodes=node_data[1],
                displayed_samples=[dict(xy=int((np.abs(s['positions'][:,2]-.5)<=.04).sum()),
                                        xz=int((np.abs(s['positions'][:,1]-.5)<=.04).sum())) for s in [a,b]])


def run(directory,output):
    meta=json.loads((directory/'metadata.json').read_text());report=json.loads((directory/'readback.json').read_text())
    if not report['passed']:raise ValueError('Require successful temporal readback.')
    hashes=dict(meta['input_sha256'])
    for name in ['metadata.json','readback.json','support_records.json.gz']:
        p=directory/name;hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p,h in hashes.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError(f'Input changed: {p}')
    with gzip.open(directory/'support_records.json.gz','rt') as f:records=json.load(f)
    output.mkdir(parents=True,exist_ok=False);audit=[]
    for name,n,mode,event,gate in CASES:
        record=choose_record(records,n,mode,event,gate);root=Path(meta['reference_directory'] if n==64 else meta['target_directory'])
        catalogs=[];labels=[]
        for i in [event,event+1]:
            with np.load(root/'online'/f'surfaces_{i:04d}.npz') as s:catalogs.append({k:s[k].copy() for k in s.files})
            with np.load(Path(meta['comparison_directory'])/f'groups_n{n:03d}_event{i:04d}.npz') as g:labels.append(g[f'{mode}_a60_labels'])
        a=render(output/f'{name}.png',record,*catalogs,*labels);a['figure']=f'{name}.png';audit.append(a)
        print(name, {kind:sum(e['kind']==kind for e in a['edges']) for kind in sorted({e['kind'] for e in a['edges']})},flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (output/'display_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (output/'metadata.json').write_text(json.dumps(dict(input_sha256=hashes,input_hashes_unchanged=True,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__),Path('experiments/astrolink/surface_associations.py')]},
        spatial_slab_half_width=.04,physical_identities_assigned=False,production_tracker_changed=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();run(a.directory,a.output_dir)
