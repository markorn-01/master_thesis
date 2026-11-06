"""Inspect small inner patches and test pooled support on saved catalogs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import importlib.metadata
import numpy as np
from experiments.astrolink.shock_connectivity import connectivity_partition
from experiments.astrolink.patch_support import pooled_support

TRANSITIONS=[7,9,12,13,14]


def write_csv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def figures(output,catalogs,child_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,6))
    for ax,event in zip(axes,[12,13,14]):
        a=catalogs[event];p=a['p'];raw=a['graphs'][60]['raw_labels'];g=a['graphs'][60]
        section=np.abs(p[:,2]-.5)<=1.5*a['dx']
        colors=np.tile([.6,.6,.6,1.],(len(p),1))
        for k,group in enumerate(np.unique(g['labels'][g['labels']>=0])):
            colors[raw==group]=plt.get_cmap('tab20')(k%20)
        ax.scatter(p[section,0],p[section,1],c=colors[section],s=14)
        ax.set(xlim=(.15,.85),ylim=(.2,.8),aspect='equal',xlabel='x',ylabel='y',
               title=f"t={a['time']:.2f}: {len(g['component_sizes'])} raw components\nGroups meeting 20-sample cutoff: {len(np.unique(g['labels'][g['labels']>=0]))}")
        ax.grid(alpha=.15)
    fig.suptitle('Inner patches are present before they reach the association size cutoff',fontsize=15)
    fig.text(.04,.055,'Spatial graph: radius 1.5 cells, signed normal angle ≤60°. Gray: raw components with fewer than 20 samples.\n'
             'Thin z section for display only; grouping uses the complete 3D catalog. Colors are event-local and do not establish identities.',fontsize=10)
    fig.tight_layout(rect=(.01,.14,.99,.91));fig.savefig(output/'patch_sequence.png',dpi=140);plt.close(fig)
    selected=[r for r in child_rows if r['parent_event']==13 and r['segmentation_angle']==60
              and r['dominance_margin']==.25 and r['child_samples']==152]
    fig,axes=plt.subplots(1,3,figsize=(13,5),sharey=True)
    for ax,distance in zip(axes,[1.5,2.5,3.5]):
        rows=[r for r in selected if r['distance_cells']==distance]
        x=np.arange(len(rows))
        ax.bar(x-.18,[r['largest_single_child_coverage'] for r in rows],.36,label='Best single patch',color='#b56a2c')
        ax.bar(x+.18,[r['union_child_coverage'] for r in rows],.36,label='Unique pooled support',color='#007f8b')
        ax.axhline(.5,color='.4',ls='--',lw=1)
        ax.set(xticks=x,xticklabels=[f"Local group {r['child_group']}" for r in rows],ylim=(0,1.08),
               title=f'Distance gate {distance:g} cells',ylabel='Fraction of child samples covered')
        for i,r in enumerate(rows):ax.text(i,.03,f"{r['eligible_parents']} parent patches",ha='center',fontsize=9)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False,fontsize=10)
    fig.suptitle('Support from t=0.13 patches for the larger t=0.14 inner groups',fontsize=14)
    fig.text(.04,.025,'Example: 60° spatial partition; association normal angle ≤60°; each parent ≥50% covered; dominance gap ≥25%.\n'
             'The pool must cover ≥50% of the child. Union counts each child sample once. These are geometric support fractions, not probabilities.',fontsize=9)
    fig.tight_layout(rect=(.01,.16,.99,.84));fig.savefig(output/'pooled_support.png',dpi=140);plt.close(fig)


def run(directory,output):
    output.mkdir(parents=True,exist_ok=False)
    catalogs={};hashes={};context=[]
    for event in sorted(set(TRANSITIONS+[e+1 for e in TRANSITIONS])):
        path=directory/f'surfaces_{event:04d}.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as a:
            if int(a['schema_version'])!=3:raise ValueError('Require multi-source catalogs.')
            p,n,dx,time=a['positions'],a['normals'],float(a['grid_spacing']),float(a['time'])
            overlap=a['upstream_inside_any_source']|a['downstream_inside_any_source']
        entry=dict(p=p,n=n,dx=dx,time=time,graphs={});saved=dict(positions=p,normals=n,grid_spacing=dx,time=time)
        for angle in [30,60]:
            g=connectivity_partition(p,n,dx,1.5,angle,20);entry['graphs'][angle]=g
            saved[f'raw_a{angle}']=g['raw_labels'];saved[f'labels_a{angle}']=g['labels']
            for group,size in enumerate(g['component_sizes']):
                points=p[g['raw_labels']==group];c=points.mean(axis=0)
                context.append(dict(event_index=event,time=time,segmentation_angle=angle,group=group,samples=int(size),
                                    passes_cutoff=bool(size>=20),centroid_x=float(c[0]),centroid_y=float(c[1]),centroid_z=float(c[2]),
                                    source_overlap_samples=int(overlap[g['raw_labels']==group].sum())))
        catalogs[event]=entry;np.savez_compressed(output/f'event_{event:04d}.npz',**saved)
    parents=[];children=[];links=[];summary=[]
    for event in TRANSITIONS:
        a,b=catalogs[event],catalogs[event+1]
        if a['dx']!=b['dx'] or b['time']<=a['time']:raise ValueError('Invalid event geometry/timing.')
        support_archive={}
        for angle in [30,60]:
            for distance in [1.5,2.5,3.5]:
                for margin in [.1,.25]:
                    config=dict(parent_event=event,child_event=event+1,parent_time=a['time'],child_time=b['time'],
                                segmentation_angle=angle,distance_cells=distance,dominance_margin=margin)
                    r=pooled_support(a['p'],a['n'],a['graphs'][angle]['labels'],b['p'],b['n'],b['graphs'][angle]['labels'],
                                     distance*a['dx'],60.,.5,.5,margin)
                    parents.extend(dict(**config,**p) for p in r['parents'])
                    for c in r['children']:
                        row=dict(c);row['parent_groups']=json.dumps(row['parent_groups']);children.append(dict(**config,**row))
                        support_archive[f'a{angle}_d{distance:g}_m{margin:g}_child{c["child_group"]}']=r['supported_child_rows'][c['child_group']]
                    links.extend(dict(**config,**s) for s in r['links'])
                    summary.append(dict(**config,accepted_links=sum(s['accepted'] for s in r['links']),
                                        accepted_pools=sum(s['accepted'] for s in r['children']),
                                        ambiguous_parents=sum(s['status']=='ambiguous_target' for s in r['parents']),
                                        insufficient_parents=sum(s['status']=='insufficient_parent_support' for s in r['parents'])))
        np.savez_compressed(output/f'support_{event:04d}_{event+1:04d}.npz',**support_archive)
        print(f'Completed patch-support comparison {event:02d} -> {event+1:02d}',flush=True)
    for name,rows in [('raw_components',context),('parents',parents),('children',children),('links',links),('summary',summary)]:write_csv(output/f'{name}.csv',rows)
    source_paths=[Path(__file__),Path('experiments/astrolink/patch_support.py'),Path('experiments/astrolink/shock_connectivity.py')]
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    meta=dict(input_sha256=hashes,input_hashes_unchanged=True,transitions=TRANSITIONS,configurations=len(summary),
              source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
              numpy=np.__version__,scipy=importlib.metadata.version('scipy'),min_parent_coverage=.5,min_child_coverage=.5,
              association_normal_angle=60.,segmentation_radius_cells=1.5,segmentation_angles=[30,60],
              association_distance_cells=[1.5,2.5,3.5],dominance_margins=[.1,.25],minimum_group_size=20,
              scope='Asymmetric pooled geometric support, not persistent physical tracking or automatic physical mergers',
              source_overlap='Recorded for inspection, not used to select associations or infer ownership')
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    figures(output,catalogs,children)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.directory,a.output_dir)
