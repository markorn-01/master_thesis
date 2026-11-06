"""Group-size sensitivity and deliberately adversarial pooled-support controls."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import importlib.metadata
import numpy as np
from experiments.astrolink.shock_connectivity import connectivity_partition
from experiments.astrolink.support_evidence import support_evidence,controlled_examples

CUTOFFS=[1,8,16,20,32]


def size_filter(raw_labels,min_size):
    if not isinstance(min_size,int) or min_size<1:raise ValueError('Require a positive integer cutoff.')
    raw=np.asarray(raw_labels);labels=raw.copy()
    ids,counts=np.unique(raw[raw>=0],return_counts=True)
    for group,size in zip(ids,counts):
        if size<min_size:labels[raw==group]=-1
    return labels


def write_csv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def plots(output,context):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for column,angle in enumerate([30,60]):
        for cutoff in CUTOFFS:
            rows=[r for r in context if r['segmentation_angle']==angle and r['min_size']==cutoff and r['time']>=.099]
            times=[r['time'] for r in rows]
            axes[0,column].plot(times,[r['retained_groups'] for r in rows],'o-',label=f'Minimum {cutoff}')
            axes[1,column].plot(times,[r['excluded_samples'] for r in rows],'o-')
        axes[0,column].set(title=f'Spatial normal-angle limit {angle}°',ylabel='Retained groups')
        axes[1,column].set(xlabel='Simulation time',ylabel='Samples excluded by size cutoff')
        for ax in axes[:,column]:ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=5,frameon=False)
    fig.suptitle('Minimum-size sensitivity: more groups do not establish more physical shocks',fontsize=15)
    fig.text(.05,.025,'Same saved point catalogs and raw spatial graphs; only the group-size cutoff changes.\n'
             'Tiny components are retained at cutoff 1. Association support remains geometric evidence, not physical identity.',fontsize=10)
    fig.tight_layout(rect=(.01,.11,.99,.86));fig.savefig(output/'size_sensitivity.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot([0,1],[0,0],color='#007f8b',lw=3,label='Stipulated true correspondence, accepted')
    ax.plot([0,1],[.1,0],color='#be4b37',lw=3,ls='--',label='Stipulated unrelated sheet, also accepted')
    ax.scatter([0,0,1],[0,.1,0],s=110,c=['#007f8b','#be4b37','#007f8b'],zorder=3)
    ax.text(-.03,.015,'Parent 0: surviving sheet',ha='right')
    ax.text(-.03,.1,'Parent 1: nearby distractor',ha='right')
    ax.text(1.03,0,'Child group',va='center')
    ax.set(xlim=(-.65,1.4),ylim=(-.04,.15),xticks=[0,1],xticklabels=['Previous event','Next event'],
           ylabel='Sheet z coordinate (toy units)',title='Counterexample: full pooled support does not prove ancestry')
    ax.legend(loc='upper right',fontsize=9)
    ax.grid(alpha=.15)
    fig.text(.04,.025,'Both parents cover the same 9 child samples: 100% union coverage, but neither adds exclusive support.\n'
             'Two parallel 3×3 sheets, normal +z, separation 0.1, distance gate 0.2. Truth is stipulated for this control.\n'
             'This exposes an information limit of geometric matching; it is not a measured false-link rate in the simulation.',fontsize=9)
    fig.tight_layout(rect=(.01,.18,.99,.97));fig.savefig(output/'parallel_sheet_control.png',dpi=140);plt.close(fig)


def run(directory,output):
    output.mkdir(parents=True,exist_ok=False)
    paths=sorted(directory.glob('surfaces_*.npz'))
    if len(paths)<2:raise ValueError('Require consecutive catalogs.')
    catalogs=[];hashes={};context=[]
    for event,path in enumerate(paths):
        if path.name!=f'surfaces_{event:04d}.npz':raise ValueError('Require contiguous events from zero.')
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as a:
            if int(a['schema_version'])!=3:raise ValueError('Require multi-source catalogs.')
            p,n=a['positions'],a['normals'];dx,time=float(a['grid_spacing']),float(a['time'])
        if catalogs and (time<=catalogs[-1]['time'] or dx!=catalogs[-1]['dx']):raise ValueError('Invalid times/grid spacing.')
        catalog=dict(p=p,n=n,dx=dx,time=time,labels={})
        saved=dict(positions=p,normals=n,grid_spacing=dx,time=time)
        for angle in [30,60]:
            graph=connectivity_partition(p,n,dx,1.5,angle,20)
            raw=graph['raw_labels'];saved[f'raw_a{angle}']=raw
            for cutoff in CUTOFFS:
                labels=size_filter(raw,cutoff)
                if cutoff==20:np.testing.assert_array_equal(labels,graph['labels'])
                catalog['labels'][angle,cutoff]=labels;saved[f'a{angle}_s{cutoff}']=labels
                context.append(dict(event_index=event,time=time,segmentation_angle=angle,min_size=cutoff,
                                    samples=len(p),raw_groups=len(graph['component_sizes']),
                                    retained_groups=len(np.unique(labels[labels>=0])),retained_samples=int((labels>=0).sum()),
                                    excluded_samples=int(((raw>=0)&(labels<0)).sum())))
        catalogs.append(catalog);np.savez_compressed(output/f'event_{event:04d}.npz',**saved)
    summaries=[];children=[];links=[]
    for event,(a,b) in enumerate(zip(catalogs,catalogs[1:])):
        for angle in [30,60]:
            for cutoff in CUTOFFS:
                for distance in [1.5,2.5]:
                    config=dict(parent_event=event,child_event=event+1,parent_time=a['time'],child_time=b['time'],
                                segmentation_angle=angle,min_size=cutoff,distance_cells=distance)
                    r=support_evidence(a['p'],a['n'],a['labels'][angle,cutoff],b['p'],b['n'],b['labels'][angle,cutoff],
                                       distance*a['dx'],max_angle_degrees=60.,min_parent_coverage=.5,min_child_coverage=.5,dominance_margin=.25)
                    for child in r['children']:
                        row=dict(child);row['parent_groups']=json.dumps(row['parent_groups']);children.append(dict(**config,**row))
                    links.extend(dict(**config,**l) for l in r['links'])
                    summaries.append(dict(**config,parent_groups=len(r['parents']),child_groups=len(r['children']),
                                          accepted_pools=sum(c['accepted'] for c in r['children']),
                                          accepted_links=sum(l['accepted'] for l in r['links']),
                                          redundant_accepted_links=sum(l['accepted'] and l['redundant_support'] for l in r['links']),
                                          ambiguous_parents=sum(p['status']=='ambiguous_target' for p in r['parents']),
                                          insufficient_parents=sum(p['status']=='insufficient_parent_support' for p in r['parents'])))
        print(f'Completed size sensitivity {event:02d} -> {event+1:02d}',flush=True)
    for name,rows in [('group_sizes',context),('children',children),('links',links),('summary',summaries)]:write_csv(output/f'{name}.csv',rows)
    controls=controlled_examples();(output/'controlled_examples.json').write_text(json.dumps(controls,indent=2)+'\n')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    sources=[Path(__file__),Path('experiments/astrolink/support_evidence.py'),Path('experiments/astrolink/patch_support.py'),Path('experiments/astrolink/shock_connectivity.py')]
    meta=dict(events=len(catalogs),adjacent_comparisons=len(summaries),cutoffs=CUTOFFS,segmentation_radius_cells=1.5,
              segmentation_angles=[30,60],distance_cells=[1.5,2.5],association_normal_angle=60.,
              min_parent_coverage=.5,min_child_coverage=.5,dominance_margin=.25,
              input_sha256=hashes,input_hashes_unchanged=True,
              source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              numpy=np.__version__,scipy=importlib.metadata.version('scipy'),
              production_tracker_changed=False,physical_identity_validation=False,
              controls='Stipulated toy correspondences; not a measured false-positive rate on hydrodynamic data.',
              conclusion='Nearby aligned distractors can pass pooled support. Keep it experimental; cutoff sensitivity does not resolve identity.')
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');plots(output,context)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.directory,a.output_dir)
