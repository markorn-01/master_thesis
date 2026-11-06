"""Bounded many-to-many association study on consecutive saved shock catalogs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import importlib.metadata
import numpy as np
from experiments.astrolink.shock_connectivity import connectivity_partition
from experiments.astrolink.surface_associations import association_scores, accepted_links


def csv_write(path, rows):
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def plot(directory, nodes, scores, states):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, angle in zip(axes, (30, 60)):
        selected = [n for n in nodes if n['segmentation_angle'] == angle]
        lookup = {(n['event_index'],n['group']):n for n in selected}
        for s in scores:
            if s['segmentation_angle'] != angle or s['distance_cells'] != 2.5: continue
            if not accepted_links([s], .25): continue
            a,b = lookup[s['parent_event'],s['parent_group']],lookup[s['child_event'],s['child_group']]
            ax.plot([a['time'],b['time']],[a['centroid_x'],b['centroid_x']],c='#007f8b',alpha=.65,lw=1.5)
        for n in selected:
            state = next(s for s in states if s['segmentation_angle']==angle and s['distance_cells']==2.5
                         and s['min_coverage']==.25 and s['event_index']==n['event_index'] and s['group']==n['group'])
            merging = state['candidate_parents'] >= 2
            ax.scatter(n['time'],n['centroid_x'],s=20+18*np.log10(n['samples']),
                       c='#c04e24' if merging else '#007f8b',zorder=3)
            if merging: ax.annotate('multiple candidate parents',(n['time'],n['centroid_x']),
                                   xytext=(12,20),textcoords='offset points',fontsize=8,
                                   arrowprops=dict(arrowstyle='-',color='.4'))
        y = [n['centroid_x'] for n in selected]
        ax.set(ylabel='Group centroid x',ylim=(min(y)-.035,max(y)+.035),title=f'Spatial grouping: radius 1.5 cells, normal angle ≤{angle}°')
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('Simulation time')
    fig.suptitle('Candidate temporal associations — no forced one-to-one identities',fontsize=16)
    fig.text(.06,.025,'Declared example: displacement gate 2.5 cells; signed normal difference ≤60°; coverage ≥25% in each direction.\n'
             'Nodes are event-local groups (size reflects sample count). Lines are geometric candidates, not material trajectories.\n'
             'Orange marks multiple incoming candidates. Nodes with equal centroid x can overlap. No matching across missing events.',fontsize=9)
    fig.tight_layout(rect=(.02,.13,.99,.94))
    fig.savefig(directory/'candidate_associations.png',dpi=140);plt.close(fig)


def run(directory, output):
    output.mkdir(parents=True,exist_ok=False)
    paths=sorted(directory.glob('surfaces_*.npz'))
    if len(paths)<2: raise ValueError('Require at least two surface catalogs.')
    catalogs=[];nodes=[];hashes={};previous_time=-np.inf
    for event,path in enumerate(paths):
        if path.name != f'surfaces_{event:04d}.npz': raise ValueError('Require a contiguous event sequence starting at zero.')
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as a:
            if int(a['schema_version'])!=3: raise ValueError('Require multi-source schema 3.')
            p,n=a['positions'],a['normals'];dx,time=float(a['grid_spacing']),float(a['time'])
        if time<=previous_time: raise ValueError('Event times must increase.')
        if catalogs and dx!=catalogs[0]['dx']: raise ValueError('Require a fixed grid spacing.')
        previous_time=time
        catalog=dict(p=p,n=n,dx=dx,time=time,labels={})
        saved=dict(positions=p,normals=n,time=time,grid_spacing=dx)
        for angle in (30,60):
            g=connectivity_partition(p,n,dx,1.5,angle,20)
            labels=g['labels'];catalog['labels'][angle]=labels
            saved[f'labels_a{angle}']=labels;saved[f'raw_labels_a{angle}']=g['raw_labels']
            for group in np.unique(labels[labels>=0]):
                selected=p[labels==group];centroid=selected.mean(axis=0)
                nodes.append(dict(segmentation_angle=angle,event_index=event,time=time,group=int(group),samples=len(selected),
                                  centroid_x=float(centroid[0]),centroid_y=float(centroid[1]),centroid_z=float(centroid[2])))
        catalogs.append(catalog)
        np.savez_compressed(output/f'event_{event:04d}.npz',**saved)
    scores=[]
    for event,(a,b) in enumerate(zip(catalogs,catalogs[1:])):
        for angle in (30,60):
            for distance in (1.5,2.5,3.5):
                pair_scores=association_scores(a['p'],a['n'],a['labels'][angle],b['p'],b['n'],b['labels'][angle],distance*a['dx'],60.)
                for score in pair_scores:
                    scores.append(dict(segmentation_angle=angle,distance_cells=distance,parent_event=event,child_event=event+1,
                                       parent_time=a['time'],child_time=b['time'],dt=b['time']-a['time'],**score))
        print(f'Compared event {event:02d} -> {event+1:02d}',flush=True)
    states=[];summary=[]
    for angle in (30,60):
        for distance in (1.5,2.5,3.5):
            selected=[s for s in scores if s['segmentation_angle']==angle and s['distance_cells']==distance]
            for coverage in (.25,.5):
                links=accepted_links(selected,coverage)
                current=[]
                for node in [n for n in nodes if n['segmentation_angle']==angle]:
                    event,group=node['event_index'],node['group']
                    parents=[s['parent_group'] for s in links if s['child_event']==event and s['child_group']==group]
                    children=[s['child_group'] for s in links if s['parent_event']==event and s['parent_group']==group]
                    state=dict(segmentation_angle=angle,distance_cells=distance,min_coverage=coverage,event_index=event,
                               time=node['time'],group=group,candidate_parents=len(parents),candidate_children=len(children),
                               parent_groups=json.dumps(parents),child_groups=json.dumps(children),
                               incoming_comparison_available=event>0,outgoing_comparison_available=event<len(catalogs)-1)
                    current.append(state);states.append(state)
                summary.append(dict(segmentation_angle=angle,distance_cells=distance,min_coverage=coverage,nodes=len(current),
                                    links=len(links),multiple_parent_nodes=sum(s['candidate_parents']>1 for s in current),
                                    multiple_child_nodes=sum(s['candidate_children']>1 for s in current),
                                    unmatched_incoming_nodes=sum(s['incoming_comparison_available'] and s['candidate_parents']==0 for s in current),
                                    unmatched_outgoing_nodes=sum(s['outgoing_comparison_available'] and s['candidate_children']==0 for s in current)))
    for name,rows in [('nodes',nodes),('pair_scores',scores),('candidate_states',states),('summary',summary)]:csv_write(output/f'{name}.csv',rows)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    source_paths=[Path(__file__),Path('experiments/astrolink/surface_associations.py'),Path('experiments/astrolink/shock_connectivity.py')]
    metadata=dict(events=len(catalogs),input_sha256=hashes,input_hashes_unchanged=True,
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                  python=platform.python_version(),numpy=np.__version__,scipy=importlib.metadata.version('scipy'),
                  segmentation_radius_cells=1.5,segmentation_angles=[30,60],min_display_group_samples=20,
                  association_distance_cells=[1.5,2.5,3.5],association_normal_angle_degrees=60,min_coverages=[.25,.5],
                  method='unique covered samples in both directions, many-to-many links, adjacent events only',
                  excluded_features='source identity, source positions, Mach, density, velocities and persistent point IDs',
                  fuzzycat_executed=False,scope='candidate geometric associations; not validated physical identities or source ancestry')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    plot(output,nodes,scores,states)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();run(args.directory,args.output_dir)
