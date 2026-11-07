"""Compare experimental temporal support in saved two-bubble graph partitions."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.support_evidence import support_evidence, controlled_examples
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write, neighborhood_settings


def support_summary(result):
    """Count supported groups and samples without counting overlapping unions twice."""
    children=result['children']; parents=result['parents']; links=result['links']
    return dict(parent_groups=len(parents),child_groups=len(children),
                accepted_child_groups=sum(c['accepted'] for c in children),
                accepted_links=sum(l['accepted'] for l in links),
                accepted_child_samples=sum(c['child_samples'] for c in children if c['accepted']),
                retained_child_samples=sum(c['child_samples'] for c in children),
                supported_child_samples=sum(c['union_covered_child_samples'] for c in children if c['accepted']),
                redundant_accepted_links=sum(l['accepted'] and l['redundant_support'] for l in links),
                ambiguous_parents=sum(p['status']=='ambiguous_target' for p in parents),
                insufficient_parents=sum(p['status']=='insufficient_parent_support' for p in parents),
                unsupported_child_groups=sum(not c['accepted'] for c in children))


def run(comparison,reference,target,output):
    previous=json.loads((comparison/'verification.json').read_text())
    if not previous['readback_passed']:raise ValueError('Require checked resolution products.')
    hashes=dict(previous['input_sha256'])
    hashes[str(comparison/'verification.json')]=hashlib.sha256((comparison/'verification.json').read_bytes()).hexdigest()
    for path,digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError(f'Input changed: {path}')
    output.mkdir(parents=True,exist_ok=False)
    summaries=[]; records=[]
    for root in [reference,target]:
        meta=json.loads((root/'metadata.json').read_text()); n=meta['num_cells']; catalogs=[]
        for event,time in enumerate(meta['actual_times']):
            path=root/'online'/f'surfaces_{event:04d}.npz'
            if str(path) not in hashes:raise ValueError('Catalog not covered by prior verification.')
            group_path=comparison/f'groups_n{n:03d}_event{event:04d}.npz'
            hashes[str(group_path)]=hashlib.sha256(group_path.read_bytes()).hexdigest()
            with np.load(path) as a, np.load(group_path) as g:
                if abs(float(a['time'])-time)>1e-7:raise ValueError('Time mismatch.')
                catalogs.append(dict(p=a['positions'],n=a['normals'],time=time,groups={k:g[k] for k in g.files}))
        for event,(a,b) in enumerate(zip(catalogs,catalogs[1:])):
            cache={}
            for mode,radius,cutoff in neighborhood_settings(n):
                for angle in [30,60]:
                    la=a['groups'][f'{mode}_a{angle}_labels'];lb=b['groups'][f'{mode}_a{angle}_labels']
                    for gate_ref_cells in [1.5,2.5]:
                        gate=gate_ref_cells/64
                        config=dict(num_cells=n,parent_event=event,child_event=event+1,parent_time=a['time'],child_time=b['time'],
                                    mode=mode,segmentation_angle=angle,segmentation_radius_cells=radius,min_samples=cutoff,
                                    gate_reference_cells=gate_ref_cells,gate_code_units=gate,gate_native_cells=gate*n)
                        key=(la.tobytes(),lb.tobytes(),gate)
                        if key not in cache:
                            result=support_evidence(a['p'],a['n'],la,b['p'],b['n'],lb,gate,
                                                    max_angle_degrees=60.,min_parent_coverage=.5,min_child_coverage=.5,dominance_margin=.25)
                            # Arrays are archived as JSON row-index lists for exact support readback.
                            cache[key]={k:({str(i):v.tolist() for i,v in val.items()} if k=='supported_child_rows' else val)
                                        for k,val in result.items()}
                        result=cache[key]
                        records.append(dict(config=config,result=result))
                        summaries.append(dict(**config,**support_summary(result),
                                               parent_excluded_samples=int((la<0).sum()),child_excluded_samples=int((lb<0).sum())))
            print(f'{n}³ temporal support: {event:02d} -> {event+1:02d}',flush=True)
    csv_write(output/'summary.csv',summaries)
    with gzip.open(output/'support_records.json.gz','wt') as f:json.dump(records,f,separators=(',',':'))
    (output/'controlled_examples.json').write_text(json.dumps(controlled_examples(),indent=2)+'\n')
    plot(output,summaries)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    sources=[Path(__file__),Path('experiments/astrolink/patch_support.py'),Path('experiments/astrolink/support_evidence.py'),
             Path('experiments/wind_bubble/compare_two_bubble_resolutions.py')]
    report=dict(comparisons=len(records),input_sha256=hashes,input_hashes_unchanged=True,
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                comparison_directory=str(comparison),reference_directory=str(reference),target_directory=str(target),
                association_normal_angle=60,min_parent_coverage=.5,min_child_coverage=.5,dominance_margin=.25,
                gates_code_units=[1.5/64,2.5/64],physical_identities_assigned=False,production_tracker_changed=False,
                scope='Adjacent-event geometric support; not persistent physical identity or a physical accuracy score.')
    (output/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')


def plot(output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specs=[(64,'native_cells'),(128,'native_cells'),(128,'native_cells_scaled_count'),(128,'common_physical_radius')]
    fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True,sharey='row')
    for col,gate in enumerate([1.5,2.5]):
        for index,(n,mode) in enumerate(specs):
            style=dict(ls=['-','-','--',':'][index],marker=['o','o','s','x'][index],
                       markersize=[7,6,5,5][index],lw=[3,2,2,1.5][index],alpha=.8)
            r=[x for x in rows if x['num_cells']==n and x['mode']==mode and x['segmentation_angle']==60 and x['gate_reference_cells']==gate]
            t=[x['child_time'] for x in r];label=f'{n}³ {mode.replace("_"," ")}; cutoff {r[0]["min_samples"]}'
            axes[0,col].plot(t,[x['child_groups'] for x in r],label=label,**style)
            axes[1,col].plot(t,[x['unsupported_child_groups'] for x in r],**style)
            axes[2,col].plot(t,[x['redundant_accepted_links'] for x in r],**style)
        axes[0,col].set_title(f'Common physical association gate: {gate}/64 = {gate/64:.5f}')
        for ax,ylabel in zip(axes[:,col],['Retained child groups','Groups without accepted support','Redundant accepted links']):
            ax.set_ylabel(ylabel);ax.grid(alpha=.2)
        axes[2,col].set_xlabel('Child simulation time')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=2,fontsize=9)
    fig.suptitle('Temporal geometric support: resolution and grouping sensitivity',fontsize=15)
    fig.text(.04,.025,'Spatial normal-angle limit 60°; association normal-angle limit 60°; both coverage thresholds 0.5; dominance margin 0.25.\n'
             'Zero unsupported groups can reflect a permissive gate or excluded fragments. Redundant links add no exclusive child samples.\n'
             'These are event-local groups and candidate correspondences, not confirmed shock identities. All 30° results are saved separately.',fontsize=9)
    fig.tight_layout(rect=(.01,.12,.99,.87));fig.savefig(output/'temporal_support_comparison.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('comparison',type=Path);p.add_argument('reference',type=Path);p.add_argument('target',type=Path)
    p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args()
    run(a.comparison,a.reference,a.target,a.output_dir)
