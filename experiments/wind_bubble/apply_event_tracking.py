"""Apply frozen event hypotheses to saved two-bubble catalogs at two distances."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from experiments.astrolink.event_tracking import event_tracking
from experiments.wind_bubble.plot_resolution_links import choose_record
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write


def check_candidates(a,b,la,lb,result):
    """Independent sparse-matrix geometry and scores; exhaustive small conflict sets."""
    p=result['parameters'];na=a['normals'].astype(float);nb=b['normals'].astype(float)
    na/=np.linalg.norm(na,axis=1)[:,None];nb/=np.linalg.norm(nb,axis=1)[:,None]
    pairs=cKDTree(a['positions']).sparse_distance_matrix(cKDTree(b['positions']),p['gate'],output_type='coo_matrix')
    ia,ib,d=pairs.row,pairs.col,pairs.data
    keep=(la[ia]>=0)&(lb[ib]>=0)&(np.einsum('ij,ij->i',na[ia],nb[ib])>=np.cos(np.deg2rad(p['normal_angle'])))
    ia,ib,d=ia[keep],ib[keep],d[keep]
    keys={(parent,child) for e in result['candidates'] for parent in e['parents'] for child in e['children']}
    support={}
    for parent,child in keys:
        mask=(la[ia]==parent)&(lb[ib]==child);distances=d[mask];per_side=[]
        for rows in [ia[mask],ib[mask]]:
            unique,inverse=np.unique(rows,return_inverse=True);minimum=np.full(len(unique),np.inf)
            np.minimum.at(minimum,inverse,distances);per_side.append(dict(zip(map(int,unique),map(float,minimum))))
        support[parent,child]=per_side
    for event in result['candidates']:
        coverages=[];cost=[]
        for side,groups,labels in [(0,event['parents'],la),(1,event['children'],lb)]:
            for group in groups:
                dictionaries=[support[group,c][0] for c in event['children']] if side==0 else [support[parent,group][1] for parent in event['parents']]
                points=set().union(*(set(x) for x in dictionaries));fraction=len(points)/int((labels==group).sum())
                assert fraction>=p['coverage'];coverages.append(fraction)
                distances=[min(x.get(row,np.inf) for x in dictionaries) for row in points]
                cost.append(fraction*float(np.mean(distances))/p['gate'])
        np.testing.assert_allclose(event['score'],sum(coverages)-p['distance_weight']*sum(cost),rtol=1e-11,atol=1e-11)
        for expected,actual in zip(event['node_coverages'],coverages):np.testing.assert_allclose(expected['coverage'],actual,atol=0,rtol=0)
        if event['kind']!='continuation':
            footprints=([set(support[parent,event['children'][0]][1]) for parent in event['parents']] if event['kind']=='combination' else
                        [set(support[event['parents'][0],child][0]) for child in event['children']])
            denominator=int((lb==event['children'][0]).sum()) if event['kind']=='combination' else int((la==event['parents'][0]).sum())
            for i,footprint in enumerate(footprints):
                others=set().union(*(s for j,s in enumerate(footprints) if i!=j))
                fraction=len(footprint-others)/denominator
                assert fraction>=p['exclusive_fraction']
                np.testing.assert_allclose(fraction,event['exclusive_central_fractions'][i],atol=0,rtol=0)
    candidates=result['candidates'];exhaustive=len(candidates)<=16
    if exhaustive:
        feasible=[]
        for mask in range(1<<len(candidates)):
            indices={i for i in range(len(candidates)) if mask&(1<<i)}
            parents=[g for i in indices for g in candidates[i]['parents']];children=[g for i in indices for g in candidates[i]['children']]
            if len(set(parents))!=len(parents) or len(set(children))!=len(children):continue
            feasible.append((sum(candidates[i]['score'] for i in indices),indices))
        optimum=max(score for score,_ in feasible)
        essential=set.intersection(*(indices for score,indices in feasible if optimum-score<=p['tie_tolerance']))
        assert essential==set(result['selection']['accepted'])
        np.testing.assert_allclose(optimum,result['selection']['optimal_score'],rtol=1e-11,atol=1e-11)
    return dict(candidate_scores_checked=len(candidates),exhaustive_conflict_check=exhaustive)


def event_key(event):return (event['kind'],tuple(event['parents']),tuple(event['children']))


def check_selection_dp(result):
    """Independent exact group-occupancy DP, including larger hypothesis lists."""
    nodes=[('p',p['parent_group']) for p in result['parents']]+[('c',c['child_group']) for c in result['children']]
    if len(nodes)>22:raise ValueError('Independent DP is limited to this small reviewed problem.')
    bits={node:1<<i for i,node in enumerate(nodes)};candidates=result['candidates']
    masks=[sum(bits['p',p] for p in e['parents'])+sum(bits['c',c] for c in e['children']) for e in candidates]
    def solve(forbid=None):
        states={0:0.}
        for i,(mask,event) in enumerate(zip(masks,candidates)):
            if i==forbid:continue
            additions={}
            for occupied,score in states.items():
                if occupied&mask:continue
                target=occupied|mask;value=score+event['score']
                additions[target]=max(additions.get(target,-np.inf),value)
            for target,value in additions.items():states[target]=max(states.get(target,-np.inf),value)
        return max(states.values())
    optimum=solve();np.testing.assert_allclose(optimum,result['selection']['optimal_score'],atol=1e-11,rtol=1e-11)
    essential=[]
    for index in result['selection']['solver_selected']:
        gap=max(0.,optimum-solve(index))
        np.testing.assert_allclose(gap,result['selection']['exclusion_gaps'][str(index)],atol=1e-11,rtol=1e-11)
        if gap>result['parameters']['tie_tolerance']:essential.append(index)
    assert essential==result['selection']['accepted']


def run(directory,controls,output):
    meta_path=directory/'metadata.json';archive=directory/'support_records.json.gz';control_report=controls/'verification.json'
    meta=json.loads(meta_path.read_text());control=json.loads(control_report.read_text())
    # The tested tracker must be byte-identical before applying its fixed parameters.
    for path,digest in control['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError(f'Controlled source changed: {path}')
    root=Path(meta['target_directory']);partitions=Path(meta['comparison_directory']);run_path=root/'metadata.json'
    run_meta=json.loads(run_path.read_text());times=run_meta['actual_times']
    assert run_meta['num_cells']==128 and len(times)==17
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [meta_path,archive,run_path,control_report]}
    with gzip.open(archive,'rt') as f:previous=json.load(f)
    catalogs=[];labels=[]
    for event,t in enumerate(times):
        source=root/'online'/f'surfaces_{event:04d}.npz';partition=partitions/f'groups_n128_event{event:04d}.npz'
        for path in [source,partition]:hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(source) as a:catalogs.append({k:a[k].copy() for k in ['positions','normals','time']})
        with np.load(partition) as a:labels.append(a['native_cells_a60_labels'].copy())
        np.testing.assert_allclose(float(catalogs[-1]['time']),t,atol=1e-7,rtol=0)
    output.mkdir(parents=True,exist_ok=False);records=[];rows=[];events=[];checks=[];agreement=[]
    for event,(a,b) in enumerate(zip(catalogs,catalogs[1:])):
        results={}
        for gate in [1.5,2.5]:
            start=time.perf_counter()
            result=event_tracking(a['positions'],a['normals'],labels[event],b['positions'],b['normals'],labels[event+1],gate/64)
            elapsed=time.perf_counter()-start
            for key,value in control['parameters'].items():assert result['parameters'][key]==value
            check=check_candidates(a,b,labels[event],labels[event+1],result);check_selection_dp(result)
            checks.append(dict(parent_event=event,gate_reference_cells=gate,independent_dp_check=True,**check))
            baseline=choose_record(previous,128,'native_cells',event,gate)['result']
            baseline_pairs={(l['parent_group'],l['child_group']) for l in baseline['links'] if l['accepted']}
            selected_pairs={(l['parent_group'],l['child_group']) for l in result['links']}
            counts=Counter(e['kind'] for e in result['events'])
            row=dict(parent_event=event,child_event=event+1,parent_time=float(a['time']),child_time=float(b['time']),
                     gate_reference_cells=gate,parent_groups=len(result['parents']),child_groups=len(result['children']),
                     candidate_events=len(result['candidates']),continuations=counts['continuation'],combinations=counts['combination'],
                     separations=counts['separation'],ambiguous_parents=sum(p['status']=='ambiguous' for p in result['parents']),
                     unmatched_parents=sum(p['status']=='unmatched' for p in result['parents']),
                     ambiguous_children=sum(c['status']=='ambiguous' for c in result['children']),
                     unmatched_children=sum(c['status']=='unmatched' for c in result['children']),
                     baseline_links=len(baseline_pairs),prototype_links=len(selected_pairs),
                     added_vs_baseline=len(selected_pairs-baseline_pairs),removed_vs_baseline=len(baseline_pairs-selected_pairs),
                     tracker_seconds=elapsed)
            rows.append(row);records.append(dict(**row,result=result,added_pairs=sorted(selected_pairs-baseline_pairs),removed_pairs=sorted(baseline_pairs-selected_pairs)))
            for selected in result['events']:
                events.append(dict(parent_event=event,child_event=event+1,gate_reference_cells=gate,kind=selected['kind'],
                                   parents=','.join(map(str,selected['parents'])),children=','.join(map(str,selected['children'])),score=selected['score'],
                                   exclusion_gap=result['selection']['exclusion_gaps'][str(selected['candidate_index'])]))
            results[gate]=result
            print(f'{event:02d}→{event+1:02d} gate {gate}: {dict(counts)}; candidates={len(result["candidates"])}; {elapsed:.2f}s',flush=True)
        left={event_key(e) for e in results[1.5]['events']};right={event_key(e) for e in results[2.5]['events']}
        agreement.append(dict(parent_event=event,child_event=event+1,event_sets_identical=left==right,
                              shared_events=len(left&right),narrow_only=len(left-right),wide_only=len(right-left)))
    csv_write(output/'summary.csv',rows);csv_write(output/'selected_events.csv',events);csv_write(output/'distance_agreement.csv',agreement)
    with gzip.open(output/'results.json.gz','wt') as f:json.dump(records,f,separators=(',',':'))
    with gzip.open(output/'results.json.gz','rt') as f:saved=json.load(f)
    assert len(saved)==len(rows)==32
    for r in saved:
        parents=[p for e in r['result']['events'] for p in e['parents']];children=[c for e in r['result']['events'] for c in e['children']]
        assert len(set(parents))==len(parents) and len(set(children))==len(children)
    plot(records,output)
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    report=dict(intervals=16,configurations=32,parameters=control['parameters'],checks=checks,
        identical_event_sets_across_distances=sum(a['event_sets_identical'] for a in agreement),
        sparse_geometry_candidate_checks=sum(c['candidate_scores_checked'] for c in checks),
        exhaustive_conflict_checks=sum(c['exhaustive_conflict_check'] for c in checks),
        independent_dp_checks=sum(c['independent_dp_check'] for c in checks),
        event_totals={str(g):dict(Counter(e['kind'] for e in events if e['gate_reference_cells']==g)) for g in [1.5,2.5]},
        input_sha256=hashes,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path('experiments/astrolink/event_tracking.py')]},
        fixed_parameters=True,original_outputs_changed=False,physical_truth_available=False,
        physical_mergers_confirmed=False,persistent_ids_assigned=False,
        note='Added/removed baseline links measure disagreement, not false/missed real-data links. Runtime is unbenchmarked CPU postprocessing.')
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['configurations','identical_event_sets_across_distances','sparse_geometry_candidate_checks','exhaustive_conflict_checks','event_totals']},indent=2))


def plot(records,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(4,2,figsize=(12,14))
    colors={'continuation':'#238b45','combination':'#8256aa','separation':'#d17824'}
    for row,event in enumerate([7,10,13,14]):
        for col,gate in enumerate([1.5,2.5]):
            ax=axes[row,col];record=next(r for r in records if r['parent_event']==event and r['gate_reference_cells']==gate)
            result=record['result'];layouts=[]
            for side,name in enumerate(['parents','children']):
                nodes=result[name];key='parent_group' if side==0 else 'child_group'
                ids=sorted(n[key] for n in nodes);ys=np.linspace(.85,.15,len(ids));layout=dict(zip(ids,ys));layouts.append(layout)
                for node in nodes:
                    group=node[key];warning=node['status']!='assigned'
                    ax.scatter(side,layout[group],s=55,c='white',edgecolors='#bf3333' if warning else 'black',zorder=3)
                    ax.text(side+(-.05 if side==0 else .05),layout[group],f'{"P" if side==0 else "C"}{group}',
                            ha='right' if side==0 else 'left',va='center',fontsize=9)
            for e in result['events']:
                for p in e['parents']:
                    for c in e['children']:ax.plot([0,1],[layouts[0][p],layouts[1][c]],color=colors[e['kind']],lw=1.7)
            ax.set(xlim=(-.3,1.3),ylim=(0,1),title=f't={record["parent_time"]:.2f} → {record["child_time"]:.2f} | distance={gate}/64');ax.axis('off')
            if event==10 and gate==2.5:
                ax.text(.5,.98,'Distance-sensitive; very small score margins',transform=ax.transAxes,
                        ha='center',va='top',fontsize=8,color='#9a531b')
    fig.suptitle('Selected group events with fixed rules: four inspected transitions')
    fig.legend(handles=[Line2D([0],[0],color=v,label=k) for k,v in colors.items()],loc='lower center',bbox_to_anchor=(.5,.055),ncol=3)
    fig.text(.025,.02,'P/C = event-local previous/current groups. Red outline: ambiguous or unmatched. Graph positions are schematic, not spatial trajectories.\n'
             'Two tested search distances; physical ancestry has no independent truth labels here. Group combinations are not confirmed physical shock mergers.',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.94));fig.savefig(output/'selected_event_review.png',dpi=135);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('outputs/two_bubbles_temporal_comparison_n064_n128'))
    parser.add_argument('--controls',type=Path,default=Path('outputs/event_tracking_controls_solver_validation'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();run(args.directory,args.controls,args.output_dir)
