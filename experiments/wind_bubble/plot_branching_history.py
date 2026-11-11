"""One reviewed two-parent group transition beside unchanged Mach histories."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.wind_bubble.plot_resolution_links import choose_record
from experiments.wind_bubble.verify_resolution_associations import enumerate_support
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write


def run(histories,directory,output):
    history_path=histories/'group_histories.csv';archive=directory/'support_records.json.gz'
    meta_path=directory/'metadata.json';meta=json.loads(meta_path.read_text())
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [history_path,archive,meta_path]}
    history=list(csv.DictReader(history_path.open()))
    with gzip.open(archive,'rt') as f:records=json.load(f)
    root=Path(meta['target_directory']);partitions=Path(meta['comparison_directory'])
    catalogs=[];labels=[]
    for event in [7,8]:
        source=root/'online'/f'surfaces_{event:04d}.npz';partition=partitions/f'groups_n128_event{event:04d}.npz'
        for path in [source,partition]:hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(source) as a:catalogs.append({k:a[k].copy() for k in ['positions','normals','mach','mach_valid','time']})
        with np.load(partition) as a:labels.append(a['native_cells_a60_labels'].copy())
    nodes={};edges={};evidence=[]
    for gate in [1.5,2.5]:
        record=choose_record(records,128,'native_cells',7,gate);result=record['result']
        parents,choices,covered=enumerate_support(*catalogs,*labels,record['config']['gate_code_units'])
        assert len(result['children'])==1 and len(result['parents'])==2
        child=result['children'][0];target=child['child_group']
        assert child['accepted'] and len(child['parent_groups'])==2
        union=set().union(*(covered[p] for p in child['parent_groups']))
        assert sorted(union)==result['supported_child_rows'][str(target)]
        child_row=next(r for r in history if int(r['event'])==8 and int(r['group'])==target)
        child_segment=int(child_row['segment_id'])
        assert child_row['inherited']=='False'
        for link in result['links']:
            p=link['parent_group'];assert link['accepted'] and not link['redundant_support']
            parent_row=next(r for r in history if int(r['event'])==7 and int(r['group'])==p)
            segment=int(parent_row['segment_id']);assert segment!=child_segment
            others=set().union(*(covered[q] for q in child['parent_groups'] if q!=p))
            exclusive=len(covered[p]-others)
            assert exclusive==link['exclusive_child_support_samples']
            assert choices[p]==target
            assert parents[p][1]==next(x['parent_coverage'] for x in result['parents'] if x['parent_group']==p)
            for row in [parent_row,child_row]:
                sid=int(row['segment_id']);series=[x for x in history if int(x['segment_id'])==sid]
                nodes[sid]=dict(segment_id=sid,first_event=int(series[0]['event']),last_event=int(series[-1]['event']),
                                start_time=float(series[0]['time']),end_time=float(series[-1]['time']))
            key=(segment,child_segment)
            edges[key]=dict(parent_segment=segment,child_segment=child_segment,parent_event=7,child_event=8,
                            relationship='candidate_predecessor',both_distances_agree=True,
                            physical_merger_confirmed=False)
            evidence.append(dict(parent_segment=segment,child_segment=child_segment,parent_group=p,child_group=target,
                                 gate_reference_cells=gate,parent_samples=int(parent_row['surface_samples']),
                                 child_samples=int(child_row['surface_samples']),parent_coverage=parents[p][1],
                                 individual_child_support_samples=len(covered[p]),exclusive_child_support_samples=exclusive,
                                 union_child_coverage=len(union)/child['child_samples']))
    assert len(nodes)==3 and len(edges)==2 and len(evidence)==4
    # Recompute the plotted medians from the original catalogs for the boundary states.
    for event,a,l in zip([7,8],catalogs,labels):
        for row in [r for r in history if int(r['event'])==event]:
            group=int(row['group']);mask=(l==group)&a['mach_valid']&np.isfinite(a['mach'])&(a['mach']>0)
            np.testing.assert_allclose(float(row['median_mach']),float(np.percentile(a['mach'][mask],50)),rtol=1e-12,atol=1e-12)
            assert int(row['surface_samples'])==int((l==group).sum())
    output.mkdir(parents=True,exist_ok=False)
    csv_write(output/'candidate_predecessor_evidence.csv',evidence)
    graph=dict(nodes=list(nodes.values()),edges=list(edges.values()),scope='One reviewed detected-group transition only',
               persistent_physical_identity_established=False)
    (output/'candidate_graph.json').write_text(json.dumps(graph,indent=2)+'\n')
    render(history,graph,evidence,output)
    with (output/'candidate_graph.json').open() as f:assert json.load(f)==graph
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    (output/'verification.json').write_text(json.dumps(dict(nodes=3,candidate_edges=2,distance_evidence_rows=4,
        support_independently_recomputed=True,boundary_mach_medians_recomputed=True,original_histories_changed=False,
        physical_merger_confirmed=False,input_sha256=hashes,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path('experiments/wind_bubble/verify_resolution_associations.py')]}
        ),indent=2)+'\n')
    print('Saved 3 segment nodes, 2 candidate predecessor edges and 4 independently checked distance records.')


def render(history,graph,evidence,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig,(left,right)=plt.subplots(1,2,figsize=(13,5.5),gridspec_kw={'width_ratios':[1,1.25]})
    parents=sorted({e['parent_segment'] for e in graph['edges']});child=graph['edges'][0]['child_segment']
    colors={parents[0]:'#e07820',parents[1]:'#298c43',child:'#237ab5'}
    locations={parents[0]:(.04,.62),parents[1]:(.04,.16),child:(.64,.39)}
    for segment,(x,y) in locations.items():
        node=next(n for n in graph['nodes'] if n['segment_id']==segment)
        event=8 if segment==child else 7
        row=next(r for r in history if int(r['segment_id'])==segment and int(r['event'])==event)
        left.add_patch(FancyBboxPatch((x,y),.30,.23,boxstyle='round,pad=0.012',facecolor='white',edgecolor=colors[segment],linewidth=2))
        left.text(x+.15,y+.115,f'S{segment:03d}\nt = {node["start_time"]:.2f}–{node["end_time"]:.2f}\n{int(row["surface_samples"]):,} samples at {float(row["time"]):.2f}',
                  ha='center',va='center',fontsize=10)
    for parent in parents:
        x,y=locations[parent]
        left.annotate('',xy=(.625,.505),xytext=(x+.32,y+.115),
                      arrowprops=dict(arrowstyle='->',color=colors[parent],linewidth=2,linestyle='--'))
    left.text(.50,.10,'Dashed arrows: candidate predecessor relationships\nBoth search distances support both links.\nNo physical merger is declared.',ha='center',va='top',fontsize=10)
    left.set(xlim=(0,1),ylim=(0,1),title='Detected groups: two become one');left.axis('off')
    for index,segment in enumerate(parents+[child]):
        rows=[r for r in history if int(r['segment_id'])==segment]
        right.plot([float(r['time']) for r in rows],[float(r['median_mach']) for r in rows],
                   color=colors[segment],marker=['o','x','o'][index],linestyle=['-',':','-'][index],
                   markerfacecolor='none' if index==0 else colors[segment],label=f'S{segment:03d}',markersize=5)
    right.axvspan(.07,.08,color='.9',zorder=0)
    right.text(.075,.97,'Grouping\ntransition',transform=right.get_xaxis_transform(),ha='center',va='top',fontsize=9)
    right.set(xlabel='Simulation time',ylabel='Median valid Mach',title='Separate statistical histories; no line across the transition')
    right.legend();right.grid(alpha=.2)
    fig.suptitle('Preserve the connection without forcing one inherited shock identity',fontsize=15)
    fig.text(.025,.025,'S labels identify provisional history segments. Sample counts are detected points, not areas or gas mass.\n'
             'The early Mach curves overlap in this symmetric setup. S002 summarizes the whole later group; its median is not an average of the earlier medians.',fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.92));fig.savefig(output/'branching_history.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--histories',type=Path,default=Path('outputs/two_bubbles_provisional_histories_n128'))
    parser.add_argument('--directory',type=Path,default=Path('outputs/two_bubbles_temporal_comparison_n064_n128'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();run(args.histories,args.directory,args.output_dir)
