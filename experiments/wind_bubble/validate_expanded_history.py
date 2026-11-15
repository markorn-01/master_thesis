"""Bounded whole-sequence comparison of frozen three- and experimental five-partner events."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.expanded_event_tracking import expanded_tracking
from experiments.astrolink.event_history import assemble_history
from experiments.wind_bubble.apply_event_tracking import check_candidates,check_selection_dp,event_key
from experiments.wind_bubble.compare_two_bubble_resolutions import load_checked,csv_write
from experiments.wind_bubble.compare_event_history_resolutions import group_statistics


def keys(result):return {event_key(e) for e in result['events']}


def describe_event(event,result):
    return dict(kind=event['kind'],parents=event['parents'],children=event['children'],score=event['score'],
        exclusion_gap=result['selection']['exclusion_gaps'][str(event['candidate_index'])],
        minimum_node_coverage=min(c['coverage'] for c in event['node_coverages']),
        minimum_exclusive_fraction=min(event['exclusive_central_fractions']) if event['exclusive_central_fractions'] else None)


def run(root,history,controls,output):
    hashes={}
    def record(path):hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for directory in [history,controls]:
        path=directory/'verification.json';record(path);v=json.loads(path.read_text())
        assert v['passed']
        for section in ['input_sha256','source_sha256']:
            for name,digest in v[section].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    meta,data,input_hashes,readback=load_checked(root);hashes.update(input_hashes)
    assert len(data)==21 and meta['num_cells']==128
    archive=history/'tracking_results.json.gz';record(archive)
    with gzip.open(archive,'rt') as f:old=json.load(f)
    assert len(old)==40
    labels=[]
    for index,event in enumerate(data):
        path=history/f'groups_{index:04d}.npz';record(path)
        with np.load(path) as p:labels.append(p['labels'].copy())
        assert len(labels[-1])==len(event['surface']['positions'])
    output.mkdir(parents=True,exist_ok=False);records=[];rows=[];changes=[];checks=[]
    for index in range(20):
        a,b=[e['surface'] for e in data[index:index+2]]
        for gate in [1.5,2.5]:
            baseline=next(r['result'] for r in old if r['parent_event']==index and r['gate_reference_cells']==gate)
            result=expanded_tracking(a['positions'],a['normals'],labels[index],b['positions'],b['normals'],labels[index+1],gate/64)
            for k,v in baseline['parameters'].items():
                if k!='max_partners':assert result['parameters'][k]==v
            candidate_map={event_key(e):e for e in result['candidates']}
            for event in baseline['candidates']:assert candidate_map[event_key(event)]==event
            check=check_candidates(a,b,labels[index],labels[index+1],result);check_selection_dp(result)
            checks.append(dict(parent_event=index,gate_reference_cells=gate,independent_dp_passed=True,**check))
            records.append(dict(parent_event=index,child_event=index+1,gate_reference_cells=gate,result=result))
            added=keys(result)-keys(baseline);removed=keys(baseline)-keys(result)
            extra=len(result['candidates'])-len(baseline['candidates'])
            if not extra:
                assert result['candidates']==baseline['candidates'] and result['selection']==baseline['selection']
            rows.append(dict(parent_event=index,parent_time=meta['actual_times'][index],child_time=meta['actual_times'][index+1],
                gate_reference_cells=gate,baseline_candidates=len(baseline['candidates']),expanded_candidates=len(result['candidates']),
                extra_candidates=extra,added_events=len(added),removed_events=len(removed),events_unchanged=not(added or removed)))
            if added or removed:
                changes.append(dict(parent_event=index,parent_time=meta['actual_times'][index],child_time=meta['actual_times'][index+1],
                    gate_reference_cells=gate,added=[describe_event(e,result) for e in result['events'] if event_key(e) in added],
                    removed=[describe_event(e,baseline) for e in baseline['events'] if event_key(e) in removed]))
        print(f'Checked {index:02d}→{index+1:02d}',flush=True)
    times=meta['actual_times'];groups=[list(map(int,np.unique(l[l>=0]))) for l in labels]
    baseline_graph=assemble_history(groups,times,old);graph=assemble_history(groups,times,records)
    stats=[]
    for index,event in enumerate(data):
        nodes=[n for n in graph['nodes'] if n['event']==index]
        for node,row in zip(nodes,group_statistics(event['surface'],labels[index],nodes)):
            mask=labels[index]==node['group'];s=event['surface']
            row.update(upstream_inside_source=int(s['upstream_inside_any_source'][mask].sum()),
                downstream_inside_source=int(s['downstream_inside_any_source'][mask].sum()))
            node.update(row);stats.append(row)
    agreements=[]
    for index in range(20):
        old_sets=[keys(r['result']) for r in old if r['parent_event']==index]
        new_sets=[keys(r['result']) for r in records if r['parent_event']==index]
        agreements.append(dict(parent_event=index,baseline_distances_agree=old_sets[0]==old_sets[1],
            expanded_distances_agree=new_sets[0]==new_sets[1],baseline_shared=len(old_sets[0]&old_sets[1]),expanded_shared=len(new_sets[0]&new_sets[1])))
    with gzip.open(output/'tracking_results.json.gz','wt') as f:json.dump(records,f)
    (output/'graph.json').write_text(json.dumps(graph,indent=2)+'\n')
    (output/'changed_events.json').write_text(json.dumps(changes,indent=2)+'\n')
    csv_write(output/'interval_comparison.csv',rows);csv_write(output/'distance_agreement.csv',agreements)
    csv_write(output/'group_statistics.csv',stats)
    render(baseline_graph,graph,changes,output)
    for name,digest in hashes.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    summary=dict(passed=True,readback=readback,comparisons=40,changed_configurations=len(changes),
        changed_intervals=sorted({c['parent_event'] for c in changes}),
        configurations_with_extra_candidates=sum(r['extra_candidates']>0 for r in rows),
        unchanged_configurations=sum(r['events_unchanged'] for r in rows),independent_checks=checks,
        baseline_distance_agreement=sum(r['baseline_distances_agree'] for r in agreements),
        expanded_distance_agreement=sum(r['expanded_distances_agree'] for r in agreements),
        baseline_event_counts=dict(Counter(e['status']+'_'+e['kind'] for e in baseline_graph['events'])),
        expanded_event_counts=dict(Counter(e['status']+'_'+e['kind'] for e in graph['events'])),
        original_history_modified=False,spatial_groups_changed=False,all_original_candidates_preserved=True,
        physical_false_link_rate_measured=False,physical_ancestry_established=False,
        input_sha256=hashes,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path('experiments/astrolink/expanded_event_tracking.py'),Path('experiments/astrolink/event_tracking.py')]})
    (output/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['changed_configurations','changed_intervals','unchanged_configurations','baseline_distance_agreement','expanded_distance_agreement','baseline_event_counts','expanded_event_counts']},indent=2))


def render(baseline,expanded,changes,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(14,10),sharex=True)
    for ax,graph,title in zip(axes,[baseline,expanded],['Original maximum three partners','Experimental maximum five partners (all intervals)']):
        lookup={n['node_id']:n for n in graph['nodes']}
        for event in graph['events']:
            nodes=[lookup[name] for name in event['parents']+event['children']]
            x=np.mean([lookup[event['parents'][0]]['time'],lookup[event['children'][0]]['time']]);y=np.mean([n['segment_id'] for n in nodes])
            for n in nodes:ax.plot([n['time'],x],[n['segment_id'],y],color='#167f88' if event['kind']=='continuation' else '#a04f8f',
                ls='-' if event['status']=='shared' else '--',lw=1.3)
        ax.scatter([n['time'] for n in graph['nodes']],[n['segment_id'] for n in graph['nodes']],s=20,color='#254960',zorder=3)
        for i in sorted({c['parent_event'] for c in changes}):ax.axvspan(i*.01,(i+1)*.01,color='#d2982e',alpha=.12)
        ax.set(title=title,ylabel='Local continuation segment',ylim=(graph['segment_count']-.5,-.7),yticks=range(graph['segment_count']))
        ax.grid(axis='x',alpha=.2)
    axes[-1].set_xlabel('Simulation time')
    fig.suptitle('Bounded validation across the complete saved unequal-bubble sequence',fontsize=16)
    fig.text(.06,.025,'Solid: shared events; dashed: distance-specific alternatives. Shading: intervals with a changed accepted event at either distance.\n'
        'Same spatial groups, coverage, exclusivity, score and tie tolerance; only maximum event arity changes. No physical-truth labels.',fontsize=10)
    fig.tight_layout(rect=(0,.1,1,.95));fig.savefig(output/'full_sequence_comparison.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/two_bubbles_detection_extension_iwr_n128'))
    parser.add_argument('--history',type=Path,default=Path('outputs/two_bubbles_detection_extension_review_n128'))
    parser.add_argument('--controls',type=Path,default=Path('outputs/five_fragment_arity_review_n128'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/expanded_full_sequence_validation_n128'))
    args=parser.parse_args();run(args.root,args.history,args.controls,args.output_dir)
