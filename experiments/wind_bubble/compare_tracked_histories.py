"""Compare two tracking modes on the same saved catalogs without physical truth claims."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.event_history import assemble_history
from experiments.wind_bubble.track_shock_history import graph_signature
from experiments.wind_bubble.apply_event_tracking import event_key
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write


def run(baseline,candidate,output):
    hashes={};bundles=[]
    for directory in [baseline,candidate]:
        v=json.loads((directory/'verification.json').read_text());assert v['passed']
        for section in ['input_sha256','source_sha256']:
            for name,digest in v[section].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
        for name in ['metadata.json','verification.json','graph.json','tracking_results.json.gz','groups.csv']:
            p=directory/name;hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        meta=json.loads((directory/'metadata.json').read_text());assert meta['status']=='complete'
        with gzip.open(directory/'tracking_results.json.gz','rt') as f:records=json.load(f)
        graph=json.loads((directory/'graph.json').read_text());counts=list(csv.DictReader((directory/'groups.csv').open()))
        groups=[]
        for index in range(meta['events']):
            path=directory/f'groups_{index:04d}.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as p:groups.append({key:p[key].copy() for key in ['labels','raw_labels','component_sizes']})
        assembled=assemble_history([list(map(int,np.unique(g['labels'][g['labels']>=0]))) for g in groups],
            [float(row['time']) for row in counts],records)
        assert graph_signature(assembled)==graph_signature(graph)
        bundles.append(dict(meta=meta,records=records,graph=graph,counts=counts,groups=groups,verification=v))
    a,b=bundles
    assert Path(a['meta']['input_directory']).resolve()==Path(b['meta']['input_directory']).resolve()
    assert a['meta']['events']==b['meta']['events']
    assert a['meta']['settings']['max_partners']==3 and b['meta']['settings']['max_partners']==5
    for key,value in a['meta']['settings'].items():
        if key!='max_partners':assert value==b['meta']['settings'][key]
    assert a['counts']==b['counts']
    for ga,gb in zip(a['groups'],b['groups']):
        for key in ga:np.testing.assert_array_equal(ga[key],gb[key])
    lookup_a={n['node_id']:n for n in a['graph']['nodes']};lookup_b={n['node_id']:n for n in b['graph']['nodes']}
    assert set(lookup_a)==set(lookup_b)
    for key,node in lookup_a.items():
        for field in ['surface_samples','valid_mach_samples','median_mach','upstream_inside_source','downstream_inside_source']:
            np.testing.assert_allclose(node[field],lookup_b[key][field],rtol=0,atol=0,equal_nan=True)
    rows=[];changes=[];agreements=[]
    for index in range(a['meta']['events']-1):
        sets=[[],[]]
        gates=sorted({r['gate_reference_cells'] for r in a['records'] if r['parent_event']==index})
        assert len(gates)==2 and gates==sorted({r['gate_reference_cells'] for r in b['records'] if r['parent_event']==index})
        for gate in gates:
            old=next(r['result'] for r in a['records'] if r['parent_event']==index and r['gate_reference_cells']==gate)
            new=next(r['result'] for r in b['records'] if r['parent_event']==index and r['gate_reference_cells']==gate)
            new_candidates={event_key(e):e for e in new['candidates']}
            for e in old['candidates']:assert new_candidates[event_key(e)]==e
            keys_old={event_key(e) for e in old['events']};keys_new={event_key(e) for e in new['events']}
            sets[0].append(keys_old);sets[1].append(keys_new)
            extra=len(new['candidates'])-len(old['candidates'])
            if extra==0:assert old['selection']==new['selection']
            rows.append(dict(parent_event=index,parent_time=float(a['counts'][index]['time']),child_time=float(a['counts'][index+1]['time']),
                gate_reference_cells=gate,extra_candidates=extra,added_events=len(keys_new-keys_old),removed_events=len(keys_old-keys_new)))
            if keys_old!=keys_new:
                def describe(event,result):return dict(kind=event['kind'],parents=event['parents'],children=event['children'],
                    child_coverages=[n['coverage'] for n in event['node_coverages'] if n['side']=='child'],
                    exclusive_fractions=event['exclusive_central_fractions'],score=event['score'],
                    exclusion_gap=result['selection']['exclusion_gaps'][str(event['candidate_index'])])
                changes.append(dict(parent_event=index,parent_time=float(a['counts'][index]['time']),child_time=float(a['counts'][index+1]['time']),
                    gate_reference_cells=gate,added=[describe(e,new) for e in new['events'] if event_key(e) not in keys_old],
                    removed=[describe(e,old) for e in old['events'] if event_key(e) not in keys_new]))
        agreements.append(dict(parent_event=index,baseline_agreement=sets[0][0]==sets[0][1],candidate_agreement=sets[1][0]==sets[1][1]))
    # Descriptors for reviewing changed events, never inputs to association.
    for change in changes:
        index=change['parent_event'];raw=Path(a['meta']['input_directory'])/'online'
        paths=[raw/f'surfaces_{i:04d}.npz' for i in [index,index+1]]
        for path in paths:hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(paths[0]) as parent,np.load(paths[1]) as child:
            for event in change['added']:
                descriptors={}
                for side,catalog,event_index,group_ids in [('parents',parent,index,event['parents']),('children',child,index+1,event['children'])]:
                    descriptors[side]=[dict(group=g,nearest_source_indices=np.unique(catalog['source_distances'][a['groups'][event_index]['labels']==g].argmin(axis=1)).tolist()) for g in group_ids]
                event['geometric_nearest_sources']=descriptors
                event['source_descriptors_used_for_matching']=False
    output.mkdir(parents=True,exist_ok=False)
    csv_write(output/'interval_comparison.csv',rows);csv_write(output/'distance_agreement.csv',agreements)
    (output/'changed_events.json').write_text(json.dumps(changes,indent=2)+'\n')
    plot(a['graph'],b['graph'],changes,output)
    for name,digest in hashes.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    result=dict(passed=True,events=a['meta']['events'],compared_configurations=len(rows),
        unchanged_configurations=sum(r['added_events']==0 and r['removed_events']==0 for r in rows),
        changed_intervals=sorted({r['parent_event'] for r in changes}),
        baseline_distance_agreement=sum(r['baseline_agreement'] for r in agreements),
        candidate_distance_agreement=sum(r['candidate_agreement'] for r in agreements),
        baseline_event_counts=dict(Counter(e['status']+'_'+e['kind'] for e in a['graph']['events'])),
        candidate_event_counts=dict(Counter(e['status']+'_'+e['kind'] for e in b['graph']['events'])),
        independent_checks_verified=sum(len(bundle['verification']['independent_checks']) for bundle in bundles),
        spatial_partitions_identical=True,group_measurements_identical=True,original_candidates_preserved=True,
        physical_false_link_rate_measured=False,input_sha256=hashes,
        source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ['input_sha256','source_sha256']},indent=2))


def plot(old,new,changes,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(13,10),sharex=True)
    intervals={(r['parent_time'],r['child_time']) for r in changes}
    for ax,graph,title in zip(axes,[old,new],['Three-partner baseline','Five-partner experimental option']):
        lookup={n['node_id']:n for n in graph['nodes']}
        for e in graph['events']:
            nodes=[lookup[n] for n in e['parents']+e['children']]
            time=np.mean([lookup[e['parents'][0]]['time'],lookup[e['children'][0]]['time']]);level=np.mean([n['segment_id'] for n in nodes])
            for n in nodes:ax.plot([n['time'],time],[n['segment_id'],level],c='#167f88' if e['kind']=='continuation' else '#9a4f91',ls='-' if e['status']=='shared' else '--',lw=1.2)
        ax.scatter([n['time'] for n in graph['nodes']],[n['segment_id'] for n in graph['nodes']],c='#23485e',s=20,zorder=3)
        for start,end in intervals:ax.axvspan(start,end,color='#d4a032',alpha=.13)
        ax.set(title=title,ylabel='Local continuation segment',ylim=(graph['segment_count']-.5,-.7),yticks=range(graph['segment_count']));ax.grid(axis='x',alpha=.2)
    axes[-1].set_xlabel('Simulation time');fig.suptitle('Tracking-mode comparison on identical saved catalogs',fontsize=16)
    fig.text(.06,.025,'Shaded: changed interval. Solid: shared choices at both distances; dashed: distance-specific alternatives.\n'
        'Same spatial groups and measurements. Geometric association and distance agreement do not establish physical ancestry.',fontsize=10)
    fig.tight_layout(rect=(0,.1,1,.95));fig.savefig(output/'history_comparison.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();run(args.baseline,args.candidate,args.output_dir)
