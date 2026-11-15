"""One entry point for saved multi-source catalogs: baseline or opt-in expanded tracking."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.surface_history import HistorySettings,analyze_surface_history
from experiments.wind_bubble.compare_two_bubble_resolutions import load_checked,csv_write
from experiments.wind_bubble.apply_event_tracking import check_candidates,check_selection_dp,event_key
from experiments.wind_bubble.compare_event_history_resolutions import group_statistics
from experiments.wind_bubble.plot_tracked_history import plot_overview


def graph_signature(graph):
    """Ignore only derived statistic columns and generated hyperedge IDs."""
    nodes=sorted((n['node_id'],n['event'],n['time'],n['group'],n['segment_id'],n['previous_group'],n['inherited']) for n in graph['nodes'])
    events=sorted((e['parent_event'],e['child_event'],e['kind'],tuple(e['parents']),tuple(e['children']),e['status'],tuple(e['selected_at_gates'])) for e in graph['events'])
    return nodes,events,graph['segment_count']


def run(root,output,max_partners=3,compare_with=None):
    settings=HistorySettings(max_partners=max_partners)
    if output.exists():raise ValueError('Use a new output directory; existing results are never overwritten.')
    meta,data,hashes,readback=load_checked(root)
    if meta.get('status')!='complete':raise ValueError('Require a completed simulation.')
    output.mkdir(parents=True)
    manifest=dict(status='running',input_directory=str(root),settings=settings.__dict__,
        mode='experimental_five_partner' if max_partners==5 else 'baseline_three_partner',
        input_num_cells=meta['num_cells'],events=len(data),analysis_during_simulation=False,
        scope='Offline geometric tracking of existing catalogs, not physical source ownership.')
    path=output/'metadata.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    try:
        catalogs=[e['surface'] for e in data]
        def progress(phase,index):
            if phase=='track':print(f'Tracked interval {index}→{index+1}',flush=True)
        result=analyze_surface_history(catalogs,meta['actual_times'],settings,progress=progress)
        checks=[];counts=[];statistics=[]
        for index,(event,partition) in enumerate(zip(data,result['partitions'])):
            np.savez_compressed(output/f'groups_{index:04d}.npz',labels=partition['labels'],
                raw_labels=partition['raw_labels'],component_sizes=partition['component_sizes'])
            labels=partition['labels'];s=event['surface'];nodes=[n for n in result['graph']['nodes'] if n['event']==index]
            for node,row in zip(nodes,group_statistics(s,labels,nodes)):
                mask=labels==node['group']
                row.update(upstream_inside_source=int(s['upstream_inside_any_source'][mask].sum()),
                    downstream_inside_source=int(s['downstream_inside_any_source'][mask].sum()))
                node.update(row);statistics.append(row)
            counts.append(dict(event=index,time=meta['actual_times'][index],retained_groups=len(nodes),
                detected_samples=len(labels),retained_samples=int((labels>=0).sum()),unretained_samples=int((labels<0).sum())))
        for record in result['records']:
            index=record['parent_event'];a,b=catalogs[index:index+2]
            check=check_candidates(a,b,result['partitions'][index]['labels'],result['partitions'][index+1]['labels'],record['result'])
            check_selection_dp(record['result'])
            checks.append(dict(parent_event=index,gate_reference_cells=record['gate_reference_cells'],independent_dp_passed=True,**check))
        agreement=[]
        for index in range(len(data)-1):
            sets=[{event_key(e) for e in r['result']['events']} for r in result['records'] if r['parent_event']==index]
            agreement.append(dict(parent_event=index,event_sets_identical=sets[0]==sets[1],shared_events=len(sets[0]&sets[1])))
        comparison=None
        if compare_with is not None:
            graph_path=compare_with/'graph.json' if compare_with.is_dir() else compare_with
            expected=json.loads(graph_path.read_text());hashes[str(graph_path)]=hashlib.sha256(graph_path.read_bytes()).hexdigest()
            if graph_signature(result['graph'])!=graph_signature(expected):raise AssertionError('History differs from comparison graph.')
            if compare_with.is_dir():
                for index,partition in enumerate(result['partitions']):
                    group_path=compare_with/f'groups_{index:04d}.npz'
                    if group_path.exists():
                        with np.load(group_path) as old:
                            for key in ['labels','raw_labels','component_sizes']:
                                if key in old:np.testing.assert_array_equal(partition[key],old[key])
                        hashes[str(group_path)]=hashlib.sha256(group_path.read_bytes()).hexdigest()
            comparison=dict(passed=True,graph=str(graph_path),all_nodes_segments_and_event_choices_match=True)
        with gzip.open(output/'tracking_results.json.gz','wt') as stream:json.dump(result['records'],stream)
        (output/'graph.json').write_text(json.dumps(result['graph'],indent=2)+'\n')
        csv_write(output/'groups.csv',counts)
        if statistics:csv_write(output/'group_statistics.csv',statistics)
        csv_write(output/'distance_agreement.csv',agreement)
        plot_overview(result['graph'],counts,manifest['mode'],output/'history_overview.png')
        for name,digest in hashes.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
        sources=[Path(__file__),Path('experiments/astrolink/surface_history.py'),Path('experiments/wind_bubble/plot_tracked_history.py')]+[
            Path('experiments/astrolink')/(name+'.py') for name in ['shock_connectivity','event_tracking','expanded_event_tracking','event_history']]
        verification=dict(passed=True,mode=manifest['mode'],settings=result['settings'],readback=readback,
            events=len(data),independent_checks=checks,comparison=comparison,
            identical_distance_intervals=sum(r['event_sets_identical'] for r in agreement),
            event_counts=dict(Counter(e['status']+'_'+e['kind'] for e in result['graph']['events'])),
            input_sha256=hashes,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            original_results_modified=False,physical_ancestry_established=False,
            independent_check_scope='Exact group-occupancy DP is limited to 22 total groups per interval.')
        (output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
        manifest.update(status='complete',comparison=comparison)
        print(json.dumps(dict(mode=manifest['mode'],events=len(data),comparison=comparison,
            identical_distance_intervals=verification['identical_distance_intervals'],event_counts=verification['event_counts']),indent=2))
    except Exception as exc:
        manifest.update(status='failed',error=str(exc));raise
    finally:path.write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True,help='Completed run directory containing metadata.json and online catalogs.')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-partners',type=int,choices=[3,5],default=3,help='Default 3 preserves baseline; 5 explicitly selects the validated experimental option.')
    parser.add_argument('--compare-with',type=Path,help='Optional saved graph or history directory for exact structural regression.')
    args=parser.parse_args();run(args.directory,args.output_dir,args.max_partners,args.compare_with)
