"""Provisional group histories from agreement of unique adjacent-event matches."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.continuation_segments import stable_pairs,assign_segments
from experiments.astrolink.controlled_tracking_cases import tracking_cases
from experiments.astrolink.check_controlled_tracking import evaluate
from experiments.wind_bubble.plot_resolution_links import choose_record
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write


def run(directory,output):
    meta_path=directory/'metadata.json';archive=directory/'support_records.json.gz'
    meta=json.loads(meta_path.read_text());root=Path(meta['target_directory']);partitions=Path(meta['comparison_directory'])
    run_path=root/'metadata.json';run_meta=json.loads(run_path.read_text());times=run_meta['actual_times']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [meta_path,archive,run_path]}
    with gzip.open(archive,'rt') as f:all_records=json.load(f)
    output.mkdir(parents=True,exist_ok=False);previous={};next_id=0;history=[];links=[];controls=[]
    for case in tracking_cases():
        results=[evaluate(case,g) for g in [.1,.2,.35]];pairs=stable_pairs(results)
        truth={tuple(x) for x in case['expected_links']}
        for result in results:
            accepted={(l['parent_group'],l['child_group']) for l in result['links'] if l['accepted']}
            assert set(pairs)<=accepted
        controls.append(dict(case=case['name'],continued_pairs=pairs,
            stipulated_truth=case['expected_links'],false_continuations=sorted(set(pairs)-truth),
            missed_stipulated_links=sorted(truth-set(pairs))))
    for event,time in enumerate(times):
        source=root/'online'/f'surfaces_{event:04d}.npz';partition=partitions/f'groups_n128_event{event:04d}.npz'
        for p in [source,partition]:hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(source) as a:data={k:a[k].copy() for k in ['positions','mach','mach_valid','time']}
        with np.load(partition) as a:labels=a['native_cells_a60_labels'].copy()
        np.testing.assert_allclose(float(data['time']),time,atol=1e-7,rtol=0)
        groups=sorted(map(int,np.unique(labels[labels>=0])))
        results=[];pairs=[]
        if event:
            records=[choose_record(all_records,128,'native_cells',event-1,g) for g in [1.5,2.5]]
            results=[r['result'] for r in records];pairs=stable_pairs(results)
            for result in results:
                assert set(previous)=={p['parent_group'] for p in result['parents']}
                assert set(groups)=={c['child_group'] for c in result['children']}
            links.extend(dict(parent_event=event-1,child_event=event,parent_group=p,child_group=c,
                              segment_id=previous[p],both_distances_agree=True) for p,c in pairs)
        current,assigned,next_id=assign_segments(previous,groups,pairs,next_id)
        for row in assigned:
            mask=labels==row['group'];valid=mask&data['mach_valid']&np.isfinite(data['mach'])&(data['mach']>0)
            values=data['mach'][valid]
            history.append(dict(event=event,time=float(data['time']),**row,
                                start_reason='continued' if row['inherited'] else 'first_observation' if not previous else 'unique_continuation_not_established',
                                surface_samples=int(mask.sum()),valid_mach_samples=len(values),
                                median_mach=float(np.median(values)) if len(values) else float('nan')))
        previous=current
    segments=[]
    for segment in sorted({r['segment_id'] for r in history}):
        rows=[r for r in history if r['segment_id']==segment];events=[r['event'] for r in rows]
        assert events==list(range(events[0],events[-1]+1))
        segments.append(dict(segment_id=segment,first_event=events[0],last_event=events[-1],observations=len(rows),
                             start_time=rows[0]['time'],end_time=rows[-1]['time'],
                             end_reason='end_of_saved_sequence' if events[-1]==len(times)-1 else 'unique_continuation_not_established'))
    csv_write(output/'group_histories.csv',history);csv_write(output/'continuation_links.csv',links);csv_write(output/'segments.csv',segments)
    (output/'controlled_checks.json').write_text(json.dumps(controls,indent=2)+'\n')
    # Independent CSV readback of statistics and adjacency; catalog row IDs are never tracked.
    reread=list(csv.DictReader((output/'group_histories.csv').open()))
    assert len(reread)==len(history)
    for event in range(len(times)):
        with np.load(root/'online'/f'surfaces_{event:04d}.npz') as a,np.load(partitions/f'groups_n128_event{event:04d}.npz') as g:
            labels=g['native_cells_a60_labels']
            for row in [r for r in reread if int(r['event'])==event]:
                selected=np.flatnonzero(labels==int(row['group']))
                assert len(selected)==int(row['surface_samples'])
                v=a['mach'][selected];good=a['mach_valid'][selected]&np.isfinite(v)&(v>0)
                assert int(good.sum())==int(row['valid_mach_samples'])
                expected=float(np.percentile(v[good],50)) if good.any() else np.nan
                np.testing.assert_allclose(float(row['median_mach']),expected,atol=1e-12,rtol=1e-12,equal_nan=True)
    for link in links:
        a=next(r for r in history if r['event']==link['parent_event'] and r['group']==link['parent_group'])
        b=next(r for r in history if r['event']==link['child_event'] and r['group']==link['child_group'])
        assert a['segment_id']==b['segment_id']==link['segment_id'] and b['inherited']
        for gate in [1.5,2.5]:
            result=choose_record(all_records,128,'native_cells',link['parent_event'],gate)['result']
            matched=next(l for l in result['links'] if l['parent_group']==link['parent_group'] and l['child_group']==link['child_group'])
            child=next(c for c in result['children'] if c['child_group']==link['child_group'])
            assert matched['accepted'] and not matched['redundant_support'] and child['parent_groups']==[link['parent_group']]
    selected=render(history,segments,output)
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    sources=[Path(__file__),Path('experiments/astrolink/continuation_segments.py'),Path('experiments/astrolink/continuation_status.py')]
    report=dict(events=len(times),group_observations=len(history),provisional_segments=len(segments),
        continuation_links=len(links),segments_with_multiple_observations=sum(s['observations']>1 for s in segments),
        plotted_segments=selected,controls=len(controls),
        input_sha256=hashes,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        group_cutoff=20,grouping_normal_angle_degrees=60,association_distances=[1.5/64,2.5/64],
        original_links_changed=False,physical_identity_established=False,merger_ancestry_assigned=False,
        physical_diagnostics_changed=False,gap_bridging_added=False)
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('sha256')},indent=2))


def render(history,segments,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # A small, deterministic view; all short/unresolved segments remain in CSV.
    selected=[s['segment_id'] for s in sorted(segments,key=lambda s:(-s['observations'],s['segment_id']))[:3]]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for index,segment in enumerate(selected):
        rows=[r for r in history if r['segment_id']==segment];time=[r['time'] for r in rows]
        label=f'S{segment:03d} ({len(rows)} observations)'
        for ax,field in zip(axes,['surface_samples','median_mach']):
            ax.plot(time,[r[field] for r in rows],marker=['o','o','x'][index],
                    linestyle=['-','-',':'][index],markerfacecolor='none' if index==1 else None,
                    markersize=5,label=label)
    for ax,title,ylabel in zip(axes,['Detected group size','Median Mach of valid group samples'],['Surface sample count','Median Mach']):
        ax.set(xlabel='Simulation time',ylabel=ylabel,title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Three longest provisional continuation segments: agreement at both tested distances')
    fig.text(.03,.025,'Segments stop at uncertain transitions; S numbers are provisional history labels, not physical shock IDs.\n'
             'The two early histories overlap in this symmetric example. Counts are not areas. All groups and short segments remain in the saved table.',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.93));fig.savefig(output/'provisional_histories.png',dpi=140);plt.close(fig)
    return selected


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('outputs/two_bubbles_temporal_comparison_n064_n128'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();run(args.directory,args.output_dir)
