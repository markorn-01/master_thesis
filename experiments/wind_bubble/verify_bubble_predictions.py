"""Read back observational prediction audit and independently check selected links."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from experiments.wind_bubble.verify_resolution_associations import enumerate_support


def verify(output):
    meta=json.loads((output/'metadata.json').read_text())
    for p,h in {**meta['input_sha256'],**meta['source_sha256']}.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    with gzip.open(output/'results.json.gz','rt') as f:records=json.load(f)
    summary_rows=list(csv.DictReader((output/'summary.csv').open()))
    assert len(summary_rows)==len(records)
    for record,row in zip(records,summary_rows):
        for key,value in row.items():assert value==('' if record['summary'][key] is None else str(record['summary'][key]))
    group_rows=list(csv.DictReader((output/'group_diagnostics.csv').open()))
    groups=[g for r in records for g in r['group_diagnostics']]
    assert len(groups)==len(group_rows)
    for group,row in zip(groups,group_rows):
        for key,value in row.items():assert value==('' if group[key] is None else str(group[key]))
    origin=Path(meta['temporal_comparison_directory'])
    with gzip.open(origin/'support_records.json.gz','rt') as f:original=json.load(f)
    root=Path(meta['target_directory']);partitions=Path(meta['partition_directory']);catalogs=[];labels=[]
    for event in range(17):
        with np.load(root/'online'/f'surfaces_{event:04d}.npz') as a:catalogs.append({k:a[k].copy() for k in a.files})
        with np.load(partitions/f'groups_n128_event{event:04d}.npz') as a:labels.append({k:a[k].copy() for k in a.files})
    baseline=0;bootstrap=0;predictions=0;selected=0;brute=0;changed=0
    assert len(records)==180
    for index,record in enumerate(records):
        c=record['config'];middle=c['middle_event'];mode=c['mode'];key=f'{mode}_a60_labels';gate=c['gate_code_units']
        a,b,d=[catalogs[i] for i in [middle-1,middle,middle+1]];l0,l1,l2=[labels[i][key] for i in [middle-1,middle,middle+1]]
        reference=lambda event:next(x['result'] for x in original if x['config']['num_cells']==128 and x['config']['mode']==mode
                                   and x['config']['parent_event']==event and x['config']['gate_reference_cells']==c['gate_reference_cells'] and x['config']['segmentation_angle']==60)
        ref0=reference(middle-1)
        for field in ['parents','children','links']:assert record['bootstrap'][field]==ref0[field]
        if record['method']=='geometry':
            for field in ['parents','children','links']:assert record['result'][field]==reference(middle)[field]
            baseline+=1;bootstrap+=1
        accepted=[x for x in ref0['links'] if x['accepted']];predicted=np.array(b['positions'],dtype=float,copy=True);dt0=c['times'][1]-c['times'][0];dt1=c['times'][2]-c['times'][1]
        for e in record['estimates']:
            group=e['group'];mask=l1==group;incoming=[l for l in accepted if l['child_group']==group]
            initialized=len(incoming)==1 and not incoming[0]['redundant_support'] and len([l for l in accepted if l['parent_group']==incoming[0]['parent_group']])==1
            assert e['initialized']==initialized
            if not initialized:assert e['velocity'] is None;continue
            parent=incoming[0]['parent_group'];prior=np.asarray(a['positions'][l0==parent],float);current=np.asarray(b['positions'][mask],float)
            center=current.mean(0);velocity=(center-prior.mean(0))/dt0
            np.testing.assert_allclose(e['velocity'],velocity,rtol=1e-12,atol=1e-12)
            if record['method']=='geometry':continue
            predicted[mask]=current+velocity*dt1
            if record['method']=='expansion':
                r0=float(np.sqrt(np.var(prior,axis=0).sum()));r1=float(np.sqrt(np.var(current,axis=0).sum()))
                np.testing.assert_allclose([e['extent_0'],e['extent_1']],[r0,r1],atol=1e-12,rtol=1e-12)
                if min(r0,r1)>1e-12 and r1+(r1-r0)/dt0*dt1>1e-12:
                    scale=(r1+(r1-r0)/dt0*dt1)/r1
                    np.testing.assert_allclose(e['scale'],scale,atol=1e-12,rtol=1e-12)
                    predicted[mask]=center+velocity*dt1+scale*(current-center)
        points=predicted[l1>=0];future=d['positions'][l2>=0]
        distances=cKDTree(future).query(points)[0] if len(future) and len(points) else np.full(len(points),np.inf)
        medians=[]
        for group in record['group_diagnostics']:
            values=distances[l1[l1>=0]==group['group']]
            if len(future):
                q=np.percentile(values,[50,84,95])/float(b['grid_spacing'])
                np.testing.assert_allclose(q,[group['nearest_median_cells'],group['nearest_p84_cells'],group['nearest_p95_cells']],atol=1e-10,rtol=1e-10)
                medians.append(q[0])
            else:assert group['nearest_median_cells'] is None
        reverse=cKDTree(points).query(future)[0] if len(points) and len(future) else np.full(len(future),np.inf)
        for name,values in [('forward_nearest',distances),('reverse_nearest',reverse)]:
            saved=record['summary'][name];finite=values[np.isfinite(values)]
            assert saved['samples']==len(values) and saved['finite_samples']==len(finite)
            if len(finite):np.testing.assert_allclose([saved[k] for k in ['median','p84','p95']],np.percentile(finite,[50,84,95]),atol=1e-12,rtol=1e-12)
            else:assert saved['median'] is None
        summary=record['summary']
        if medians:np.testing.assert_allclose(summary['median_group_nearest_distance_cells'],np.median(medians),atol=1e-10,rtol=1e-10)
        else:assert summary['median_group_nearest_distance_cells'] is None
        predictions+=1
        if middle in [7,12,14]:
            parents,choices,covered=enumerate_support(dict(positions=predicted,normals=b['normals']),d,l1,l2,gate)
            for p in record['result']['parents']:assert parents[p['parent_group']]==(p['best_child_group'],p['parent_coverage'],p['runner_up_parent_coverage'],p['status'])
            for child in record['result']['children']:
                group=child['child_group'];eligible=[p for p,target in choices.items() if target==group]
                union=set().union(*(covered[p] for p in eligible));coverage=len(union)/int((l2==group).sum())
                assert coverage==child['union_child_coverage'] and bool(eligible and coverage>=.5)==child['accepted']
                for link in [l for l in record['result']['links'] if l['child_group']==group]:
                    others=set().union(*(covered[p] for p in eligible if p!=link['parent_group']))
                    assert len(covered[link['parent_group']]-others)==link['exclusive_child_support_samples']
            if len(points) and len(future):
                for i in np.unique(np.linspace(0,len(points)-1,5,dtype=int)):
                    exact=np.sqrt(np.sum((future-points[i])**2,axis=1)).min()
                    np.testing.assert_allclose(exact,distances[i],atol=1e-12,rtol=1e-12);brute+=1
            selected+=1
        if index%3==0:
            group=records[index:index+3]
            pairs=lambda x:sorted((l['parent_group'],l['child_group']) for l in x['result']['links'] if l['accepted'])
            changed+=pairs(group[0])!=pairs(group[1]) or pairs(group[0])!=pairs(group[2])
    report=dict(passed=True,geometry_baselines_reproduced=baseline,bootstrap_records_reproduced=bootstrap,
                prediction_and_distance_summaries_checked=predictions,independent_geometric_comparisons=selected,
                brute_force_nearest_distance_checks=brute,configurations_with_changed_accepted_links=changed,
                physical_identity_validated=False)
    (output/'readback.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);verify(p.parse_args().output)
