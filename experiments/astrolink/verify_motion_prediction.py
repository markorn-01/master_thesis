"""Dense-distance verification of saved bootstrap, prediction and final links."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def dense_support(pa,na,la,pb,nb,lb,gate):
    na=na/np.linalg.norm(na,axis=1)[:,None];nb=nb/np.linalg.norm(nb,axis=1)[:,None]
    adjacency=(np.linalg.norm(pa[:,None,:]-pb[None,:,:],axis=2)<=gate)&((na@nb.T)>=np.cos(np.deg2rad(60)))
    choices={};supports={};parents={};children={};accepted=set();redundant={}
    for parent in np.unique(la[la>=0]):
        indices=np.flatnonzero(la==parent);rank=[]
        for child in np.unique(lb[lb>=0]):
            targets=np.flatnonzero(lb==child);block=adjacency[np.ix_(indices,targets)]
            rank.append((float(block.any(axis=1).mean()),int(child),set(map(int,targets[block.any(axis=0)]))))
        rank.sort(key=lambda x:(-x[0],x[1]));best=rank[0];runner=rank[1][0] if len(rank)>1 else 0.
        status='insufficient_parent_support' if best[0]<.5 else ('ambiguous_target' if best[0]-runner<.25 else 'eligible')
        parents[int(parent)]=(best[1],best[0],runner,status)
        if status=='eligible':choices[int(parent)]=best[1];supports[int(parent)]=best[2]
    for child in np.unique(lb[lb>=0]):
        selected=[p for p,c in choices.items() if c==child];union=set().union(*(supports[p] for p in selected))
        coverage=len(union)/int((lb==child).sum());valid=bool(selected) and coverage>=.5
        children[int(child)]=(coverage,valid)
        for parent in selected:
            others=set().union(*(supports[p] for p in selected if p!=parent))
            redundant[parent,int(child)]=len(supports[parent]-others)==0
            if valid:accepted.add((parent,int(child)))
    return parents,children,accepted,redundant


def verify(directory):
    meta=json.loads((directory/'metadata.json').read_text())
    for p,h in meta['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    records=json.loads((directory/'results.json').read_text());assert len(records)==meta['cases']*6
    observed={};bootstrap_checked=0
    for record in records:
        name=record['case']
        if name not in observed:
            with np.load(directory/f'{name}.npz') as a:data={k:a[k].copy() for k in a.files}
            def args(i):return tuple(data[f'{k}_{i}'] for k in ['positions','normals','labels'])
            _,_,accepted,redundant=dense_support(*args(0),*args(1),meta['bootstrap_gate'])
            expected=data['positions_1'].copy();estimates={}
            for group in np.unique(data['labels_1']):
                incoming=[p for p,c in accepted if c==group];initialized=False;velocity=None;parent=None
                if len(incoming)==1:
                    predecessor=incoming[0];outgoing=[c for p,c in accepted if p==predecessor]
                    if len(outgoing)==1 and not redundant[predecessor,int(group)]:
                        parent=predecessor;initialized=True
                        velocity=(data['positions_1'][data['labels_1']==group].mean(0)-data['positions_0'][data['labels_0']==parent].mean(0))/(data['times'][1]-data['times'][0])
                        expected[data['labels_1']==group]+=velocity*(data['times'][2]-data['times'][1])
                estimates[int(group)]=(initialized,parent,velocity)
            np.testing.assert_allclose(expected,data['predicted_positions'],rtol=0,atol=0)
            observed[name]=(data,estimates);bootstrap_checked+=1
        data,estimates=observed[name]
        for estimate in record['estimates']:
            initialized,parent,velocity=estimates[estimate['group']]
            assert (estimate['initialized'],estimate['predecessor_group'])==(initialized,parent)
            if initialized:np.testing.assert_allclose(estimate['velocity'],velocity,rtol=0,atol=0)
            else:assert estimate['velocity'] is None
        p=data['positions_1'] if record['method']=='geometry' else data['predicted_positions']
        parents,children,accepted,_=dense_support(p,data['normals_1'],data['labels_1'],data['positions_2'],data['normals_2'],data['labels_2'],record['gate'])
        for parent in record['parents']:
            assert parents[parent['parent_group']]==(parent['best_child_group'],parent['parent_coverage'],parent['runner_up_parent_coverage'],parent['status'])
        for child in record['children']:assert children[child['child_group']]==(child['union_child_coverage'],child['accepted'])
        truth={tuple(x) for x in record['expected_links']}
        for key,value in [('accepted_links',accepted),('false_links',accepted-truth),('missed_links',truth-accepted)]:
            assert sorted(value)==[tuple(x) for x in record[key]]
    report=dict(passed=True,independent_dense_bootstrap_checks=bootstrap_checked,independent_final_comparisons=len(records),
                centroid_predictions_match=True,known_failures_preserved=True,physical_tracker_validated=False)
    (directory/'readback.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);verify(p.parse_args().directory)
