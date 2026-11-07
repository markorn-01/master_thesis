"""Independent dense-distance readback of the small synthetic catalogs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def verify(directory):
    meta=json.loads((directory/'metadata.json').read_text())
    for p,h in meta['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    records=json.loads((directory/'results.json').read_text());summary=list(csv.DictReader((directory/'summary.csv').open()))
    assert len(records)==len(summary)==27;checked=0
    for record,row in zip(records,summary):
        assert record['case']==row['case'] and record['gate']==float(row['gate'])
        with np.load(directory/f"{record['case']}.npz") as a:
            pa,pb=a['positions_a'],a['positions_b'];na,nb=a['normals_a'],a['normals_b'];la,lb=a['labels_a'],a['labels_b']
        na=na/np.linalg.norm(na,axis=1)[:,None];nb=nb/np.linalg.norm(nb,axis=1)[:,None]
        adjacency=(np.linalg.norm(pa[:,None,:]-pb[None,:,:],axis=2)<=record['gate'])&((na@nb.T)>=np.cos(np.deg2rad(60)))
        choices={};support={}
        for parent in record['parents']:
            group=parent['parent_group'];idx=np.flatnonzero(la==group);rank=[]
            for child in np.unique(lb):
                target=np.flatnonzero(lb==child);block=adjacency[np.ix_(idx,target)]
                rank.append((float(block.any(axis=1).mean()),int(child),set(map(int,target[block.any(axis=0)]))))
            rank.sort(key=lambda x:(-x[0],x[1]));best=rank[0];runner=rank[1][0] if len(rank)>1 else 0.
            status='insufficient_parent_support' if best[0]<.5 else ('ambiguous_target' if best[0]-runner<.25 else 'eligible')
            assert (best[1],best[0],runner,status)==(parent['best_child_group'],parent['parent_coverage'],parent['runner_up_parent_coverage'],parent['status'])
            if status=='eligible':choices[group]=best[1];support[group]=best[2]
        accepted=set()
        for child in record['children']:
            group=child['child_group'];parents=[p for p,c in choices.items() if c==group]
            union=set().union(*(support[p] for p in parents));coverage=len(union)/int((lb==group).sum())
            valid=bool(parents) and coverage>=.5
            assert coverage==child['union_child_coverage'] and valid==child['accepted']
            for parent in parents:
                if valid:accepted.add((parent,group))
                link=next(l for l in record['links'] if l['parent_group']==parent and l['child_group']==group)
                other=set().union(*(support[p] for p in parents if p!=parent))
                assert len(support[parent]-other)==link['exclusive_child_support_samples']
        truth={tuple(x) for x in record['expected_links']}
        for key,pairs in [('accepted_links',accepted),('correct_links',accepted&truth),('false_links',accepted-truth),('missed_links',truth-accepted)]:
            assert sorted(pairs)==[tuple(x) for x in record[key]]
            if key!='accepted_links':assert int(row[key])==len(pairs)
        checked+=1
    report=dict(passed=True,dense_geometry_comparisons=checked,cases=9,known_failures_preserved=True,
                correctness_of_calculation_verified=True,physical_tracker_validated=False)
    (directory/'readback.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);verify(p.parse_args().directory)
