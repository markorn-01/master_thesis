"""Read back temporal comparison products and independently enumerate selected support."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from experiments.wind_bubble.compare_resolution_associations import support_summary


def enumerate_support(a,b,la,lb,gate):
    """Query each parent point, independently of the sparse-matrix implementation."""
    pa,na=a['positions'],a['normals'];pb,nb=b['positions'],b['normals']
    na=na/np.linalg.norm(na,axis=1)[:,None];nb=nb/np.linalg.norm(nb,axis=1)[:,None]
    parents=np.unique(la[la>=0]);children=np.unique(lb[lb>=0]);tree=cKDTree(pb)
    covered={}; choices={};records={}
    for parent in parents:
        rows=np.flatnonzero(la==parent); hits={int(child):[set(),set()] for child in children}
        for i,near in zip(rows,tree.query_ball_point(pa[rows],gate)):
            near=np.asarray(near,dtype=int)
            near=near[(lb[near]>=0)&((nb[near]@na[i])>=np.cos(np.deg2rad(60.)))]
            for child in np.unique(lb[near]):
                hits[int(child)][0].add(int(i));hits[int(child)][1].update(map(int,near[lb[near]==child]))
        ranking=sorted([(len(v[0])/len(rows),child,v[1]) for child,v in hits.items()],key=lambda x:(-x[0],x[1]))
        best=ranking[0] if ranking else (0.,-1,set());second=ranking[1][0] if len(ranking)>1 else 0.
        status='insufficient_parent_support' if best[0]<.5 else ('ambiguous_target' if best[0]-second<.25 else 'eligible')
        records[int(parent)]=(best[1],best[0],second,status)
        if status=='eligible':choices[int(parent)]=best[1];covered[int(parent)]=best[2]
    return records,choices,covered


def run(output):
    meta=json.loads((output/'metadata.json').read_text())
    for path,digest in {**meta['input_sha256'],**meta['source_sha256']}.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    with gzip.open(output/'support_records.json.gz','rt') as f:records=json.load(f)
    summary=list(csv.DictReader((output/'summary.csv').open()));assert len(records)==len(summary)==384
    selected=0;parent_checks=0;child_checks=0
    for record,row in zip(records,summary):
        c=record['config'];r=record['result']
        for key,value in c.items():assert str(value)==row[key]
        for key,value in support_summary(r).items():assert int(row[key])==value
        assert c['gate_code_units']==c['gate_reference_cells']/64
        assert c['gate_native_cells']==c['gate_code_units']*c['num_cells']
        for child in r['children']:
            support=r['supported_child_rows'][str(child['child_group'])]
            assert len(support)==len(set(support))==child['union_covered_child_samples']
            assert child['accepted']==(bool(child['parent_groups']) and child['union_child_coverage']>=.5)
        if c['segmentation_angle']!=60 or c['parent_event'] not in [12,13,15]:continue
        root=Path(meta['reference_directory'] if c['num_cells']==64 else meta['target_directory'])
        comparison=Path(meta['comparison_directory'])
        with np.load(root/'online'/f"surfaces_{c['parent_event']:04d}.npz") as a, np.load(root/'online'/f"surfaces_{c['child_event']:04d}.npz") as b, \
             np.load(comparison/f"groups_n{c['num_cells']:03d}_event{c['parent_event']:04d}.npz") as ga, \
             np.load(comparison/f"groups_n{c['num_cells']:03d}_event{c['child_event']:04d}.npz") as gb:
            key=f"{c['mode']}_a60_labels";la,lb=ga[key],gb[key]
            parents,choices,covered=enumerate_support(a,b,la,lb,c['gate_code_units'])
            for parent in r['parents']:
                target,cov,runner,status=parents[parent['parent_group']]
                assert (target,cov,runner,status)==(parent['best_child_group'],parent['parent_coverage'],parent['runner_up_parent_coverage'],parent['status'])
                parent_checks+=1
            for child in r['children']:
                group=child['child_group'];eligible=[p for p,t in choices.items() if t==group]
                union=set().union(*(covered[p] for p in eligible))
                assert sorted(eligible)==child['parent_groups']
                assert sorted(union)==r['supported_child_rows'][str(group)]
                assert len(union)/int((lb==group).sum())==child['union_child_coverage']
                for link in [x for x in r['links'] if x['child_group']==group]:
                    p=link['parent_group'];others=set().union(*(covered[q] for q in eligible if q!=p))
                    exclusive=len(covered[p]-others)
                    assert exclusive==link['exclusive_child_support_samples']
                    assert (exclusive==0)==link['redundant_support']
                child_checks+=1
        selected+=1
    controls=json.loads((output/'controlled_examples.json').read_text())
    assert controls[0]['unexpected_parent_groups']==[1] # Known geometric false-link control still fails.
    report=dict(passed=True,saved_summaries_checked=len(summary),independent_geometric_comparisons=selected,
                independent_parent_checks=parent_checks,independent_child_union_checks=child_checks,
                known_parallel_sheet_false_link_remains=True,physical_identity_validated=False)
    (output/'readback.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);run(p.parse_args().output)
