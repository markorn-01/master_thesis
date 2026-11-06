"""Verify cutoff accounting and independently inspect selected support overlaps."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def verify(directory):
    meta=json.loads((directory/'metadata.json').read_text())
    for name,h in {**meta['input_sha256'],**meta['source_sha256']}.items():
        path=Path(name)
        if not path.exists() and '/experiments/' in name:path=Path('experiments')/name.split('/experiments/',1)[1]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==h,name
    tables={name:list(csv.DictReader((directory/f'{name}.csv').open())) for name in ['group_sizes','children','links','summary']}
    checked=0
    for name in meta['input_sha256']:
        event=int(Path(name).stem.split('_')[-1])
        with np.load(name) as source,np.load(directory/f'event_{event:04d}.npz') as a:
            for key in ['positions','normals','time','grid_spacing']:np.testing.assert_array_equal(a[key],source[key])
            for r in [r for r in tables['group_sizes'] if int(r['event_index'])==event]:
                angle,cutoff=int(r['segmentation_angle']),int(r['min_size'])
                raw,labels=a[f'raw_a{angle}'],a[f'a{angle}_s{cutoff}']
                ids,counts=np.unique(raw[raw>=0],return_counts=True)
                kept=ids[counts>=cutoff]
                expected=raw.copy();expected[(raw>=0)&~np.isin(raw,kept)]=-1
                np.testing.assert_array_equal(labels,expected)
                assert len(kept)==int(r['retained_groups'])
                assert int((labels>=0).sum())==int(r['retained_samples'])
                assert int(((raw>=0)&(labels<0)).sum())==int(r['excluded_samples'])
                checked+=1
    for summary in tables['summary']:
        keys=['parent_event','segmentation_angle','min_size','distance_cells']
        match=lambda r:all(r[k]==summary[k] for k in keys)
        children=[r for r in tables['children'] if match(r)]
        links=[r for r in tables['links'] if match(r)]
        assert len(children)==int(summary['child_groups'])
        assert sum(r['accepted']=='True' for r in children)==int(summary['accepted_pools'])
        assert sum(r['accepted']=='True' for r in links)==int(summary['accepted_links'])
        assert sum(r['accepted']=='True' and r['redundant_support']=='True' for r in links)==int(summary['redundant_accepted_links'])
    independent=0
    for event in [12,14]:
        with np.load(directory/f'event_{event:04d}.npz') as a,np.load(directory/f'event_{event+1:04d}.npz') as b:
            pa,pb=a['positions'].astype(float),b['positions'].astype(float)
            na,nb=a['normals'].astype(float),b['normals'].astype(float)
            na/=np.linalg.norm(na,axis=1)[:,None];nb/=np.linalg.norm(nb,axis=1)[:,None]
            neighbors=cKDTree(pa).query_ball_tree(cKDTree(pb),2.5*float(a['grid_spacing']))
            for cutoff in meta['cutoffs']:
                la,lb=a[f'a60_s{cutoff}'],b[f'a60_s{cutoff}']
                support={}
                for i,candidates in enumerate(neighbors):
                    if la[i]<0:continue
                    for j in candidates:
                        if lb[j]>=0 and np.dot(na[i],nb[j])>=np.cos(np.deg2rad(60.)):
                            support.setdefault((int(la[i]),int(lb[j])),set()).add(j)
                children=[r for r in tables['children'] if int(r['parent_event'])==event and r['segmentation_angle']=='60'
                          and int(r['min_size'])==cutoff and r['distance_cells']=='2.5']
                for child in children:
                    target=int(child['child_group']);parents=json.loads(child['parent_groups'])
                    sets={p:support[(p,target)] for p in parents}
                    union=set().union(*sets.values())
                    assert len(union)==int(child['union_covered_child_samples'])
                    selected=[r for r in tables['links'] if int(r['parent_event'])==event and r['segmentation_angle']=='60'
                              and int(r['min_size'])==cutoff and r['distance_cells']=='2.5' and int(r['child_group'])==target]
                    assert len(selected)==len(parents)
                    for link in selected:
                        p=int(link['parent_group']);others=set().union(*(s for g,s in sets.items() if g!=p))
                        exclusive=len(sets[p]-others)
                        assert exclusive==int(link['exclusive_child_support_samples'])
                        assert (exclusive==0)==(link['redundant_support']=='True')
                        independent+=1
    controls=json.loads((directory/'controlled_examples.json').read_text())
    near=next(c for c in controls if c['case']=='near_parallel_distractor')
    assert near['unexpected_parent_groups']==[1] and near['redundant_accepted_parents']==2
    for case in controls[1:]:assert case['unexpected_parent_groups']==[] and case['missed_parent_groups']==[]
    assert len(tables['summary'])==320
    result=dict(software_and_readback_checks_passed=True,cutoff_partitions_checked=checked,adjacent_configurations=320,
                independently_checked_link_overlap_counts=independent,inputs_and_sources_unchanged=True,
                near_parallel_false_link_control_passed=False,known_counterexample_recorded=True,
                ready_for_physical_identity_assignment=False,
                scope='Checks verify recorded behavior, including a known false link; they do not make the method physically valid.')
    (directory/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    verify(p.parse_args().directory)
