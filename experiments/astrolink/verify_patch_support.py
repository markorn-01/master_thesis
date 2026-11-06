"""Read back pooled supports and independently reconstruct their geometry."""
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
    tables={name:list(csv.DictReader((directory/f'{name}.csv').open())) for name in ['parents','children','links','summary','raw_components']}
    for source in meta['input_sha256']:
        event=int(Path(source).stem.split('_')[-1])
        with np.load(source) as original,np.load(directory/f'event_{event:04d}.npz') as saved:
            for key in ['positions','normals','time','grid_spacing']:
                np.testing.assert_array_equal(saved[key],original[key])
            for row in [r for r in tables['raw_components'] if int(r['event_index'])==event]:
                mask=saved[f"raw_a{row['segmentation_angle']}"]==int(row['group'])
                assert int(mask.sum())==int(row['samples'])
                assert (mask.sum()>=20)==(row['passes_cutoff']=='True')
                np.testing.assert_allclose(saved['positions'][mask].mean(axis=0),
                                           [float(row[f'centroid_{axis}']) for axis in 'xyz'],rtol=0,atol=0)
    checks=0;parent_checks=0;cache={}
    for summary in tables['summary']:
        event,angle=int(summary['parent_event']),int(summary['segmentation_angle'])
        distance,margin=float(summary['distance_cells']),float(summary['dominance_margin'])
        keys=['parent_event','segmentation_angle','distance_cells','dominance_margin']
        subset=lambda name:[r for r in tables[name] if all(r[k]==summary[k] for k in keys)]
        parents,children,links=subset('parents'),subset('children'),subset('links')
        cache_key=(event,angle,distance)
        if cache_key not in cache:
            with np.load(directory/f'event_{event:04d}.npz') as a,np.load(directory/f'event_{event+1:04d}.npz') as b:
                pa,pb=a['positions'].astype(float),b['positions'].astype(float)
                na,nb=a['normals'].astype(float),b['normals'].astype(float)
                na/=np.linalg.norm(na,axis=1)[:,None];nb/=np.linalg.norm(nb,axis=1)[:,None]
                la,lb=a[f'labels_a{angle}'].copy(),b[f'labels_a{angle}'].copy()
                neighbors=cKDTree(pa).query_ball_tree(cKDTree(pb),distance*float(a['grid_spacing']))
                support={}
                for i,candidates in enumerate(neighbors):
                    if la[i]<0:continue
                    candidates=np.asarray(candidates,dtype=int)
                    candidates=candidates[(lb[candidates]>=0)&((nb[candidates]@na[i])>=np.cos(np.deg2rad(60.)))]
                    for j in candidates:
                        pair=(int(la[i]),int(lb[j]))
                        support.setdefault(pair,[set(),set()])[0].add(i)
                        support[pair][1].add(int(j))
            cache[cache_key]=(la,lb,support)
        la,lb,support=cache[cache_key]
        for p in parents:
            group=int(p['parent_group']);size=int((la==group).sum())
            candidates=sorted([(len(support.get((group,int(b)),[(),()])[0])/size,int(b)) for b in np.unique(lb[lb>=0])],
                              key=lambda x:(-x[0],x[1]))
            best=candidates[0] if candidates else (0.,-1)
            second=candidates[1][0] if len(candidates)>1 else 0.
            expected='insufficient_parent_support' if best[0]<.5 else 'ambiguous_target' if best[0]-second<margin else 'eligible'
            assert p['status']==expected and int(p['best_child_group'])==best[1]
            assert float(p['parent_coverage'])==best[0] and float(p['runner_up_parent_coverage'])==second
            parent_checks+=1
        with np.load(directory/f'support_{event:04d}_{event+1:04d}.npz') as saved:
            for child in children:
                target=int(child['child_group']);size=int((lb==target).sum())
                selected=[int(p['parent_group']) for p in parents if p['status']=='eligible' and int(p['best_child_group'])==target]
                assert selected==json.loads(child['parent_groups'])
                sets=[support[(p,target)][1] for p in selected]
                union=set().union(*sets) if sets else set()
                key=f'a{angle}_d{distance:g}_m{margin:g}_child{target}'
                np.testing.assert_array_equal(saved[key],np.array(sorted(union),dtype=int))
                assert len(union)==int(child['union_covered_child_samples'])
                assert len(union)/size==float(child['union_child_coverage'])
                assert sum(map(len,sets))==int(child['summed_child_support_counts'])
                assert max((len(s)/size for s in sets),default=0.)==float(child['largest_single_child_coverage'])
                accepted=bool(selected and len(union)/size>=.5)
                assert accepted==(child['accepted']=='True')
                matching=[l for l in links if int(l['child_group'])==target]
                assert [int(l['parent_group']) for l in matching]==selected
                assert all((l['accepted']=='True')==accepted for l in matching)
                checks+=1
        assert sum(l['accepted']=='True' for l in links)==int(summary['accepted_links'])
        assert sum(c['accepted']=='True' for c in children)==int(summary['accepted_pools'])
        assert sum(p['status']=='ambiguous_target' for p in parents)==int(summary['ambiguous_parents'])
    assert len(tables['summary'])==60
    result=dict(passed=True,configurations=60,independently_checked_parent_choices=parent_checks,
                independently_checked_child_unions=checks,inputs_and_sources_unchanged=True,
                scope='Geometry and saved accounting; not physical identity validation.')
    (directory/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    verify(p.parse_args().directory)
