"""Readback checks for candidate temporal association experiments."""
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
        if not path.exists() and '/experiments/' in name:
            path=Path('experiments')/name.split('/experiments/',1)[1]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==h,name
    rows={name:list(csv.DictReader((directory/f'{name}.csv').open()))
          for name in ['nodes','pair_scores','candidate_states','summary']}
    assert len(rows['summary'])==12
    for node in rows['nodes']:
        with np.load(directory/f"event_{int(node['event_index']):04d}.npz") as a:
            points=a['positions'][a[f"labels_a{node['segmentation_angle']}"]==int(node['group'])]
            assert len(points)==int(node['samples'])
            np.testing.assert_allclose(points.mean(axis=0),[float(node[f'centroid_{c}']) for c in 'xyz'],rtol=0,atol=0)
    for s in rows['pair_scores']:
        assert int(s['child_event'])==int(s['parent_event'])+1
        for tag in ['parent','child']:
            assert float(s[f'{tag}_coverage'])==int(s[f'covered_{tag}_samples'])/int(s[f'{tag}_samples'])
            assert 0<=float(s[f'{tag}_coverage'])<=1
    for config in rows['summary']:
        angle,distance,coverage=config['segmentation_angle'],config['distance_cells'],float(config['min_coverage'])
        scores=[s for s in rows['pair_scores'] if s['segmentation_angle']==angle and s['distance_cells']==distance
                and float(s['parent_coverage'])>=coverage and float(s['child_coverage'])>=coverage]
        states=[s for s in rows['candidate_states'] if s['segmentation_angle']==angle and s['distance_cells']==distance
                and float(s['min_coverage'])==coverage]
        assert len(scores)==int(config['links']) and len(states)==int(config['nodes'])
        for state in states:
            parents=[int(s['parent_group']) for s in scores if s['child_event']==state['event_index'] and s['child_group']==state['group']]
            children=[int(s['child_group']) for s in scores if s['parent_event']==state['event_index'] and s['parent_group']==state['group']]
            assert parents==json.loads(state['parent_groups']) and children==json.loads(state['child_groups'])
            assert len(parents)==int(state['candidate_parents']) and len(children)==int(state['candidate_children'])
        assert sum(int(s['candidate_parents'])>1 for s in states)==int(config['multiple_parent_nodes'])
        assert sum(int(s['candidate_children'])>1 for s in states)==int(config['multiple_child_nodes'])
        for tag,available in [('incoming','incoming_comparison_available'),('outgoing','outgoing_comparison_available')]:
            count='candidate_parents' if tag=='incoming' else 'candidate_children'
            assert sum(s[available]=='True' and int(s[count])==0 for s in states)==int(config[f'unmatched_{tag}_nodes'])
    # Independent sparse-distance pair enumeration checks selected merge and
    # fragmentation transitions. This does not reuse association_scores().
    checked=0
    for event in [7,9,13]:
        with np.load(directory/f'event_{event:04d}.npz') as a,np.load(directory/f'event_{event+1:04d}.npz') as b:
            pa,pb=a['positions'].astype(float),b['positions'].astype(float)
            na,nb=a['normals'].astype(float),b['normals'].astype(float)
            na/=np.linalg.norm(na,axis=1)[:,None];nb/=np.linalg.norm(nb,axis=1)[:,None]
            for distance in [2.5,3.5]:
                pairs=cKDTree(pa).sparse_distance_matrix(cKDTree(pb),distance*float(a['grid_spacing']),output_type='coo_matrix')
                ia,ib=pairs.row,pairs.col
                good=np.einsum('ij,ij->i',na[ia],nb[ib])>=np.cos(np.deg2rad(60.))
                ia,ib=ia[good],ib[good]
                for angle in [30,60]:
                    la,lb=a[f'labels_a{angle}'],b[f'labels_a{angle}']
                    selected=[s for s in rows['pair_scores'] if int(s['parent_event'])==event
                              and int(s['segmentation_angle'])==angle and float(s['distance_cells'])==distance]
                    for s in selected:
                        keep=(la[ia]==int(s['parent_group']))&(lb[ib]==int(s['child_group']))
                        assert len(np.unique(ia[keep]))==int(s['covered_parent_samples'])
                        assert len(np.unique(ib[keep]))==int(s['covered_child_samples'])
                        checked+=1
    result=dict(passed=True,configurations=len(rows['summary']),nodes=len(rows['nodes']),pair_scores=len(rows['pair_scores']),
                candidate_states=len(rows['candidate_states']),independently_recomputed_pair_scores=checked,
                inputs_and_sources_unchanged=True,scope='Geometry/accounting consistency, not validated physical tracking.')
    (directory/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    verify(p.parse_args().directory)
