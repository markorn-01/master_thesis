"""Dense geometry and pair-distance-moment verification of expansion experiments."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.verify_motion_prediction import dense_support


def pair_extent(p):
    return float(np.sqrt(np.sum((p[:,None,:]-p[None,:,:])**2)/(2*len(p)**2)))


def verify(directory):
    meta=json.loads((directory/'metadata.json').read_text())
    for p,h in meta['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    records=json.loads((directory/'results.json').read_text());assert len(records)==meta['cases']*9
    cache={}
    for record in records:
        name=record['case']
        if name not in cache:
            with np.load(directory/f'{name}.npz') as a:data={k:a[k].copy() for k in a.files}
            def args(i):return tuple(data[f'{field}_{i}'] for field in ['positions','normals','labels'])
            _,_,accepted,redundant=dense_support(*args(0),*args(1),meta['bootstrap_gate'])
            translation=data['positions_1'].copy();expansion=translation.copy();models={}
            dt0,dt1=np.diff(data['times'])
            for group in np.unique(data['labels_1']):
                incoming=[p for p,c in accepted if c==group];initialized=False;extent=False;parent=None;velocity=None;r0=r1=rate=scale=None
                reason='no_motion_initialization';mask=data['labels_1']==group
                if len(incoming)==1:
                    predecessor=incoming[0];outgoing=[c for p,c in accepted if p==predecessor]
                    if len(outgoing)==1 and not redundant[predecessor,int(group)]:
                        initialized=True;parent=predecessor;prior=data['positions_0'][data['labels_0']==parent];current=data['positions_1'][mask]
                        center=current.mean(0);velocity=(center-prior.mean(0))/dt0
                        translation[mask]+=velocity*dt1;expansion[mask]=translation[mask]
                        r0,r1=pair_extent(prior),pair_extent(current)
                        reason='translation_fallback_unresolved_extent'
                        if min(r0,r1)>1e-12:
                            rate=(r1-r0)/dt0;future=r1+rate*dt1;reason='translation_fallback_nonpositive_predicted_extent'
                            if future>1e-12:
                                scale=future/r1;extent=True;reason='observed_centroid_and_rms_extent'
                                expansion[mask]=center+velocity*dt1+scale*(current-center)
                models[int(group)]=(initialized,extent,parent,velocity,r0,r1,rate,scale,reason)
            np.testing.assert_allclose(translation,data['translation_positions'],atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(expansion,data['expansion_positions'],atol=1e-12,rtol=1e-12)
            cache[name]=(data,models)
        data,models=cache[name]
        for e in record['estimates']:
            initialized,extent,parent,v,r0,r1,rate,scale,reason=models[e['group']]
            assert e['initialized']==initialized and e['predecessor_group']==parent
            if initialized:np.testing.assert_allclose(e['velocity'],v,atol=1e-12,rtol=1e-12)
            if record['method']=='isotropic_expansion':
                assert e['expansion_initialized']==extent and e['expansion_reason']==reason
                for k,value in [('extent_0',r0),('extent_1',r1),('extent_rate',rate),('scale',scale)]:
                    if value is None:assert e[k] is None
                    else:np.testing.assert_allclose(e[k],value,atol=1e-12,rtol=1e-12)
        positions=data[{'geometry':'positions_1','centroid_translation':'translation_positions','isotropic_expansion':'expansion_positions'}[record['method']]]
        parents,children,accepted,_=dense_support(positions,data['normals_1'],data['labels_1'],data['positions_2'],data['normals_2'],data['labels_2'],record['gate'])
        for p in record['parents']:assert parents[p['parent_group']]==(p['best_child_group'],p['parent_coverage'],p['runner_up_parent_coverage'],p['status'])
        for c in record['children']:assert children[c['child_group']]==(c['union_child_coverage'],c['accepted'])
        truth={tuple(x) for x in record['expected_links']}
        for key,value in [('accepted_links',accepted),('false_links',accepted-truth),('missed_links',truth-accepted)]:assert sorted(value)==[tuple(x) for x in record[key]]
    report=dict(passed=True,independent_dense_bootstrap_checks=len(cache),pair_distance_moment_checks=len(cache),
                independent_final_comparisons=len(records),known_failures_preserved=True,physical_tracker_validated=False)
    (directory/'readback.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);verify(p.parse_args().directory)
