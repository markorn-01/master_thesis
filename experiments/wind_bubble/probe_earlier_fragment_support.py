"""Independent dense inspection of the earlier seven-patch transition."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist


def run(root,history,validation):
    paths=[root/'online/surfaces_0010.npz',root/'online/surfaces_0011.npz',history/'groups_0010.npz',history/'groups_0011.npz',validation/'tracking_results.json.gz']
    report=json.loads((validation/'verification.json').read_text());assert report['passed']
    for path in paths:
        if str(path) in report['input_sha256']:assert hashlib.sha256(path.read_bytes()).hexdigest()==report['input_sha256'][str(path)]
    with gzip.open(paths[-1],'rt') as stream:records=json.load(stream)
    with np.load(paths[0]) as a,np.load(paths[1]) as b,np.load(paths[2]) as ga,np.load(paths[3]) as gb:
        la=ga['labels'];lb=gb['labels'];ma=np.isin(la,range(1,8));mb=lb==1
        pa=a['positions'][ma].astype(float);pb=b['positions'][mb].astype(float)
        na=a['normals'][ma].astype(float);nb=b['normals'][mb].astype(float)
        na/=np.linalg.norm(na,axis=1)[:,None];nb/=np.linalg.norm(nb,axis=1)[:,None]
        distance=cdist(pa,pb);distance[(na@nb.T)<np.cos(np.deg2rad(60))]=np.inf
        rows=[]
        for gate in [1.5,2.5]:
            good=distance<=gate/64
            footprints=[good[la[ma]==g].any(axis=0) for g in range(1,8)]
            unique=[float((f&~np.logical_or.reduce([other for j,other in enumerate(footprints) if j!=i])).mean()) for i,f in enumerate(footprints)]
            record=next(r for r in records if r['parent_event']==10 and r['gate_reference_cells']==gate)
            selected=next(e for e in record['result']['events'] if e['children']==[1])
            omitted=[]
            for group in [1,7]:
                candidates=[e for e in record['result']['candidates'] if group in e['parents'] and e['children']==[1]]
                best=max(candidates,key=lambda e:e['score'])
                omitted.append(dict(group=group,samples=int((la==group).sum()),admissible_candidates=len(candidates),
                    best_candidate_parents=best['parents'],score_deficit=selected['score']-best['score']))
            rows.append(dict(gate_reference_cells=gate,parents=list(range(1,8)),
                all_seven_child_coverage=float(np.logical_or.reduce(footprints).mean()),exclusive_child_fractions=unique,
                full_seven_would_pass_exclusivity=all(x>=.1 for x in unique),
                selected_five=selected['parents'],omitted_patch_audit=omitted))
    result=dict(results=rows,physical_ancestry_established=False,
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[validation/'verification.json']},
        source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (validation/'earlier_seven_fragment_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/two_bubbles_detection_extension_iwr_n128'))
    parser.add_argument('--history',type=Path,default=Path('outputs/two_bubbles_detection_extension_review_n128'))
    parser.add_argument('--validation',type=Path,default=Path('outputs/expanded_full_sequence_validation_n128'))
    args=parser.parse_args();run(args.root,args.history,args.validation)
