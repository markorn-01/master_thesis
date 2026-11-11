"""Compare pooled matching and event optimization on stipulated catalog cases."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from experiments.astrolink.controlled_tracking_cases import tracking_cases,evaluate_links
from experiments.astrolink.check_controlled_tracking import evaluate
from experiments.astrolink.event_tracking import event_tracking
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write

KEYS=['positions_a','normals_a','labels_a','positions_b','normals_b','labels_b']
GATES=[.1,.2,.35]


def cases():
    original=tracking_cases()
    for case in original:case['suite']='existing'
    near=copy.deepcopy(next(c for c in original if c['name']=='nearby_sheet_disappears'))
    near.update(name='same_geometry_different_ancestry',suite='stress',expected_links=[[1,10]],
                meaning='Identical inputs to nearby_sheet_disappears, opposite stipulated ancestry. Geometry cannot resolve both truths.')
    equal=copy.deepcopy(near);plane=near['positions_b']-[0,0,.03]
    equal.update(name='equal_distance_alternatives',positions_a=np.vstack((plane+[0,0,-.1],plane+[0,0,.1])),
                 positions_b=plane,expected_links=[[0,10]],meaning='Equal-distance alternatives must be left ambiguous.')
    tiny=copy.deepcopy(next(c for c in original if c['name']=='two_fragments_form_one_group'))
    tiny.update(name='tiny_complementary_fragment',suite='stress',labels_a=np.array([0]+[1]*15),
                meaning='One of sixteen complementary samples belongs to a small fragment, below the declared 10% exclusive fraction.')
    return original+[near,equal,tiny]


def independent_check(case,result):
    """Dense geometry, exhaustive event generation, and brute-force optimization."""
    pa,na,la,pb,nb,lb=[np.asarray(case[k]) for k in KEYS];params=result['parameters']
    na=na/np.linalg.norm(na,axis=1)[:,None];nb=nb/np.linalg.norm(nb,axis=1)[:,None]
    distances=cdist(pa,pb);compatible=(distances<=params['gate'])&(na@nb.T>=np.cos(np.deg2rad(params['normal_angle'])))
    parents=sorted(map(int,np.unique(la[la>=0])));children=sorted(map(int,np.unique(lb[lb>=0])))
    def hypothesis(ps,cs,kind):
        coverages=[];cost=[]
        for side,groups in [(0,ps),(1,cs)]:
            for group in groups:
                ia=np.flatnonzero(la==group if side==0 else np.isin(la,ps))
                ib=np.flatnonzero(np.isin(lb,cs) if side==0 else lb==group)
                matrix=compatible[np.ix_(ia,ib)];d=distances[np.ix_(ia,ib)]
                covered=matrix.any(axis=1 if side==0 else 0);fraction=float(covered.mean())
                if fraction<params['coverage']:return None
                minimum=np.where(matrix,d,np.inf).min(axis=1 if side==0 else 0)
                coverages.append(fraction);cost.append(fraction*float(minimum[covered].mean())/params['gate'])
        if kind!='continuation':
            if kind=='combination':
                ib=np.flatnonzero(lb==cs[0]);footprints=[compatible[np.ix_(np.flatnonzero(la==p),ib)].any(0) for p in ps]
            else:
                ia=np.flatnonzero(la==ps[0]);footprints=[compatible[np.ix_(ia,np.flatnonzero(lb==c))].any(1) for c in cs]
            for i,footprint in enumerate(footprints):
                others=np.logical_or.reduce([f for j,f in enumerate(footprints) if j!=i])
                if float((footprint&~others).mean())<params['exclusive_fraction']:return None
        return sum(coverages)-params['distance_weight']*sum(cost)
    expected={}
    for p in parents:
        for c in children:
            score=hypothesis([p],[c],'continuation')
            if score is not None:expected[('continuation',(p,),(c,))]=score
    for count in range(2,params['max_partners']+1):
        for ps in combinations(parents,count):
            for c in children:
                score=hypothesis(ps,[c],'combination')
                if score is not None:expected[('combination',ps,(c,))]=score
        for cs in combinations(children,count):
            for p in parents:
                score=hypothesis([p],cs,'separation')
                if score is not None:expected[('separation',(p,),cs)]=score
    actual={(e['kind'],tuple(e['parents']),tuple(e['children'])):e['score'] for e in result['candidates']}
    assert actual.keys()==expected.keys()
    for key,value in actual.items():np.testing.assert_allclose(value,expected[key],atol=1e-12,rtol=1e-12)
    candidates=result['candidates'];assert len(candidates)<=18
    feasible=[]
    for mask in range(1<<len(candidates)):
        indices={i for i in range(len(candidates)) if mask&(1<<i)}
        p=[x for i in indices for x in candidates[i]['parents']];c=[x for i in indices for x in candidates[i]['children']]
        if len(p)!=len(set(p)) or len(c)!=len(set(c)):continue
        feasible.append((sum(candidates[i]['score'] for i in indices),indices))
    optimum=max(score for score,_ in feasible)
    near=[indices for score,indices in feasible if optimum-score<=params['tie_tolerance']]
    essential=set.intersection(*near)
    assert essential==set(result['selection']['accepted'])
    np.testing.assert_allclose(result['selection']['optimal_score'],optimum,atol=1e-12,rtol=1e-12)
    return len(candidates)


def run(output):
    output.mkdir(parents=True,exist_ok=False);rows=[];details=[];candidate_checks=0
    for case in cases():
        np.savez_compressed(output/f'{case["name"]}.npz',**{k:case[k] for k in KEYS})
        for gate in GATES:
            baseline=evaluate(case,gate);prototype=event_tracking(*(case[k] for k in KEYS),gate)
            candidate_checks+=independent_check(case,prototype)
            rng=np.random.default_rng(42);a=rng.permutation(len(case['positions_a']));b=rng.permutation(len(case['positions_b']))
            shuffled=event_tracking(case['positions_a'][a],case['normals_a'][a],case['labels_a'][a],
                                     case['positions_b'][b],case['normals_b'][b],case['labels_b'][b],gate)
            assert evaluate_links(shuffled,case['expected_links'])==evaluate_links(prototype,case['expected_links'])
            assert [(e['kind'],e['parents'],e['children']) for e in shuffled['events']]==[(e['kind'],e['parents'],e['children']) for e in prototype['events']]
            for method,result in [('pooled_baseline',baseline),('event_prototype',prototype)]:
                metrics=evaluate_links(result,case['expected_links'])
                rows.append(dict(suite=case['suite'],case=case['name'],gate=gate,method=method,
                                 correct_links=len(metrics['correct_links']),false_links=len(metrics['false_links']),
                                 missed_links=len(metrics['missed_links']),exact_correspondence=metrics['exact_correspondence'],
                                 continuations=sum(e['kind']=='continuation' for e in result.get('events',[])),
                                 combinations=sum(e['kind']=='combination' for e in result.get('events',[])),
                                 separations=sum(e['kind']=='separation' for e in result.get('events',[]))))
                details.append(dict(case=case['name'],suite=case['suite'],gate=gate,method=method,meaning=case['meaning'],
                                    **metrics,**({k:prototype[k] for k in ['candidates','selection','events','parents','children','parameters']} if method=='event_prototype' else
                                                {k:baseline[k] for k in ['parents','children','links']})))
    csv_write(output/'summary.csv',rows);(output/'results.json').write_text(json.dumps(details,indent=2)+'\n')
    aggregate=[]
    for suite in ['existing','stress']:
        for method in ['pooled_baseline','event_prototype']:
            selected=[r for r in rows if r['suite']==suite and r['method']==method]
            aggregate.append(dict(suite=suite,method=method,configurations=len(selected),
                                  exact_configurations=sum(r['exact_correspondence'] for r in selected),
                                  false_links=sum(r['false_links'] for r in selected),missed_links=sum(r['missed_links'] for r in selected)))
    csv_write(output/'aggregate.csv',aggregate);plot(rows,output)
    # Read metrics back from disk, not only in-memory solver results.
    saved=json.loads((output/'results.json').read_text())
    for result,row in zip(saved,list(csv.DictReader((output/'summary.csv').open()))):
        assert len(result['false_links'])==int(row['false_links']) and len(result['missed_links'])==int(row['missed_links'])
    sources=[Path(__file__)]+[Path('experiments/astrolink')/f'{name}.py' for name in
        ['event_tracking','controlled_tracking_cases','check_controlled_tracking','patch_support','support_evidence']]
    (output/'verification.json').write_text(json.dumps(dict(configurations=36,methods=2,
        independent_dense_and_bruteforce_checks=36,independently_checked_hypotheses=candidate_checks,shuffle_checks=36,
        parameters=dict(coverage=.5,exclusive_fraction=.1,distance_weight=.25,max_partners=3,normal_angle=60,tie_tolerance=1e-8),
        aggregate=aggregate,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        real_bubble_results_changed=False,validated_on_hydrodynamic_ancestry=False,
        literature_note='Original one-stage set-packing prototype informed by feature-event conflict graphs; not a reproduction of Schnorr et al. two-step optimization.'),indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))


def plot(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(dict.fromkeys(r['case'] for r in rows));fig,axes=plt.subplots(2,2,figsize=(15,12))
    for row,(field,title) in enumerate([('false_links','False links'),('missed_links','Missed stipulated links')]):
        for col,method in enumerate(['pooled_baseline','event_prototype']):
            ax=axes[row,col];matrix=np.array([[next(r[field] for r in rows if r['case']==name and r['method']==method and r['gate']==g) for g in GATES] for name in names])
            ax.imshow(matrix,cmap='YlOrRd',vmin=0,vmax=2,aspect='auto')
            for (i,j),value in np.ndenumerate(matrix):ax.text(j,i,str(value),ha='center',va='center')
            ax.set(xticks=range(3),xticklabels=GATES,yticks=range(len(names)),yticklabels=[n.replace('_',' ') for n in names],
                   xlabel='Distance gate (toy units)',title=f'{method.replace("_"," ")}: {title}')
            ax.axhline(8.5,color='black',linewidth=1)
    fig.suptitle('Event-aware prototype versus pooled matching: stipulated catalog correspondences')
    fig.text(.025,.02,'First nine rows: existing cases. Last three: additional stress cases. The same-geometry ancestry control exposes missing information; the closest-distance rule fails there.\n'
             'Weights/thresholds fixed across all cases. These synthetic results do not validate physical shock ancestry or general performance on colliding bubbles.',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(output/'event_tracking_comparison.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args().output_dir)
