"""Offline bounded arity experiment; frozen production tracker stays unchanged."""
from itertools import combinations
import numpy as np
from experiments.astrolink.event_tracking import event_tracking,_event_candidates,_tracking_result


def expanded_tracking(pa,na,la,pb,nb,lb,gate,*,max_partners=5):
    if type(max_partners) is not int or max_partners not in (3,4,5):
        raise ValueError('This bounded experiment supports maximum arity 3, 4 or 5.')
    if max_partners==3:return event_tracking(pa,na,la,pb,nb,lb,gate)
    candidates,parents,children,p,hits,sizes_p,sizes_c=_event_candidates(pa,na,la,pb,nb,lb,gate)
    sizes=[sizes_p,sizes_c]
    def support(parent,child,side):return hits.get((parent,child),({},{}))[side]
    audits=[]
    for kind,central,partners in [('combination',children,parents),('separation',parents,children)]:
        for center in central:
            side=0 if kind=='combination' else 1
            eligible=[g for g in partners if len(support(g,center,0) if side==0 else support(center,g,1))/sizes[side][g]>=p['coverage']]
            for count in range(4,min(max_partners,len(eligible))+1):
                for group_set in combinations(eligible,count):
                    ps=list(group_set) if side==0 else [center];cs=[center] if side==0 else list(group_set)
                    coverages=[];distances=[]
                    for s,groups in [(0,ps),(1,cs)]:
                        for g in groups:
                            dictionaries=[support(g,c,0) for c in cs] if s==0 else [support(parent,g,1) for parent in ps]
                            union={}
                            for d in dictionaries:
                                for row,distance in d.items():union[row]=min(union.get(row,np.inf),distance)
                            fraction=len(union)/sizes[s][g]
                            coverages.append(dict(side='parent' if s==0 else 'child',group=g,coverage=fraction))
                            distances.append(fraction*np.mean(list(union.values()))/gate if union else 0.)
                    footprints=[set(support(g,center,1)) if side==0 else set(support(center,g,0)) for g in group_set]
                    denominator=sizes[1-side][center]
                    unique=[len(f-set().union(*(other for j,other in enumerate(footprints) if j!=i)))/denominator for i,f in enumerate(footprints)]
                    coverage_ok=all(c['coverage']>=p['coverage'] for c in coverages)
                    unique_ok=all(u>=p['exclusive_fraction'] for u in unique)
                    score=sum(c['coverage'] for c in coverages)-p['distance_weight']*sum(distances)
                    audits.append(dict(kind=kind,parents=ps,children=cs,node_coverages=coverages,
                        exclusive_central_fractions=unique,coverage_pass=coverage_ok,exclusivity_pass=unique_ok,
                        admitted=coverage_ok and unique_ok,score=float(score)))
                    if len(audits)>5000:raise ValueError('Too many high-arity hypotheses for this bounded experiment.')
                    if coverage_ok and unique_ok:
                        candidates.append(dict(kind=kind,parents=ps,children=cs,score=float(score),
                            node_coverages=coverages,exclusive_central_fractions=unique))
                        if len(candidates)>5000:raise ValueError('Too many hypotheses for this bounded experiment.')
    candidates.sort(key=lambda e:(e['kind'],e['parents'],e['children']))
    result=_tracking_result(candidates,parents,children,dict(p,max_partners=max_partners))
    result['higher_arity_audit']=audits
    return result
