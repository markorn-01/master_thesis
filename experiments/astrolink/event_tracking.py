"""Experimental conflict-free event matching for fresh surface catalogs.

This is a geometric baseline, not a physical ancestry estimator or an
implementation of the published two-step feature-tracking algorithm.
"""
from itertools import combinations
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import milp, Bounds, LinearConstraint


def select_events(candidates, parents, children, tie_tolerance=1e-8):
    """Maximum-score set packing; report events needed in every near-optimum.

    Each group participates in at most one selected event. Forbid each
    optimal event in turn: if an alternative is within tie_tolerance of
    the optimum, its identity is unresolved and is not accepted.
    """
    nodes=[('p',int(p)) for p in parents]+[('c',int(c)) for c in children]
    if not candidates:
        return dict(optimal_score=0.,solver_selected=[],accepted=[],ambiguous=[],exclusion_gaps={})
    scores=np.array([e['score'] for e in candidates])
    matrix=np.zeros((len(nodes),len(candidates)))
    for j,event in enumerate(candidates):
        for i,(side,group) in enumerate(nodes):
            matrix[i,j]=group in event['parents' if side=='p' else 'children']
    def solve(forbid=None):
        upper=np.ones(len(candidates))
        if forbid is not None:upper[forbid]=0
        # HiGHS absolute objective tolerances can otherwise exceed our 1e-8
        # score-tie tolerance. Scale only the solver objective, not the model.
        result=milp(-scores*1e6,integrality=np.ones(len(scores)),bounds=Bounds(np.zeros(len(scores)),upper),
                    constraints=LinearConstraint(matrix,np.zeros(len(nodes)),np.ones(len(nodes))),
                    options={'mip_rel_gap':0.,'time_limit':10.})
        if result.status!=0:raise RuntimeError(f'Event optimization did not finish exactly: {result.message}')
        selected=np.flatnonzero(result.x>.5).tolist()
        assert (matrix[:,selected].sum(axis=1)<=1).all()
        return selected,float(scores[selected].sum())
    selected,optimum=solve();accepted=[];ambiguous=set();gaps={}
    for j in selected:
        alternative,score=solve(j);gap=max(0.,optimum-score);gaps[str(j)]=gap
        if gap<=tie_tolerance:
            ambiguous.update(set(selected)^set(alternative))
        else:accepted.append(j)
    return dict(optimal_score=optimum,solver_selected=selected,accepted=accepted,
                ambiguous=sorted(ambiguous),exclusion_gaps=gaps)


def event_tracking(pa,na,la,pb,nb,lb,gate,*,coverage=.5,exclusive_fraction=.1,
                   distance_weight=.25,max_partners=3,normal_angle=60.,tie_tolerance=1e-8):
    """Build continuation/combination/separation hypotheses and optimize conflicts.

    A one-to-one candidate covers at least coverage of each group. For a
    many-to-one event every parent meets that threshold, their union covers
    the child, and every parent contributes exclusive_fraction of child
    samples not covered by its partners. One-to-many is the reverse rule.
    Candidate events with zero unique contributions cannot claim a merger.
    """
    pa,na,pb,nb=[np.array(a,dtype=float,copy=True) for a in [pa,na,pb,nb]]
    la,lb=np.asarray(la),np.asarray(lb)
    for p,n,l in [(pa,na,la),(pb,nb,lb)]:
        if p.ndim!=2 or p.shape[1]!=3 or n.shape!=p.shape or l.shape!=(len(p),):raise ValueError('Invalid catalog shapes.')
        if not np.isfinite(p).all() or not np.isfinite(n).all() or (np.linalg.norm(n,axis=1)<=0).any():raise ValueError('Require finite geometry and nonzero normals.')
    if not np.isfinite(gate) or gate<=0 or not 0<coverage<=1 or not 0<exclusive_fraction<=1 or not 0<=distance_weight<1 or max_partners not in [2,3] or not 0<normal_angle<90:
        raise ValueError('Invalid event matching parameters.')
    if not np.isfinite(tie_tolerance) or tie_tolerance<0:raise ValueError('Invalid tie tolerance.')
    na/=np.linalg.norm(na,axis=1)[:,None];nb/=np.linalg.norm(nb,axis=1)[:,None]
    parents=sorted(map(int,np.unique(la[la>=0])));children=sorted(map(int,np.unique(lb[lb>=0])))
    sizes_p={p:int((la==p).sum()) for p in parents};sizes_c={c:int((lb==c).sum()) for c in children}
    hits={};tree=cKDTree(pb)
    for i in np.flatnonzero(la>=0):
        near=np.asarray(tree.query_ball_point(pa[i],gate),int)
        near=near[(lb[near]>=0)&((nb[near]@na[i])>=np.cos(np.deg2rad(normal_angle)))]
        for j in near:
            key=(int(la[i]),int(lb[j]));a,b=hits.setdefault(key,({},{}));distance=float(np.linalg.norm(pa[i]-pb[j]))
            a[int(i)]=min(a.get(int(i),float('inf')),distance)
            b[int(j)]=min(b.get(int(j),float('inf')),distance)
    def support(p,c,side):return hits.get((p,c),({},{}))[side]
    candidates=[]
    def add(ps,cs,kind):
        ps=sorted(ps);cs=sorted(cs);node_coverages=[];distance_terms=[];exclusive=[]
        for side,groups,sizes in [(0,ps,sizes_p),(1,cs,sizes_c)]:
            for group in groups:
                dictionaries=[support(group,c,0) for c in cs] if side==0 else [support(p,group,1) for p in ps]
                union={}
                for d in dictionaries:
                    for row,distance in d.items():union[row]=min(union.get(row,float('inf')),distance)
                fraction=len(union)/sizes[group]
                if fraction<coverage:return
                node_coverages.append(dict(side='parent' if side==0 else 'child',group=group,coverage=fraction))
                distance_terms.append(fraction*np.mean(list(union.values()))/gate if union else 0.)
        if kind!='continuation':
            sets=[set(support(p,cs[0],1)) for p in ps] if kind=='combination' else [set(support(ps[0],c,0)) for c in cs]
            denominator=sizes_c[cs[0]] if kind=='combination' else sizes_p[ps[0]]
            for i,s in enumerate(sets):
                others=set().union(*(q for j,q in enumerate(sets) if i!=j))
                fraction=len(s-others)/denominator
                if fraction<exclusive_fraction:return
                exclusive.append(fraction)
        score=sum(v['coverage'] for v in node_coverages)-distance_weight*sum(distance_terms)
        candidates.append(dict(kind=kind,parents=ps,children=cs,score=float(score),node_coverages=node_coverages,
                               exclusive_central_fractions=exclusive))
        if len(candidates)>5000:raise ValueError('Too many hypotheses for this bounded prototype.')
    for p in parents:
        for c in children:
            if (p,c) in hits:add([p],[c],'continuation')
    for c in children:
        eligible=[p for p in parents if len(support(p,c,0))/sizes_p[p]>=coverage]
        for count in range(2,min(max_partners,len(eligible))+1):
            for ps in combinations(eligible,count):add(ps,[c],'combination')
    for p in parents:
        eligible=[c for c in children if len(support(p,c,1))/sizes_c[c]>=coverage]
        for count in range(2,min(max_partners,len(eligible))+1):
            for cs in combinations(eligible,count):add([p],cs,'separation')
    candidates.sort(key=lambda e:(e['kind'],e['parents'],e['children']))
    selection=select_events(candidates,parents,children,tie_tolerance)
    events=[dict(candidate_index=i,**candidates[i]) for i in selection['accepted']]
    links=[dict(parent_group=p,child_group=c,accepted=True,event_index=e['candidate_index'],event_kind=e['kind'])
           for e in events for p in e['parents'] for c in e['children']]
    used_p={p for e in events for p in e['parents']};used_c={c for e in events for c in e['children']}
    ambiguous_p={p for i in selection['ambiguous'] for p in candidates[i]['parents']}
    ambiguous_c={c for i in selection['ambiguous'] for c in candidates[i]['children']}
    return dict(candidates=candidates,selection=selection,events=events,links=links,
        parents=[dict(parent_group=p,status='assigned' if p in used_p else 'ambiguous' if p in ambiguous_p else 'unmatched') for p in parents],
        children=[dict(child_group=c,status='assigned' if c in used_c else 'ambiguous' if c in ambiguous_c else 'unmatched') for c in children],
        parameters=dict(gate=gate,coverage=coverage,exclusive_fraction=exclusive_fraction,distance_weight=distance_weight,
                        max_partners=max_partners,normal_angle=normal_angle,tie_tolerance=tie_tolerance))
