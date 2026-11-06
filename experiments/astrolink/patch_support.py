"""Experimental pooled geometric support; no physical identity assignment."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def pooled_support(pa, na, la, pb, nb, lb, max_distance,
                   max_angle_degrees=60., min_parent_coverage=.5,
                   min_child_coverage=.5, dominance_margin=.25):
    """Pool unique child samples after each parent selects a dominant target.

    Require adequate parent coverage and a gap from its runner-up target.
    The union of supported child rows must cover min_child_coverage of the
    child. This is asymmetric and tests fragment-to-group correspondence.
    No summing of overlapping support, forced ties or persistent IDs.
    """
    pa,pb,na,nb=map(lambda x:np.asarray(x,dtype=float),(pa,pb,na,nb))
    la,lb=np.asarray(la),np.asarray(lb)
    for p,n,l in [(pa,na,la),(pb,nb,lb)]:
        if p.ndim!=2 or p.shape[1]!=3 or n.shape!=p.shape or l.shape!=(len(p),):
            raise ValueError('Require positions/normals (N,3) and labels (N,).')
        if not np.isfinite(p).all() or not np.isfinite(n).all() or (np.linalg.norm(n,axis=1)<=0).any():
            raise ValueError('Require finite geometry and nonzero normals.')
    if (not np.isfinite(max_distance) or max_distance<=0 or not np.isfinite(max_angle_degrees)
            or not 0<max_angle_degrees<90 or not 0<min_parent_coverage<=1
            or not 0<min_child_coverage<=1 or not 0<dominance_margin<=1):
        raise ValueError('Invalid support thresholds.')
    ids_a,counts_a=np.unique(la[la>=0],return_counts=True)
    ids_b,counts_b=np.unique(lb[lb>=0],return_counts=True)
    na=na/np.linalg.norm(na,axis=1)[:,None]
    nb=nb/np.linalg.norm(nb,axis=1)[:,None]
    # Enumerate once in original catalog row coordinates.
    pairs=cKDTree(pa).sparse_distance_matrix(cKDTree(pb),max_distance,output_type='coo_matrix')
    ia,ib=pairs.row,pairs.col
    good=(la[ia]>=0)&(lb[ib]>=0)&(np.einsum('ij,ij->i',na[ia],nb[ib])>=np.cos(np.deg2rad(max_angle_degrees)))
    ia,ib=ia[good],ib[good]
    parents=[];supports={};choices={}
    for a,size in zip(ids_a,counts_a):
        candidates=[]
        for b in ids_b:
            keep=(la[ia]==a)&(lb[ib]==b)
            covered_a,covered_b=np.unique(ia[keep]),np.unique(ib[keep])
            candidates.append((len(covered_a)/int(size),int(b),covered_b))
        candidates.sort(key=lambda x:(-x[0],x[1]))
        best=candidates[0] if candidates else (0.,-1,np.array([],dtype=int))
        second=candidates[1][0] if len(candidates)>1 else 0.
        status='eligible'
        if best[0]<min_parent_coverage:status='insufficient_parent_support'
        elif best[0]-second<dominance_margin:status='ambiguous_target'
        row=dict(parent_group=int(a),parent_samples=int(size),best_child_group=best[1],
                 parent_coverage=best[0],runner_up_parent_coverage=second,
                 dominance_gap=best[0]-second,status=status)
        parents.append(row)
        if status=='eligible':
            choices[int(a)]=best[1];supports[int(a)]=best[2]
    children=[];links=[];saved_support={}
    for b,size in zip(ids_b,counts_b):
        group_parents=[a for a,target in choices.items() if target==b]
        union=np.unique(np.concatenate([supports[a] for a in group_parents])) if group_parents else np.array([],dtype=int)
        union_coverage=len(union)/int(size)
        individual=[len(supports[a])/int(size) for a in group_parents]
        accepted=bool(group_parents and union_coverage>=min_child_coverage)
        children.append(dict(child_group=int(b),child_samples=int(size),eligible_parents=len(group_parents),
                             parent_groups=group_parents,union_covered_child_samples=len(union),
                             union_child_coverage=union_coverage,
                             largest_single_child_coverage=max(individual,default=0.),
                             summed_child_support_counts=sum(len(supports[a]) for a in group_parents),
                             accepted=accepted))
        saved_support[int(b)]=union
        for a in group_parents:
            links.append(dict(parent_group=a,child_group=int(b),accepted=accepted,
                              individual_child_coverage=len(supports[a])/int(size),union_child_coverage=union_coverage))
    return dict(parents=parents,children=children,links=links,supported_child_rows=saved_support)
