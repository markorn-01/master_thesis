"""Advisory overlap diagnostics and explicit controls for pooled support."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from experiments.astrolink.patch_support import pooled_support


def support_evidence(pa,na,la,pb,nb,lb,max_distance,**kwargs):
    """Append evidence diagnostics without changing pooled acceptance decisions.

    Zero exclusive support means other candidate parents cover every child
    sample this parent supports. It is an ambiguity flag, not proof of a false
    match or a reason to reject every redundant parent automatically.
    """
    result=pooled_support(pa,na,la,pb,nb,lb,max_distance,**kwargs)
    pa,pb,na,nb=map(lambda x:np.array(x,dtype=float,copy=True),(pa,pb,na,nb))
    la,lb=np.asarray(la),np.asarray(lb)
    na/=np.linalg.norm(na,axis=1)[:,None];nb/=np.linalg.norm(nb,axis=1)[:,None]
    pairs=cKDTree(pa).sparse_distance_matrix(cKDTree(pb),max_distance,output_type='coo_matrix')
    ia,ib=pairs.row,pairs.col
    keep=np.einsum('ij,ij->i',na[ia],nb[ib])>=np.cos(np.deg2rad(kwargs.get('max_angle_degrees',60.)))
    ia,ib=ia[keep],ib[keep]
    for child in result['children']:
        target=child['child_group'];parents=child['parent_groups']
        sets={a:set(map(int,ib[(la[ia]==a)&(lb[ib]==target)])) for a in parents}
        redundant=0
        for link in [l for l in result['links'] if l['child_group']==target]:
            a=link['parent_group']
            others=set().union(*(sets[b] for b in parents if b!=a))
            exclusive=len(sets[a]-others)
            link['exclusive_child_support_samples']=exclusive
            link['exclusive_child_support_fraction']=exclusive/child['child_samples']
            link['redundant_support']=exclusive==0
            redundant+=exclusive==0
        child['redundant_candidate_parents']=redundant
    return result


def controlled_examples():
    """Known correspondences are stipulated for toy cases, not inferred physics."""
    xy=np.array([[x,y,0.] for x in (0.,1.,2.) for y in (0.,1.,2.)])
    n=lambda count:np.tile([0.,0.,1.],(count,1))
    cases=[]
    # Parent 0 persists. Parent 1 belongs to a different parallel sheet that
    # disappears in this stipulated example. The target supplies no velocity
    # or ancestry cue that lets the geometric rule infer that distinction.
    for name,offset,opposite in [('near_parallel_distractor',.1,False),
                                 ('far_parallel_distractor',1.,False),
                                 ('opposite_normal_distractor',.1,True)]:
        pa=np.vstack((xy,xy+[0,0,offset]));na=n(len(pa))
        if opposite:na[len(xy):]*=-1
        result=support_evidence(pa,na,np.repeat([0,1],len(xy)),xy,n(len(xy)),np.zeros(len(xy),int),.2)
        accepted=sorted(l['parent_group'] for l in result['links'] if l['accepted'])
        truth=[0]
        cases.append(dict(case=name,expected_parent_groups=truth,accepted_parent_groups=accepted,
                          unexpected_parent_groups=sorted(set(accepted)-set(truth)),
                          missed_parent_groups=sorted(set(truth)-set(accepted)),
                          redundant_accepted_parents=sum(l['accepted'] and l['redundant_support'] for l in result['links'])))
    pa=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[3.,0.,0.]])
    result=support_evidence(pa,n(4),[0,0,1,1],pa,n(4),[0]*4,.2)
    accepted=sorted(l['parent_group'] for l in result['links'] if l['accepted'])
    cases.append(dict(case='complementary_patch_support',expected_parent_groups=[0,1],accepted_parent_groups=accepted,
                      unexpected_parent_groups=sorted(set(accepted)-{0,1}),missed_parent_groups=sorted({0,1}-set(accepted)),
                      redundant_accepted_parents=sum(l['accepted'] and l['redundant_support'] for l in result['links'])))
    return cases
