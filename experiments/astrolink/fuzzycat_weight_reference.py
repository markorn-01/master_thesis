"""NumPy audit of saved FuzzyCat memberships; does not change the library."""
from __future__ import annotations
import numpy as np


def reconstruct_memberships(cluster_files,clusters,ordering,fuzzy_clusters,weights,stabilities,n_samples,n_points):
    """Reconstruct documented reciprocal-weight accumulation from saved inputs.

    Passing archived integer weights checks the unmodified library output.
    Passing floating reciprocal weights is a diagnostic counterfactual for the
    same selected clusters, not a validated replacement physical estimator.
    """
    count=len(fuzzy_clusters)
    membership=np.zeros((count,n_points));correction=membership.copy()
    hierarchy=np.zeros((count,count));which=np.full(len(cluster_files),-1,dtype=int)
    for j,(start,end) in enumerate(fuzzy_clusters):which[np.asarray(ordering)[int(start):int(end)]]=j
    bases=[str(name).removesuffix('.npy')+'-' for name in cluster_files]
    for i,name in enumerate(cluster_files):
        group=which[i]
        if group<0:continue
        sample=int(str(name).split('_')[0]);rows=clusters[str(name).removesuffix('.npy')]
        membership[group,rows]+=weights[sample,group]
        parents=np.unique([which[j] for j,base in enumerate(bases) if bases[i].startswith(base)])
        for parent in parents:
            if parent>=0 and parent!=group:
                weight=weights[sample,group]*weights[sample,parent]
                correction[parent,rows]+=weight;hierarchy[parent,group]+=weight
    if count:
        normalization=n_samples*np.asarray(stabilities)
        membership/=normalization[:,None];correction/=normalization[:,None];hierarchy/=normalization[None,:]
    return membership,membership-correction,hierarchy
