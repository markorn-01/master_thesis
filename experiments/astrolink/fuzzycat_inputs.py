"""Map one event's clustering ensemble to common catalog-row memberships."""
from __future__ import annotations
import numpy as np


def hierarchy_memberships(ordering_catalog,intervals,ids,n_points,include_root=False):
    ordering=np.asarray(ordering_catalog)
    intervals=np.asarray(intervals);ids=np.asarray(ids)
    if (ordering.ndim!=1 or not np.issubdtype(ordering.dtype,np.integer)
            or len(np.unique(ordering))!=len(ordering) or (ordering<0).any() or (ordering>=n_points).any()):
        raise ValueError('Require unique valid original catalog row indices.')
    if intervals.shape!=(len(ids),2) or len(set(map(str,ids)))!=len(ids):
        raise ValueError('Require unique hierarchy IDs and matching intervals.')
    result={}
    for interval,identifier in zip(intervals,ids):
        identifier=str(identifier)
        if not all(part.isdigit() for part in identifier.split('-')):raise ValueError('Invalid hierarchical ID.')
        start,end=map(int,interval)
        if not 0<=start<end<=len(ordering):raise ValueError('Invalid hierarchy interval.')
        if identifier=='1' and not include_root:continue
        result[identifier]=np.sort(ordering[start:end]).astype(np.int64)
    return result


def partition_memberships(labels):
    labels=np.asarray(labels)
    if labels.ndim!=1 or not np.issubdtype(labels.dtype,np.integer):raise ValueError('Require integer labels.')
    return {f'1-{j+1}':np.flatnonzero(labels==group).astype(np.int64)
            for j,group in enumerate(np.unique(labels[labels>=0]))}


def display_memberships(memberships_flat,threshold=.5):
    """Illustrative max-membership labels, retaining low values and ties separately."""
    m=np.asarray(memberships_flat,float)
    if m.ndim!=2 or not np.isfinite(m).all() or not 0<threshold<=1:raise ValueError('Invalid membership array/threshold.')
    labels=np.full(m.shape[1],-1,dtype=np.int32)
    if not len(m):return labels,np.zeros(m.shape[1])
    maximum=m.max(axis=0)
    ties=np.isclose(m,maximum[None,:],atol=1e-10,rtol=0).sum(axis=0)>1
    selected=(maximum>=threshold)&~ties
    labels[selected]=np.argmax(m[:,selected],axis=0)
    labels[(maximum>=threshold)&ties]=-3
    return labels,maximum


def sample_multiplicities(cluster_files,ordering,fuzzy_clusters,n_samples):
    """Independent counts used to audit the library's reciprocal weights."""
    samples=np.array([int(str(name).split('_')[0]) for name in cluster_files])
    if (samples<0).any() or (samples>=n_samples).any():raise ValueError('Sample index out of range.')
    counts=np.zeros((n_samples,len(fuzzy_clusters)),dtype=int)
    for j,(start,end) in enumerate(fuzzy_clusters):
        counts[:,j]=np.bincount(samples[np.asarray(ordering)[int(start):int(end)]],minlength=n_samples)
    reference=np.divide(1.,counts,out=np.zeros_like(counts,dtype=float),where=counts>0)
    return counts,reference
