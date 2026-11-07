"""Experimental centroid-translation prediction from observed candidate matches."""
from __future__ import annotations
import numpy as np
from experiments.astrolink.support_evidence import support_evidence


def predict_positions(p0,n0,l0,p1,n1,l1,times,bootstrap_gate=.2):
    """Use only the first two observations; no truth or future positions.

    Estimate velocity only for unique, nonredundant accepted one-to-one links.
    Uninitialized groups retain their observed positions and are explicitly
    marked as geometric fallback. This is not persistent physical tracking.
    """
    times=np.asarray(times,float)
    if times.shape!=(3,) or not np.isfinite(times).all() or not (np.diff(times)>0).all():
        raise ValueError('Require three finite increasing times.')
    p0,p1=np.asarray(p0,float),np.asarray(p1,float);l0,l1=np.asarray(l0),np.asarray(l1)
    seed=support_evidence(p0,n0,l0,p1,n1,l1,bootstrap_gate)
    accepted=[x for x in seed['links'] if x['accepted']]
    predicted=p1.copy();estimates=[]
    for group in np.unique(l1[l1>=0]):
        incoming=[x for x in accepted if x['child_group']==group]
        estimate=dict(group=int(group),initialized=False,reason='no_unique_nonredundant_initial_link',velocity=None,
                      predecessor_group=None)
        if len(incoming)==1:
            link=incoming[0];outgoing=[x for x in accepted if x['parent_group']==link['parent_group']]
            if len(outgoing)==1 and not link['redundant_support']:
                parent=link['parent_group']
                velocity=(p1[l1==group].mean(axis=0)-p0[l0==parent].mean(axis=0))/(times[1]-times[0])
                predicted[l1==group]+=velocity*(times[2]-times[1])
                estimate.update(initialized=True,reason='observed_one_to_one_centroid_difference',
                                velocity=velocity.tolist(),predecessor_group=int(parent))
        estimates.append(estimate)
    return predicted,estimates,seed
