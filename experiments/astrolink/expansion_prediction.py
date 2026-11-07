"""Experimental translation plus isotropic expansion from catalog moments."""
from __future__ import annotations
import numpy as np
from experiments.astrolink.motion_prediction import predict_positions


def rms_extent(positions):
    p=np.asarray(positions,float)
    return float(np.sqrt(np.mean(np.sum((p-p.mean(axis=0))**2,axis=1))))


def predict_expanding_positions(p0,n0,l0,p1,n1,l1,times,bootstrap_gate=.2):
    """Extrapolate centroid and RMS extent using only the first two catalogs.

    Extent is a sample-cloud moment, not necessarily a physical shock radius.
    Positive isotropic scaling preserves normal directions. No matching against
    future positions is used to choose the scale, velocity or fallback.
    """
    predicted,estimates,seed=predict_positions(p0,n0,l0,p1,n1,l1,times,bootstrap_gate)
    p0,p1=np.asarray(p0,float),np.asarray(p1,float);l0,l1=np.asarray(l0),np.asarray(l1)
    dt0=float(times[1]-times[0]);dt1=float(times[2]-times[1])
    for estimate in estimates:
        estimate.update(expansion_initialized=False,extent_0=None,extent_1=None,extent_rate=None,scale=None,
                        expansion_reason='no_motion_initialization')
        if not estimate['initialized']:continue
        group=estimate['group'];parent=estimate['predecessor_group'];mask=l1==group
        r0,r1=rms_extent(p0[l0==parent]),rms_extent(p1[mask])
        estimate.update(extent_0=r0,extent_1=r1)
        if min(r0,r1)<=1e-12:
            estimate['expansion_reason']='translation_fallback_unresolved_extent';continue
        rate=(r1-r0)/dt0;future=r1+rate*dt1
        estimate['extent_rate']=rate
        if future<=1e-12:
            estimate['expansion_reason']='translation_fallback_nonpositive_predicted_extent';continue
        scale=future/r1;center=p1[mask].mean(axis=0)
        predicted[mask]=center+np.asarray(estimate['velocity'])*dt1+scale*(p1[mask]-center)
        estimate.update(expansion_initialized=True,scale=scale,expansion_reason='observed_centroid_and_rms_extent')
    return predicted,estimates,seed
