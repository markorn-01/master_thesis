"""Stipulated surface correspondences for auditing the geometric baseline.

Truth labels are construction metadata, never passed into geometric matching.
These are synthetic catalog tests, not solutions of the hydrodynamic equations.
"""
from __future__ import annotations
import numpy as np


def tracking_cases():
    plane=np.array([[x,y,0.] for x in range(4) for y in range(4)])
    normals=lambda count:np.tile([0.,0.,1.],(count,1))
    cases=[]
    def add(name,pa,la,pb,lb,truth,meaning,na=None,nb=None):
        cases.append(dict(name=name,positions_a=np.array(pa,float),normals_a=normals(len(pa)) if na is None else na,
                          labels_a=np.array(la,int),positions_b=np.array(pb,float),
                          normals_b=normals(len(pb)) if nb is None else nb,labels_b=np.array(lb,int),
                          expected_links=[list(pair) for pair in truth],meaning=meaning))
    add('translation_within_gate',plane,[0]*16,plane+[0,0,.08],[10]*16,[(0,10)],
        'One sheet translates normally by .08; its stipulated identity is retained.')
    add('translation_beyond_gate',plane,[0]*16,plane+[0,0,.30],[10]*16,[(0,10)],
        'The same sheet moves .30, beyond the .20 baseline gate; a missed link is expected.')
    # New positions have no exact coordinate matches with the previous catalog.
    add('fresh_sampling',plane,[0]*16,plane+[.03,-.02,.08],[10]*16,[(0,10)],
        'The sheet persists but is sampled at different positions; catalog row identity is irrelevant.')
    pa=np.vstack((plane+[0,0,-.18],plane+[0,0,.18]))
    pb=np.vstack((plane+[0,0,-.06],plane+[0,0,.06]))
    add('approaching_aligned_sheets',pa,[0]*16+[1]*16,pb,[10]*16+[11]*16,[(0,10),(1,11)],
        'Two sheets approach without crossing or merging. Their ancestry is stipulated by construction.')
    na=normals(32);na[16:]*=-1
    add('approaching_opposite_normals',pa,[0]*16+[1]*16,pb,[10]*16+[11]*16,[(0,10),(1,11)],
        'Same approach, but opposite signed normals distinguish the sheets.',na=na,nb=na.copy())
    split=np.where(plane[:,0]<1.5,10,11)
    add('one_group_splits_into_two',plane,[0]*16,plane,split,[(0,10),(0,11)],
        'One catalog group becomes two groups covering its two halves; not a physical shock-splitting model.')
    merge=np.where(plane[:,0]<1.5,0,1)
    add('two_fragments_form_one_group',plane,merge,plane,[10]*16,[(0,10),(1,10)],
        'Two catalog fragments become one group; tests representation changes, not physical merger dynamics.')
    for name,offset in [('nearby_sheet_disappears',.10),('distant_sheet_disappears',1.)]:
        add(name,np.vstack((plane,plane+[0,0,offset])),[0]*16+[1]*16,plane+[0,0,.03],[10]*16,[(0,10)],
            'Parent 0 persists and parent 1 disappears. Parent 1 must not be attached to the surviving child.')
    return cases


def evaluate_links(result,expected):
    accepted={tuple((l['parent_group'],l['child_group'])) for l in result['links'] if l['accepted']}
    truth={tuple(x) for x in expected}
    return dict(accepted_links=[list(x) for x in sorted(accepted)],expected_links=[list(x) for x in sorted(truth)],
                correct_links=[list(x) for x in sorted(accepted&truth)],
                false_links=[list(x) for x in sorted(accepted-truth)],missed_links=[list(x) for x in sorted(truth-accepted)],
                exact_correspondence=accepted==truth)
