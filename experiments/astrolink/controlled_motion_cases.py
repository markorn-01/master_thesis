"""Three-observation toy sequences with stipulated final correspondence."""
import numpy as np


def motion_cases():
    plane=np.array([[x,y,0.] for x in range(4) for y in range(4)])
    n=lambda p:np.tile([0.,0.,1.],(len(p),1))
    result=[]
    def add(name,p0,l0,p1,l1,p2,l2,truth,meaning,times=(0.,.2,1.2)):
        result.append(dict(name=name,positions_0=np.array(p0,float),positions_1=np.array(p1,float),positions_2=np.array(p2,float),
                           normals_0=n(p0),normals_1=n(p1),normals_2=n(p2),labels_0=np.array(l0,int),
                           labels_1=np.array(l1,int),labels_2=np.array(l2,int),times=np.array(times),
                           expected_links=[list(x) for x in truth],meaning=meaning))
    add('constant_translation',plane,[0]*16,plane+[0,0,.06],[10]*16,plane+[0,0,.36],[20]*16,[(10,20)],
        'Constant velocity .3; the short initial interval permits bootstrap, the longer final interval moves .3.')
    for name,z0,z1,z2,meaning in [
        ('passing_sheets',.18,.12,-.18,'Two synthetic sheets pass through; stipulated ancestry follows their motion, not nearest position.'),
        ('approaching_sheets',.42,.36,.06,'Two sheets approach without crossing; broad residual gates can still confuse nearby predictions.')]:
        stack=lambda z:np.vstack((plane+[0,0,-z],plane+[0,0,z]))
        add(name,stack(z0),[0]*16+[1]*16,stack(z1),[10]*16+[11]*16,stack(z2),[20]*16+[21]*16,[(10,20),(11,21)],meaning)
    add('abrupt_stop',plane,[0]*16,plane+[0,0,.06],[10]*16,plane+[0,0,.06],[20]*16,[(10,20)],
        'The sheet stops after the second observation; constant-velocity extrapolation is wrong.')
    add('group_split',plane,[0]*16,plane,[10]*16,plane,np.where(plane[:,0]<1.5,20,21),[(10,20),(10,21)],
        'A catalog group splits into equal halves; translation does not resolve branching.')
    halves=np.where(plane[:,0]<1.5,0,1)
    add('fragment_pooling',plane,halves,plane,halves+10,plane,[20]*16,[(10,20),(11,20)],
        'Two catalog fragments form one group; bootstrap retains separate fragments.')
    two=np.vstack((plane,plane+[0,0,.1]))
    add('unresolved_nearby_disappearance',two,[0]*16+[1]*16,two,[10]*16+[11]*16,plane,[20]*16,[(10,20)],
        'Nearby aligned sheets are ambiguous at bootstrap; parent 11 disappears. Motion is unavailable.')
    partial=plane[plane[:,0]>=2]
    add('changing_sample_coverage',plane,[0]*16,partial,[10]*8,plane,[20]*16,[(10,20)],
        'A stationary sheet is partly sampled at observation 1. Centroid change falsely resembles motion.')
    add('no_motion_initialization',plane,[0]*16,plane+[0,0,.3],[10]*16,plane+[0,0,.6],[20]*16,[(10,20)],
        'Even bootstrap displacement exceeds the gate; no velocity may be supplied from truth.',times=(0.,.2,.4))
    sphere=np.vstack((np.eye(3),-np.eye(3)))
    add('symmetric_expansion',sphere*.5,[0]*6,sphere*.56,[10]*6,sphere*.86,[20]*6,[(10,20)],
        'A symmetric six-sample sphere expands about a fixed center; centroid translation cannot predict radial growth.')
    return result
