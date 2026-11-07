"""Inherited motion controls plus curved-surface expansion counterexamples."""
import numpy as np
from experiments.astrolink.controlled_motion_cases import motion_cases


def expansion_cases():
    cases=motion_cases()
    directions=np.array([[x,y,z] for x in [-1.,0.,1.] for y in [-1.,0.,1.] for z in [-1.,0.,1.] if x or y or z])
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    def add(name,positions,normals,meaning):
        case=dict(name=name,times=np.array([0.,.2,1.2]),expected_links=[[10,20]],meaning=meaning)
        for i,(p,n) in enumerate(zip(positions,normals)):
            case[f'positions_{i}']=p;case[f'normals_{i}']=n;case[f'labels_{i}']=np.full(len(p),[0,10,20][i],int)
        cases.append(case)
    radii=[.5,.545,.77];centers=[np.zeros(3),np.array([.04,-.02,.01]),np.array([.24,-.12,.06])]
    add('translating_expanding_sphere',[c+r*directions for c,r in zip(centers,radii)],[directions.copy() for _ in range(3)],
        '26 symmetric directions; center and radius evolve linearly. No true center/radius is passed to prediction.')
    add('decelerating_expansion',[r*directions for r in [.5,.56,.59]],[directions.copy() for _ in range(3)],
        'Growth slows after t1; extrapolated constant extent-rate overshoots.')
    positions=[directions*np.array([rx,.5,.5]) for rx in [.5,.56,.86]]
    normals=[]
    for rx in [.5,.56,.86]:
        n=directions/np.array([rx,.5,.5]);normals.append(n/np.linalg.norm(n,axis=1)[:,None])
    add('anisotropic_expansion',positions,normals,
        'Only the x extent grows. Isotropic scaling cannot predict the full directional deformation.')
    return cases
