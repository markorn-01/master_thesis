"""Display-only oriented disks, inspired by surfels (Pfister et al., 2000).

No connectivity, reconstruction, EWA filtering, or change to physical normals.
"""
import numpy as np


def disk_vertices(positions,normals,radius,sides=10):
    p=np.asarray(positions,float);n=np.asarray(normals,float)
    if (p.ndim!=2 or p.shape[1]!=3 or n.shape!=p.shape or not np.isfinite(p).all()
            or not np.isfinite(n).all() or (np.linalg.norm(n,axis=1)<=0).any()
            or not np.isfinite(radius) or radius<=0 or not isinstance(sides,int) or sides<3):
        raise ValueError('Require finite Nx3 geometry, nonzero normals and positive disk radius.')
    n=n/np.linalg.norm(n,axis=1)[:,None]
    reference=np.eye(3)[np.argmin(np.abs(n),axis=1)]
    u=np.cross(n,reference);u/=np.linalg.norm(u,axis=1)[:,None]
    v=np.cross(n,u);angles=np.arange(sides)*2*np.pi/sides
    return p[:,None,:]+radius*(np.cos(angles)[None,:,None]*u[:,None,:]+np.sin(angles)[None,:,None]*v[:,None,:])
