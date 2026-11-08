"""Optional Open3D ball-pivoting candidate with explicit original-index guards."""
from __future__ import annotations
from collections import Counter
import numpy as np
from experiments.wind_bubble.surface_mesh import mesh_topology


def guard_faces(positions,normals,labels,faces,dx,max_edge_cells=2.):
    """Same geometric limits as the local prototype, without its proposal votes."""
    p=np.asarray(positions,float);n=np.asarray(normals,float);n=n/np.linalg.norm(n,axis=1)[:,None];labels=np.asarray(labels)
    faces=np.asarray(faces,int).reshape(-1,3);accepted=[];rejected=Counter();seen=set()
    for f in faces:
        key=tuple(sorted(map(int,f)))
        if len(set(key))!=3 or key in seen:rejected['duplicate_or_invalid']+=1;continue
        seen.add(key)
        if (labels[f]<0).any() or np.ptp(labels[f]):rejected['group']+=1;continue
        tri=p[f];norms=n[f];pairs=np.array([[0,1],[1,2],[2,0]]);edges=tri[pairs[:,1]]-tri[pairs[:,0]];length=np.linalg.norm(edges,axis=1)
        if (length>max_edge_cells*dx*(1+1e-12)).any() or (length<=dx*1e-10).any():rejected['edge_length']+=1;continue
        if (np.einsum('ij,ij->i',norms[pairs[:,0]],norms[pairs[:,1]])<np.cos(np.deg2rad(45))).any():rejected['normal']+=1;continue
        if any((np.abs(np.einsum('ij,ij->i',edges,norms[pairs[:,j]]))>np.sin(np.deg2rad(20))*length).any() for j in [0,1]):rejected['tangent']+=1;continue
        cross=np.cross(tri[1]-tri[0],tri[2]-tri[0]);area2=np.linalg.norm(cross)
        if area2<=dx*dx*1e-10:rejected['degenerate']+=1;continue
        normal=cross/area2
        if normal@norms.mean(0)<0:f=f[[0,2,1]];normal=-normal
        if (norms@normal<np.cos(np.deg2rad(30))).any():rejected['face_normal']+=1;continue
        angles=[]
        for i in range(3):
            a=tri[(i+1)%3]-tri[i];b=tri[(i+2)%3]-tri[i]
            angles.append(np.rad2deg(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))
        if min(angles)<12:rejected['minimum_angle']+=1;continue
        accepted.append(f)
    f=np.asarray(accepted,int).reshape(-1,3)
    counts=Counter(tuple(sorted((int(a),int(b)))) for face in f for a,b in zip(face,np.roll(face,-1)))
    keep=np.array([all(counts[tuple(sorted((int(a),int(b))))]<=2 for a,b in zip(face,np.roll(face,-1))) for face in f],bool)
    rejected['nonmanifold_edge_faces']=int((~keep).sum());return f[keep],dict(rejected)


def ball_pivot_samples(positions,normals,labels,dx,radius_factors=(.8,1.,1.4)):
    import open3d as o3d
    p=np.asarray(positions,float);n=np.asarray(normals,float);labels=np.asarray(labels)
    if p.ndim!=2 or p.shape[1]!=3 or n.shape!=p.shape or labels.shape!=(len(p),):raise ValueError('Require matching finite (N,3) geometry and (N,) labels.')
    if not np.isfinite(p).all() or not np.isfinite(n).all() or (np.linalg.norm(n,axis=1)<=0).any():raise ValueError('Require finite points and nonzero normals.')
    radii=np.asarray(radius_factors,float)
    if not np.isfinite(dx) or dx<=0 or radii.ndim!=1 or not len(radii) or not np.isfinite(radii).all() or (radii<=0).any():raise ValueError('Require positive finite spacing and radii.')
    n=n/np.linalg.norm(n,axis=1)[:,None];raw=[]
    for group in np.unique(labels[labels>=0]):
        rows=np.flatnonzero(labels==group)
        if len(rows)<3:continue
        cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(p[rows]);cloud.normals=o3d.utility.Vector3dVector(n[rows])
        mesh=o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud,o3d.utility.DoubleVector(np.array(radius_factors)*dx))
        # Refuse a result that moves/reorders vertices rather than guessing a map.
        np.testing.assert_array_equal(np.asarray(mesh.vertices),p[rows])
        raw.extend(rows[np.asarray(mesh.triangles,dtype=int)])
    raw=np.asarray(raw,int).reshape(-1,3);faces,rejected=guard_faces(p,n,labels,raw,dx)
    return faces,dict(method='Open3D ball pivoting plus original-prototype geometric guards',open3d_version=o3d.__version__,
                      radius_factors=list(radius_factors),raw_faces=len(raw),faces=len(faces),rejected=rejected,
                      topology=mesh_topology(faces),vertices_moved=False),raw
