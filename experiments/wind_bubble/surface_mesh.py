"""Conservative display-only local triangulation of oriented surface samples.

Vertices are original samples. This is not global surface reconstruction,
physical shock identification, or a replacement for diagnostic area estimates.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree,Delaunay,QhullError


def mesh_topology(faces):
    faces=np.asarray(faces,int).reshape(-1,3)
    edges=Counter(tuple(sorted((int(a),int(b)))) for f in faces for a,b in zip(f,np.roll(f,-1)))
    directed=Counter((int(a),int(b)) for f in faces for a,b in zip(f,np.roll(f,-1)))
    return dict(edges=len(edges),boundary_edges=sum(v==1 for v in edges.values()),
                nonmanifold_edges=sum(v>2 for v in edges.values()),
                inconsistent_shared_edge_winding=sum(v==2 and (directed[a,b]==2 or directed[b,a]==2) for (a,b),v in edges.items()))


def triangulate_samples(positions,normals,labels,dx,max_edge_cells=2.,normal_angle=45.,tangent_angle=20.,face_angle=30.,min_angle=12.):
    p=np.asarray(positions,float);n=np.asarray(normals,float);labels=np.asarray(labels)
    if p.ndim!=2 or p.shape[1]!=3 or n.shape!=p.shape or labels.shape!=(len(p),):raise ValueError('Require positions/normals (N,3), labels (N,).')
    if not np.isfinite(p).all() or not np.isfinite(n).all() or (np.linalg.norm(n,axis=1)<=0).any():raise ValueError('Require finite geometry and nonzero normals.')
    if not np.isfinite(dx) or dx<=0 or not np.isfinite(max_edge_cells) or max_edge_cells<=0:raise ValueError('Require positive finite lengths.')
    if not all(0<a<90 for a in [normal_angle,tangent_angle,face_angle]) or not 0<min_angle<60:raise ValueError('Invalid angle thresholds.')
    n=n/np.linalg.norm(n,axis=1)[:,None];limit=max_edge_cells*dx;votes=Counter();qhull_failures=0;projected_duplicates=0
    # Tiny deterministic perturbation is used only in projected triangle proposals
    # to resolve cocircular grid diagonals. Original 3D mesh vertices never move.
    rank=np.empty(len(p),int);rank[np.lexsort((p[:,2],p[:,1],p[:,0]))]=np.arange(len(p))
    proposal=p+dx*1e-7*np.sin((rank[:,None]+1)*np.array([1.3,2.7,3.9]))
    normal_cos=np.cos(np.deg2rad(normal_angle));tangent_sin=np.sin(np.deg2rad(tangent_angle));face_cos=np.cos(np.deg2rad(face_angle))
    for group in np.unique(labels[labels>=0]):
        rows=np.flatnonzero(labels==group);tree=cKDTree(p[rows]);neighborhoods=tree.query_ball_point(p[rows],limit)
        for local,neighbors in enumerate(neighborhoods):
            center=rows[local];neighbors=rows[np.asarray(neighbors,int)];delta=p[neighbors]-p[center];length=np.linalg.norm(delta,axis=1)
            # Require tangent compatibility at both endpoints as well as signed normals.
            tangent=(length==0)|((np.abs(delta@n[center])<=tangent_sin*length)&(np.abs(np.einsum('ij,ij->i',delta,n[neighbors]))<=tangent_sin*length))
            neighbors=neighbors[tangent&((n[neighbors]@n[center])>=normal_cos)]
            if len(neighbors)<3:continue
            axis=np.eye(3)[np.argmin(np.abs(n[center]))];u=np.cross(n[center],axis);u/=np.linalg.norm(u);v=np.cross(n[center],u)
            projected=(p[neighbors]-p[center])@np.column_stack((u,v))
            _,unique=np.unique(projected,axis=0,return_index=True)
            if len(unique)!=len(projected):projected_duplicates+=1;continue # Skip exact projection duplicates rather than silently dropping a point.
            projected=(proposal[neighbors]-proposal[center])@np.column_stack((u,v))
            try:triangles=Delaunay(projected).simplices
            except QhullError:qhull_failures+=1;continue
            for triangle in triangles:
                face=neighbors[triangle]
                if center in face:votes[tuple(sorted(map(int,face)))]+=1
    reasons=Counter();accepted=[]
    for key,count in sorted(votes.items()):
        if count!=3:reasons['incomplete_three_vertex_agreement']+=1;continue
        f=np.array(key);points=p[f];norms=n[f]
        pairs=np.array([[0,1],[1,2],[2,0]]);d=points[pairs[:,1]]-points[pairs[:,0]];length=np.linalg.norm(d,axis=1)
        if (length>limit*(1+1e-12)).any() or (length<=dx*1e-10).any():reasons['edge_length']+=1;continue
        if (np.einsum('ij,ij->i',norms[pairs[:,0]],norms[pairs[:,1]])<normal_cos).any():reasons['normal_disagreement']+=1;continue
        if ((np.abs(np.einsum('ij,ij->i',d,norms[pairs[:,0]]))>tangent_sin*length)|(np.abs(np.einsum('ij,ij->i',d,norms[pairs[:,1]]))>tangent_sin*length)).any():reasons['nontangent_edge']+=1;continue
        cross=np.cross(points[1]-points[0],points[2]-points[0]);area2=np.linalg.norm(cross)
        if area2<=dx*dx*1e-10:reasons['degenerate']+=1;continue
        face_normal=cross/area2
        if face_normal@norms.mean(axis=0)<0:f=f[[0,2,1]];face_normal=-face_normal
        if (norms@face_normal<face_cos).any():reasons['face_normal']+=1;continue
        # Interior angles opposite the three edge lengths.
        cosines=[(length[(i+1)%3]**2+length[(i+2)%3]**2-length[i]**2)/(2*length[(i+1)%3]*length[(i+2)%3]) for i in range(3)]
        angles=np.rad2deg(np.arccos(np.clip(cosines,-1,1)))
        if angles.min()<min_angle:reasons['small_angle']+=1;continue
        accepted.append(f)
    faces=np.asarray(accepted,int).reshape(-1,3)
    # Reject all faces incident to an over-subscribed edge rather than choosing a greedy winner.
    counts=Counter(tuple(sorted((int(a),int(b)))) for f in faces for a,b in zip(f,np.roll(f,-1)))
    keep=np.array([all(counts[tuple(sorted((int(a),int(b))))]<=2 for a,b in zip(f,np.roll(f,-1))) for f in faces],bool)
    reasons['nonmanifold_edge_faces']=int((~keep).sum());faces=faces[keep]
    report=dict(input_samples=len(p),retained_samples=int((labels>=0).sum()),candidate_faces=len(votes),faces=len(faces),
                used_vertices=len(np.unique(faces)),qhull_failures=qhull_failures,projected_duplicate_neighborhoods=projected_duplicates,
                rejected=dict(reasons),topology=mesh_topology(faces),max_edge_cells=max_edge_cells,
                normal_angle=normal_angle,tangent_angle=tangent_angle,face_angle=face_angle,min_angle=min_angle,
                vertices_moved=False,proposal_tie_perturbation_cells=1e-7,physical_area_recomputed=False)
    return faces,report
