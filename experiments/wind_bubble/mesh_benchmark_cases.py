"""Known curved, gapped and nearby-sheet inputs; truth is evaluation-only."""
import numpy as np
from scipy.spatial import cKDTree


def cases():
    result=[];count=1600;i=np.arange(count);z=1-2*(i+.5)/count;phi=i*np.pi*(3-np.sqrt(5))
    sphere=np.column_stack((np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z))
    dx=float(np.median(cKDTree(sphere).query(sphere,k=2)[0][:,1]))
    for name,keep,area in [('sphere',np.ones(count,bool),4*np.pi),('sphere_cap_hole',z<=.6,2*np.pi*1.6)]:
        p=sphere[keep];result.append(dict(name=name,positions=p,normals=p.copy(),labels=np.zeros(len(p),int),truth_sheet=np.zeros(len(p),int),dx=dx,reference_area=area))
    grid=np.array([[x,y,0.] for x in np.arange(-8,9) for y in np.arange(-8,9)])
    grid=grid[np.linalg.norm(grid[:,:2],axis=1)>=3]
    result.append(dict(name='planar_hole',positions=grid,normals=np.tile([0,0,1.],(len(grid),1)),labels=np.zeros(len(grid),int),truth_sheet=np.zeros(len(grid),int),dx=1.,reference_area=None))
    sheet=np.array([[x,y,0.] for x in range(9) for y in range(9)])
    p=np.vstack((sheet,sheet+[.35,.35,.10]));truth=np.repeat([0,1],len(sheet));normals=np.tile([0,0,1.],(len(p),1))
    for name,labels in [('near_sheets_separate_groups',truth),('near_sheets_same_group_counterexample',np.zeros(len(p),int))]:
        result.append(dict(name=name,positions=p,normals=normals,labels=labels,truth_sheet=truth,dx=1.,reference_area=None))
    return result


def evaluate_mesh(case,faces):
    p=case['positions'];f=np.asarray(faces,int).reshape(-1,3);tri=p[f]
    area=float(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2)
    # Evaluate ten barycentric interior/edge locations per triangle. This is a
    # sampled intrusion check, not an exact continuous triangle-hole intersection.
    weights=np.array([[a,b,1-a-b] for a in [0.,1/3,2/3,1.] for b in [0.,1/3,2/3,1.] if a+b<=1+1e-12])
    probes=np.einsum('kj,fjd->fkd',weights,tri)
    intrusion=np.zeros(len(f),bool)
    if case['name']=='sphere_cap_hole' and len(f):
        unit=probes/np.linalg.norm(probes,axis=2)[:,:,None];intrusion=(unit[:,:,2]>.62).any(axis=1)
    if case['name']=='planar_hole' and len(f):intrusion=(np.linalg.norm(probes[:,:,:2],axis=2)<2.5).any(axis=1)
    mixed=int((np.ptp(case['truth_sheet'][f],axis=1)>0).sum()) if len(f) else 0
    return dict(faces=len(f),used_vertices=len(np.unique(f)),input_vertices=len(p),mesh_area=area,
                area_fraction=area/case['reference_area'] if case['reference_area'] else None,
                sampled_hole_intrusion_faces=int(intrusion.sum()),cross_stipulated_sheet_faces=mixed)
