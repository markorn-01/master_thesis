"""Independent face-geometry, vertex-readback and PLY checks for the prototype."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np


def verify(output):
    meta=json.loads((output/'metadata.json').read_text())
    for p,h in {**meta['input_sha256'],**meta['source_sha256']}.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    source_path=next(Path(p) for p in meta['input_sha256'] if 'surfaces_' in p)
    group_path=next(Path(p) for p in meta['input_sha256'] if 'groups_' in p)
    with np.load(source_path) as a:source={k:a[k].copy() for k in a.files}
    with np.load(group_path) as a:labels=a['native_cells_a60_labels']
    checked=[]
    for name,report in zip(['mesh_edge_1p5.npz','mesh_edge_2p0.npz'],meta['reports']):
        with np.load(output/name) as a:data={k:a[k].copy() for k in a.files}
        for field in ['positions','normals','mach']:np.testing.assert_array_equal(data[field],source[field])
        np.testing.assert_array_equal(data['labels'],labels)
        p=data['positions'].astype(float);n=data['normals'].astype(float);n/=np.linalg.norm(n,axis=1)[:,None];f=data['faces']
        assert f.ndim==2 and f.shape[1]==3 and f.min()>=0 and f.max()<len(p)
        assert len({tuple(sorted(face)) for face in f})==len(f)
        assert (labels[f]>=0).all() and (np.ptp(labels[f],axis=1)==0).all()
        np.testing.assert_array_equal(data['face_groups'],labels[f[:,0]])
        tri=p[f];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area2=np.linalg.norm(normal,axis=1)
        assert (area2>float(source['grid_spacing'])**2*1e-10).all()
        normal/=area2[:,None]
        assert (np.einsum('fij,fj->fi',n[f],normal)>=np.cos(np.deg2rad(30))-1e-12).all()
        angles=[];maximum=0
        for j in range(3):
            a,b=tri[:,j],tri[:,(j+1)%3];edge=b-a;length=np.linalg.norm(edge,axis=1);maximum=max(maximum,float(length.max()))
            assert (length<=report['max_edge_cells']*float(source['grid_spacing'])*(1+1e-12)).all()
            assert (np.einsum('ij,ij->i',n[f[:,j]],n[f[:,(j+1)%3]])>=np.cos(np.deg2rad(45))-1e-12).all()
            for idx in [j,(j+1)%3]:assert (np.abs(np.einsum('ij,ij->i',edge,n[f[:,idx]]))<=np.sin(np.deg2rad(20))*length+1e-12).all()
            e1=tri[:,(j+1)%3]-tri[:,j];e2=tri[:,(j+2)%3]-tri[:,j]
            angles.append(np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i',e1,e2)/(np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)),-1,1))))
        minimum=float(np.min(angles));assert minimum>=12-1e-10
        edges=Counter(tuple(sorted((int(a),int(b)))) for face in f for a,b in zip(face,np.roll(face,-1)))
        assert max(edges.values())<=2
        assert sum(v==1 for v in edges.values())==report['topology']['boundary_edges']
        checked.append(dict(file=name,faces=len(f),vertices_unchanged=True,cross_group_faces=0,
                            maximum_edge_cells=maximum/float(source['grid_spacing']),minimum_triangle_angle=minimum,
                            boundary_edges=sum(v==1 for v in edges.values()),nonmanifold_edges=0))
        if name=='mesh_edge_2p0.npz':ply_data=data
    with (output/'surface_mesh.ply').open() as stream:
        header=[]
        while True:
            line=stream.readline().strip();header.append(line)
            if line=='end_header':break
            assert line
        assert f"element vertex {len(ply_data['positions'])}" in header and f"element face {len(ply_data['faces'])}" in header
        for p in ply_data['positions']:np.testing.assert_array_equal(np.array(stream.readline().split()[:3],float),p)
        for face in ply_data['faces']:np.testing.assert_array_equal(np.array(stream.readline().split(),int),np.r_[3,face])
        assert stream.read()==''
    controls=json.loads((output/'controls.json').read_text())
    by_name={x['case']:x for x in controls}
    for name in ['wide_gap_same_group','near_sheets_separate_groups','opposite_normals_same_group']:assert by_name[name]['cross_stipulated_sheet_faces']==0
    assert by_name['near_sheets_same_group_counterexample']['cross_stipulated_sheet_faces']>0
    result=dict(passed=True,meshes=checked,ply_vertices_and_faces_match=True,
                known_same_group_parallel_sheet_bridging_preserved=True,global_self_intersections_checked=False,
                physical_surface_topology_validated=False,physical_diagnostics_changed=False)
    (output/'readback.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);verify(p.parse_args().output)
