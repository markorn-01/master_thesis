"""Independent geometry/readback checks of the frozen three-event display."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
from experiments.wind_bubble.verify_mesh_comparison import check_geometry
from experiments.wind_bubble.surface_mesh import mesh_topology


def face_components(faces):
    faces=np.asarray(faces,int).reshape(-1,3);parent=np.arange(len(faces));owner={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,face in enumerate(faces):
        for a,b in zip(face,np.roll(face,-1)):
            key=tuple(sorted((int(a),int(b))))
            if key in owner:
                ra,rb=find(i),find(owner[key])
                if ra!=rb:parent[ra]=rb
            else:owner[key]=i
    sizes=np.unique([find(i) for i in range(len(faces))],return_counts=True)[1]
    return dict(edge_connected_components=len(sizes),largest_component_faces=int(max(sizes,default=0)),
                singleton_face_components=int((sizes==1).sum()))


def verify(output):
    meta=json.loads((output/'metadata.json').read_text())
    for p,h in {**meta['input_sha256'],**meta['source_sha256']}.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    assert (meta['radius_factor'],meta['max_edge_cells'],meta['cutoff'])==(1.,2.,20)
    rows=list(csv.DictReader((output/'summary.csv').open()));assert len(rows)==3
    group_rows=list(csv.DictReader((output/'group_statistics.csv').open()));assert len(group_rows)==sum(len(r['group_statistics']) for r in meta['records'])
    root=Path(meta['root']);partitions=Path(meta['partitions']);pair_checks=0;segment_checks=0;counts=[]
    for record,row in zip(meta['records'],rows):
        event=record['event']
        for key,value in row.items():assert value==str(record[key])
        with np.load(root/'online'/f'surfaces_{event:04d}.npz') as s,np.load(partitions/f'groups_n128_event{event:04d}.npz') as g,np.load(output/f'mesh_{event:04d}.npz') as m:
            for field in ['positions','normals','mach','time','grid_spacing']:np.testing.assert_array_equal(m[field],s[field])
            labels=m['labels'];np.testing.assert_array_equal(labels,g['native_cells_a60_labels'])
            p=m['positions'].astype(float);f=m['faces'];dx=float(s['grid_spacing']);raw=m['raw_faces']
            check_geometry(p,m['normals'],labels,f,dx)
            raw_keys={tuple(sorted(x)) for x in raw};assert all(tuple(sorted(x)) in raw_keys for x in f)
            assert len(f)==record['faces'] and len(raw)==record['raw_faces']
            topo=mesh_topology(f)
            for key,value in topo.items():assert value==record[key]
            assert topo['nonmanifold_edges']==topo['inconsistent_shared_edge_winding']==0
            assert int((labels<0).sum())==record['excluded_samples']
            for group in record['group_statistics']:
                group_id=group['group'];selected=f[labels[f[:,0]]==group_id]
                assert group['samples']==int((labels==group_id).sum()) and group['used_vertices']==len(np.unique(selected)) and group['faces']==len(selected)
                line=next(x for x in group_rows if int(x['event'])==event and int(x['group'])==group_id)
                for key,value in line.items():assert value==str(group[key])
            for pair in record['separation']:
                a=p[labels==pair['group_a']];b=p[labels==pair['group_b']]
                # Independent dense pairwise check, chunked to bound memory.
                minimum=min(float(cdist(a[i:i+256],b).min()) for i in range(0,len(a),256))
                np.testing.assert_allclose(minimum,pair['minimum_vertex_distance'],atol=1e-12,rtol=1e-12)
                bounds=[]
                for group in [pair['group_a'],pair['group_b']]:
                    tri=p[f[labels[f[:,0]]==group]]
                    bounds.append(float(np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2).max()) if len(tri) else 0.)
                np.testing.assert_allclose(sum(bounds),pair['sum_maximum_triangle_edges'],atol=1e-12,rtol=1e-12)
                assert pair['separation_bound_passed']==(minimum>sum(bounds));pair_checks+=1
            if record['groups']==1:assert record['separation']==[]
            tri=p[f];offset=tri[:,:,2]-.5;coplanar=np.all(offset==0,axis=1)
            crossing=((offset>0).any(axis=1)&(offset<0).any(axis=1))|((np.sum(offset==0,axis=1)>=2)&~coplanar)
            np.testing.assert_array_equal(m['segment_face_rows'],np.flatnonzero(crossing))
            assert int(coplanar.sum())==record['coplanar_triangles_skipped']
            assert len(m['plane_segments'])==record['central_plane_segments']
            for segment,index in zip(m['plane_segments'],m['segment_face_rows']):
                points=np.column_stack((segment,np.full(2,.5)))
                matrix=np.vstack((tri[index].T,np.ones(3)))
                for point in points:
                    weights,_,_,_=np.linalg.lstsq(matrix,np.r_[point,1.],rcond=None)
                    assert (weights>=-1e-8).all() and (weights<=1+1e-8).all()
                    np.testing.assert_allclose(weights@tri[index],point,atol=1e-10,rtol=1e-10)
                segment_checks+=1
            if event==16:
                with np.load('outputs/two_bubbles_mesh_method_comparison/bubble_ball_single.npz') as previous:np.testing.assert_array_equal(f,previous['faces'])
            counts.append(dict(event=event,faces=len(f),cross_group_faces=0,boundary_edges=topo['boundary_edges'],
                               **face_components(f),group_connectivity=[dict(group=int(group),**face_components(f[labels[f[:,0]]==group])) for group in np.unique(labels[labels>=0])]))
    for name,size in [('temporal_mesh_comparison.png',(2280,1680)),('central_mesh_sections.png',(1560,1950))]:
        with Image.open(output/name) as image:assert image.size==size;image.verify()
    report=dict(passed=True,events_checked=3,frozen_settings=True,meshes=counts,
                cross_group_separation_bounds_independently_checked=pair_checks,mesh_plane_segments_checked=segment_checks,
                previous_event16_mesh_reproduced=True,original_vertices_and_values_unchanged=True,
                physical_openings_validated=False,within_group_topology_validated=False,temporal_flicker_validated=False)
    (output/'readback.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);verify(p.parse_args().output)
