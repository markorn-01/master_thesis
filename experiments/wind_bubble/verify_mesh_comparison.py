"""Readback of method comparison, control metrics and unchanged bubble vertices."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
from experiments.wind_bubble.mesh_benchmark_cases import cases,evaluate_mesh
from experiments.wind_bubble.surface_mesh import mesh_topology


def check_geometry(p,n,labels,f,dx):
    p=np.asarray(p,float);n=np.asarray(n,float);n/=np.linalg.norm(n,axis=1)[:,None];f=np.asarray(f,int).reshape(-1,3)
    assert len({tuple(sorted(x)) for x in f})==len(f)
    if not len(f):return
    assert f.min()>=0 and f.max()<len(p) and (labels[f]>=0).all() and (np.ptp(labels[f],axis=1)==0).all()
    tri=p[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area2=np.linalg.norm(cross,axis=1)
    assert (area2>dx*dx*1e-10).all();cross/=area2[:,None]
    assert (np.einsum('fij,fj->fi',n[f],cross)>=np.cos(np.deg2rad(30))-1e-12).all()
    for j in range(3):
        edge=tri[:,(j+1)%3]-tri[:,j];length=np.linalg.norm(edge,axis=1)
        assert (length<=2*dx*(1+1e-12)).all()
        assert (np.einsum('ij,ij->i',n[f[:,j]],n[f[:,(j+1)%3]])>=np.cos(np.deg2rad(45))-1e-12).all()
        for end in [j,(j+1)%3]:assert (np.abs(np.einsum('ij,ij->i',edge,n[f[:,end]]))<=np.sin(np.deg2rad(20))*length+1e-12).all()
        e1=tri[:,(j+1)%3]-tri[:,j];e2=tri[:,(j+2)%3]-tri[:,j]
        angle=np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i',e1,e2)/(np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)),-1,1)))
        assert (angle>=12-1e-10).all()
    topology=mesh_topology(f);assert topology['nonmanifold_edges']==0


def verify(output):
    meta=json.loads((output/'metadata.json').read_text())
    for path,digest in {**meta['source_sha256'],**meta['input_sha256']}.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    rows=list(csv.DictReader((output/'summary.csv').open()));control_cases=cases();checks=0
    for case in control_cases:
        with np.load(output/f"input_{case['name']}.npz") as a:
            for key in ['positions','normals','labels','truth_sheet']:np.testing.assert_array_equal(a[key],case[key])
            assert float(a['dx'])==case['dx']
        for method in ['local','ball_single','ball_multi']:
            with np.load(output/f"{case['name']}_{method}.npz") as a:f=a['faces']
            check_geometry(case['positions'],case['normals'],case['labels'],f,case['dx'])
            metrics=evaluate_mesh(case,f);row=next(x for x in rows if x['case']==case['name'] and x['method']==method)
            for key,value in metrics.items():
                if value is None:assert row[key]==''
                elif isinstance(value,(int,np.integer)):assert int(row[key])==value
                else:np.testing.assert_allclose(float(row[key]),value,atol=1e-12,rtol=1e-12)
            for key,value in mesh_topology(f).items():assert int(row[key])==value
            # Independent area calculation from side lengths (Heron's formula).
            tri=case['positions'][f]
            lengths=np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2);semi=lengths.sum(1)/2
            area=np.sqrt(np.maximum(0,semi*np.prod(semi[:,None]-lengths,axis=1))).sum()
            np.testing.assert_allclose(area,metrics['mesh_area'],atol=1e-10,rtol=1e-10)
            checks+=1
    surface=next(Path(p) for p in meta['input_sha256'] if 'surfaces_' in p)
    with np.load(surface) as a:source={k:a[k].copy() for k in a.files}
    selection=json.loads((output/'selection.json').read_text());eligible=[]
    for method in ['local','ball_single','ball_multi']:
        good=True
        for case in control_cases:
            a=next(r for r in rows if r['case']==case['name'] and r['method']==method)
            b=next(r for r in rows if r['case']==case['name'] and r['method']=='local')
            if int(a['sampled_hole_intrusion_faces'])>int(b['sampled_hole_intrusion_faces']):good=False
            if case['name']=='near_sheets_separate_groups' and int(a['cross_stipulated_sheet_faces']):good=False
            if int(a['nonmanifold_edges']) or int(a['inconsistent_shared_edge_winding']):good=False
        score=-sum(abs(1-float(r['area_fraction'])) for r in rows if r['method']==method and r['area_fraction'])
        if good:eligible.append((score,method))
    chosen=max(eligible,key=lambda x:(x[0],x[1]=='local'))[1] if eligible else 'local'
    assert chosen==selection['chosen']
    bubble_checks=0
    for path in output.glob('bubble_*.npz'):
        with np.load(path) as a:
            for key in ['positions','normals','mach']:np.testing.assert_array_equal(a[key],source[key])
            check_geometry(a['positions'],a['normals'],a['labels'],a['faces'],float(source['grid_spacing']))
            bubble_checks+=1
    report=dict(passed=True,controlled_meshes_checked=checks,bubble_meshes_checked=bubble_checks,
                original_bubble_vertices_preserved=True,chosen_from_controls=selection['chosen'],
                global_self_intersections_checked=False,physical_topology_validated=False)
    (output/'readback.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);verify(p.parse_args().output)
