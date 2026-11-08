"""Full-event display comparison with original and fitted geometric normals."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.wind_bubble.compare_patch_normals import plane_normals
from experiments.wind_bubble.ball_pivot_mesh import ball_pivot_samples
from experiments.wind_bubble.surface_mesh import mesh_topology
from experiments.wind_bubble.verify_temporal_mesh import face_components
from experiments.wind_bubble.verify_mesh_comparison import check_geometry
from experiments.wind_bubble.check_temporal_mesh import group_separation, plane_segments
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write


def metrics(p, labels, faces):
    used = np.unique(faces)
    return dict(samples=int((labels>=0).sum()), faces=len(faces), used_vertices=len(used),
                unused_samples=int((labels>=0).sum())-len(used), **mesh_topology(faces), **face_components(faces))


def run(mesh, output):
    digest=hashlib.sha256(mesh.read_bytes()).hexdigest()
    with np.load(mesh) as a: data={k:a[k].copy() for k in a.files}
    p=data['positions'].astype(float); saved=data['normals'].astype(float); labels=data['labels']; dx=float(data['grid_spacing'])
    output.mkdir(parents=True,exist_ok=False); summaries=[]; groups=[]; records=[]; display=[]; svd_checks=0
    for mode,radius in [('saved',None),('plane_2cells',2),('plane_3cells',3)]:
        if radius is None:
            normals=saved.copy();valid=np.zeros(len(p),bool);counts=np.zeros(len(p),int)
        else:
            normals,valid,counts,_=plane_normals(p,saved,labels,radius*dx)
            # Independent dense-neighborhood/SVD checks on a deterministic sample of each group.
            for group in np.unique(labels[labels>=0]):
                rows=np.flatnonzero(labels==group)
                for row in rows[np.linspace(0,len(rows)-1,min(100,len(rows)),dtype=int)]:
                    points=p[rows[np.linalg.norm(p[rows]-p[row],axis=1)<=radius*dx]]
                    assert len(points)==counts[row]
                    if valid[row]:
                        _,_,vh=np.linalg.svd(points-points.mean(0),full_matrices=False)
                        np.testing.assert_allclose(abs(vh[-1]@normals[row]),1.,atol=1e-10);svd_checks+=1
        faces,construction,raw=ball_pivot_samples(p,normals,labels,dx,radius_factors=(1.,))
        if mode=='saved':np.testing.assert_array_equal(faces,data['faces'])
        check_geometry(p,normals,labels,faces,dx)
        segments,segment_faces,coplanar=plane_segments(p,faces)
        separation=group_separation(p,labels,faces)
        row=dict(mode=mode,raw_faces=len(raw),invalid_fits=None if radius is None else int(((labels>=0)&~valid).sum()),
                 **metrics(p,labels,faces))
        summaries.append(row)
        for group in np.unique(labels[labels>=0]):
            gfaces=faces[labels[faces[:,0]]==group];mask=labels==group
            deviation=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',normals[mask],saved[mask])/
                          (np.linalg.norm(normals[mask],axis=1)*np.linalg.norm(saved[mask],axis=1)),-1,1)))
            # Negative labels for other samples, so the metrics count only this group.
            groups.append(dict(mode=mode,group=int(group),invalid_fits=None if radius is None else int((~valid[mask]).sum()),
                               median_normal_deviation_degrees=float(np.median(deviation)),
                               **metrics(p,np.where(mask,group,-1),gfaces)))
        np.savez_compressed(output/f'{mode}.npz',positions=data['positions'],saved_normals=data['normals'],
                            display_normals=normals,mach=data['mach'],labels=labels,faces=faces,raw_faces=raw,
                            fit_valid=valid,neighbor_count=counts,time=data['time'],grid_spacing=data['grid_spacing'],
                            plane_segments=segments,segment_face_rows=segment_faces)
        with np.load(output/f'{mode}.npz') as a:
            for k,original in [('positions','positions'),('saved_normals','normals'),('mach','mach'),('labels','labels')]:
                np.testing.assert_array_equal(a[k],data[original])
            check_geometry(a['positions'],a['display_normals'],a['labels'],a['faces'],dx)
            assert metrics(p,labels,a['faces'])==metrics(p,labels,faces)
            # Heron area cross-check, strictly a display metric.
            tri=p[a['faces']];length=np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2);semi=length.sum(1)/2
            heron=np.sqrt(np.maximum(0,semi*np.prod(semi[:,None]-length,axis=1))).sum()
            cross=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2
            np.testing.assert_allclose(heron,cross,atol=1e-10,rtol=1e-10)
        records.append(dict(**row,construction=construction,cross_group_separation=separation,
                            central_segments=len(segments),coplanar_skipped=coplanar))
        display.append((mode,faces,segments,segment_faces));print(row,flush=True)
    csv_write(output/'summary.csv',summaries);csv_write(output/'group_statistics.csv',groups)
    render(p,data['mach'],labels,display,output)
    assert hashlib.sha256(mesh.read_bytes()).hexdigest()==digest
    sources=[Path(__file__)]+[Path('experiments/wind_bubble')/f'{name}.py' for name in
        ['compare_patch_normals','ball_pivot_mesh','surface_mesh','check_temporal_mesh','verify_temporal_mesh','verify_mesh_comparison']]
    import open3d
    (output/'readback.json').write_text(json.dumps(dict(input=str(mesh),input_sha256=digest,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        open3d_version=open3d.__version__,records=records,independent_sampled_svd_checks=svd_checks,
        original_arrays_exact=True,baseline_faces_exact=True,physical_diagnostics_changed=False,
        production_method_changed=False,guard_thresholds_changed=False,ball_radius_cells=1.,
        limitations=['Within-group false bridges and global self-intersections are not excluded.',
                     'Connected mesh pieces are not physical shocks or persistent identities.',
                     'Boundary-edge counts measure open mesh edges, not physical shock openings.',
                     'Single-event comparison does not validate temporal stability.']),indent=2)+'\n')


def render(p,mach,labels,display,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    norm=Normalize(0,16);cmap=plt.get_cmap('viridis');fig=plt.figure(figsize=(16,14))
    for col,(mode,faces,_,_) in enumerate(display):
        for row in range(3):
            ax=fig.add_subplot(3,3,row*3+col+1,projection='3d')
            mask=np.ones(len(p),bool) if row==0 else p[:,2]<=.5 if row==1 else labels>0
            f=faces[np.all(mask[faces],axis=1)]
            ax.add_collection3d(Poly3DCollection(p[f],facecolors=cmap(norm(mach[f].mean(1))),edgecolors='none'))
            unused=mask&~np.isin(np.arange(len(p)),np.unique(faces));ax.scatter(*p[unused].T,c='.6',s=2,depthshade=False)
            bounds=(.1,.9) if row<2 else (.23,.77)
            ax.set(xlim=bounds,ylim=bounds,zlim=bounds,box_aspect=(1,1,1),xlabel='x',ylabel='y',zlabel='z')
            ax.view_init(elev=25,azim=-65)
            ax.set_title(mode.replace('_',' ')+[' | full',' | z ≤ .5 cutaway',' | inner groups only'][row],fontsize=11)
    fig.suptitle('Full saved event t=0.16: single-radius reconstruction with three display-normal settings',fontsize=14)
    fig.text(.025,.02,'Original refined positions and Mach values; fixed camera and color scale. Gray: samples unused by the full mesh.\n'
             'Cutaway removes faces for display only. Fitted normals do not change physical diagnostics. Gaps and false same-group bridges remain possible.',fontsize=10)
    fig.subplots_adjust(left=.01,right=.93,bottom=.08,top=.94,wspace=.02,hspace=.14)
    cb=fig.add_axes([.96,.25,.012,.5]);fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cb,label='Mean vertex Mach')
    fig.savefig(output/'event_normal_comparison.png',dpi=125);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,(mode,faces,segments,segment_faces) in zip(axes,display):
        for group in np.unique(labels[labels>=0]):
            keep=labels[faces[segment_faces,0]]==group
            ax.add_collection(LineCollection(segments[keep],colors=[plt.get_cmap('tab10')(int(group))],linewidths=1))
        ax.set(xlim=(.1,.9),ylim=(.1,.9),aspect='equal',xlabel='x',ylabel='y',title=mode.replace('_',' '));ax.grid(alpha=.15)
    fig.suptitle('Actual triangle intersections with z=0.5; same bounds and event-local group colors')
    fig.text(.025,.025,'Lines are triangle-plane intersections, not a projected point slab. Line gaps do not establish physical openings. No persistent IDs assigned.',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.93));fig.savefig(output/'central_sections.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',type=Path,default=Path('outputs/two_bubbles_temporal_mesh_n128/mesh_0016.npz'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();run(args.mesh,args.output_dir)
