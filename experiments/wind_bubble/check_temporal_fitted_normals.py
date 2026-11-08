"""Three saved-time comparison of original and two-cell display normal fits."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.wind_bubble.compare_patch_normals import plane_normals
from experiments.wind_bubble.ball_pivot_mesh import ball_pivot_samples
from experiments.wind_bubble.compare_event_normals import metrics
from experiments.wind_bubble.verify_mesh_comparison import check_geometry
from experiments.wind_bubble.check_temporal_mesh import group_separation, plane_segments
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write

EVENTS = [8, 12, 16]


def run(root, prior, output):
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}; summary = []; group_rows = []; records = []; displays = []; checks = 0
    for event in EVENTS:
        path = root/f'mesh_{event:04d}.npz'; hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as a: data = {k:a[k].copy() for k in a.files}
        np.testing.assert_allclose(float(data['time']),event*.01,atol=1e-7,rtol=0)
        p = data['positions'].astype(float); saved = data['normals'].astype(float)
        labels = data['labels']; dx = float(data['grid_spacing'])
        for mode in ['saved','plane_2cells']:
            if mode == 'saved':
                normals = saved.copy(); valid = np.zeros(len(p),bool); counts = np.zeros(len(p),int)
                faces = data['faces']; raw = data['raw_faces']; construction = None
            else:
                normals,valid,counts,_ = plane_normals(p,saved,labels,2*dx)
                for group in np.unique(labels[labels>=0]):
                    rows = np.flatnonzero(labels==group)
                    for row in rows[np.linspace(0,len(rows)-1,min(100,len(rows)),dtype=int)]:
                        points = p[rows[np.linalg.norm(p[rows]-p[row],axis=1)<=2*dx]]
                        assert len(points) == counts[row]
                        if valid[row]:
                            _,_,vh = np.linalg.svd(points-points.mean(0),full_matrices=False)
                            np.testing.assert_allclose(abs(vh[-1]@normals[row]),1.,atol=1e-10); checks += 1
                faces,construction,raw = ball_pivot_samples(p,normals,labels,dx,radius_factors=(1.,))
            check_geometry(p,normals,labels,faces,dx)
            row = dict(event=event,time=float(data['time']),mode=mode,
                       invalid_fits=None if mode=='saved' else int(((labels>=0)&~valid).sum()),
                       excluded_samples=int((labels<0).sum()),**metrics(p,labels,faces))
            summary.append(row)
            for group in np.unique(labels[labels>=0]):
                mask = labels==group; f = faces[labels[faces[:,0]]==group]
                group_rows.append(dict(event=event,time=float(data['time']),mode=mode,group=int(group),
                                       **metrics(p,np.where(mask,group,-1),f)))
            separation = group_separation(p,labels,faces)
            segments,segment_faces,coplanar = plane_segments(p,faces)
            target = output/f'mesh_{event:04d}_{mode}.npz'
            np.savez_compressed(target,positions=data['positions'],saved_normals=data['normals'],mach=data['mach'],
                                labels=labels,display_normals=normals,fit_valid=valid,neighbor_count=counts,
                                faces=faces,raw_faces=raw,time=data['time'],grid_spacing=data['grid_spacing'],
                                plane_segments=segments,segment_face_rows=segment_faces)
            with np.load(target) as a:
                for key,original in [('positions','positions'),('saved_normals','normals'),('mach','mach'),('labels','labels')]:
                    np.testing.assert_array_equal(a[key],data[original])
                check_geometry(a['positions'],a['display_normals'],a['labels'],a['faces'],dx)
                assert metrics(p,labels,a['faces']) == metrics(p,labels,faces)
                if mode=='saved': np.testing.assert_array_equal(a['faces'],data['faces'])
            if event==16:
                reference = prior/f'{mode}.npz'; hashes[str(reference)] = hashlib.sha256(reference.read_bytes()).hexdigest()
                with np.load(reference) as a:
                    np.testing.assert_array_equal(faces,a['faces'])
                    np.testing.assert_array_equal(normals,a['display_normals'])
            records.append(dict(**row,construction=construction,separation=separation,
                                central_segments=len(segments),coplanar_skipped=coplanar))
            displays.append((event,mode,data,faces,segments,segment_faces))
            print(row,flush=True)
    csv_write(output/'summary.csv',summary);csv_write(output/'group_statistics.csv',group_rows)
    render(displays,output)
    for path,digest in hashes.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    sources = [Path(__file__)]+[Path('experiments/wind_bubble')/f'{name}.py' for name in
        ['compare_patch_normals','ball_pivot_mesh','compare_event_normals','verify_mesh_comparison','check_temporal_mesh','verify_temporal_mesh']]
    import open3d
    (output/'readback.json').write_text(json.dumps(dict(events=EVENTS,records=records,
        independent_sampled_svd_checks=checks,input_sha256=hashes,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        open3d_version=open3d.__version__,original_arrays_exact=True,original_faces_exact=True,event16_prior_exact=True,
        fit_radius_cells=2,ball_radius_cells=1,guard_thresholds_changed=False,production_method_changed=False,
        physical_diagnostics_changed=False,temporal_flicker_validated=False,
        limitations=['Three snapshots do not validate intervening frames or temporal flicker.',
                     'Mesh components and event-local groups are not persistent physical shock IDs.',
                     'Same-group bridges, physical hole coverage and global self-intersections remain unvalidated.']),indent=2)+'\n')


def render(displays,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    norm = Normalize(0,16); cmap = plt.get_cmap('viridis'); fig = plt.figure(figsize=(13,15))
    for i,(event,mode,data,faces,segments,segment_faces) in enumerate(displays):
        ax = fig.add_subplot(3,2,i+1,projection='3d'); p = data['positions']; mask = p[:,2]<=.5
        f = faces[np.all(mask[faces],axis=1)]
        ax.add_collection3d(Poly3DCollection(p[f],facecolors=cmap(norm(data['mach'][f].mean(1))),edgecolors='none'))
        unused = mask&~np.isin(np.arange(len(p)),np.unique(faces));ax.scatter(*p[unused].T,c='.6',s=2,depthshade=False)
        ax.set(xlim=(.1,.9),ylim=(.1,.9),zlim=(.1,.9),box_aspect=(1,1,1),xlabel='x',ylabel='y',zlabel='z')
        ax.view_init(elev=25,azim=-65);ax.set_title(f't={float(data["time"]):.2f} | {mode.replace("_"," ")} | z ≤ .5')
    fig.suptitle('Fixed display reconstruction at three saved times: original versus fitted normals')
    fig.text(.025,.02,'Same original refined positions, Mach values, camera, bounds and color scale. Cutaways remove faces for display only.\n'
             'Gray: samples unused by the full mesh, including excluded small groups. Three snapshots do not validate temporal flicker or physical topology.',fontsize=10)
    fig.subplots_adjust(left=.01,right=.9,bottom=.08,top=.94,wspace=.02,hspace=.15)
    cb = fig.add_axes([.95,.25,.012,.5]);fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cb,label='Mean vertex Mach')
    fig.savefig(output/'temporal_fitted_comparison.png',dpi=125);plt.close(fig)
    fig,axes = plt.subplots(3,2,figsize=(10,14))
    for ax,(event,mode,data,faces,segments,segment_faces) in zip(axes.flat,displays):
        labels = data['labels']
        for group in np.unique(labels[labels>=0]):
            keep = labels[faces[segment_faces,0]]==group
            ax.add_collection(LineCollection(segments[keep],colors=[plt.get_cmap('tab10')(int(group))],linewidths=1))
        ax.set(xlim=(.1,.9),ylim=(.1,.9),aspect='equal',xlabel='x',ylabel='y',
               title=f't={float(data["time"]):.2f} | {mode.replace("_"," ")}');ax.grid(alpha=.15)
    fig.suptitle('Actual triangle intersections with z=0.5')
    fig.text(.025,.02,'Colors identify event-local groups, not persistent IDs. Line gaps are not validated physical openings.\n'
             'No positional smoothing, physical-normal replacement or tracking changes.',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(output/'central_sections.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/two_bubbles_temporal_mesh_n128'))
    parser.add_argument('--prior',type=Path,default=Path('outputs/two_bubbles_event_normals_n128'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args();run(args.root,args.prior,args.output_dir)
