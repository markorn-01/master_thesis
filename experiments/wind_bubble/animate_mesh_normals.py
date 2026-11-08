"""Animate original versus experimental fitted-normal display reconstructions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image
from experiments.wind_bubble.compare_patch_normals import plane_normals
from experiments.wind_bubble.ball_pivot_mesh import ball_pivot_samples
from experiments.wind_bubble.compare_event_normals import metrics
from experiments.wind_bubble.verify_mesh_comparison import check_geometry
from experiments.wind_bubble.check_temporal_mesh import group_separation
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write


def render(path, data, meshes, rows, event, total):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    p=data['positions'];norm=Normalize(0,16);cmap=plt.get_cmap('viridis')
    fig=plt.figure(figsize=(13,7))
    fig.suptitle(f'Saved event {event+1:02d}/{total} | t = {float(data["time"]):.2f} | same original surface samples',fontsize=14)
    for col,(mode,faces) in enumerate(meshes.items()):
        ax=fig.add_subplot(1,2,col+1,projection='3d');mask=p[:,2]<=.5
        shown=faces[np.all(mask[faces],axis=1)]
        ax.add_collection3d(Poly3DCollection(p[shown],facecolors=cmap(norm(data['mach'][shown].mean(1))),edgecolors='none'))
        unused=mask&~np.isin(np.arange(len(p)),np.unique(faces));ax.scatter(*p[unused].T,c='.6',s=2,depthshade=False)
        ax.set(xlim=(.1,.9),ylim=(.1,.9),zlim=(.1,.9),box_aspect=(1,1,1),xlabel='x',ylabel='y',zlabel='z')
        ax.view_init(elev=25,azim=-65)
        ax.set_title('Original shock normals' if mode=='saved' else 'Experimental display normals: 2-cell plane fit',fontsize=11)
        row=rows[col]
        fig.text(.04+.47*col,.14,f'Used samples: {row["used_vertices"]:,} | Mesh pieces: {row["edge_connected_components"]:,}\n'
                 f'Open mesh edges: {row["boundary_edges"]:,} | Accepted faces: {row["faces"]:,}',fontsize=10)
    fig.text(.04,.065,'Cutaway z ≤ 0.5; gray: unused or grouping-excluded samples. Original positions and physical diagnostics unchanged.\n'
             'Open edges are not validated physical holes. Nearby same-group sheets can connect incorrectly. No temporal interpolation or persistent IDs.',fontsize=9)
    fig.subplots_adjust(left=.01,right=.92,top=.88,bottom=.22,wspace=.02)
    cb=fig.add_axes([.93,.3,.012,.48]);fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cb,label='Mean vertex Mach')
    fig.savefig(path,dpi=110);plt.close(fig)


def run(root,partitions,prior,output,fps):
    if fps not in [1,2,5,10]:raise ValueError('Use 1, 2, 5 or 10 FPS for exact GIF frame durations.')
    metadata=root/'metadata.json';meta=json.loads(metadata.read_text());times=meta['actual_times']
    if len(times)!=17 or meta['num_cells']!=128:raise ValueError('Require the completed 17-event 128³ run.')
    if meta['status']!='complete':raise ValueError('Require a complete saved run.')
    output.mkdir(parents=True,exist_ok=False);hashes={str(metadata):hashlib.sha256(metadata.read_bytes()).hexdigest()}
    summary=[];records=[];frames=[];duration=1000//fps
    with TemporaryDirectory(prefix='shock-mesh-normal-frames-') as temporary:
        for event,time in enumerate(times):
            source=root/'online'/f'surfaces_{event:04d}.npz';partition=partitions/f'groups_n128_event{event:04d}.npz'
            for path in [source,partition]:hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(source) as a:data={k:a[k].copy() for k in a.files}
            with np.load(partition) as a:labels=a['native_cells_a60_labels'].copy()
            np.testing.assert_allclose(float(data['time']),time,atol=1e-7,rtol=0)
            p=data['positions'].astype(float);saved=data['normals'].astype(float);dx=float(data['grid_spacing'])
            if labels.shape!=(len(p),):raise ValueError('Catalog/partition mismatch.')
            fitted,valid,counts,_=plane_normals(p,saved,labels,2*dx)
            meshes={};rows=[];arrays={}
            for mode,normals in [('saved',saved),('plane_2cells',fitted)]:
                faces,construction,raw=ball_pivot_samples(p,normals,labels,dx,radius_factors=(1.,))
                check_geometry(p,normals,labels,faces,dx)
                row=dict(event=event,time=float(data['time']),mode=mode,excluded_samples=int((labels<0).sum()),
                         invalid_fits=None if mode=='saved' else int(((labels>=0)&~valid).sum()),**metrics(p,labels,faces))
                if event in [8,12,16]:
                    reference=prior/f'mesh_{event:04d}_{mode}.npz';hashes[str(reference)]=hashlib.sha256(reference.read_bytes()).hexdigest()
                    with np.load(reference) as a:
                        np.testing.assert_array_equal(faces,a['faces']);np.testing.assert_array_equal(normals,a['display_normals'])
                meshes[mode]=faces;rows.append(row);summary.append(row)
                arrays[f'{mode}_faces']=faces;arrays[f'{mode}_raw_faces']=raw
                records.append(dict(**row,construction=construction,separation=group_separation(p,labels,faces)))
            target=output/f'meshes_{event:04d}.npz'
            np.savez_compressed(target,positions=data['positions'],saved_normals=data['normals'],mach=data['mach'],labels=labels,
                                display_normals=fitted,fit_valid=valid,neighbor_count=counts,time=data['time'],grid_spacing=data['grid_spacing'],**arrays)
            with np.load(target) as a:
                for field,original in [('positions','positions'),('saved_normals','normals'),('mach','mach')]:
                    np.testing.assert_array_equal(a[field],data[original])
                np.testing.assert_array_equal(a['labels'],labels)
                for mode,normals in [('saved',a['saved_normals']),('plane_2cells',a['display_normals'])]:
                    check_geometry(a['positions'],normals,a['labels'],a[f'{mode}_faces'],dx)
                    assert metrics(p,labels,a[f'{mode}_faces'])==metrics(p,labels,meshes[mode])
            path=Path(temporary)/f'frame_{event:04d}.png';render(path,data,meshes,rows,event,len(times))
            with Image.open(path) as image:frames.append(image.convert('RGB'))
            if event in [0,8,12,16]:frames[-1].save(output/f'frame_{event:04d}.png')
            print(f'Event {event:02d}: saved/fitted pieces {rows[0]["edge_connected_components"]}/{rows[1]["edge_connected_components"]}',flush=True)
    gif=output/'mesh_normal_comparison.gif'
    frames[0].save(gif,save_all=True,append_images=frames[1:],duration=duration,loop=0,disposal=2,optimize=False)
    with Image.open(gif) as image:
        assert image.n_frames==len(frames);durations=[]
        for index in range(image.n_frames):
            image.seek(index);image.load();assert image.size==frames[index].size
            durations.append(image.info['duration'])
            # The changing saved-event/time header must match the correct rendered frame.
            actual=np.asarray(image.convert('RGB').crop((0,0,image.width,60)),float)
            errors=[np.mean(abs(actual-np.asarray(f.crop((0,0,f.width,60)),float))) for f in frames]
            assert int(np.argmin(errors))==index
    assert durations==[duration]*len(frames)
    csv_write(output/'summary.csv',summary)
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    sources=[Path(__file__)]+[Path('experiments/wind_bubble')/f'{name}.py' for name in
        ['compare_patch_normals','ball_pivot_mesh','compare_event_normals','verify_mesh_comparison','check_temporal_mesh','verify_temporal_mesh']]
    import open3d
    report=dict(events=len(times),decoded_frames=len(frames),frame_size=list(frames[0].size),fps=fps,
                frame_durations_ms=durations,header_frame_order_verified=True,records=records,
                original_arrays_exact=True,three_time_references_exact=True,input_sha256=hashes,
                source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                open3d_version=open3d.__version__,physical_diagnostics_changed=False,production_method_changed=False,
                temporal_identity_tracking_added=False,temporal_interpolation_added=False,
                temporal_flicker_quantitatively_validated=False,within_group_topology_validated=False)
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['events','decoded_frames','frame_size','fps','header_frame_order_verified']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/two_bubbles_budget_matched_iwr_compgpu3_n128'))
    parser.add_argument('--partitions',type=Path,default=Path('outputs/two_bubbles_resolution_comparison_n064_n128'))
    parser.add_argument('--prior',type=Path,default=Path('outputs/two_bubbles_temporal_fitted_normals_n128'))
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--fps',type=int,default=2)
    args=parser.parse_args();run(args.root,args.partitions,args.prior,args.output_dir,args.fps)
