"""Three-event visualization check with frozen single-radius mesh settings."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from experiments.wind_bubble.ball_pivot_mesh import ball_pivot_samples
from experiments.wind_bubble.surface_mesh import mesh_topology
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write

EVENTS=[8,12,16]


def group_separation(p,labels,faces):
    """Sufficient cross-group nonintersection bound, not a general topology test.

    Every point of a triangle lies within its maximum edge length of every
    vertex. If the two groups' closest vertices exceed the sum of their maximum
    triangle edges, their triangles cannot intersect. A failed bound is inconclusive.
    """
    p=np.asarray(p,float);labels=np.asarray(labels);faces=np.asarray(faces,int).reshape(-1,3)
    groups=np.unique(labels[labels>=0]);maximum={};rows=[]
    for group in groups:
        f=faces[labels[faces[:,0]]==group]
        maximum[int(group)]=float(np.linalg.norm(p[f]-p[np.roll(f,1,axis=1)],axis=2).max()) if len(f) else 0.
    for i,a in enumerate(groups):
        for b in groups[i+1:]:
            distance=float(cKDTree(p[labels==a]).query(p[labels==b])[0].min())
            bound=maximum[int(a)]+maximum[int(b)]
            rows.append(dict(group_a=int(a),group_b=int(b),minimum_vertex_distance=distance,
                             sum_maximum_triangle_edges=bound,separation_bound_passed=distance>bound))
    return rows


def plane_segments(p,faces,z=.5):
    segments=[];face_rows=[];coplanar=0
    for i,face in enumerate(faces):
        tri=p[face];offset=tri[:,2]-z;crossings=[]
        if np.all(offset==0):coplanar+=1;continue
        for a,b in [(0,1),(1,2),(2,0)]:
            if offset[a]==0:crossings.append(tri[a,:2])
            if offset[a]*offset[b]<0:
                t=-offset[a]/(offset[b]-offset[a]);crossings.append(tri[a,:2]+t*(tri[b,:2]-tri[a,:2]))
        if crossings:
            unique=np.unique(np.asarray(crossings),axis=0)
            if len(unique)==2 and np.linalg.norm(unique[1]-unique[0])>1e-12:
                segments.append(unique);face_rows.append(i)
    return np.asarray(segments,float).reshape(-1,2,2),np.asarray(face_rows,int),coplanar


def run(root,partitions,comparison,output):
    selection=json.loads((comparison/'selection.json').read_text())
    if selection['chosen']!='ball_single':raise ValueError('Require the previous single-radius display choice.')
    prior=json.loads((comparison/'metadata.json').read_text())
    for p,h in prior['source_sha256'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError(f'Comparison source changed: {p}')
    hashes={};records=[];summary=[];display=[];output.mkdir(parents=True,exist_ok=False)
    for name in ['metadata.json','selection.json','readback.json']:
        p=comparison/name;hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    for event in EVENTS:
        surface=root/'online'/f'surfaces_{event:04d}.npz';partition=partitions/f'groups_n128_event{event:04d}.npz'
        for path in [surface,partition]:hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(surface) as a:s={k:a[k].copy() for k in a.files}
        with np.load(partition) as a:labels=a['native_cells_a60_labels'].copy()
        if s['grid_shape'].tolist()!=[128]*3 or labels.shape!=(len(s['positions']),):raise ValueError('Require matching 128³ catalogs and partitions.')
        np.testing.assert_allclose(float(s['time']),event*.01,rtol=0,atol=1e-7)
        faces,construction,raw=ball_pivot_samples(s['positions'],s['normals'],labels,float(s['grid_spacing']),radius_factors=(1.,))
        p=s['positions'].astype(float);separation=group_separation(p,labels,faces);segments,segment_faces,coplanar=plane_segments(p,faces)
        group_rows=[]
        for group in np.unique(labels[labels>=0]):
            selected=faces[labels[faces[:,0]]==group];used=np.unique(selected)
            group_rows.append(dict(event=event,time=float(s['time']),group=int(group),samples=int((labels==group).sum()),
                                   used_vertices=len(used),faces=len(selected),**mesh_topology(selected)))
        np.savez_compressed(output/f'mesh_{event:04d}.npz',positions=s['positions'],normals=s['normals'],mach=s['mach'],labels=labels,
                            faces=faces,raw_faces=raw,time=s['time'],grid_spacing=s['grid_spacing'],plane_segments=segments,segment_face_rows=segment_faces)
        row=dict(event=event,time=float(s['time']),samples=len(labels),retained_samples=int((labels>=0).sum()),excluded_samples=int((labels<0).sum()),
                 groups=len(group_rows),raw_faces=len(raw),faces=len(faces),used_vertices=len(np.unique(faces)),
                 unused_retained_samples=int((labels>=0).sum())-len(np.unique(faces)),**mesh_topology(faces))
        records.append(dict(**row,construction=construction,group_statistics=group_rows,separation=separation,
                            central_plane_segments=len(segments),coplanar_triangles_skipped=coplanar))
        summary.append(row);display.append((s,labels,faces,segments,segment_faces));print(row,flush=True)
    csv_write(output/'summary.csv',summary);csv_write(output/'group_statistics.csv',[g for r in records for g in r['group_statistics']])
    render(output,display)
    for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    sources=[Path(__file__),Path('experiments/wind_bubble/ball_pivot_mesh.py'),Path('experiments/wind_bubble/surface_mesh.py')]
    meta=dict(events=EVENTS,input_sha256=hashes,input_hashes_unchanged=True,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              root=str(root),partitions=str(partitions),method='Open3D single-radius ball pivoting plus fixed geometric guards',
              radius_factor=1.,max_edge_cells=2.,grouping='native_cells_a60_labels',cutoff=20,normal_angle=45,tangent_angle=20,face_angle=30,min_angle=12,
              figures=dict(camera=[25,-65],bounds=[.1,.9],mach_limits=[0,16],cutaway_z=.5,plane_z=.5,point_slab_half_width=.0078125),
              records=records,vertices_moved=False,physical_diagnostics_changed=False,physical_identities_assigned=False,
              temporal_flicker_validated=False)
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


def render(output,display):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    norm=Normalize(0,16);cmap=plt.get_cmap('viridis');fig=plt.figure(figsize=(19,14))
    for row,(s,labels,faces,segments,segment_faces) in enumerate(display):
        p=s['positions'];keep=p[:,2]<=.5
        for col in range(4):
            ax=fig.add_subplot(3,4,row*4+col+1,projection='3d');cut=col>=2;mask=keep if cut else np.ones(len(p),bool)
            if col%2==0:
                ax.scatter(*p[mask].T,c=s['mach'][mask],cmap=cmap,norm=norm,s=2,depthshade=False)
            else:
                f=faces[np.all(mask[faces],axis=1)];color=cmap(norm(np.mean(s['mach'][f],axis=1)))
                ax.add_collection3d(Poly3DCollection(p[f],facecolors=color,edgecolors='none'))
                unused=mask&~np.isin(np.arange(len(p)),np.unique(faces));ax.scatter(*p[unused].T,c='.6',s=2,depthshade=False)
            ax.set(xlim=(.1,.9),ylim=(.1,.9),zlim=(.1,.9),box_aspect=(1,1,1),xlabel='x',ylabel='y',zlabel='z')
            ax.view_init(elev=25,azim=-65);ax.set_title(f't={float(s["time"]):.2f} | '+('original points' if col%2==0 else 'fixed single-radius mesh')+(' | cutaway' if cut else ''),fontsize=10)
    fig.suptitle('Three-event check: fixed reconstruction, camera, spatial bounds and Mach scale',fontsize=15)
    fig.text(.025,.02,'Every vertex and Mach value is unchanged. Gray: original samples not used by triangles (including size-excluded groups). Cutaway removes points/faces above z=.5 for display only.\n'
             'Many open boundaries remain; surfaces seen through gaps do not imply transparent gas. Group labels are event-local and do not establish physical identity.\n'
             'These three snapshots cannot establish temporal flicker, correct topology within a group, or a converged physical surface.',fontsize=10)
    fig.subplots_adjust(left=.01,right=.94,top=.94,bottom=.08,wspace=.02,hspace=.16)
    cb=fig.add_axes([.965,.23,.008,.54]);fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cb,label='Mach (mesh: mean vertex value)')
    fig.savefig(output/'temporal_mesh_comparison.png',dpi=120);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,15))
    for row,(s,labels,faces,segments,segment_faces) in enumerate(display):
        p=s['positions'];mask=np.abs(p[:,2]-.5)<=float(s['grid_spacing']);colors=plt.get_cmap('tab10')
        for j,group in enumerate(np.unique(labels[labels>=0])):
            selected=mask&(labels==group);axes[row,0].scatter(p[selected,0],p[selected,1],s=7,color=colors(j),label=f'local group {group}')
            keep=labels[faces[segment_faces,0]]==group
            axes[row,1].add_collection(LineCollection(segments[keep],colors=[colors(j)],linewidths=1.))
        excluded=mask&(labels<0);axes[row,0].scatter(p[excluded,0],p[excluded,1],s=7,c='.6',marker='x')
        for col in range(2):
            ax=axes[row,col];ax.scatter(*s['source_positions'][:,:2].T,marker='*',c='black',s=65)
            ax.set(xlim=(.1,.9),ylim=(.1,.9),aspect='equal',xlabel='x',ylabel='y',title=f't={float(s["time"]):.2f} | '+('points within ±1 cell of z=.5' if col==0 else 'triangle intersections with z=.5'))
            ax.grid(alpha=.15)
        axes[row,0].legend(fontsize=8,loc='upper right')
    fig.suptitle('Central geometry: projected sample slab versus actual mesh-plane intersections',fontsize=14)
    fig.text(.035,.02,'Colors are assigned locally at each event; they are not persistent shock IDs. Stars mark wind sources.\n'
             'A line gap can reflect rejected/missing triangles, not a physical opening. The point slab and triangle-plane intersections are different selections.\n'
             'No spatial or temporal interpolation is added. Coplanar triangles, if any, are counted and skipped in the intersection view.',fontsize=10)
    fig.tight_layout(rect=(.01,.09,.99,.95));fig.savefig(output/'central_mesh_sections.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('outputs/two_bubbles_budget_matched_iwr_compgpu3_n128'))
    p.add_argument('--partitions',type=Path,default=Path('outputs/two_bubbles_resolution_comparison_n064_n128'))
    p.add_argument('--comparison',type=Path,default=Path('outputs/two_bubbles_mesh_method_comparison'))
    a=p.parse_args();run(a.root,a.partitions,a.comparison,a.output_dir)
