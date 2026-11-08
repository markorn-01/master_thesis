"""One-event display-only mesh prototype with unchanged diagnostic samples."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.wind_bubble.surface_mesh import triangulate_samples


def controls():
    plane=np.array([[x,y,0.] for x in range(7) for y in range(7)])
    normals=lambda count:np.tile([0.,0.,1.],(count,1))
    inputs=[]
    inputs.append(('planar_grid',plane,normals(len(plane)),np.zeros(len(plane),int),np.zeros(len(plane),int)))
    gap=plane[(plane[:,0]<=1)|(plane[:,0]>=5)]
    inputs.append(('wide_gap_same_group',gap,normals(len(gap)),np.zeros(len(gap),int),(gap[:,0]>=5).astype(int)))
    stacked=np.vstack((plane,plane+[.35,.35,.10]));truth=np.repeat([0,1],len(plane))
    inputs.append(('near_sheets_separate_groups',stacked,normals(len(stacked)),truth,truth))
    inputs.append(('near_sheets_same_group_counterexample',stacked,normals(len(stacked)),np.zeros(len(stacked),int),truth))
    opposing=normals(len(stacked));opposing[len(plane):]*=-1
    inputs.append(('opposite_normals_same_group',stacked,opposing,np.zeros(len(stacked),int),truth))
    records=[]
    for name,p,n,labels,truth in inputs:
        f,r=triangulate_samples(p,n,labels,1.)
        mixed=int((np.ptp(truth[f],axis=1)>0).sum()) if len(f) else 0
        records.append(dict(case=name,faces=len(f),cross_stipulated_sheet_faces=mixed,report=r,
                            expected_limitation='Aligned close sheets in one group may be bridged.' if 'counterexample' in name else None))
    return records


def render(output,s,labels,faces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    p=s['positions'];mach=s['mach'];norm=Normalize(0,16);cmap=plt.get_cmap('viridis')
    fig=plt.figure(figsize=(13,11));counts=[]
    for row,cut in enumerate([False,True]):
        keep=p[:,2]<=.5 if cut else np.ones(len(p),bool)
        shown=faces[np.all(keep[faces],axis=1)];counts.append(dict(cutaway=cut,points=int(keep.sum()),faces=len(shown)))
        for col in [0,1]:
            ax=fig.add_subplot(2,2,row*2+col+1,projection='3d')
            if col==0:ax.scatter(*p[keep].T,c=mach[keep],cmap=cmap,norm=norm,s=2,alpha=.8,depthshade=False)
            elif len(shown):
                values=np.mean(mach[shown],axis=1);colors=cmap(norm(values));colors[~np.isfinite(values)]=[.6,.6,.6,1]
                collection=Poly3DCollection(p[shown],facecolors=colors,edgecolors='none',linewidths=0,alpha=1.)
                ax.add_collection3d(collection)
                unused=keep&~np.isin(np.arange(len(p)),np.unique(faces))
                if unused.any():ax.scatter(*p[unused].T,c='.5',s=2,depthshade=False)
            ax.scatter(*s['source_positions'].T,marker='*',s=70,c='black',depthshade=False)
            ax.set(xlim=(.1,.9),ylim=(.1,.9),zlim=(.1,.9),xlabel='x',ylabel='y',zlabel='z',box_aspect=(1,1,1))
            ax.view_init(elev=25,azim=-65)
            ax.set_title(('Original samples' if col==0 else 'Local triangles; gray = unused samples')+(' | z ≤ .5 cutaway' if cut else ' | full view'),fontsize=11)
    fig.suptitle(f"Display-only triangulation | 128³, t={float(s['time']):.2f} | {len(p):,} unchanged vertices, {len(faces):,} triangles",fontsize=14)
    cb=fig.add_axes([.92,.24,.015,.5]);fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cb,label='Mach (triangle colors: mean of vertex values)')
    fig.text(.035,.025,'No vertex smoothing or new samples; no global hole filling. Physical diagnostics unchanged. Triangles stay within existing spatial groups.\n'
             'Cutaway removes points/triangles for display only and creates an artificial open rim. Surface holes are retained.\n'
             'Local guards do not establish correct topology: close aligned surfaces in the same group can still be bridged.',fontsize=9)
    fig.subplots_adjust(left=.02,right=.88,bottom=.11,top=.92,hspace=.14,wspace=.01)
    fig.savefig(output/'surface_mesh_comparison.png',dpi=140);plt.close(fig)
    return counts


def run(surface,partitions,output):
    inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [surface,partitions]}
    with np.load(surface) as a:s={k:a[k].copy() for k in a.files}
    with np.load(partitions) as a:labels=a['native_cells_a60_labels'].copy()
    if s['grid_shape'].tolist()!=[128]*3 or labels.shape!=(len(s['positions']),):raise ValueError('Require matching 128³ catalogs and partitions.')
    output.mkdir(parents=True,exist_ok=False);reports=[]
    for limit in [1.5,2.]:
        faces,report=triangulate_samples(s['positions'],s['normals'],labels,float(s['grid_spacing']),limit)
        name=f'mesh_edge_{str(limit).replace(".","p")}.npz'
        np.savez_compressed(output/name,positions=s['positions'],normals=s['normals'],mach=s['mach'],labels=labels,
                            faces=faces,face_groups=labels[faces[:,0]],time=s['time'],grid_spacing=s['grid_spacing'])
        for group in np.unique(labels[labels>=0]):
            mask=labels==group;used=np.unique(faces[labels[faces[:,0]]==group]);report.setdefault('groups',[]).append(dict(group=int(group),samples=int(mask.sum()),used_vertices=len(used),faces=int((labels[faces[:,0]]==group).sum())))
        reports.append(report);print(json.dumps(report),flush=True)
        if limit==2.:
            display=render(output,s,labels,faces)
            # Standard ASCII PLY with original vertices and triangle indices for external inspection.
            with (output/'surface_mesh.ply').open('x') as f:
                f.write(f'ply\nformat ascii 1.0\ncomment display-only original-sample mesh\nelement vertex {len(labels)}\nproperty double x\nproperty double y\nproperty double z\nproperty float mach\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
                for p,m in zip(s['positions'],s['mach']):f.write(f'{float(p[0]):.17g} {float(p[1]):.17g} {float(p[2]):.17g} {float(m):.9g}\n')
                for face in faces:f.write('3 '+' '.join(map(str,face))+'\n')
    (output/'controls.json').write_text(json.dumps(controls(),indent=2)+'\n')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in inputs.items())
    meta=dict(input_sha256=inputs,input_hashes_unchanged=True,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [Path(__file__),Path('experiments/wind_bubble/surface_mesh.py')]},reports=reports,display_counts=display,
              vertex_positions_unchanged=True,physical_diagnostics_changed=False,physical_identities_assigned=False,
              algorithm='Experimental local tangent-plane Delaunay, three-vertex agreement and geometric guards.',
              scipy_delaunay_reference='https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html')
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('surface',type=Path);p.add_argument('partitions',type=Path);p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args();run(a.surface,a.partitions,a.output_dir)
