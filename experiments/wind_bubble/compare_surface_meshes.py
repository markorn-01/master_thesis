"""Compare local triangulation and ball pivoting before choosing a bubble display."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from experiments.wind_bubble.surface_mesh import triangulate_samples,mesh_topology
from experiments.wind_bubble.ball_pivot_mesh import ball_pivot_samples
from experiments.wind_bubble.mesh_benchmark_cases import cases,evaluate_mesh
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write

METHODS=['local','ball_single','ball_multi']
DISPLAY_NAMES={'points':'Original samples','local':'Local triangle prototype','ball_single':'Ball pivoting (one radius)','ball_multi':'Ball pivoting (three radii)'}


def build(case,method):
    args=[case[k] for k in ['positions','normals','labels','dx']]
    if method=='local':f,r=triangulate_samples(*args);return f,r
    f,r,_=ball_pivot_samples(*args,radius_factors=(1.,) if method=='ball_single' else (.8,1.,1.4));return f,r


def plot_controls(output,inputs,meshes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(15,17))
    for row,case in enumerate(inputs):
        p=case['positions']
        for col,method in enumerate(['points']+METHODS):
            ax=fig.add_subplot(len(inputs),4,row*4+col+1,projection='3d')
            if method=='points':ax.scatter(*p.T,s=1,c=case['truth_sheet'],cmap='coolwarm',vmin=0,vmax=1,depthshade=False)
            else:
                f=meshes[case['name'],method]
                if len(f):
                    mixed=np.ptp(case['truth_sheet'][f],axis=1)>0
                    color=np.tile([.15,.55,.65,1.],(len(f),1));color[mixed]=[.85,.2,.15,1.]
                    ax.add_collection3d(Poly3DCollection(p[f],facecolors=color,edgecolors='none'))
                unused=np.setdiff1d(np.arange(len(p)),np.unique(f));ax.scatter(*p[unused].T,c='.6',s=1,depthshade=False)
            lo=p.min(0);hi=p.max(0);center=(lo+hi)/2;radius=max(np.ptp(p,axis=0).max()/2,.1)*1.08
            ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius),box_aspect=(1,1,1))
            ax.view_init(elev=60 if 'hole' in case['name'] else 25,azim=-65)
            ax.set_title(f'{case["name"].replace("_"," ")}\n{DISPLAY_NAMES[method]}',fontsize=9)
            ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
    fig.suptitle('Same original vertices; reconstruction methods compared on known geometry',fontsize=15)
    fig.text(.025,.015,'Teal: triangle faces. Red: faces bridging stipulated sheets. Gray: unused original samples. No vertex smoothing.\n'
             'Both ball-pivoting outputs retain the original prototype edge/normal/tangent/angle guards; same-group sheet confusion remains an explicit control.',fontsize=10)
    fig.subplots_adjust(left=.01,right=.99,top=.94,bottom=.055,wspace=.01,hspace=.25);fig.savefig(output/'controlled_mesh_comparison.png',dpi=130);plt.close(fig)


def plot_bubble(output,s,chosen):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    methods=list(dict.fromkeys(['local',chosen]));names=['points']+methods;p=s['positions'];norm=Normalize(0,16);cmap=plt.get_cmap('viridis')
    fig=plt.figure(figsize=(6*len(names),10))
    for row,cut in enumerate([False,True]):
        keep=p[:,2]<=.5 if cut else np.ones(len(p),bool)
        for col,method in enumerate(names):
            ax=fig.add_subplot(2,len(names),row*len(names)+col+1,projection='3d')
            if method=='points':ax.scatter(*p[keep].T,c=s['mach'][keep],cmap=cmap,norm=norm,s=2,depthshade=False)
            else:
                with np.load(output/f'bubble_{method}.npz') as a:f=a['faces']
                shown=f[np.all(keep[f],axis=1)]
                ax.add_collection3d(Poly3DCollection(p[shown],facecolors=cmap(norm(np.mean(s['mach'][shown],axis=1))),edgecolors='none'))
                unused=keep&~np.isin(np.arange(len(p)),np.unique(f));ax.scatter(*p[unused].T,c='.6',s=2,depthshade=False)
            ax.set(xlim=(.1,.9),ylim=(.1,.9),zlim=(.1,.9),box_aspect=(1,1,1),xlabel='x',ylabel='y',zlabel='z')
            ax.view_init(elev=25,azim=-65);ax.set_title(DISPLAY_NAMES[method]+(' | z ≤ .5 cutaway' if cut else ' | full view'))
    fig.suptitle(f'128³, t=.16: unchanged samples | reconstruction choice from controlled inputs: {DISPLAY_NAMES[chosen]}',fontsize=14)
    fig.text(.025,.025,'Same vertices and fixed Mach scale 0–16; face color is mean vertex Mach. Cutaways remove faces only for display.\n'
             'No smoothing of vertex coordinates or changes to diagnostic area, heating, histograms or tracking. Known topology limitations remain.',fontsize=10)
    fig.subplots_adjust(left=.015,right=.92,bottom=.12,top=.91,wspace=.03,hspace=.1)
    cb=fig.add_axes([.95,.25,.012,.5]);fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cb,label='Mach')
    fig.savefig(output/'bubble_mesh_comparison.png',dpi=130);plt.close(fig)


def run(output,surface,partitions):
    output.mkdir(parents=True,exist_ok=False);inputs=cases();rows=[];reports=[];meshes={}
    for case in inputs:
        np.savez_compressed(output/f"input_{case['name']}.npz",**{k:v for k,v in case.items() if isinstance(v,np.ndarray)},dx=case['dx'])
        for method in METHODS:
            start=time.perf_counter();f,r=build(case,method)
            metrics=evaluate_mesh(case,f);row=dict(case=case['name'],method=method,**metrics,**mesh_topology(f))
            rows.append(row);reports.append(dict(case=case['name'],method=method,construction=r,metrics=metrics,seconds=time.perf_counter()-start))
            meshes[case['name'],method]=f;np.savez_compressed(output/f"{case['name']}_{method}.npz",faces=f)
            print(row,flush=True)
    plot_controls(output,inputs,meshes);csv_write(output/'summary.csv',rows)
    # Freeze a display candidate using sampled hole guards, separately labeled
    # sheets, edge topology and analytic sphere/cap area error. The close
    # same-group counterexample remains a separate unresolved limitation.
    eligible=[]
    for method in METHODS:
        good=True
        for case in inputs:
            baseline=next(r for r in rows if r['case']==case['name'] and r['method']=='local')
            tested=next(r for r in rows if r['case']==case['name'] and r['method']==method)
            if tested['sampled_hole_intrusion_faces']>baseline['sampled_hole_intrusion_faces']:good=False
            if case['name']=='near_sheets_separate_groups' and tested['cross_stipulated_sheet_faces']:good=False
            if tested['nonmanifold_edges'] or tested['inconsistent_shared_edge_winding']:good=False
        score=-sum(abs(1-r['area_fraction']) for r in rows if r['method']==method and r['area_fraction'] is not None)
        if good:eligible.append((score,method))
    if not eligible:chosen='local'
    else:chosen=max(eligible,key=lambda x:(x[0],x[1]=='local'))[1]
    selection=dict(chosen=chosen,eligible=eligible,criterion='No extra sampled known-hole intrusions versus local, no separately labeled sheet bridges or edge topology violations; then minimum sphere/cap area-fraction error. Same-group ambiguity remains a separate failure control.',
                   truth_used_for_bubble_reconstruction=False,physical_reconstruction_validated=False)
    (output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [surface,partitions]}
    with np.load(surface) as a:s={k:a[k].copy() for k in a.files}
    with np.load(partitions) as a:labels=a['native_cells_a60_labels']
    bubble=dict(positions=s['positions'],normals=s['normals'],labels=labels,dx=float(s['grid_spacing']))
    # Same event is shown for local and the selected alternative (if distinct).
    for method in dict.fromkeys(['local',chosen]):
        f,r=build(bubble,method)
        np.savez_compressed(output/f'bubble_{method}.npz',positions=s['positions'],normals=s['normals'],mach=s['mach'],labels=labels,faces=f,time=s['time'])
        reports.append(dict(case='bubble_event0016',method=method,construction=r))
    plot_bubble(output,s,chosen)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (output/'construction.json').write_text(json.dumps(reports,indent=2)+'\n')
    import open3d as o3d
    sources=[Path(__file__)]+[Path('experiments/wind_bubble')/f'{n}.py' for n in ['ball_pivot_mesh','mesh_benchmark_cases','surface_mesh']]
    (output/'metadata.json').write_text(json.dumps(dict(input_sha256=hashes,input_hashes_unchanged=True,open3d=o3d.__version__,
        runtime={name:importlib.metadata.version(name) for name in ['numpy','scipy','matplotlib','open3d','dash','plotly','pandas']},
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},vertices_moved=False,
        physical_diagnostics_changed=False,physical_identities_assigned=False),indent=2)+'\n')
    print('Selected from controls:',selection,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--surface',type=Path,default=Path('outputs/two_bubbles_budget_matched_iwr_compgpu3_n128/online/surfaces_0016.npz'))
    p.add_argument('--partitions',type=Path,default=Path('outputs/two_bubbles_resolution_comparison_n064_n128/groups_n128_event0016.npz'))
    a=p.parse_args();run(a.output_dir,a.surface,a.partitions)
