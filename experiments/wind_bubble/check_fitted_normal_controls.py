"""Known-geometry controls for display-normal fitting, including full BPA reruns."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.wind_bubble.compare_patch_normals import plane_normals
from experiments.wind_bubble.mesh_benchmark_cases import cases, evaluate_mesh
from experiments.wind_bubble.ball_pivot_mesh import ball_pivot_samples, guard_faces
from experiments.wind_bubble.surface_mesh import mesh_topology
from experiments.wind_bubble.verify_mesh_comparison import check_geometry
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write


def run(output, reference):
    output.mkdir(parents=True, exist_ok=False)
    rows = []; images = []; reports = []; reference_hashes = {}
    for case in cases():
        p, saved, labels, dx = [case[k] for k in ['positions','normals','labels','dx']]
        baseline, baseline_report, raw = ball_pivot_samples(p, saved, labels, dx, radius_factors=(1.,))
        prior = reference / f"{case['name']}_ball_single.npz"
        reference_hashes[str(prior)] = hashlib.sha256(prior.read_bytes()).hexdigest()
        with np.load(prior) as a:
            assert {tuple(sorted(f)) for f in baseline} == {tuple(sorted(f)) for f in a['faces']}
        np.savez_compressed(output/f"input_{case['name']}.npz", positions=p, saved_normals=saved,
                            labels=labels, truth_sheet=case['truth_sheet'], dx=dx, original_raw_faces=raw)
        for mode, radius in [('saved',None),('plane_2cells',2),('plane_3cells',3)]:
            if radius is None:
                normals = saved.copy(); valid = np.ones(len(p), bool); counts = np.zeros(len(p),int)
                faces, report, new_raw = baseline, baseline_report, raw
            else:
                normals, valid, counts, _ = plane_normals(p, saved, labels, radius*dx)
                faces, report, new_raw = ball_pivot_samples(p, normals, labels, dx, radius_factors=(1.,))
            fixed_faces, fixed_report = guard_faces(p,normals,labels,raw,dx)
            for stage, tested in [('fixed_proposals',fixed_faces),('rebuilt_bpa',faces)]:
                # Independently check every geometric guard against the normal set actually used.
                check_geometry(p,normals,labels,tested,dx)
                metrics = evaluate_mesh(case,tested)
                tri = p[tested]; edges = np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2)
                semi = edges.sum(axis=1)/2
                heron = np.sqrt(np.maximum(0,semi*np.prod(semi[:,None]-edges,axis=1))).sum()
                np.testing.assert_allclose(heron,metrics['mesh_area'],rtol=1e-10,atol=1e-10)
                rows.append(dict(case=case['name'],mode=mode,stage=stage,invalid_fits=int((~valid).sum()),
                                 **metrics,**mesh_topology(tested)))
            np.savez_compressed(output/f"{case['name']}_{mode}.npz",positions=p,saved_normals=saved,
                                display_normals=normals,fit_valid=valid,neighbor_count=counts,
                                faces=faces,raw_faces=new_raw,fixed_proposal_faces=fixed_faces)
            reports.append(dict(case=case['name'],mode=mode,rebuilt=report,fixed_guard_rejections=fixed_report))
            images.append((case,mode,faces))
            print(rows[-1],flush=True)
    csv_write(output/'summary.csv',rows)
    (output/'construction.json').write_text(json.dumps(reports,indent=2)+'\n')
    render(images,output)
    # Reopen every artifact, independently regenerate the known input, and check unchanged points/normals.
    checks = 0
    for case in cases():
        for mode in ['saved','plane_2cells','plane_3cells']:
            with np.load(output/f"{case['name']}_{mode}.npz") as a:
                np.testing.assert_array_equal(a['positions'],case['positions'])
                np.testing.assert_array_equal(a['saved_normals'],case['normals'])
                for stage,key in [('fixed_proposals','fixed_proposal_faces'),('rebuilt_bpa','faces')]:
                    check_geometry(a['positions'],a['display_normals'],case['labels'],a[key],case['dx'])
                    expected = next(r for r in rows if r['case']==case['name'] and r['mode']==mode and r['stage']==stage)
                    for k,v in evaluate_mesh(case,a[key]).items():
                        if v is not None: np.testing.assert_allclose(v,expected[k],atol=1e-12,rtol=1e-12)
                    checks += 1
    for path,digest in reference_hashes.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    import open3d
    sources = [Path(__file__)]+[Path('experiments/wind_bubble')/f'{name}.py' for name in
                               ['compare_patch_normals','ball_pivot_mesh','mesh_benchmark_cases','surface_mesh']]
    metadata = dict(checked_case_mode_stage_combinations=checks,original_arrays_exact=True,
                    original_baseline_faces_reproduced=True,open3d_version=open3d.__version__,
                    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    reference_sha256=reference_hashes,ball_radius_cells=1.,thresholds_changed=False,
                    production_method_changed=False,physical_diagnostics_changed=False,
                    limits=['Hole intrusion uses sampled barycentric probes, not exact continuous intersections.',
                            'Same-group nearby sheets are an explicit known failure control.',
                            'Analytic area error applies only to sphere and spherical cap.',
                            'No global self-intersection or vertex-manifold proof.'])
    (output/'readback.json').write_text(json.dumps(metadata,indent=2)+'\n')


def render(images,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(12,17))
    for index,(case,mode,faces) in enumerate(images):
        ax=fig.add_subplot(5,3,index+1,projection='3d');p=case['positions']
        mixed=np.ptp(case['truth_sheet'][faces],axis=1)>0
        colors=np.tile([.1,.5,.55,1.],(len(faces),1));colors[mixed]=[.85,.2,.15,1.]
        ax.add_collection3d(Poly3DCollection(p[faces],facecolors=colors,edgecolors='none'))
        unused=np.setdiff1d(np.arange(len(p)),np.unique(faces));ax.scatter(*p[unused].T,c='.5',s=2)
        center=(p.min(0)+p.max(0))/2;radius=max(np.ptp(p,axis=0).max()/2,.1)*1.08
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),
               zlim=(center[2]-radius,center[2]+radius),box_aspect=(1,1,1))
        ax.view_init(elev=60 if 'hole' in case['name'] else 25,azim=-65)
        ax.set_title(f'{case["name"].replace("_"," ")}\n{mode.replace("_"," ")} | {len(faces)} faces',fontsize=9)
        ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
    fig.suptitle('Known-geometry controls: rerun single-radius ball pivoting with each normal set')
    fig.text(.025,.015,'Same original positions and fixed guards. Red: triangles connecting different stipulated sheets; teal: other faces; gray: unused points.\n'
             'Same-group sheets remain a failure control. Fitted directions are display-only; truth labels are used only for evaluation.',fontsize=9)
    fig.subplots_adjust(left=.01,right=.99,bottom=.055,top=.95,wspace=.02,hspace=.3)
    fig.savefig(output/'fitted_normal_controls.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=Path('outputs/two_bubbles_mesh_method_comparison'))
    args=parser.parse_args();run(args.output_dir,args.reference)
