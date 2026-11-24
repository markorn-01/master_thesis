"""Apply fixed reference-calibrated, read-only jump diagnostics to saved bubbles."""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates

from experiments.shock.check_colliding_shocks import fingerprint,write_csv
from experiments.shock.pressure_velocity_diagnostic import DiagnosticSettings,analyze_state,summarize,jump_diagnostics
from experiments.wind_bubble.audit_collision_detection import BOUNDS
from experiments.wind_bubble.three_bubble_setup import SOURCES

ROOTS=(Path('outputs/three_bubble_collision_detection_audit_iwr_compgpu8_n064'),
       Path('outputs/three_bubble_collision_resolution_iwr_compgpu8_n128'))
OUTER_BOUNDS=np.array([[.05,.3125],[.335,.415],[.46,.54]])


def independent_side_probe(state,records,dx,settings):
    """Bounded sampling check on four outer rays closest to the source axis.

    Try the same declared distance set independently on each side. No
    thresholds are relaxed. This probe is a sensitivity check, not another
    search for missed shocks or a replacement of the guarded candidate test.
    """
    by_candidate={}
    for row in records:
        by_candidate.setdefault(row['candidate_index'],row)
    chosen=sorted(by_candidate.values(),key=lambda r:(r['y']-.375)**2+(r['z']-.5)**2)[:4]
    def sample(point):
        return np.array([map_coordinates(field,point[:,None],order=1,mode='nearest')[0] for field in state])
    rows=[]
    for r in chosen:
        center=np.array([r['x'],r['y'],r['z']])/dx-.5
        normal=np.array([r['normal_'+axis] for axis in 'xyz'])
        for a,b in itertools.product(settings.half_widths,repeat=2):
            up,down=sample(center-a*normal),sample(center+b*normal)
            outer_up=sample(center-(a+settings.endpoint_step)*normal)
            outer_down=sample(center+(b+settings.endpoint_step)*normal)
            metrics=jump_diagnostics(up,down,normal,settings.gamma)
            dp=max(abs(down[4]-up[4]),1e-30)
            du=max(abs((up[1:4]-down[1:4])@normal),1e-30)
            plateau=max(abs(outer_up[4]-up[4])/dp,abs(outer_down[4]-down[4])/dp,
                abs((outer_up[1:4]-up[1:4])@normal)/du,
                abs((outer_down[1:4]-down[1:4])@normal)/du)
            error=max(metrics['pressure_velocity_relative_error'],metrics['density_jump_relative_error'],
                metrics['momentum_relative_error'],metrics['enthalpy_relative_error'],
                metrics['tangential_jump_over_upstream_sound_speed'],plateau)
            rows.append(dict(candidate_index=r['candidate_index'],x=r['x'],y=r['y'],z=r['z'],
                upstream_distance_cells=a,downstream_distance_cells=b,
                largest_relative_error=error,endpoint_pressure_velocity_change=plateau,**metrics))
    return rows


def run(args):
    if args.output_dir.exists():
        raise ValueError('Use a fresh output directory.')
    settings=DiagnosticSettings()
    sources=[Path(__file__),Path('experiments/shock/pressure_velocity_diagnostic.py'),
        Path('experiments/wind_bubble/assess_collision_physics.py'),
        Path('experiments/wind_bubble/three_bubble_setup.py'),
        Path('astronomix/_physics_modules/_shock_finder/_shock_surface.py')]
    source_hashes={str(p):fingerprint(p) for p in sources};input_hashes={}
    args.output_dir.mkdir(parents=True)
    summaries=[];records=[];windows=[];figures=[];side_probes=[]
    for root in args.roots:
        meta_path=root/'metadata.json';meta=json.loads(meta_path.read_text())
        input_hashes[str(meta_path)]=fingerprint(meta_path)
        audit_path=root/'audit'/'verification.json';audit=json.loads(audit_path.read_text())
        input_hashes[str(audit_path)]=fingerprint(audit_path)
        expected={Path(p).name:h for p,h in audit['state_sha256'].items()}
        if not audit['passed'] or meta['status']!='complete':
            raise ValueError('Require completed, verified saved-state runs.')
        n=meta['num_cells'];dx=1/n
        np.testing.assert_array_equal(meta['source_positions'],SOURCES)
        for path in sorted((root/'audit').glob('state_*.npy')):
            digest=fingerprint(path)
            if digest!=expected[path.name]:
                raise AssertionError(f'Saved state differs from original audit: {path}')
            input_hashes[str(path)]=digest
            state=np.load(path,allow_pickle=False)
            event=int(path.stem.split('_')[-1]);time=float(meta['actual_times'][event])
            catalog_path=root/'online'/f'surfaces_{event:04d}.npz'
            input_hashes[str(catalog_path)]=fingerprint(catalog_path)
            with np.load(catalog_path) as catalog:
                tree=cKDTree(catalog['positions'])
            for region,bounds in (('central',BOUNDS),('outer_control',OUTER_BOUNDS)):
                rows,geometry=analyze_state(state,dx,settings,sources=SOURCES,
                    source_radius=meta['injection_radius'],candidate_bounds=bounds)
                summary=dict(resolution=n,event=event,time=time,region=region,**summarize(rows,geometry))
                summary['maximum_sampled_pressure_mach']=max((r['pressure_mach'] for r in rows),default=None)
                summaries.append(summary)
                for r in rows:
                    r.update(resolution=n,event=event,time=time,region=region,
                        nearest_original_surface_cells=float(tree.query([r['x'],r['y'],r['z']])[0]/dx))
                records.extend(rows)
                if region=='outer_control' and n==128 and event==16:
                    side_probes=independent_side_probe(state,rows,dx,settings)
                for w in geometry['sampling_windows']:
                    windows.append(dict(resolution=n,event=event,time=time,region=region,**w))
                if region=='central' and event in (4,8,16):
                    points=(geometry['coordinates']+.5)*dx
                    pv={r['candidate_index'] for r in rows if r['pressure_velocity_10pct']}
                    full={r['candidate_index'] for r in rows if r['full_jump_10pct']}
                    figures.append(dict(resolution=n,event=event,time=time,density=state[0,:,:,n//2],
                        points=points,pv=pv,full=full,summary=summary))
                print(f'{n}³ t={time:.2f} {region}: geometric={summary["geometric_candidates"]}, '
                      f'PV={summary["pressure_velocity_candidates_10pct"]}, full={summary["full_jump_candidates_10pct"]}',flush=True)
    write_csv(args.output_dir/'summary.csv',summaries)
    if records:
        write_csv(args.output_dir/'state_pairs.csv',records)
    write_csv(args.output_dir/'sampling_windows.csv',windows)
    if side_probes:
        write_csv(args.output_dir/'outer_independent_side_probe.csv',side_probes)
    plot(args.output_dir,figures)
    for path,expected in {**source_hashes,**input_hashes}.items():
        if fingerprint(path)!=expected:
            raise AssertionError(f'Input/source changed: {path}')
    central_late=[r for r in summaries if r['region']=='central' and r['event']>=8]
    outer=[r for r in summaries if r['region']=='outer_control']
    outer_full=sum(r['full_jump_candidates_10pct'] for r in outer)
    result=dict(completed=True,diagnostic_only=True,production_changed=False,simulation_evolved=False,
        settings=asdict(settings),saved_states=len(summaries)//2,
        late_central_pressure_velocity_candidates_10pct=sum(r['pressure_velocity_candidates_10pct'] for r in central_late),
        late_central_full_jump_candidates_10pct=sum(r['full_jump_candidates_10pct'] for r in central_late),
        late_central_candidates_without_valid_sampling_window=sum(r['candidates_with_no_valid_window'] for r in central_late),
        outer_control_pressure_velocity_candidates_10pct=sum(r['pressure_velocity_candidates_10pct'] for r in outer),
        outer_control_full_jump_candidates_10pct=outer_full,
        full_jump_gate_passes_bubble_positive_control=outer_full>0,
        central_non_detection_can_rule_out_shocks=False,
        independent_side_probe_pairs=len(side_probes),
        independent_side_probe_best_relative_error=min((r['largest_relative_error'] for r in side_probes),default=None),
        input_sha256=input_hashes,source_sha256=source_hashes,
        scope='Pressure/convergence candidates and necessary jump compatibility on saved states; no physical-shock proof, new production detection, Mach output, or thermalization output.')
    (args.output_dir/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('sha256')},indent=2))
    return result


def plot(output,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    resolutions=sorted({r['resolution'] for r in data})
    events=sorted({r['event'] for r in data})
    fig,axes=plt.subplots(len(resolutions),len(events),figsize=(4*len(events),3.8*len(resolutions)),
                          dpi=150,squeeze=False)
    fig.subplots_adjust(left=.07,right=.98,top=.86,bottom=.20,wspace=.26,hspace=.52)
    for row,n in zip(axes,resolutions):
        for ax,event in zip(row,events):
            snap=next(r for r in data if r['resolution']==n and r['event']==event)
            ax.imshow(snap['density'].T,origin='lower',extent=(0,1,0,1),cmap='gray_r')
            points=snap['points'];plane=abs(points[:,2]-(n//2+.5)/n)<=.75/n if len(points) else np.zeros(0,bool)
            for group,color,marker,label in ((set(range(len(points))),'#d98a27','o','Pressure/compression location'),
                (snap['pv'],'#b544a5','s','Pressure/velocity compatible'),
                (snap['full'],'#087f8c','x','Full jump compatible')):
                selected=[i for i in sorted(group) if plane[i]]
                if selected:
                    ax.scatter(points[selected,0],points[selected,1],s=22,color=color,marker=marker,label=label)
            ax.add_patch(Rectangle((.45,.335),.10,.08,fill=False,color='#087f8c',lw=1))
            summary=snap['summary']
            ax.set(xlim=(.40,.60),ylim=(.28,.47),xlabel='x',ylabel='y',
                title=f'{n}³ t={snap["time"]:.2f}\n3D box: candidates={summary["geometric_candidates"]}, '
                      f'PV={summary["pressure_velocity_candidates_10pct"]}, full={summary["full_jump_candidates_10pct"]}')
    handles,labels=[],[]
    for ax in axes.flat:
        h,l=ax.get_legend_handles_labels()
        for handle,label in zip(h,l):
            if label not in labels:handles.append(handle);labels.append(label)
    if handles:fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.09),fontsize=8)
    fig.suptitle('Separate pressure/velocity diagnostic on saved bubble states | no production changes',fontsize=14)
    fig.text(.07,.025,'Gray: density slice. Symbols: candidates within 0.75 cell of this slice. Counts: full counted 3D box.\n'
        'PV-only candidates can also occur in smooth compression; even full jump compatibility is necessary evidence, not proof of a shock.\n'
        'Pairs crossing source spheres, domain bounds, or opposing pressure directions cannot supply accepted evidence.',fontsize=8.5)
    fig.savefig(output/'pressure_velocity_overlay.png');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',type=Path,nargs='+',default=list(ROOTS))
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args())
