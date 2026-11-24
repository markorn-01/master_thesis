"""Compare old/corrected detection on the same saved three-bubble gas states.

No integration, grouping, or smoothing is run. The historical modules are
loaded from an explicit Git revision and verified against the frozen analytic
benchmark's source hashes. Old and corrected selection then run on the same
backend; archived GPU masks are an additional comparison, not silently assumed
to be bitwise reproducible on CPU.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import types

import jax
import jax.numpy as jnp
import numpy as np

from astronomix._physics_modules._shock_finder._gradients import _calculate_shock_direction
from experiments.shock.check_colliding_shocks import fingerprint, write_csv
from experiments.wind_bubble.audit_collision_detection import BOUNDS
from experiments.wind_bubble.audit_detection_stages import stage_fields
from experiments.wind_bubble.run_single_bubble import build_problem

BASELINE_REVISION='07a8ec16a32ee50a3f444008942594d68aea54f8'
ROOTS=(Path('outputs/three_bubble_collision_detection_audit_iwr_compgpu8_n064'),
       Path('outputs/three_bubble_collision_resolution_iwr_compgpu8_n128'))
MODULE_ROOT='astronomix/_physics_modules/_shock_finder/'


def load_baseline(revision,filename,expected):
    import hashlib
    path=MODULE_ROOT+filename
    source=subprocess.run(['git','show',f'{revision}:{path}'],check=True,
                          capture_output=True).stdout
    if hashlib.sha256(source).hexdigest()!=expected:
        raise AssertionError(f'Historical source differs from frozen benchmark: {path}')
    module=types.ModuleType('collision_baseline_'+Path(filename).stem)
    exec(compile(source,f'{revision}:{path}','exec'),module.__dict__)
    return module,source.decode()


def function_ast(source,name):
    return ast.dump(next(node for node in ast.parse(source).body
                         if isinstance(node,ast.FunctionDef) and node.name==name))


def run(args):
    if args.output_dir.exists():
        raise ValueError('Use a fresh output directory.')
    baseline_path=Path('outputs/colliding_shocks_reference_n032_n064/verification.json')
    frozen=json.loads(baseline_path.read_text())
    old_zones,zone_source=load_baseline(args.baseline_revision,'_shock_zones.py',
        frozen['source_sha256'][MODULE_ROOT+'_shock_zones.py'])
    old_surface,_=load_baseline(args.baseline_revision,'_shock_surface.py',
        frozen['source_sha256'][MODULE_ROOT+'_shock_surface.py'])
    current_zones=Path(MODULE_ROOT+'_shock_zones.py').read_text()
    for name in ('_shock_zone_criterion_converging_flow','_shock_zone_criterion_aligned_gradients',
                 '_shock_zone_criterion_minimum_mach','identify_shock_zones','get_post_pre_shock_values'):
        if function_ast(zone_source,name)!=function_ast(current_zones,name):
            raise AssertionError(f'Earlier detection criteria changed: {name}')
    gradient_path=MODULE_ROOT+'_gradients.py'
    if fingerprint(gradient_path)!=frozen['source_sha256'][gradient_path]:
        raise AssertionError('Reference normal/gradient calculation changed.')
    source_paths=[Path(__file__),Path('experiments/wind_bubble/audit_detection_stages.py'),
                  Path('experiments/wind_bubble/run_single_bubble.py')]
    source_paths+=sorted(Path(MODULE_ROOT).glob('*.py'))
    source_hashes={str(p):fingerprint(p) for p in source_paths}
    inputs={str(baseline_path):fingerprint(baseline_path)}
    args.output_dir.mkdir(parents=True)
    rows=[];archive_differences=[];plot_data=[]
    for root in args.roots:
        meta_path=root/'metadata.json';inputs[str(meta_path)]=fingerprint(meta_path)
        meta=json.loads(meta_path.read_text());n=meta['num_cells']
        verification_path=root/'audit'/'verification.json'
        verification=json.loads(verification_path.read_text())
        inputs[str(verification_path)]=fingerprint(verification_path)
        expected_states={Path(k).name:v for k,v in verification['state_sha256'].items()}
        _,config,_,registry,helper=build_problem(argparse.Namespace(num_cells=n,
            num_snapshots=17,t_end=meta['t_end'],num_injection_cells=n//16,plasma_beta=None))
        centers=np.asarray(helper.geometric_centers)
        roi=((centers>=BOUNDS[:,0])&(centers<=BOUNDS[:,1])).all(axis=-1)
        plane=np.zeros(roi.shape,bool);plane[:,:,n//2]=True
        def analyze(state):
            direction=_calculate_shock_direction(state[registry.pressure_index],state[registry.density_index],config)
            zone=old_zones.identify_shock_zones(state,config,registry,helper,direction)
            surface=old_surface.identify_shock_surface(state,zone,direction,config,registry)
            return stage_fields(state,config,registry,helper),zone,surface
        compiled=jax.jit(analyze)
        for path in sorted((root/'audit').glob('state_*.npy')):
            index=int(path.stem.split('_')[-1]);digest=fingerprint(path)
            if digest!=expected_states[path.name]:
                raise AssertionError(f'Saved gas state changed: {path}')
            inputs[str(path)]=digest
            state=np.load(path,allow_pickle=False)
            fields,zone_before,surface_before=compiled(jnp.asarray(state))
            fields={k:np.asarray(v) for k,v in fields.items()}
            zone_before=np.asarray(zone_before);surface_before=np.asarray(surface_before)
            np.testing.assert_array_equal(zone_before,fields['shock_zone'])
            if (fields['shock_surface']&~fields['shock_zone']).any():
                raise AssertionError('A corrected surface escaped the shock-zone mask.')
            time=float(meta['actual_times'][index])
            for name,mask in (('central_volume',roi),('central_slice',roi&plane)):
                row=dict(resolution=n,event=index,time=time,region=name,cells=int(mask.sum()),
                    compression=int((fields['compression']&mask).sum()),
                    gradient_alignment=int((fields['gradient_alignment']&mask).sum()),
                    pressure_jump=int((fields['pressure_jump_pass']&mask).sum()),
                    temperature_jump=int((fields['temperature_jump_pass']&mask).sum()),
                    zone_before=int((zone_before&mask).sum()),zone_after=int((fields['shock_zone']&mask).sum()),
                    surface_before=int((surface_before&mask).sum()),surface_after=int((fields['shock_surface']&mask).sum()),
                    domain_surface_before=int(surface_before.sum()),domain_surface_after=int(fields['shock_surface'].sum()))
                rows.append(row)
            crop_path=root/'audit'/f'collision_crop_{index:04d}.npz'
            inputs[str(crop_path)]=fingerprint(crop_path)
            with np.load(crop_path) as archive:
                ijk=np.rint(archive['centers']/float(config.grid_spacing)-.5).astype(int)
                crop_index=tuple(ijk[...,axis] for axis in range(3))
                archive_differences.append(dict(resolution=n,event=index,
                    zone_differences_old_cpu_vs_archived=int(np.count_nonzero(zone_before[crop_index]!=archive['shock_zone'])),
                    surface_differences_old_cpu_vs_archived=int(np.count_nonzero(surface_before[crop_index]!=archive['shock_surface']))))
            if index in (8,16):
                iz=n//2
                plot_data.append(dict(resolution=n,event=index,time=time,
                    density=state[registry.density_index,:,:,iz],old=surface_before[:,:,iz],
                    new=fields['shock_surface'][:,:,iz],zone=zone_before[:,:,iz]))
            print(f'{n}³ t={time:.2f}: central volume zone={rows[-2]["zone_after"]}; '
                  f'surface {rows[-2]["surface_before"]} → {rows[-2]["surface_after"]}',flush=True)
        del compiled,fields,state
        jax.clear_caches()
    write_csv(args.output_dir/'central_stage_counts.csv',rows)
    write_csv(args.output_dir/'archived_backend_comparison.csv',archive_differences)
    plot(args.output_dir,plot_data)
    for path,expected in {**inputs,**source_hashes}.items():
        if fingerprint(path)!=expected:
            raise AssertionError(f'Input or source changed during review: {path}')
    later=[r for r in rows if r['event']>=8]
    report=dict(completed=True,backend=jax.default_backend(),baseline_revision=args.baseline_revision,
        saved_states_checked=len(rows)//2,threshold_function_asts_unchanged=True,
        full_domain_zone_masks_exact_before_after=True,
        central_counts_unchanged=all(r['surface_before']==r['surface_after'] for r in rows),
        later_central_zones_empty=all(r['zone_after']==0 for r in later),
        earlier_stage_rejection_remains=all(r['zone_after']==r['surface_after']==0 for r in later),
        simulation_evolved=False,input_sha256=inputs,source_sha256=source_hashes,
        scope='Same saved physical states, same-backend old/new detector comparison. '
              'Not a new simulation or a proof that no physical shock is present in the rejected layer.')
    (args.output_dir/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('sha256')},indent=2))
    return report


def plot(output,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(len(data),3,figsize=(10,2.7*len(data)),dpi=140,squeeze=False)
    fig.subplots_adjust(left=.08,right=.97,top=.91,bottom=.08,hspace=.65,wspace=.30)
    for row_index,(row,snap) in enumerate(zip(axes,data)):
        n=snap['resolution'];x=(np.arange(n)+.5)/n
        for ax,key,title in zip(row,('old','new','zone'),('Before: selected surface','After: selected surface','Shock zone (unchanged)')):
            ax.imshow(snap['density'].T,origin='lower',extent=(0,1,0,1),cmap='gray_r')
            xx,yy=np.nonzero(snap[key]);ax.scatter(x[xx],x[yy],s=3,color='#d8582b')
            ax.add_patch(Rectangle((.45,.335),.10,.08,fill=False,color='#087f8c',lw=1.2))
            ax.set(xlim=(.40,.60),ylim=(.28,.47),
                   xlabel='x' if row_index==len(data)-1 else '',ylabel='y',
                   title=f'{title}\n{n}³, t={snap["time"]:.2f}')
    fig.suptitle('Saved three-bubble collision: opposing-front corrections do not fill the central gap',fontsize=13)
    fig.text(.08,.025,'Gray: density. Orange: selected cells / zone. Teal box: counted central region.\n'
        'Old and corrected finders use identical saved gas states on the same backend; shock-zone criteria are unchanged.',fontsize=9)
    fig.savefig(output/'bubble_before_after.png');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',type=Path,nargs='+',default=list(ROOTS))
    parser.add_argument('--baseline-revision',default=BASELINE_REVISION)
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args())
