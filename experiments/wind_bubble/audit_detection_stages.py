"""Capture selected physical states and audit the unchanged shock-finder stages.

All ordinary diagnostic events and solver timestep limits remain unchanged.
Full-state storage is deliberate validation output, not routine compact output.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from experiments.wind_bubble.run_two_bubbles import run as run_simulation, SOURCES, INJECTION_RADIUS
from experiments.wind_bubble.run_single_bubble import build_problem
from astronomix._physics_modules._shock_finder._gradients import _calculate_shock_direction, _calculate_velocity_divergence
from astronomix._physics_modules._shock_finder._shock_zones import (
    _shock_zone_criterion_converging_flow, _shock_zone_criterion_aligned_gradients,
    _shock_zone_criterion_minimum_mach, identify_shock_zones, get_post_pre_shock_values,
    _make_interior_mask,
)
from astronomix._physics_modules._shock_finder._shock_surface import identify_shock_surface, calculate_shock_surface_offsets


def selected_indices(indices,count):
    values=sorted(set(indices))
    if not values or values[0]<0 or values[-1]>=count:
        raise ValueError('Capture indices must be within the unchanged analysis schedule.')
    return values


class StateCapture:
    def __init__(self,directory,times):
        self.directory=directory;self.times=np.asarray(times);self.saved=[]

    def receive(self,time,state):
        index=int(np.argmin(abs(self.times-float(time))))
        if abs(self.times[index]-float(time))>1e-6:raise ValueError('Unexpected capture time.')
        if index in self.saved:raise ValueError('Duplicate capture.')
        array=np.asarray(state)
        if array.ndim!=4 or not np.isfinite(array).all():raise ValueError('Invalid physical state.')
        self.directory.mkdir(parents=True,exist_ok=True)
        np.save(self.directory/f'state_{index:04d}.npy',array,allow_pickle=False)
        self.saved.append(index)

    def observer(self,indices):
        targets=jnp.asarray(self.times[indices])
        def observe(time,physical):
            def save(_):
                jax.debug.callback(self.receive,time,physical,ordered=True)
            jax.lax.cond(jnp.any(jnp.abs(time-targets)<1e-6),save,lambda _:None,operand=None)
        return observe


def stage_fields(state,config,registry,helper):
    """Use production tests; expose measured jumps without changing thresholds."""
    pressure=state[registry.pressure_index];density=state[registry.density_index]
    direction=_calculate_shock_direction(pressure,density,config)
    compression=_shock_zone_criterion_converging_flow(state,config,registry)
    gradient=_shock_zone_criterion_aligned_gradients(pressure,density,config)
    jump=_shock_zone_criterion_minimum_mach(state,config,registry,helper,direction)
    zone=identify_shock_zones(state,config,registry,helper,direction)
    surface=identify_shock_surface(state,zone,direction,config,registry)
    offsets=calculate_shock_surface_offsets(state,surface,direction,config,registry)
    p2,p1,t2,t1=get_post_pre_shock_values(direction,pressure,pressure/density)
    logp=jnp.log(jnp.maximum(p2,1e-30))-jnp.log(jnp.maximum(p1,1e-30))
    logt=jnp.log(jnp.maximum(t2,1e-30))-jnp.log(jnp.maximum(t1,1e-30))
    # Mirror only the production threshold decomposition; assert exact agreement below.
    gamma=5/3;mach2=1.3**2
    pratio=(2*gamma*mach2-(gamma-1))/(gamma+1)
    tratio=pratio*((gamma-1)*mach2+2)/((gamma+1)*mach2)
    interior=_make_interior_mask(pressure.shape)
    return dict(compression=compression,gradient_alignment=gradient,minimum_jump=jump,shock_zone=zone,
        pressure_jump_pass=interior&(logp>=jnp.log(pratio)),temperature_jump_pass=interior&(logt>=jnp.log(tratio)),
        shock_surface=surface,surface_offsets=offsets,direction=jnp.moveaxis(direction,0,-1),
        divergence=_calculate_velocity_divergence(state,config,registry),
        log_pressure_jump=logp,log_temperature_jump=logt)


def audit_states(args,indices):
    _,config,_,registry,helper=build_problem(argparse.Namespace(num_cells=args.num_cells,
        num_snapshots=args.num_analysis,t_end=args.t_end,num_injection_cells=args.num_cells//16,plasma_beta=None))
    # Finder uses geometry/registry, not wind-source parameters. Same builder as simulation.
    compiled=jax.jit(lambda state:stage_fields(state,config,registry,helper))
    centers=np.asarray(helper.geometric_centers);dx=float(config.grid_spacing);rows=[];hashes={}
    for index in indices:
        path=args.output_dir/'audit'/f'state_{index:04d}.npy'
        state=np.load(path,allow_pickle=False);assert state.shape[1:]==centers.shape[:-1]
        fields={k:np.asarray(v) for k,v in compiled(jnp.asarray(state)).items()}
        np.testing.assert_array_equal(fields['shock_zone'],fields['compression']&fields['gradient_alignment']&fields['minimum_jump'])
        np.testing.assert_array_equal(fields['minimum_jump'],fields['pressure_jump_pass']&fields['temperature_jump_pass'])
        assert not (fields['shock_surface']&~fields['shock_zone']).any()
        positions=centers+dx*fields['surface_offsets'][...,None]*fields['direction']
        distances=np.linalg.norm(positions[...,None,:]-np.asarray(SOURCES),axis=-1)
        exported=fields['shock_surface']&np.isfinite(distances).all(axis=-1)&(distances>INJECTION_RADIUS).all(axis=-1)
        with np.load(args.output_dir/'online'/f'surfaces_{index:04d}.npz') as catalog:
            np.testing.assert_array_equal(np.argwhere(exported),catalog['grid_indices'])
            np.testing.assert_array_equal(positions[exported],catalog['positions'])
            time=float(catalog['time'])
        for source,center in enumerate(SOURCES):
            radius=np.linalg.norm(centers-np.asarray(center),axis=-1)
            radial=(centers-center)/np.maximum(radius[...,None],1e-30)
            inward=(fields['direction']*radial).sum(axis=-1)<-.5
            region=radius<=2*INJECTION_RADIUS
            injection=radius<=INJECTION_RADIUS-dx/2
            for name,roi in [('all_local',region),('local_inward',region&inward),
                             ('local_outside_source',region&~injection),('local_inward_outside_source',region&inward&~injection)]:
                row=dict(event=index,time=time,source=source,region=name,cells=int(roi.sum()))
                for key in ['compression','gradient_alignment','pressure_jump_pass','temperature_jump_pass','minimum_jump','shock_zone','shock_surface']:
                    row[key]=int((roi&fields[key]).sum())
                both=fields['compression']&fields['gradient_alignment']
                row['compression_and_gradient']=int((roi&both).sum())
                row['jump_fails_after_first_two']=int((roi&both&~fields['minimum_jump']).sum())
                row['zone_not_surface']=int((roi&fields['shock_zone']&~fields['shock_surface']).sum())
                row['surface_removed_by_export']=int((roi&fields['shock_surface']&~exported).sum())
                row['surface_inside_exclusion_sphere']=int((roi&fields['shock_surface']&(distances<=INJECTION_RADIUS).any(axis=-1)).sum())
                row['exported_surface']=int((roi&exported).sum());rows.append(row)
                code=fields['compression'].astype(int)+2*fields['gradient_alignment']+4*fields['minimum_jump']
                for pattern in range(8):row[f'criterion_pattern_{pattern:03b}']=int((roi&(code==pattern)).sum())
            # Full local primitive fields and criterion arrays, including injection interior.
            bounds=[np.flatnonzero(region.any(axis=tuple(j for j in range(3) if j!=i))) for i in range(3)]
            box=tuple(slice(int(b.min()),int(b.max())+1) for b in bounds)
            local={k:v[box] for k,v in fields.items()}
            np.savez_compressed(args.output_dir/'audit'/f'stages_{index:04d}_source{source}.npz',
                **local,state=state[(slice(None),)+box],centers=centers[box],exported_surface=exported[box],
                injection_mask=injection[box],region_mask=region[box],time=time,grid_spacing=dx,
                inside_export_exclusion_sphere=(distances<=INJECTION_RADIUS).any(axis=-1)[box],
                density_index=registry.density_index,pressure_index=registry.pressure_index)
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        print(f'Audited event {index}: production zone tests and exported surface positions match.',flush=True)
    with (args.output_dir/'audit'/'stage_counts.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return hashes


def compare_reference(root,reference):
    """Check replay products at all events, not only captured states."""
    a=json.loads((root/'metadata.json').read_text());b=json.loads((reference/'metadata.json').read_text())
    for key in ['num_cells','actual_times','source_strengths','source_positions','discrete_mass_rates',
                'discrete_energy_rates','injection_radius','wind_terminal_velocity','ambient_density','ambient_pressure']:
        np.testing.assert_allclose(a[key],b[key],rtol=1e-7,atol=1e-8,err_msg=key)
    for index in range(len(a['actual_times'])):
        for prefix in ['surfaces','histograms','slice']:
            name=f'{prefix}_{index:04d}.npz'
            with np.load(root/'online'/name) as x,np.load(reference/'online'/name) as y:
                assert set(x.files)==set(y.files)
                for key in x.files:
                    if x[key].dtype.kind=='f':np.testing.assert_allclose(x[key],y[key],rtol=1e-6,atol=1e-8,equal_nan=True,err_msg=f'{name}:{key}')
                    else:np.testing.assert_array_equal(x[key],y[key])
    return dict(passed=True,events=len(a['actual_times']),scope='Replay product agreement, not full-state online/offline equivalence.')


def run(args):
    indices=selected_indices(args.capture_events,args.num_analysis)
    # Parent directory is created by the simulation before any callbacks run.
    capture=StateCapture(args.output_dir/'audit',np.asarray(jnp.linspace(0.,args.t_end,args.num_analysis)))
    run_simulation(args,state_observer=capture.observer(indices))
    assert sorted(capture.saved)==indices
    hashes=audit_states(args,indices)
    reference=compare_reference(args.output_dir,args.reference_run) if args.reference_run else None
    sources=[Path(__file__),Path('experiments/wind_bubble/run_two_bubbles.py')]
    sources+=list(Path('astronomix/_physics_modules/_shock_finder').glob('*.py'))
    result=dict(passed=True,captured_events=indices,capture_times=capture.times[indices].tolist(),
        production_zone_masks_match=True,exported_indices_and_positions_match=True,reference_replay=reference,
        full_state_bytes=sum((args.output_dir/'audit'/f'state_{i:04d}.npy').stat().st_size for i in indices),
        state_sha256=hashes,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        physics_or_thresholds_changed=False,schedule_changed=False,
        scope='Diagnostic replay; cell counts are not physical shock counts, source ownership, or causal proof.')
    (args.output_dir/'audit'/'verification.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-cells',type=int,default=128)
    parser.add_argument('--num-analysis',type=int,default=17)
    parser.add_argument('--t-end',type=float,default=.16)
    parser.add_argument('--source-strengths',type=float,nargs=2,default=[1.,.5])
    parser.add_argument('--match-source-budget',type=Path,default=Path('outputs/two_bubbles_iwr_compgpu9_n064/metadata.json'))
    parser.add_argument('--capture-events',type=int,nargs='+',default=[10,12,14,15,16])
    parser.add_argument('--reference-run',type=Path)
    parser.add_argument('--require-gpu',action='store_true')
    parser.add_argument('--verify-offline',action='store_true',help='Only supported up to 64³.')
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args())
