"""Inspect reverse-candidate thermodynamic samples without changing shock physics.

Reruns the 21-event temporal setup; saves profiles and surface samples, not gas
snapshots. CPU output is separate evidence from the preserved GPU simulation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astronomix import time_integration
from experiments.shock.analyze_state import analyze_shock_state_device
from experiments.wind_bubble.run_single_bubble import (
    build_problem, split_radial_shock_candidates, _spherical_bin_median,
)


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract(state, analysis, registry, helper, config):
    result, mach, _, samples = analysis(state)
    p2, p1, rho2, rho1, valid, post, pre = map(np.asarray, samples)
    mask = np.asarray(result.shock_surface_cells, dtype=bool)
    normals = np.moveaxis(np.asarray(result.shock_direction), 0, -1)
    dx = float(config.grid_spacing)
    centers = np.asarray(helper.geometric_centers)
    positions = centers + dx*np.asarray(result.shock_surface_offsets)[..., None]*normals
    radii = np.linalg.norm(positions-.5, axis=-1)
    # Distances are relative to the refined center, in cell units.
    upstream = positions + dx*pre[..., None]*normals
    downstream = positions - dx*post[..., None]*normals
    _, _, separation = split_radial_shock_candidates(radii[mask], .0625, dx)
    rho = np.asarray(state[registry.density_index])
    pressure = np.asarray(state[registry.pressure_index])
    v = np.stack([np.asarray(state[i]) for i in
                  (registry.velocity_index.x, registry.velocity_index.y, registry.velocity_index.z)], -1)
    grid_r = np.linalg.norm(centers-.5, axis=-1)
    vr = np.sum(v*(centers-.5), axis=-1)/np.maximum(grid_r, 1e-30)
    flow_mach = np.abs(vr)/np.sqrt((5/3)*pressure/rho)
    edges = np.arange(0, .5+dx, dx)
    bins = np.digitize(grid_r, edges)-1
    fields = dict(density=rho, pressure=pressure, radial_velocity=vr, radial_flow_mach=flow_mach)
    medians = {k:_spherical_bin_median(x, bins, len(edges)-1) for k,x in fields.items()}
    profile = [dict(radius=float(r), **{k:float(x[i]) for k,x in medians.items()})
               for i,r in enumerate((edges[:-1]+edges[1:])/2)]
    surface = []
    for ijk in np.argwhere(mask & (radii>.0625)):
        i=tuple(ijk)
        band='reverse' if np.isfinite(separation) and radii[i]<=separation else 'forward'
        radial=(positions[i]-.5)/radii[i]
        surface.append(dict(ix=int(ijk[0]), iy=int(ijk[1]), iz=int(ijk[2]), band=band,
            radius=float(radii[i]), alignment=float(np.dot(normals[i],radial)),
            mach=float(mach[i]), sampling_valid=bool(valid[i]),
            upstream_radius=float(np.linalg.norm(upstream[i]-.5)),
            downstream_radius=float(np.linalg.norm(downstream[i]-.5)),
            rho1=float(rho1[i]), rho2=float(rho2[i]), p1=float(p1[i]), p2=float(p2[i]),
            upstream_inside_injection=bool(np.linalg.norm(upstream[i]-.5)<=.0625),
            downstream_inside_injection=bool(np.linalg.norm(downstream[i]-.5)<=.0625)))
    return profile, surface


def plot(events, path, *, num_cells=64):
    fig,axes=plt.subplots(4,3,figsize=(13,11),sharex=True)
    for col,(time,profile,surface) in enumerate(events):
        radius=np.array([r['radius'] for r in profile])
        rev=[r for r in surface if r['band']=='reverse']
        # Reproducible illustrative sample: valid reverse point nearest median radius.
        valid=[r for r in rev if r['sampling_valid']]
        sample=min(valid,key=lambda r:abs(r['radius']-np.median([v['radius'] for v in valid]))) if valid else None
        for row,(field,label) in enumerate(zip(
            ('density','pressure','radial_velocity','radial_flow_mach'),
            ('Density','Pressure','Radial velocity','Radial flow Mach'))):
            ax=axes[row,col]
            ax.plot(radius,[r[field] for r in profile],color='0.2',label='Shell median')
            ax.axvspan(0,.0625,color='0.9',label='Injection region')
            for band,color in [('reverse','#07818c'),('forward','#cc6b28')]:
                selected=[r for r in surface if r['band']==band]
                if selected:
                    ax.axvline(np.median([r['radius'] for r in selected]),color=color,ls='--',label=f'{band.capitalize()} radius')
            if sample:
                for side,color,marker in [('upstream','#8353b1','<'),('downstream','#cc3566','>')]:
                    r=sample[f'{side}_radius']
                    ax.axvline(r,color=color,lw=.8,alpha=.65)
                    if row<2:
                        key=('rho' if row==0 else 'p')+('1' if side=='upstream' else '2')
                        ax.scatter(r,sample[key],color=color,marker=marker,s=45,zorder=4,label=f'{side.capitalize()} sample')
            if row==3:
                ax.scatter([r['radius'] for r in rev],[r['mach'] for r in rev],s=8,color='#07818c',alpha=.4,label='Reverse shock Mach')
            if row<2: ax.set_yscale('log')
            ax.set_xlim(0,.34); ax.grid(alpha=.2)
            if col==0: ax.set_ylabel(label+' (code units)' if row<3 else label)
            if row==0: ax.set_title(f't = {time:.2f}')
            if row==3: ax.set_xlabel('Radius (code units)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    handles2,labels2=axes[3,0].get_legend_handles_labels()
    unique=dict(zip(labels+labels2,handles+handles2))
    fig.legend(unique.values(),unique.keys(),loc='upper center',bbox_to_anchor=(.5,.965),ncol=4,frameon=False)
    fig.suptitle(rf'Reverse-candidate inspection: ${num_cells}^3$, 21-event schedule',fontsize=15)
    fig.text(.5,.015,'Sample markers show one actual valid surface sample per time; curves are spherical shell medians.\nRadial flow Mach is measured in the simulation frame and differs from reconstructed shock Mach.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.90))
    fig.savefig(path,dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-cells',type=int,default=64)
    parser.add_argument('--require-gpu',action='store_true')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.num_cells<32 or args.num_cells%16: raise ValueError('Use a multiple of 16, at least 32.')
    if args.require_gpu and jax.default_backend()!='gpu': raise RuntimeError('GPU required.')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    setup=argparse.Namespace(num_cells=args.num_cells,num_snapshots=21,t_end=.2,
                             num_injection_cells=args.num_cells//16,plasma_beta=None)
    state,config,params,registry,helper=build_problem(setup)
    times=jnp.linspace(0.,.2,21)
    config=config._replace(use_specific_snapshot_timepoints=True,print_elapsed_time=False,progress_bar=False)
    params=params._replace(snapshot_timepoints=times,dt_max=.2/21)
    metadata=dict(status='running',backend=jax.default_backend(),jax=jax.__version__,python=platform.python_version(),
        num_cells=args.num_cells,requested_times=np.asarray(times).tolist(),dt_max=.2/21,injection_radius=.0625,
        units='dimensionless baseline code units',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).relative_to(Path.cwd()),Path('experiments/shock/analyze_state.py'),Path('experiments/wind_bubble/run_single_bubble.py')]})
    metadata_path=args.output_dir/'metadata.json'
    metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
    snapshots=time_integration(state,config,params,registry)
    analyze=jax.jit(lambda s:analyze_shock_state_device(s,config,registry,helper))
    events=[];summary=[]
    for index in (12,15,20):
        time=float(snapshots.time_points[index])
        profile,surface=extract(snapshots.states[index],analyze,registry,helper,config)
        write_csv(args.output_dir/f'profile_{index:04d}.csv',profile)
        if surface: write_csv(args.output_dir/f'samples_{index:04d}.csv',surface)
        rev=[r for r in surface if r['band']=='reverse']; valid=[r for r in rev if r['sampling_valid']]
        row=dict(time=time,reverse_samples=len(rev),valid_samples=len(valid),
            compressive_density_samples=sum(r['rho2']>r['rho1'] for r in valid),
            pressure_increase_samples=sum(r['p2']>r['p1'] for r in valid),
            upstream_inside_injection=sum(r['upstream_inside_injection'] for r in valid),
            downstream_inside_injection=sum(r['downstream_inside_injection'] for r in valid))
        summary.append(row);events.append((time,profile,surface))
        print(row,flush=True)
    plot(events,args.output_dir/'reverse_profiles.png',num_cells=args.num_cells)
    write_csv(args.output_dir/'sample_checks.csv',summary)
    metadata.update(status='complete',actual_times=np.asarray(snapshots.time_points).tolist(),
                    full_gas_snapshots_saved=False,interpretation='Inspection only; no shock physics or estimator changed.')
    metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__': main()
