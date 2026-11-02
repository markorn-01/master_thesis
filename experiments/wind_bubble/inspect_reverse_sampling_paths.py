"""Read-only inspection of normal sampling paths at t=0.20.

Uses the 21-event wind-bubble setup. No detector, Mach or energy rule changes.
Records all valid reverse-candidate paths and local endpoint variations.
"""
from __future__ import annotations
import argparse
import csv
from functools import partial
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astronomix import time_integration
from experiments.shock.analyze_state import analyze_shock_state_device
from experiments.wind_bubble.run_single_bubble import build_problem, split_radial_shock_candidates


@partial(jax.jit, static_argnames=('order',))
def interpolate_rays(fields, base_indices, normals, distances, *, order=1):
    """Sample M normal rays at M×K signed cell distances; mask out-of-grid values."""
    coords = base_indices[:, None, :] + distances[:, :, None]*normals[:, None, :]
    valid = jnp.all((coords >= 0) & (coords <= jnp.array(fields.shape[1:])-1), axis=-1)
    values = jax.vmap(lambda field: map_coordinates(field, jnp.moveaxis(coords, -1, 0),
                                                   order=order, mode='nearest'))(fields)
    return jnp.where(valid[None, ...], values, jnp.nan), valid


def fractional_span(values):
    """Relative endpoint-to-endpoint change across a centered one-cell interval."""
    return np.abs(values[:, 2]-values[:, 0])/np.maximum(np.abs(values[:, 1]), 1.e-30)


def inspect(state, result, mach, samples, registry, dx):
    p2, p1, rho2, rho1, valid, post, pre = map(np.asarray, samples)
    mask = np.asarray(result.shock_surface_cells, bool)
    indices = np.argwhere(mask)
    normals = np.moveaxis(np.asarray(result.shock_direction), 0, -1)[mask]
    offsets = np.asarray(result.shock_surface_offsets)[mask]
    base = indices.astype(np.float32) + offsets[:, None]*normals
    positions = (base+.5)*dx
    radii = np.linalg.norm(positions-.5, axis=1)
    _, _, separation = split_radial_shock_candidates(radii, .0625, dx)
    selected = (radii > .0625) & (radii <= separation) & valid[mask]
    if not selected.any(): raise RuntimeError('No valid reverse candidates to inspect.')
    selected_indices = indices[selected]
    base, normals, positions = base[selected], normals[selected], positions[selected]
    radii = radii[selected]
    pre, post = pre[mask][selected], post[mask][selected]
    m = np.asarray(mach)[mask][selected]
    reference = np.stack([p1[mask][selected], p2[mask][selected],
                          rho1[mask][selected], rho2[mask][selected]], axis=1)
    fields = jnp.stack([state[registry.pressure_index], state[registry.density_index],
                       state[registry.velocity_index.x], state[registry.velocity_index.y],
                       state[registry.velocity_index.z]])
    base_j, normals_j = jnp.asarray(base), jnp.asarray(normals)

    def sample(s, *, zone=False):
        if zone:
            values, _ = interpolate_rays(jnp.asarray(result.shock_zones)[None].astype(jnp.float32),
                                         base_j, normals_j, jnp.asarray(s), order=0)
        else:
            values, _ = interpolate_rays(fields, base_j, normals_j, jnp.asarray(s))
        return np.asarray(values)

    endpoints = np.stack([pre, -post], axis=1)
    endpoint_values = sample(endpoints)
    measured = np.stack([endpoint_values[0, :, 0], endpoint_values[0, :, 1],
                         endpoint_values[1, :, 0], endpoint_values[1, :, 1]], axis=1)
    np.testing.assert_allclose(measured, reference, rtol=1e-5, atol=1e-7,
                               err_msg='Independent endpoint interpolation differs from analyzer.')
    endpoint_max_difference = float(np.max(np.abs(measured-reference)))
    endpoint_zone = sample(endpoints, zone=True)[0] > .5
    endpoint_radius = np.linalg.norm(positions[:, None, :] + dx*endpoints[:, :, None]*normals[:, None, :]-.5, axis=-1)
    half_width = float(np.ceil(max(np.max(pre), np.max(post)))+1)
    s = np.linspace(-half_width, half_width, int(half_width*16)+1, dtype=np.float32)
    paths = sample(np.broadcast_to(s, (len(base), len(s))))
    zones = sample(np.broadcast_to(s, (len(base), len(s))), zone=True)[0]
    path_radius = np.linalg.norm(positions[:, None, :]+dx*s[None, :, None]*normals[:, None, :]-.5,axis=-1)
    velocity = np.sum(paths[2:]*normals.T[:, :, None], axis=0)

    # Reproduce the sampler's first nearest-neighbor zone exits along each side.
    steps = np.arange(.25, 8.001, .25, dtype=np.float32)
    exits = []
    for sign in (1., -1.):
        zone_values = sample(np.broadcast_to(sign*steps,(len(base),len(steps))),zone=True)[0]
        outside = np.isfinite(zone_values) & (zone_values <= .5)
        if not outside.any(axis=1).all(): raise AssertionError('Missing zone exit on inspected path.')
        exits.append(steps[np.argmax(outside, axis=1)])
    pre_exit, post_exit = exits
    offsets = np.arange(.25, 1.501, .25, dtype=np.float32)
    candidate_s = np.concatenate([pre_exit[:, None]+offsets, -(post_exit[:, None]+offsets)],axis=1)
    candidates = sample(candidate_s)
    candidate_pre = np.argmin(candidates[0, :, :6],axis=1)
    candidate_post = np.argmax(candidates[0, :, 6:],axis=1)
    expected_pre = pre_exit+offsets[candidate_pre]
    expected_post = post_exit+offsets[candidate_post]
    np.testing.assert_allclose(pre,expected_pre,rtol=0,atol=1e-6)
    np.testing.assert_allclose(post,expected_post,rtol=0,atol=1e-6)

    neighborhood_s = np.concatenate([pre[:, None]+np.array([-.5,0,.5]),
                                     -post[:, None]+np.array([-.5,0,.5])],axis=1)
    neighbors = sample(neighborhood_s)
    rho_jump = reference[:, 3]/reference[:, 2]
    predicted_rho = 4*m*m/(m*m+3)
    residual = rho_jump/predicted_rho-1
    rows = []
    for i, ijk in enumerate(selected_indices):
        rows.append(dict(ix=int(ijk[0]),iy=int(ijk[1]),iz=int(ijk[2]),radius=float(radii[i]),
            mach=float(m[i]),density_jump=float(rho_jump[i]),rh_density_residual=float(residual[i]),
            upstream_distance_cells=float(pre[i]),downstream_distance_cells=float(post[i]),
            upstream_radius=float(endpoint_radius[i,0]),downstream_radius=float(endpoint_radius[i,1]),
            upstream_in_zone=bool(endpoint_zone[i,0]),downstream_in_zone=bool(endpoint_zone[i,1]),
            upstream_candidate_index=int(candidate_pre[i]),downstream_candidate_index=int(candidate_post[i]),
            upstream_pressure_one_cell_change=float(fractional_span(neighbors[0,:,:3])[i]),
            downstream_pressure_one_cell_change=float(fractional_span(neighbors[0,:,3:])[i]),
            upstream_density_one_cell_change=float(fractional_span(neighbors[1,:,:3])[i]),
            downstream_density_one_cell_change=float(fractional_span(neighbors[1,:,3:])[i])))
    summary = dict(reverse_samples=len(rows),endpoint_interpolation_matches=True,
                   endpoint_max_absolute_difference=endpoint_max_difference,
                   pressure_candidate_selection_matches=True,
                   upstream_in_zone_count=int(endpoint_zone[:,0].sum()),
                   downstream_in_zone_count=int(endpoint_zone[:,1].sum()),
                   upstream_inside_injection_count=int((endpoint_radius[:,0]<=.0625).sum()),
                   downstream_inside_injection_count=int((endpoint_radius[:,1]<=.0625).sum()))
    for field in ('rh_density_residual','upstream_pressure_one_cell_change','downstream_pressure_one_cell_change',
                  'upstream_density_one_cell_change','downstream_density_one_cell_change'):
        summary[field+'_p16_median_p84'] = np.percentile([r[field] for r in rows],[16,50,84]).tolist()
    summary['upstream_candidate_index_counts'] = np.bincount(candidate_pre,minlength=6).tolist()
    summary['downstream_candidate_index_counts'] = np.bincount(candidate_post,minlength=6).tolist()
    arrays = dict(s_cells=s,pressure=paths[0],density=paths[1],normal_velocity=velocity,
        shock_zone=zones,path_radius=path_radius,grid_indices=selected_indices,
        surface_positions=positions,normals=normals,upstream_distance_cells=pre,
        downstream_distance_cells=post,selected_pressure=reference[:,:2],selected_density=reference[:,2:],
        candidate_s_cells=candidate_s,candidate_pressure=candidates[0],candidate_density=candidates[1])
    return rows, summary, arrays


def plot(arrays, rows, output, n):
    # Illustrative path chosen by median density-RH residual, not by good agreement.
    residual=np.array([r['rh_density_residual'] for r in rows])
    i=int(np.argmin(abs(residual-np.median(residual))))
    s=arrays['s_cells']
    fig,axes=plt.subplots(3,1,figsize=(9,9),sharex=True)
    for ax,key,label in zip(axes,('pressure','density','normal_velocity'),
                           ('Pressure (code units)','Density (code units)','Normal velocity (code units)')):
        ax.plot(s,arrays[key][i],color='black')
        ax.fill_between(s,0,1,where=arrays['shock_zone'][i]>.5,transform=ax.get_xaxis_transform(),
                        color='#07818c',alpha=.12,label='Shock zone')
        ax.fill_between(s,0,1,where=arrays['path_radius'][i]<=.0625,transform=ax.get_xaxis_transform(),
                        color='gray',alpha=.2,label='Injection region')
        for side,color,sign in [('upstream','#8353b1',1),('downstream','#cc3566',-1)]:
            distance=sign*arrays[f'{side}_distance_cells'][i]
            ax.axvline(distance,color=color,label=side.capitalize())
            if key!='normal_velocity':
                column=0 if side=='upstream' else 1
                value=arrays['selected_'+key][i,column]
                ax.scatter(distance,value,color=color,zorder=3)
        if key!='normal_velocity':
            ax.scatter(arrays['candidate_s_cells'][i],arrays['candidate_'+key][i],
                       facecolors='none',edgecolors='gray',s=24,label='Pressure-search candidates')
            ax.set_yscale('log')
        ax.axvline(0,color='0.5',ls=':');ax.set_ylabel(label);ax.grid(alpha=.2)
        ax.set_xlim(-arrays['downstream_distance_cells'][i]-1,arrays['upstream_distance_cells'][i]+1)
    axes[0].legend(ncol=3,fontsize=8,frameon=False)
    axes[-1].set_xlabel('Signed distance from refined surface (cells): downstream ← 0 → upstream')
    fig.suptitle(f'Reverse-candidate sampling path: {n}³, t=0.20\nRepresentative index {tuple(arrays["grid_indices"][i])}; median-residual selection')
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(output,dpi=180);plt.close(fig)
    return i


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
    source_paths=[Path(__file__).relative_to(Path.cwd()),Path('experiments/shock/analyze_state.py'),
                  Path('experiments/wind_bubble/run_single_bubble.py'),
                  Path('astronomix/_physics_modules/_shock_finder/_shock_zones.py')]
    metadata=dict(status='running',backend=jax.default_backend(),jax=jax.__version__,python=platform.python_version(),
        num_cells=args.num_cells,injection_radius=.0625,dt_max=.2/21,requested_times=np.asarray(times).tolist(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        purpose='Normal-path inspection only; no changes to physical estimators or filtering.')
    path=args.output_dir/'metadata.json';path.write_text(json.dumps(metadata,indent=2)+'\n')
    try:
        snapshots=time_integration(state,config,params,registry)
        final=snapshots.states[-1]
        analyze=jax.jit(lambda s:analyze_shock_state_device(s,config,registry,helper))
        result,mach,_,samples=analyze(final)
        rows,summary,arrays=inspect(final,result,mach,samples,registry,float(config.grid_spacing))
        with (args.output_dir/'endpoint_checks.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        np.savez_compressed(args.output_dir/'normal_paths.npz',**arrays)
        summary['illustrative_path_index']=plot(arrays,rows,args.output_dir/'normal_path.png',args.num_cells)
        (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        metadata.update(status='complete',actual_times=np.asarray(snapshots.time_points).tolist(),full_gas_snapshots_saved=False)
        print(json.dumps(summary,indent=2),flush=True)
    except Exception as exc:
        metadata.update(status='failed',error=str(exc));raise
    finally: path.write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__': main()
