"""Explain primary-sampler endpoints in planar known-reference cases.

The 1D trace uses the production dimension-independent sampler and is checked
against the saved 3D Mach field; no detector or sampler implementation changes.
"""
import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from astronomix._physics_modules._shock_finder._shock_zones import get_profile_aware_post_pre_shock_values
from experiments.shock.check_colliding_shocks import fingerprint


def surface_comparisons(fields):
    """Reproduce planar comparison operands and cross-check the saved 3D result."""
    n=len(fields['x']);index=np.arange(n);normal=fields['normal_x']
    backward=index-normal;forward=index+normal
    back_d=np.interp(backward,index,fields['divergence'])
    forward_d=np.interp(forward,index,fields['divergence'])
    back_idx=np.clip(np.floor(backward+.5).astype(int),0,n-1)
    forward_idx=np.clip(np.floor(forward+.5).astype(int),0,n-1)
    back_zone=fields['shock_zone'][back_idx]&(backward>=0)&(backward<=n-1)
    forward_zone=fields['shock_zone'][forward_idx]&(forward>=0)&(forward<=n-1)
    better_back=back_zone&(back_d<=fields['divergence'])
    better_forward=forward_zone&(forward_d<fields['divergence'])
    selected=fields['shock_zone']&(abs(normal)>0)&~better_back&~better_forward
    np.testing.assert_array_equal(selected,fields['shock_surface'])
    rows=[]
    for i in np.flatnonzero(fields['shock_zone']):
        rows.append(dict(cell=int(i),x=float(fields['x'][i]),divergence=float(fields['divergence'][i]),
            normal_x=float(normal[i]),backward_cell=int(back_idx[i]),
            backward_divergence=float(back_d[i]),backward_in_zone=bool(back_zone[i]),
            backward_opposite_direction=bool(normal[i]*normal[back_idx[i]]<0),
            rejected_by_backward_comparison=bool(better_back[i]),
            rejected_by_forward_comparison=bool(better_forward[i]),selected=bool(selected[i])))
    return rows


def run(args):
    if args.output_dir.exists():raise ValueError('Use a fresh output directory.')
    report=json.loads((args.root/'verification.json').read_text())
    for path,expected in report['source_sha256'].items():
        if fingerprint(path)!=expected:raise ValueError(f'Benchmark source changed: {path}')
    rows=[];plots=[];inputs={str(args.root/'verification.json'):fingerprint(args.root/'verification.json')}
    for name in ('n064_incoming_s8_f1','n064_incoming_s2_f1'):
        path=args.root/'profiles'/f'{name}.npz'
        inputs[str(path)]=fingerprint(path)
        with np.load(path) as data:fields={k:data[k] for k in data.files}
        n=len(fields['x']);dx=1/n
        direction=jnp.asarray(fields['normal_x'][None],jnp.float32)
        pressure=jnp.asarray(fields['pressure'],jnp.float32)
        density=jnp.asarray(fields['density'],jnp.float32)
        sampled=jax.jit(lambda d,z,p,t,o:get_profile_aware_post_pre_shock_values(
            d,z,p,t,max_steps=8,center_offsets=d*o[None]))(
                direction,jnp.asarray(fields['shock_zone']),pressure,pressure/density,
                jnp.asarray(fields['surface_offsets']))
        p2,p1,t2,t1,valid,post,pre=map(np.asarray,sampled)
        reconstructed=np.where(valid,np.sqrt((4*np.maximum(p2/np.maximum(p1,1e-30),1)+1)/5),0.)
        selected=fields['shock_surface']
        np.testing.assert_allclose(reconstructed[selected],fields['mach'][selected],rtol=1e-6,atol=1e-7)
        case_rows=[]
        for i in np.flatnonzero(selected):
            normal=fields['normal_x'][i]
            refined=fields['x'][i]+dx*fields['surface_offsets'][i]*normal
            pre_x=refined+dx*pre[i]*normal
            post_x=refined-dx*post[i]*normal
            positions=np.arange(0,float(pre[i])+.125,.25)
            ray_x=refined+positions*dx*normal
            ray_direction=np.interp(ray_x,fields['x'],fields['normal_x'])
            flipped=bool((ray_direction*normal<-.5).any())
            row=dict(case=name,surface_cell=int(i),reference_mach=float(fields['reference_mach']),
                measured_mach=float(fields['mach'][i]),refined_x=float(refined),
                nominal_pre_x=float(pre_x),nominal_post_x=float(post_x),
                sampled_pre_pressure=float(p1[i]),sampled_post_pressure=float(p2[i]),
                expected_pre_pressure=1.,expected_post_pressure=report['incident_downstream_state'][2],
                upstream_endpoint_between_exact_fronts=bool(fields['reference_positions'][0]<pre_x<fields['reference_positions'][1]),
                upstream_ray_crosses_opposing_direction=flipped)
            rows.append(row);case_rows.append(row)
        plots.append((name,fields,case_rows))
    args.output_dir.mkdir(parents=True)
    with (args.output_dir/'sampling_trace.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),dpi=150)
    fig.subplots_adjust(left=.07,right=.97,bottom=.26,top=.80,wspace=.27)
    for ax,(name,fields,case_rows) in zip(axes,plots):
        fronts=fields['reference_positions'];x=fields['x'];p=fields['pressure']
        ax.axvspan(*fronts,color='#9bbcad',alpha=.18,label='True upstream gas between fronts')
        ax.plot(x,p,'o-',c='#087f8c',ms=4,lw=1,label='Pressure from conservative averages')
        for j,front in enumerate(fronts):ax.axvline(front,c='#bb4d34',ls='--',lw=1,label='Exact shock front' if j==0 else None)
        left=min(case_rows,key=lambda r:r['refined_x'])
        start=left['refined_x'];end=left['nominal_pre_x'];pv=left['sampled_pre_pressure']
        start_pressure=np.interp(start,x,p)
        ax.scatter([start],[start_pressure],marker='v',s=70,c='#713e99',label='Detected left front',zorder=5)
        ax.scatter([end],[pv],marker='s',s=45,c='#e68c18',label='Sampler’s upstream endpoint',zorder=6)
        ax.annotate('',xy=(end,pv),xytext=(start,start_pressure),arrowprops=dict(arrowstyle='->',color='#e68c18',lw=2))
        ax.text(.04,.86,f'Expected M = 3\nReported M = {left["measured_mach"]:.3g}\nSampled upstream P = {pv:.3g}',transform=ax.transAxes,fontsize=10,
                bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
        gap=(fronts[1]-fronts[0])*len(x)
        ax.set(xlim=(fronts[0]-3/len(x),fronts[1]+3/len(x)),ylim=(0,14),xlabel='x',ylabel='Pressure',title=f'Exact shock separation: {gap:g} cells')
        ax.grid(alpha=.2)
    axes[1].legend(loc='lower right',fontsize=7,framealpha=.9)
    fig.suptitle('A known Mach-reconstruction error when two shock zones join',fontsize=15)
    fig.text(.07,.065,'Both fronts have exact Mach number 3. The physical upstream gas is the cold P=1 region between them.\n'
        'At 2-cell separation, the sampler walks through the joined zone into hot P=11 gas beyond the opposite front.\n'
        'It therefore uses P₂/P₁=11/11 and returns M=1. The detector and sampler remain unchanged.',fontsize=10)
    fig.savefig(args.output_dir/'sampling_crosses_opposite_shock.png');plt.close(fig)

    # Independent surface-stage failure: opposing fronts share a flat minimum.
    name='n064_outgoing_s2_f1'
    path=args.root/'profiles'/f'{name}.npz';inputs[str(path)]=fingerprint(path)
    with np.load(path) as data:fields={k:data[k] for k in data.files}
    comparisons=surface_comparisons(fields)
    if len(comparisons)!=4 or any(r['selected'] for r in comparisons):
        raise AssertionError('Expected exact outgoing two-cell characterization changed.')
    with (args.output_dir/'surface_comparisons.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    fig,axes=plt.subplots(2,1,figsize=(9,7),dpi=150,sharex=True)
    fig.subplots_adjust(left=.10,right=.96,top=.83,bottom=.20,hspace=.35)
    x=fields['x'];zone=fields['shock_zone'];fronts=fields['reference_positions']
    axes[0].plot(x,fields['pressure'],'o-',c='#087f8c',ms=4,lw=1.2)
    axes[0].scatter(x[zone],fields['pressure'][zone],s=65,facecolors='none',edgecolors='#e68c18',label='Shock-zone candidates')
    axes[0].set(ylabel='Pressure',title='Both physical pressure jumps are present; no surface cell survives')
    axes[0].legend(fontsize=9)
    axes[1].plot(x,fields['divergence'],'o-',c='#087f8c',ms=4,lw=1.2)
    axes[1].scatter(x[zone],fields['divergence'][zone],s=65,facecolors='none',edgecolors='#e68c18')
    for row in comparisons:
        start=row['x'];end=x[row['backward_cell']]
        axes[1].annotate('',xy=(end,row['divergence']-7),xytext=(start,row['divergence']-7),
                         arrowprops=dict(arrowstyle='->',lw=1.3,color='#713e99'))
    axes[1].set(xlim=(fronts[0]-3/64,fronts[1]+3/64),ylim=(-105,10),
                xlabel='x',ylabel='Velocity divergence',title='Equal backward neighbors reject all four candidates')
    for ax in axes:
        for front in fronts:ax.axvline(front,c='#bb4d34',ls='--',lw=1)
        ax.grid(alpha=.2)
    fig.suptitle('A known surface-selection failure after the shock collision',fontsize=15)
    fig.text(.10,.05,'Exact outgoing Mach number ≈1.915 on both sides; shock separation = 2 cells.\n'
        'The inner two candidates have opposite normals but disqualify each other as equal backward minima.\n'
        'Red lines: physical fronts. Purple arrows: backward comparisons. Production code is unchanged.',fontsize=9.5)
    fig.savefig(args.output_dir/'surface_selection_opposing_tie.png');plt.close(fig)
    for path,expected in inputs.items():
        if fingerprint(path)!=expected:raise AssertionError('Trace input changed.')
    result=dict(trace_completed=True,production_changed=False,one_dimensional_sampler_matches_3d_mach=True,
        rows=rows,surface_comparisons= comparisons,
        outgoing_two_cell_zone_candidates=len(comparisons),outgoing_two_cell_surface_samples=int(fields['shock_surface'].sum()),
        opposite_direction_backward_comparisons=sum(r['backward_opposite_direction'] for r in comparisons),
        surface_comparison_operands_match_saved_3d_result=True,
        input_sha256=inputs,source_sha256={str(Path(__file__)):fingerprint(__file__)},
        scope='Known-reference state-sampling and surface-selection failures; these do not by themselves explain the earlier bubble shock-zone rejection.')
    (args.output_dir/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('sha256')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/colliding_shocks_reference_n032_n064'))
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args())
