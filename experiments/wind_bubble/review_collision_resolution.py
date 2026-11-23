"""Locate marginal state-pair fits and compare the verified 64/128-cubed cases.

Read-only review: no new simulation, production mask, estimator or threshold.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from experiments.wind_bubble.assess_collision_physics import ROOT,BOUNDS,pair_metrics,sha

FINE=Path('outputs/three_bubble_collision_resolution_iwr_compgpu8_n128')


def verify_inputs(root):
    audit=json.loads((root/'audit/verification.json').read_text())
    if not audit['passed']:raise ValueError(f'Audit failed: {root}')
    fingerprints={str(root/'metadata.json'):sha(root/'metadata.json'),
                  str(root/'audit/verification.json'):sha(root/'audit/verification.json')}
    # The audit records the intentional wrapper repair separately; validate its
    # postprocessing source hashes, not the superseded integration wrapper hash.
    for group in ('state_sha256','source_sha256','saved_simulation_sha256','reference_sha256'):
        for name,expected in audit.get(group,{}).items():
            path=Path(name.split('master_thesis/')[-1])
            if group=='state_sha256':path=root/'audit'/path.name
            if sha(path)!=expected:raise ValueError(f'Audit input changed: {path}')
            fingerprints[str(path)]=expected
    assessment=root/'physics/assessment.json'
    # The 64^3 saved-data analysis lives outside its original run directory.
    if not assessment.exists():assessment=Path('outputs/three_bubble_collision_physics_n064/assessment.json')
    report=json.loads(assessment.read_text())
    for name,expected in {**report['input_sha256'],**report['source_sha256']}.items():
        path=Path(name.split('master_thesis/')[-1])
        if sha(path)!=expected:raise ValueError(f'Physical assessment input changed: {path}')
        fingerprints[str(path)]=expected
    fingerprints[str(assessment)]=sha(assessment)
    return fingerprints


def sample(state,points,dx):
    return np.stack([map_coordinates(field,(points/dx-.5).T,order=1,mode='nearest') for field in state],axis=1)


def review_candidates(root):
    rows=[];fingerprints={};dx=1/128
    for filename,kind in [('pressure_normal_state_pairs.csv','pressure_normal'),
                          ('axis_state_pairs.csv','axial')]:
        path=root/'physics'/filename
        fingerprints[str(path)]=sha(path)
        all_rows=list(csv.DictReader(path.open()))
        for event in (6,8,12,16):
            selected=[r for r in all_rows if int(r['event'])==event and r['compatible_20pct']=='True'
                      and (kind!='axial' or r['region']=='central_volume')]
            if not selected:continue
            state=np.load(root/'audit'/f'state_{event:04d}.npy').astype(float)
            with np.load(root/'online'/f'surfaces_{event:04d}.npz') as catalog:
                points=catalog['positions'];tree=cKDTree(points)
                lookup={tuple(ijk):i for i,ijk in enumerate(catalog['grid_indices'])}
                positions=points.copy();mach=catalog['mach'].copy()
            with np.load(root/'audit'/f'collision_crop_{event:04d}.npz') as c:
                crop={key:c[key] for key in c.files}
            origin=crop['centers'][0,0,0]
            for record in selected:
                if kind=='pressure_normal':
                    center=np.array([float(record['center_'+k]) for k in 'xyz'])
                    normal=np.array([float(record['normal_'+k]) for k in 'xyz'])
                    half=float(record['half_width_cells'])
                    up,down=center-half*dx*normal,center+half*dx*normal
                    matching=[r for r in all_rows if int(r['event'])==event and all(
                        r['center_'+k]==record['center_'+k] for k in 'xyz')]
                    passing_widths=[float(r['half_width_cells']) for r in matching if r['compatible_10pct']=='True']
                    probe=center
                else:
                    up=np.array([float(record['upstream_x']),float(record['y']),float(record['z'])])
                    down=np.array([float(record['downstream_x']),float(record['y']),float(record['z'])])
                    center=(up+down)/2
                    normal=np.array([np.sign(down[0]-up[0]),0.,0.])
                    half=abs(up[0]-down[0])/(2*dx);passing_widths=[];probe=down
                q1,q2=sample(state,np.stack([up,down]),dx)
                metrics=pair_metrics(q1,q2,normal)
                for key in ('density_ratio','pressure_ratio','entropy_ratio','mass_inferred_speed',
                            'momentum_relative_error','enthalpy_relative_error'):
                    np.testing.assert_allclose(metrics[key],float(record[key]),rtol=1e-9,atol=1e-10)
                ijk=np.rint((probe-origin)/dx).astype(int)
                nearest_distance,nearest=tree.query(center)
                downstream_index=tuple(np.rint(down/dx-.5).astype(int)) if kind=='axial' else None
                exported=downstream_index in lookup if downstream_index is not None else False
                exact_index=lookup[downstream_index] if exported else None
                if kind=='axial':
                    i1,i2=np.rint(np.array([up[0],down[0]])/dx-.5).astype(int)
                    j,k=np.rint(down[1:]/dx-.5).astype(int)
                    covered=any((i,j,k) in lookup for i in range(min(i1,i2),max(i1,i2)+1))
                else:
                    covered=tuple(np.rint(center/dx-.5).astype(int)) in lookup
                pressure_mach=np.sqrt((4*metrics['pressure_ratio']+1)/5)
                row=dict(event=event,kind=kind,x=center[0],y=center[1],z=center[2],
                    compatible_10pct=record['compatible_10pct']=='True',compatible_20pct=True,
                    half_width_cells=half,inside_central_box=bool(((center>=BOUNDS[:,0])&(center<=BOUNDS[:,1])).all()),
                    pressure_inferred_mach=pressure_mach,mass_inferred_mach=metrics['upstream_shock_frame_mach'],
                    pressure_ratio=metrics['pressure_ratio'],density_ratio=metrics['density_ratio'],
                    entropy_ratio=metrics['entropy_ratio'],passing_half_widths=json.dumps(passing_widths),
                    probe_kind='downstream_cell' if kind=='axial' else 'ray_center',
                    probe_compression=bool(crop['compression'][tuple(ijk)]),
                    probe_alignment=bool(crop['gradient_alignment'][tuple(ijk)]),
                    probe_pressure_jump=bool(crop['pressure_jump_pass'][tuple(ijk)]),
                    probe_zone=bool(crop['shock_zone'][tuple(ijk)]),
                    probe_surface=bool(crop['shock_surface'][tuple(ijk)]),
                    exact_downstream_cell_exported=exported,
                    existing_export_on_axis_segment_or_ray_center=covered,
                    exact_exported_mach=float(mach[exact_index]) if exported else None,
                    exact_exported_x=float(positions[exact_index,0]) if exported else None,
                    nearest_surface_distance_cells=float(nearest_distance/dx))
                rows.append(row)
            for path in (root/'online'/f'surfaces_{event:04d}.npz',root/'audit'/f'collision_crop_{event:04d}.npz'):
                fingerprints[str(path)]=sha(path)
    axial=[r for r in rows if r['kind']=='axial' and r['compatible_10pct']]
    pressure=[r for r in rows if r['kind']=='pressure_normal' and r['compatible_10pct']]
    if len(axial)!=10 or not all(r['exact_downstream_cell_exported'] for r in axial):
        raise AssertionError('Axial candidates no longer all map to existing exports.')
    if len(pressure)!=8 or not all(r['pressure_inferred_mach']<1.3 for r in pressure):
        raise AssertionError('Weak pressure-pair finding changed.')
    above=[r for r in rows if r['pressure_inferred_mach']>=1.3]
    if len(above)!=54 or not all(r['existing_export_on_axis_segment_or_ray_center'] for r in above):
        raise AssertionError('An above-reference pressure contrast is no longer covered by existing detections.')
    return rows,fingerprints


def figures(coarse,fine,output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(2,3,figsize=(12,7.3),dpi=150)
    fig.subplots_adjust(left=.075,right=.98,bottom=.20,top=.88,wspace=.32,hspace=.42)
    common=[]
    for row,event in enumerate([8,16]):
        for n,root,marker in [(64,coarse,'o'),(128,fine,'s')]:
            path=root/'audit'/f'state_{event:04d}.npy'
            state=np.load(path).astype(float);x=(np.arange(n)+.5)/n
            points=np.stack([x,np.full(n,.375),np.full(n,.5)],axis=1)
            values=sample(state,points,1/n);rho,p=values[:,0],values[:,4]
            for i in np.flatnonzero((x>=.42)&(x<=.58)):
                common.append(dict(num_cells=n,event=event,x=x[i],density=rho[i],pressure=p[i],temperature=p[i]/rho[i]))
            for ax,v,title in zip(axes[row],(rho,p,p/rho),('Density','Pressure','Temperature proxy P/ρ')):
                visible=(x>=.42)&(x<=.58)
                ax.plot(x[visible],v[visible],marker+'-',ms=3,lw=1.2,label=f'{n}³')
                ax.set(xlim=(.42,.58),xlabel='x',ylabel=title,title=f'{title} | t={event/100:.2f}')
                ax.axvline(.5,color='gray',ls=':',lw=.8);ax.axvspan(.45,.55,color='gray',alpha=.08)
                ax.grid(alpha=.2)
        axes[row,0].legend(fontsize=9)
    fig.suptitle('Same physical axis at both resolutions: y=0.375, z=0.5',fontsize=15)
    fig.text(.075,.05,'Grid fields sampled on the same axis by multilinear interpolation. Shading: checked central x interval.\n'
             'The central density ridge remains cooler than its surroundings. It is not established as one isolated shock.\n'
             'Prescribed source budgets and physical injection radius match; two resolutions do not prove convergence.',fontsize=9.5)
    fig.savefig(output/'matched_axis_profiles.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,6.3),dpi=150)
    fig.subplots_adjust(left=.075,right=.96,top=.82,bottom=.23,wspace=.37)
    dx=1/128
    configs=[(8,1,.35546875,'weak off-plane comparisons'),
             (16,2,.52734375,'comparisons already detected')]
    for ax,(event,normal_axis,plane,title) in zip(axes,configs):
        state=np.load(fine/'audit'/f'state_{event:04d}.npy')
        index=int(round(plane/dx-.5))
        if normal_axis==1:
            density=state[0,:,index,:];haxis=2;ylabel='z';xlim=(.38,.62);ylim=(.38,.62)
        else:
            density=state[0,:,:,index];haxis=1;ylabel='y';xlim=(.36,.64);ylim=(.235,.515)
        image=ax.imshow(density.T,origin='lower',extent=(0,1,0,1),interpolation='nearest',
                        cmap='magma',norm=LogNorm(.02,3.))
        with np.load(fine/'online'/f'surfaces_{event:04d}.npz') as catalog:
            pts=catalog['positions'];near=abs(pts[:,normal_axis]-plane)<=dx/2
            ax.scatter(pts[near,0],pts[near,haxis],s=8,facecolors='none',edgecolors='#42dce8',linewidths=.6,label='Exported surface points')
        chosen=[r for r in rows if r['event']==event and r['compatible_10pct']
                and abs(r['yz'[normal_axis-1]]-plane)<1.e-9]
        ax.scatter([r['x'] for r in chosen],[r[ylabel] for r in chosen],s=60,marker='x',c='#00ff9c',lw=1.8,label='State-pair centers')
        if normal_axis==1:
            ax.add_patch(Rectangle((.45,.42),.1,.16,fill=False,edgecolor='white',lw=1.1))
            ax.axhline(.50390625,color='white',ls='--',lw=1,label='128³ central slice')
            ax.set_xlabel('x');ax.set_ylabel('z')
        else:
            ax.add_patch(Rectangle((.45,.335),.1,.08,fill=False,edgecolor='white',lw=1.1))
            ax.set_xlabel('x');ax.set_ylabel('y')
        ax.set(xlim=xlim,ylim=ylim,title=f't={event/100:.2f}: {title}\n'+('XZ slice, y=' if normal_axis==1 else 'XY slice, z=')+f'{plane:.7f}')
        fig.colorbar(image,ax=ax,pad=.02,label='Density (code units)')
        ax.legend(loc='upper left',fontsize=7,framealpha=.85)
    fig.suptitle('Where the marginal fits occur | 128³ saved states',fontsize=15)
    fig.text(.075,.055,'Left: representative 4 of 8 weak comparisons; pressure-based M≈1.10, below the detection threshold 1.3.\n'
        'Right: representative 6 of 10 axial comparisons. Every downstream cell is already an exported shock surface.\n'
        'White boxes: original audit region. Cyan overlays use a half-cell slab, not an all-depth projection.',fontsize=9.5)
    fig.savefig(output/'candidate_locations.png');plt.close(fig)
    return common


def run(args):
    if args.output_dir.exists():raise ValueError('Use a fresh output directory.')
    inputs={**verify_inputs(args.coarse),**verify_inputs(args.fine)}
    rows,extra=review_candidates(args.fine);inputs.update(extra)
    args.output_dir.mkdir(parents=True)
    common=figures(args.coarse,args.fine,args.output_dir,rows)
    for name,records in [('candidate_review.csv',rows),('common_axis_profiles.csv',common)]:
        with (args.output_dir/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    for path,h in inputs.items():
        if sha(path)!=h:raise AssertionError(f'Input changed during review: {path}')
    pressure=[r for r in rows if r['kind']=='pressure_normal' and r['compatible_10pct']]
    axial=[r for r in rows if r['kind']=='axial' and r['compatible_10pct']]
    above=[r for r in rows if r['pressure_inferred_mach']>=1.3]
    result=dict(review_completed=True,input_hashes_match=True,physics_changed=False,new_simulation=False,
        reviewed_axial_pairs=len(axial),axial_pairs_already_exported=sum(r['exact_downstream_cell_exported'] for r in axial),
        reviewed_pressure_pairs=len(pressure),pressure_pair_mach_range=[min(r['pressure_inferred_mach'] for r in pressure),max(r['pressure_inferred_mach'] for r in pressure)],
        mass_pair_mach_range=[min(r['mass_inferred_mach'] for r in pressure),max(r['mass_inferred_mach'] for r in pressure)],
        mach_min=1.3,pressure_pairs_passing_only_half_width_2=all(r['passing_half_widths']=='[2.0]' for r in pressure),
        reviewed_20pct_pairs=len(rows),above_reference_pressure_jump_pairs_20pct=len(above),
        above_reference_pressure_jump_pairs_already_covered=sum(r['existing_export_on_axis_segment_or_ray_center'] for r in above),
        missed_above_threshold_central_shock_established=False,physical_absence_proven=False,
        conclusion='The central ridge is not established as a missed isolated shock. The stronger marginal comparisons are already detected; '
                   'the remaining weak, off-plane comparisons do not establish a missed above-threshold central front.',
        scope='Bounded saved-data comparison at 64/128 cubed, not proof of physical absence, convergence, or complete shock recovery.',
        input_sha256=inputs,source_sha256={str(Path(__file__)):sha(__file__)})
    (args.output_dir/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('sha256')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coarse',type=Path,default=ROOT)
    parser.add_argument('--fine',type=Path,default=FINE)
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args())
