"""Compare the unchanged 3D finder with an exact symmetric shock collision.

No time integrator is used here. Width=1 is the exact conservative cell
average; larger odd widths are explicitly synthetic broadening controls.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from astronomix import (CARTESIAN,SimulationConfig,construct_primitive_state,
    finalize_config,get_helper_data,get_registered_variables)
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import find_shocks_pfrommer
from experiments.wind_bubble.audit_detection_stages import stage_fields
from experiments.shock.colliding_shocks_reference import (
    SymmetricShockCollision,contact_control,isolated_control,
)


def fingerprint(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_analyzer(resolution):
    config=SimulationConfig(geometry=CARTESIAN,dimensionality=3,box_size=1.,num_cells=resolution,mhd=False)
    variables=get_registered_variables(config);helper=get_helper_data(config)
    ones=jnp.ones((resolution,)*3);zeros=jnp.zeros_like(ones)
    template=construct_primitive_state(config=config,registered_variables=variables,
        density=ones,gas_pressure=ones,velocity_x=zeros,velocity_y=zeros,velocity_z=zeros)
    config=finalize_config(config,template.shape)
    @jax.jit
    def analyze(state):
        result=find_shocks_pfrommer(state,config,variables,helper,profile_aware_mach=True)
        fields=stage_fields(state,config,variables,helper)
        fields['mach']=result.mach_numbers
        fields['mask_mismatch_count']=jnp.sum(result.shock_zones!=fields['shock_zone'])+jnp.sum(
            result.shock_surface_cells!=fields['shock_surface'])
        return fields
    def evaluate(profile):
        n=resolution
        state=template.at[variables.density_index].set(jnp.broadcast_to(jnp.asarray(profile[:,0,None,None],jnp.float32),(n,)*3))
        state=state.at[variables.pressure_index].set(jnp.broadcast_to(jnp.asarray(profile[:,2,None,None],jnp.float32),(n,)*3))
        state=state.at[variables.velocity_index.x].set(jnp.broadcast_to(jnp.asarray(profile[:,1,None,None],jnp.float32),(n,)*3))
        fields={key:np.asarray(value) for key,value in analyze(state).items()}
        if fields['mask_mismatch_count']!=0:raise AssertionError('Instrumented masks differ from production.')
        return fields
    return evaluate


def score_fronts(fields,structure,resolution):
    n=resolution;dx=1/n;x=(np.arange(n)+.5)*dx
    normal_x=fields['direction'][...,0]
    positions=x[:,None,None]+dx*fields['surface_offsets']*normal_x
    interior=np.zeros((n,)*3,bool);interior[2:-2,2:-2,2:-2]=True
    surface=fields['shock_surface']&interior
    rows=[]
    targets=structure['positions'] if structure['phase']!='meeting' else []
    for index,target in enumerate(targets):
        expected_normal=structure['hot_to_cold_directions'][index]
        assigned=np.ones((n,)*3,bool)
        if len(targets)==2:
            assigned=positions<.5 if index==0 else positions>.5
        candidate=surface&assigned&(normal_x*expected_normal>.9)
        error=abs(positions-target)/dx
        row=dict(front_index=index,reference_x=float(target),reference_mach=structure['mach'])
        for tolerance in (1,2):
            selected=candidate&(error<=tolerance)
            row[f'coverage_{tolerance}cell']=float(selected.any(axis=0)[2:-2,2:-2].mean())
        selected=candidate&(error<=2)
        values=fields['mach'][selected]
        good=np.isfinite(values)&(values>0)
        row.update(surface_samples=int(selected.sum()),finite_positive_mach_fraction=float(good.mean()) if len(values) else None,
            mach_median=float(np.median(values[good])) if good.any() else None,
            position_error_cells_median=float(np.median(error[selected])) if selected.any() else None)
        row['mach_relative_error']=row['mach_median']/structure['mach']-1 if row['mach_median'] is not None else None
        rows.append(row)
    return rows,int(surface.sum())


def evaluate_case(evaluate,profile,structure,resolution,name,filter_cells,separation,time,output):
    fields=evaluate(profile)
    rows,count=score_fronts(fields,structure,resolution)
    recovery=bool(rows and all(r['coverage_2cell']>=.99 for r in rows))
    accurate=bool(recovery and all(r['mach_relative_error'] is not None and abs(r['mach_relative_error'])<=.05 for r in rows))
    summary=dict(case=name,resolution=resolution,phase=structure['phase'],filter_cells=filter_cells,
        separation_cells=separation,time=time,reference_mach=structure['mach'],
        expected_distinct_fronts=len(rows),surface_samples=count,all_fronts_located_2cell=recovery,
        all_fronts_located_and_mach_within_5pct=accurate,
        minimum_front_coverage=min((r['coverage_2cell'] for r in rows),default=None),
        worst_absolute_mach_relative_error=max((abs(r['mach_relative_error']) for r in rows if r['mach_relative_error'] is not None),default=None))
    i=resolution//2
    arrays={key:value[:,i,i] for key,value in fields.items() if value.ndim==3}
    arrays['normal_x']=fields['direction'][:,i,i,0]
    arrays.update(x=(np.arange(resolution)+.5)/resolution,density=profile[:,0],velocity_x=profile[:,1],pressure=profile[:,2],
        temperature=profile[:,2]/profile[:,0],reference_positions=structure['positions'],
        reference_mach=np.asarray(structure['mach'] if structure['mach'] is not None else np.nan),
        reference_time=np.asarray(time if time is not None else np.nan))
    np.savez_compressed(output/'profiles'/f'{name}.npz',**arrays)
    return summary,[dict(case=name,resolution=resolution,phase=structure['phase'],filter_cells=filter_cells,
                         separation_cells=separation,**row) for row in rows]


def write_csv(path,rows):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plot_results(output,summary,resolutions,widths,separations,reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),dpi=150)
    fig.subplots_adjust(left=.09,right=.97,top=.86,bottom=.21,hspace=.32,wspace=.25)
    colors={w:plt.get_cmap('tab10')(i) for i,w in enumerate(widths)}
    for col,phase in enumerate(['incoming','outgoing']):
        for n in resolutions:
            for width in widths:
                rows=sorted([r for r in summary if r['phase']==phase and r['resolution']==n and r['filter_cells']==width],key=lambda r:r['separation_cells'])
                xx=[r['separation_cells'] for r in rows]
                coverage=[r['minimum_front_coverage'] for r in rows]
                bias=[r['worst_absolute_mach_relative_error'] if r['all_fronts_located_2cell'] else np.nan for r in rows]
                style='-' if n==max(resolutions) else '--'
                label=f'{n}³, filter {width}'
                axes[0,col].plot(xx,coverage,'o'+style,c=colors[width],ms=4,label=label)
                axes[1,col].plot(xx,bias,'o'+style,c=colors[width],ms=4)
        mach=reference.incoming_mach if phase=='incoming' else reference.outgoing_mach
        axes[0,col].set(title=f'{phase.capitalize()} reference fronts: M={mach:.3f}',ylim=(-.05,1.05),ylabel='Minimum coverage of the two fronts')
        axes[1,col].axhline(.05,c='gray',ls=':',lw=1)
        axes[1,col].set(ylabel='Worst |Mach / reference Mach − 1|',xlabel='Exact front separation (cells)')
        for ax in axes[:,col]:
            ax.set_xscale('log',base=2);ax.set_xticks(separations,labels=[f'{v:g}' for v in separations]);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('Unchanged 3D finder on a known symmetric shock collision',fontsize=15)
    fig.text(.09,.055,'Filter 1: exact conservative cell averages. Larger filters: controlled box averaging of conserved fields.\n'
        'Coverage requires separate fronts, correct normal signs and locations within 2 cells. Mach errors shown only when both are located.\n'
        'This tests detection/reconstruction, without evolving the gas. The meeting instant has no two spatially separated fronts.',fontsize=9)
    fig.savefig(output/'recovery_vs_separation.png');plt.close(fig)

    n=max(resolutions)
    chosen=[]
    for phase,sep,width in [('incoming',min(separations),min(widths)),('outgoing',min(separations),min(widths)),
                            ('outgoing',max(separations),min(widths)),('outgoing',max(separations),max(widths))]:
        chosen.append(next(r for r in summary if r['resolution']==n and r['phase']==phase and
                           r['filter_cells']==width and r['separation_cells']==sep))
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),dpi=150)
    fig.subplots_adjust(left=.08,right=.98,top=.87,bottom=.18,hspace=.45,wspace=.25)
    for ax,row in zip(axes.flat,chosen):
        with np.load(output/'profiles'/f'{row["case"]}.npz') as data:
            x=data['x'];p=data['pressure'];fronts=data['reference_positions'];surface=data['shock_surface']
            for j,front in enumerate(fronts):ax.axvline(front,c='#c74a32',ls='--',lw=1.2,label='Exact shock' if j==0 else None)
            ax.plot(x,p,'o-',c='#087f8c',ms=3,lw=1,label='Analyzed pressure')
            ax.scatter(x[surface],p[surface],s=75,marker='v',c='#713e99',label='Selected surface cell',zorder=5)
            ax.fill_between(x,0,1,where=data['shock_zone'],transform=ax.get_xaxis_transform(),alpha=.10,color='#087f8c')
            margin=max(.06,4/n)
            ax.set(xlim=(fronts.min()-margin,fronts.max()+margin),xlabel='x',ylabel='Pressure',
                   title=f'{row["phase"]}: separation={row["separation_cells"]:g} cells, filter={row["filter_cells"]}')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'Known front locations versus selected cells | {n}³',fontsize=15)
    fig.text(.08,.045,'Red dashed lines: exact physical reference positions. Purple triangles: selected grid-cell centers, before sub-cell refinement.\n'
        'Teal shading: production shock zone. Broadening controls are not additional exact Euler solutions.',fontsize=9)
    fig.savefig(output/'known_front_profiles.png');plt.close(fig)


def run(args):
    if args.output_dir.exists():raise ValueError('Use a fresh output directory.')
    if args.require_gpu and jax.default_backend()!='gpu':raise RuntimeError('GPU requested but unavailable.')
    resolutions=sorted(set(args.resolutions));widths=sorted(set(args.filter_cells));separations=sorted(set(args.separations))
    if any(n<32 or n%2 for n in resolutions) or not resolutions:raise ValueError('Use even resolutions of at least 32.')
    if any(w<1 or w%2!=1 for w in widths) or not widths:raise ValueError('Filters must be positive odd cell widths.')
    if not separations or min(separations)<=0:raise ValueError('Separations must be positive.')
    reference=SymmetricShockCollision(args.incoming_mach)
    residuals=reference.flux_residuals()
    np.testing.assert_allclose(residuals,0,rtol=0,atol=1e-12)
    sources=[Path(__file__),Path('experiments/shock/colliding_shocks_reference.py'),
             Path('experiments/wind_bubble/audit_detection_stages.py')]
    sources+=sorted(Path('astronomix/_physics_modules/_shock_finder').glob('*.py'))
    hashes={str(p):fingerprint(p) for p in sources}
    (args.output_dir/'profiles').mkdir(parents=True)
    summary=[];front_rows=[]
    for n in resolutions:
        evaluate=make_analyzer(n)
        for width in widths:
            for phase in ['incoming','outgoing']:
                profile,structure=isolated_control(reference,n,phase,width)
                name=f'n{n:03d}_isolated_{phase}_f{width}'
                a,b=evaluate_case(evaluate,profile,structure,n,name,width,None,None,args.output_dir)
                summary.append(a);front_rows.extend(b)
            profile,structure=contact_control(n,width)
            name=f'n{n:03d}_contact_f{width}'
            a,b=evaluate_case(evaluate,profile,structure,n,name,width,None,None,args.output_dir)
            summary.append(a);front_rows.extend(b)
            profile,structure=reference.cell_averages(n,reference.collision_time,width)
            name=f'n{n:03d}_meeting_f{width}'
            a,b=evaluate_case(evaluate,profile,structure,n,name,width,0.,reference.collision_time,args.output_dir)
            summary.append(a);front_rows.extend(b)
            for phase in ['incoming','outgoing']:
                for sep in separations:
                    time=reference.time_at_separation(phase,sep,n)
                    profile,structure=reference.cell_averages(n,time,width)
                    name=f'n{n:03d}_{phase}_s{sep:g}_f{width}'
                    a,b=evaluate_case(evaluate,profile,structure,n,name,width,sep,time,args.output_dir)
                    summary.append(a);front_rows.extend(b)
                print(f'{n}³ filter={width}, {phase}: reference M={structure["mach"]:.4f}; separation sweep complete.',flush=True)
    write_csv(args.output_dir/'summary.csv',summary);write_csv(args.output_dir/'front_metrics.csv',front_rows)
    plot_results(args.output_dir,summary,resolutions,widths,separations,reference)
    for path,h in hashes.items():
        if fingerprint(path)!=h:raise AssertionError(f'Production/reference source changed: {path}')
    baseline=[r for r in summary if r['filter_cells']==1 and
              (r['phase'].startswith('isolated') or r['separation_cells']==max(separations))]
    report=dict(benchmark_completed=True,backend=jax.default_backend(),jax=jax.__version__,
        production_changed=False,simulation_evolved=False,reference='Exact symmetric two-shock collision, then symmetric two-shock Riemann solution.',
        gamma=reference.gamma,incoming_mach=reference.incoming_mach,outgoing_mach=reference.outgoing_mach,
        incoming_speed=reference.incoming_speed,outgoing_speed=reference.outgoing_speed,collision_time=reference.collision_time,
        ambient_state=reference.ambient.tolist(),incident_downstream_state=reference.incoming_left.tolist(),
        final_central_state=reference.central.tolist(),reference_flux_residuals=residuals.tolist(),
        cases=len(summary),front_measurements=len(front_rows),
        isolated_and_widest_exact_fronts_located=all(r['all_fronts_located_2cell'] for r in baseline),
        isolated_and_widest_exact_mach_within_5pct=all(r['all_fronts_located_and_mach_within_5pct'] for r in baseline),
        contact_surface_samples=sum(r['surface_samples'] for r in summary if r['phase']=='contact'),
        widths=widths,separations_cells=separations,source_sha256=hashes,
        scope='Planar detection/primary Mach reconstruction test. Controlled filters are not exact Euler solutions; '
              'this does not validate the hydrodynamic solver, curved shocks, contacts mixed with shocks, or the bubble interpretation.',
        references=['https://www.clawpack.org/riemann_book/html/Euler.html', 'https://arxiv.org/html/1407.4117'])
    (args.output_dir/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','references','reference_flux_residuals')},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolutions',type=int,nargs='+',default=[32,64])
    parser.add_argument('--incoming-mach',type=float,default=3.)
    parser.add_argument('--filter-cells',type=int,nargs='+',default=[1,3,5])
    parser.add_argument('--separations',type=float,nargs='+',default=[.5,1,2,4,8,16])
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--require-gpu',action='store_true')
    run(parser.parse_args())
