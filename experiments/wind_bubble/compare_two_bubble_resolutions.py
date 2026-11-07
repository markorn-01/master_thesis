"""Readback and morphology comparison of budget-matched two-bubble catalogs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.shock_connectivity import connectivity_partition

INSPECT=[4,8,12,13,14,16]


def neighborhood_settings(n,reference_n=64):
    return [('native_cells',1.5,20),('native_cells_scaled_count',1.5,int(round(20*(n/reference_n)**2))),
            ('common_physical_radius',1.5*n/reference_n,int(round(20*(n/reference_n)**2)))]


def validate_pair(a,b):
    if a['status']!='complete' or b['status']!='complete':raise ValueError('Require completed runs.')
    if b['num_cells']<=a['num_cells']:raise ValueError('Require increasing resolution.')
    for key in ['source_positions','injection_radius','wind_terminal_velocity','ambient_density','ambient_pressure','gamma','domain']:
        if not np.allclose(a[key],b[key],rtol=0,atol=1e-12):raise ValueError(f'Physical setting differs: {key}')
    for key in ['requested_times','actual_times']:
        if np.shape(a[key])!=np.shape(b[key]) or not np.allclose(a[key],b[key],rtol=0,atol=1e-6):raise ValueError(f'Event schedules differ: {key}')
    for key in ['discrete_mass_rates','discrete_energy_rates']:
        if not np.allclose(a[key],b[key],rtol=1e-7,atol=1e-12):raise ValueError(f'Prescribed source budgets differ: {key}')
    if a['injection_prescription']!=b['injection_prescription']:raise ValueError('Injection prescriptions differ.')
    if not b.get('source_budget_plan',{}).get('budget_matched'):raise ValueError('Target does not record explicit budget matching.')


def check_distribution(values,edges,counts,row,prefix):
    good=np.isfinite(values)&(values>0);v=values[good]
    np.testing.assert_array_equal(np.histogram(v,bins=edges)[0],counts)
    expected=dict(valid_count=int(good.sum()),invalid_count=int((~good).sum()),
                  underflow=int((v<edges[0]).sum()),overflow=int((v>edges[-1]).sum()))
    for key,value in expected.items():assert int(row[f'{prefix}_{key}'])==value
    q=np.percentile(v,[16,50,84]) if len(v) else [np.nan]*3
    np.testing.assert_allclose(q,[float(row[f'{prefix}_{key}']) for key in ['p16','median','p84']],rtol=1e-6,atol=1e-8,equal_nan=True)
    assert len(values)==expected['invalid_count']+expected['underflow']+expected['overflow']+int(counts.sum())


def load_checked(root):
    meta=json.loads((root/'metadata.json').read_text());n=meta['num_cells'];data=[];hashes={}
    def record(p):hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    record(root/'metadata.json');record(root/'online/history.csv')
    rows=list(csv.DictReader((root/'online/history.csv').open()))
    assert len(rows)==len(meta['actual_times'])
    for prefix in ['surfaces','histograms','slice']:assert len(list((root/'online').glob(f'{prefix}_*.npz')))==len(rows)
    mismatch=[]
    for name,h in meta['source_sha256'].items():
        path=Path(name)
        if not path.exists() and '/master_thesis/' in name:path=Path(name.split('/master_thesis/',1)[1])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=h:mismatch.append(str(path))
    # The runner was extended for budget matching after the historical 64³ run.
    unexpected=[p for p in mismatch if p!='experiments/wind_bubble/run_two_bubbles.py']
    if unexpected:raise ValueError(f'Unreviewed analysis/physics source mismatch: {unexpected}')
    minimum=1.;max_unknown=0
    for i,row in enumerate(rows):
        arrays=[]
        for prefix in ['surfaces','histograms','slice']:
            path=root/'online'/f'{prefix}_{i:04d}.npz';record(path)
            with np.load(path) as a:arrays.append({key:a[key].copy() for key in a.files})
        s,h,background=arrays;p=s['positions'];dx=float(s['grid_spacing'])
        assert int(s['schema_version'])==3 and s['grid_shape'].tolist()==[n]*3
        assert len(p)==int(row['surface_cells_outside_injection']) and int(row['surface_cells_all'])>=len(p)
        for a in arrays:np.testing.assert_allclose(float(a['time']),float(row['time']),atol=1e-7,rtol=0)
        np.testing.assert_allclose(float(row['time']),meta['actual_times'][i],atol=1e-7,rtol=0)
        np.testing.assert_allclose(dx,1/n,atol=0,rtol=0)
        np.testing.assert_array_equal(s['source_positions'],meta['source_positions'])
        distance=np.linalg.norm(p[:,None,:]-np.asarray(meta['source_positions'])[None,:,:],axis=-1)
        np.testing.assert_array_equal(s['source_distances'],distance)
        assert np.isfinite(p).all() and (distance>meta['injection_radius']).all()
        assert np.isfinite(s['normals']).all() and (np.linalg.norm(s['normals'],axis=1)>0).all()
        for side in ['upstream','downstream']:
            end=np.linalg.norm(s[f'{side}_positions'][:,None,:]-np.asarray(meta['source_positions'])[None,:,:],axis=-1)
            flags=s['sample_positions_valid'][:,None]&(end<=meta['injection_radius'])
            np.testing.assert_array_equal(flags,s[f'{side}_inside_sources'])
            np.testing.assert_array_equal(flags.any(axis=1),s[f'{side}_inside_any_source'])
            assert int(flags.any(axis=1).sum())==int(row[f'{side}_inside_any_source'])
        unknown=int((~s['sample_positions_valid']).sum());assert unknown==int(row['sample_positions_unknown']);max_unknown=max(max_unknown,unknown)
        expected_valid=s['sampling_valid']&np.isfinite(s['rho_upstream'])&np.isfinite(s['rho_downstream'])&(s['rho_upstream']>0)&(s['rho_downstream']>0)
        np.testing.assert_array_equal(expected_valid,s['density_jump_valid'])
        np.testing.assert_array_equal(np.isfinite(s['mach'])&(s['mach']>0),s['mach_valid'])
        expected_ratio=np.full(len(p),np.nan);np.divide(s['rho_downstream'],s['rho_upstream'],out=expected_ratio,where=expected_valid)
        np.testing.assert_allclose(expected_ratio,s['density_jump'],rtol=0,atol=0,equal_nan=True)
        for field in ['mach','density_jump']:
            check_distribution(s[field],h[f'{field}_edges'],h[f'{field}_counts'],row,field)
        if len(p):
            clearance=float(np.minimum(p,1-p).min());minimum=min(minimum,clearance)
            assert clearance==float(row['minimum_surface_boundary_clearance'])
        for field in ['density','pressure']:assert background[field].shape==(n,n) and np.isfinite(background[field]).all() and (background[field]>0).all()
        assert float(background['z_coordinate'])==(n//2+.5)/n
        data.append(dict(surface=s,histogram=h,background=background,row=row))
    report=dict(num_cells=n,events=len(rows),readback_passed=True,source_hash_mismatches=mismatch,
                expected_historical_runner_change=bool(mismatch),minimum_boundary_clearance=minimum,
                minimum_boundary_clearance_cells=minimum*n,max_unknown_sample_positions=max_unknown,
                offline_equivalence_repeated=False,full_gas_states_rechecked=False)
    return meta,data,hashes,report


def csv_write(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def spatial_plots(output,runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm,Normalize
    limits={field:(min(float(data[i]['background'][field].min()) for _,data in runs for i in INSPECT),
                   max(float(data[i]['background'][field].max()) for _,data in runs for i in INSPECT)) for field in ['pressure','density']}
    mach_high=max(16.,max(float(np.ceil(data[i]['surface']['mach'][data[i]['surface']['mach_valid']].max())) for _,data in runs for i in INSPECT))
    for event in INSPECT:
        fig,axes=plt.subplots(2,3,figsize=(14,11.3))
        for row,(meta,data) in enumerate(runs):
            a=data[event];s=a['surface'];p=s['positions'];slab=np.abs(p[:,2]-.5)<=1.5/64
            for col,field in enumerate(['pressure','density']):
                ax=axes[row,col];lo,hi=limits[field]
                im=ax.imshow(a['background'][field].T,origin='lower',extent=(0,1,0,1),cmap='magma',norm=LogNorm(lo,hi))
                ax.scatter(p[slab,0],p[slab,1],s=2,c='cyan',alpha=.5)
                fig.colorbar(im,ax=ax,pad=.02,shrink=.8)
                ax.set_title(f"{meta['num_cells']}³ {field}; z={float(a['background']['z_coordinate']):.5f}",fontsize=11)
            ax=axes[row,2];good=slab&s['mach_valid']
            im=ax.scatter(p[good,0],p[good,1],c=s['mach'][good],cmap='viridis',norm=Normalize(0,mach_high),s=6)
            ax.scatter(p[slab&~s['mach_valid'],0],p[slab&~s['mach_valid'],1],c='.6',s=6)
            fig.colorbar(im,ax=ax,pad=.02,shrink=.8,label='Mach')
            ax.set_title(f"{meta['num_cells']}³ refined surface samples",fontsize=11)
            for ax in axes[row]:
                sources=np.asarray(meta['source_positions']);ax.scatter(sources[:,0],sources[:,1],marker='*',c='white',edgecolors='black',s=60)
                ax.set(xlim=(0,1),ylim=(0,1),xlabel='x',ylabel='y',aspect='equal')
        fig.suptitle(f'Prescribed-source-budget matched resolution comparison: t={float(runs[0][1][event]["surface"]["time"]):.2f}',fontsize=14)
        fig.text(.04,.025,'Shared color scales across both resolutions and all six displayed events. Cyan points are a projected slab within ±0.0234375 of z=0.5.\n'
                 'The saved gas planes differ slightly between resolutions. No pixelwise error or point correspondence is inferred.\n'
                 'Mass and mechanical-energy source coefficients match; spatial injection profiles and net gas-energy deposition need not match.',fontsize=9)
        fig.subplots_adjust(left=.055,right=.96,bottom=.15,top=.89,hspace=.43,wspace=.35)
        fig.savefig(output/f'event_{event:04d}_comparison.png',dpi=130);plt.close(fig)
    return dict(pressure=limits['pressure'],density=limits['density'],mach=[0,mach_high],display_slab_half_width=1.5/64)


def history_plot(output,runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for (meta,data),color in zip(runs,['#007f8b','#bf6b2b']):
        t=np.array([float(a['row']['time']) for a in data]);label=f"{meta['num_cells']}³"
        for ax,field in zip(axes[0],['mach','density_jump']):
            v=np.array([[float(a['row'][f'{field}_{q}']) for q in ['p16','median','p84']] for a in data])
            ax.plot(t,v[:,1],label=label,c=color);ax.fill_between(t,v[:,0],v[:,2],alpha=.12,color=color)
            ax.set(xlabel='Simulation time',ylabel='Mach' if field=='mach' else 'Sampled density ratio',title=f"{field.replace('_',' ').capitalize()}: median and 16–84% range")
        counts=np.array([len(a['surface']['positions']) for a in data])
        axes[1,0].plot(t,counts,label=label,c=color)
        for side,style in [('upstream','-'),('downstream','--')]:
            fraction=np.divide([int(a['row'][f'{side}_inside_any_source']) for a in data],counts,out=np.zeros(len(counts),float),where=counts>0)
            fraction[counts==0]=np.nan
            axes[1,1].plot(t,fraction,label=f'{label} {side}',c=color,ls=style)
    axes[0,1].axhline(1,c='.5',ls=':',lw=1)
    axes[1,0].set(title='Detected sample count (resolution dependent)',xlabel='Simulation time',ylabel='Surface samples')
    axes[1,1].set(title='Endpoint source overlap (advisory)',xlabel='Simulation time',ylabel='Fraction of selected samples')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Saved diagnostics at matching times',fontsize=15)
    fig.text(.04,.025,'Distributions count samples, not physical surface area; percentile bands are population ranges, not error bars.\n'
             'Direct sampled density ratios are retained below one. Matching prescribed budgets does not establish convergence or shock identities.',fontsize=9)
    fig.tight_layout(rect=(.01,.11,.99,.93));fig.savefig(output/'diagnostic_histories.png',dpi=140);plt.close(fig)


def graph_plots(output,runs,partitions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specs=[(0,'native_cells',20),(1,'native_cells',20),(1,'native_cells_scaled_count',80),(1,'common_physical_radius',80)]
    fig,axes=plt.subplots(4,3,figsize=(15,15))
    for row,(run_index,mode,cutoff) in enumerate(specs):
        meta,data=runs[run_index];n=meta['num_cells']
        for col,event in enumerate([12,13,14]):
            p=data[event]['surface']['positions'];labels=partitions[n,event,mode,60]
            slab=np.abs(p[:,2]-.5)<=1.5/64;colors=np.tile([.65,.65,.65,1.],(len(p),1))
            ids=np.unique(labels[labels>=0])
            for j,group in enumerate(ids):colors[labels==group]=plt.get_cmap('tab20')(j%20)
            ax=axes[row,col];ax.scatter(p[slab,0],p[slab,1],c=colors[slab],s=8)
            ax.set(xlim=(.12,.88),ylim=(.2,.8),xlabel='x',ylabel='y',aspect='equal')
            ax.set_title(f'{n}³, t={float(data[event]["surface"]["time"]):.2f}, groups retained: {len(ids)}\n{mode.replace("_"," ")}; cutoff {cutoff}',fontsize=10)
            ax.grid(alpha=.12)
    fig.suptitle('Inner-patch grouping depends on resolution and neighborhood scale',fontsize=15)
    fig.text(.04,.025,'All graphs use a 60° signed-normal limit and all 3D samples. Native radius: 1.5 cells. Common physical radius: 1.5/64 code units.\n'
             'Cutoff 80 is the 20-sample cutoff scaled by N²: a sampling heuristic, not a measured area threshold. Gray: excluded small components.\n'
             'Same physical display slab at both resolutions. Colors are local labels, not shock identities. Raw sizes and 30° results are saved separately.',fontsize=9)
    fig.tight_layout(rect=(.01,.1,.99,.95));fig.savefig(output/'inner_patch_sensitivity.png',dpi=130);plt.close(fig)


def run(reference,target,output):
    ma=json.loads((reference/'metadata.json').read_text());mb=json.loads((target/'metadata.json').read_text());validate_pair(ma,mb)
    if mb['source_budget_reference']['sha256']!=hashlib.sha256((reference/'metadata.json').read_bytes()).hexdigest():raise ValueError('Target matched a different reference metadata file.')
    output.mkdir(parents=True,exist_ok=False)
    runs=[];hashes={};reports=[]
    for root in [reference,target]:
        meta,data,input_hashes,report=load_checked(root);runs.append((meta,data));hashes.update(input_hashes);reports.append(report)
    statistics=[];components=[];partitions={}
    for meta,data in runs:
        n=meta['num_cells']
        for event,a in enumerate(data):
            s=a['surface'];cache={};saved={}
            for mode,radius,cutoff in neighborhood_settings(n,ma['num_cells']):
                for angle in [30,60]:
                    key=(radius,angle)
                    if key not in cache:cache[key]=connectivity_partition(s['positions'],s['normals'],float(s['grid_spacing']),radius,angle,1)
                    g=cache[key];raw=g['raw_labels'];sizes=g['component_sizes'];labels=raw.copy()
                    for group,size in enumerate(sizes):
                        if size<cutoff:labels[raw==group]=-1
                    partitions[n,event,mode,angle]=labels
                    saved[f'{mode}_a{angle}_labels']=labels;saved[f'{mode}_a{angle}_raw']=raw
                    statistics.append(dict(num_cells=n,event_index=event,time=float(s['time']),mode=mode,radius_cells=radius,
                                           radius_code_units=radius/n,normal_angle_degrees=angle,min_samples=cutoff,
                                           raw_components=len(sizes),retained_components=int((sizes>=cutoff).sum()),
                                           excluded_samples=int(sizes[sizes<cutoff].sum()),eligible_samples=len(g['eligible_rows']),
                                           largest_component_samples=int(max(sizes,default=0))))
                    for group,size in enumerate(sizes):components.append(dict(num_cells=n,event_index=event,mode=mode,normal_angle_degrees=angle,
                                                                               group=group,samples=int(size),retained=bool(size>=cutoff)))
            np.savez_compressed(output/f'groups_n{n:03d}_event{event:04d}.npz',**saved)
        print(f'Checked {n}³: {len(data)} events and graph-scale sensitivity',flush=True)
    csv_write(output/'grouping_summary.csv',statistics);csv_write(output/'raw_component_sizes.csv',components)
    scalar=[]
    for meta,data in runs:
        scalar.extend(dict(num_cells=meta['num_cells'],**a['row']) for a in data)
    csv_write(output/'diagnostic_histories.csv',scalar)
    limits=spatial_plots(output,runs);history_plot(output,runs);graph_plots(output,runs,partitions)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(readback_passed=True,runs=reports,matching_prescribed_budgets=True,matching_analysis_times=True,
                source_budget_scope='Mass rate and mechanical-energy coefficient, not measured cumulative net energy deposition.',
                physical_convergence_established=False,physical_identities_assigned=False,
                display_limits=limits,input_sha256=hashes,input_hashes_unchanged=True,
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [Path(__file__),Path('experiments/astrolink/shock_connectivity.py')]})
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['readback_passed','runs','matching_prescribed_budgets','physical_convergence_established']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path);p.add_argument('target',type=Path)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.reference,a.target,a.output_dir)
