"""Read back the GPU detection audit and summarize rejection evidence, not truth labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def thresholds():
    gamma=5/3; m2=1.3**2
    pressure=(2*gamma*m2-(gamma-1))/(gamma+1)
    temperature=pressure*((gamma-1)*m2+2)/((gamma+1)*m2)
    return pressure,temperature


def stencil_probe(full,crop,meta,eligible):
    """Offline sensitivity only; interpolate full fields at selected cell-centered rays.

    Check the production one-cell jumps before exploring larger separations.
    No alternative zone/surface catalog or Mach estimate is generated.
    """
    from scipy.ndimage import map_coordinates
    points=crop['centers'][eligible];normals=crop['direction'][eligible]
    dx=float(crop['grid_spacing']);indices=np.rint(points/dx-.5).astype(np.float32)
    pressure=full[int(crop['pressure_index'])];temperature=pressure/full[int(crop['density_index'])]
    sources=np.asarray(meta['source_positions']);radius=meta['injection_radius'];pr,tr=thresholds();rows=[]
    for step in [1,2,3,4]:
        pre=indices+step*normals;post=indices-step*normals
        def jumps(field):
            a=map_coordinates(field,post.T,order=1,mode='nearest',prefilter=False)
            b=map_coordinates(field,pre.T,order=1,mode='nearest',prefilter=False)
            return np.log(np.maximum(a,1e-30))-np.log(np.maximum(b,1e-30))
        lp,lt=jumps(pressure),jumps(temperature)
        if step==1:
            np.testing.assert_allclose(lp,crop['log_pressure_jump'][eligible],rtol=1e-4,atol=3e-6)
            np.testing.assert_allclose(lt,crop['log_temperature_jump'][eligible],rtol=1e-4,atol=3e-6)
        passed=(lp>=np.log(pr))&(lt>=np.log(tr))
        outside=np.ones(len(points),bool);support=outside.copy()
        for coordinates in [pre,post]:
            locations=(coordinates+.5)*dx
            outside&=(np.linalg.norm(locations[:,None,:]-sources[None,:,:],axis=-1)>radius).all(axis=1)
            # All eight interpolation corners must avoid the actual discrete injection mask.
            lower=np.floor(coordinates).astype(int)
            for corner in np.ndindex(2,2,2):
                corner_index=lower+corner
                valid=((corner_index>=0)&(corner_index<full.shape[1])).all(axis=1)
                locations=(corner_index+.5)*dx
                support&=valid&(np.linalg.norm(locations[:,None,:]-sources[None,:,:],axis=-1)>radius-dx/2).all(axis=1)
        center_outside=(np.linalg.norm(points[:,None,:]-sources[None,:,:],axis=-1)>radius).all(axis=1)
        rows.append(dict(step_each_side_cells=step,eligible_cells=len(points),joint_pass=int(passed.sum()),
            pass_with_both_endpoints_outside_exclusion=int((passed&outside).sum()),
            pass_with_interpolation_corners_outside_injection=int((passed&support).sum()),
            pass_with_center_endpoints_and_corners_outside=int((passed&outside&support&center_outside).sum())))
    return rows


def run(root,reference,output):
    verification=root/'audit/verification.json';v=json.loads(verification.read_text());assert v['passed']
    hashes={}
    def record(path):hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    record(verification)
    for name,digest in v['source_sha256'].items():
        p=Path(name.split('/master_thesis/',1)[-1]);assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
    for name,digest in v['state_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    for path in [root/'metadata.json',reference/'metadata.json',root/'audit/stage_counts.csv']:record(path)
    meta=json.loads((root/'metadata.json').read_text());prior=json.loads((reference/'metadata.json').read_text())
    for key in ['num_cells','actual_times','source_positions','source_strengths','discrete_energy_rates','injection_radius']:
        np.testing.assert_allclose(meta[key],prior[key],rtol=1e-7,atol=1e-8)
    # Independent local readback of the reported replay comparison.
    for i in range(len(meta['actual_times'])):
        for prefix in ['surfaces','histograms','slice']:
            paths=[p/'online'/f'{prefix}_{i:04d}.npz' for p in [root,reference]]
            for path in paths:record(path)
            with np.load(paths[0]) as a,np.load(paths[1]) as b:
                assert set(a.files)==set(b.files)
                for key in a.files:
                    if a[key].dtype.kind=='f':np.testing.assert_allclose(a[key],b[key],rtol=1e-6,atol=1e-8,equal_nan=True)
                    else:np.testing.assert_array_equal(a[key],b[key])
    counts=list(csv.DictReader((root/'audit/stage_counts.csv').open()));rows=[];frames=[];slices=[];stencils=[]
    pmin,tmin=thresholds()
    for index in v['captured_events']:
        full=np.load(root/'audit'/f'state_{index:04d}.npy',mmap_mode='r')
        for source in [0,1]:
            path=root/'audit'/f'stages_{index:04d}_source{source}.npz';record(path)
            with np.load(path) as saved:d={k:saved[k] for k in saved.files}
            np.testing.assert_array_equal(d['shock_zone'],d['compression']&d['gradient_alignment']&d['minimum_jump'])
            np.testing.assert_array_equal(d['minimum_jump'],d['pressure_jump_pass']&d['temperature_jump_pass'])
            grid=np.rint(d['centers']/float(d['grid_spacing'])-.5).astype(int)
            np.testing.assert_array_equal(d['state'],full[:,grid[...,0],grid[...,1],grid[...,2]])
            center=np.asarray(meta['source_positions'][source]);delta=d['centers']-center
            radii=np.linalg.norm(delta,axis=-1)
            inward=(d['direction']*delta/np.maximum(radii[...,None],1e-30)).sum(axis=-1)<-.5
            region=d['region_mask'];outside=~d['injection_mask']
            for name,mask in [('all_local',region),('local_inward',region&inward),
                              ('local_outside_source',region&outside),('local_inward_outside_source',region&outside&inward)]:
                original=next(r for r in counts if int(r['event'])==index and int(r['source'])==source and r['region']==name)
                for key in ['compression','gradient_alignment','pressure_jump_pass','temperature_jump_pass','minimum_jump','shock_zone','shock_surface','exported_surface']:
                    assert int((d[key]&mask).sum())==int(original[key]),(index,source,name,key)
            eligible=region&inward&outside&d['compression']&d['gradient_alignment']
            for row in stencil_probe(full,d,meta,eligible):
                stencils.append(dict(event=index,time=float(d['time']),source=source,**row))
            original=next(r for r in counts if int(r['event'])==index and int(r['source'])==source and r['region']=='local_inward_outside_source')
            rows.append(dict(event=index,time=float(d['time']),source=source,compression_and_alignment=int(eligible.sum()),
                pressure_jump_pass=int((eligible&d['pressure_jump_pass']).sum()),
                temperature_jump_pass=int((eligible&d['temperature_jump_pass']).sum()),
                joint_jump_pass=int((eligible&d['minimum_jump']).sum()),
                maximum_local_pressure_ratio=float(np.exp(d['log_pressure_jump'][eligible]).max()) if eligible.any() else None,
                maximum_local_temperature_ratio=float(np.exp(d['log_temperature_jump'][eligible]).max()) if eligible.any() else None,
                shock_zone=int(original['shock_zone']),shock_surface=int(original['shock_surface']),
                removed_by_export=int(original['surface_removed_by_export'])))
            if source==1:
                frames.append(d)
                # Illustrative +x ray: actual nearest grid centers in y and z, not exact center interpolation.
                iy=int(np.argmin(abs(d['centers'][0,:,0,1]-center[1])))
                iz=int(np.argmin(abs(d['centers'][0,0,:,2]-center[2])))
                x=d['centers'][:,iy,iz,0];take=(x>=center[0])&(x<=center[0]+2*meta['injection_radius'])
                for j in np.flatnonzero(take):
                    slices.append(dict(event=index,time=float(d['time']),radius=float(x[j]-center[0]),
                        y=float(d['centers'][j,iy,iz,1]),z=float(d['centers'][j,iy,iz,2]),
                        density=float(d['state'][int(d['density_index']),j,iy,iz]),pressure=float(d['state'][int(d['pressure_index']),j,iy,iz]),
                        temperature=float(d['state'][int(d['pressure_index']),j,iy,iz]/d['state'][int(d['density_index']),j,iy,iz]),
                        divergence=float(d['divergence'][j,iy,iz]),log_pressure_jump=float(d['log_pressure_jump'][j,iy,iz]),
                        log_temperature_jump=float(d['log_temperature_jump'][j,iy,iz])))
    output.mkdir(parents=True,exist_ok=False)
    for filename,data in [('eligible_cell_summary.csv',rows),('illustrative_ray.csv',slices),('stencil_sensitivity.csv',stencils)]:
        with (output/filename).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    render(rows,slices,meta,output)
    for name,digest in hashes.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    summary=dict(passed=True,local_replay_events_checked=len(meta['actual_times']),captured_states_checked=len(v['captured_events']),
        crops_checked=2*len(v['captured_events']),stage_count_regions_checked=len(counts),
        pressure_ratio_threshold=pmin,temperature_ratio_threshold=tmin,weak_source=[r for r in rows if r['source']==1],
        one_cell_scipy_jump_crosscheck_passed=True,stencil_sensitivity_is_offline_only=True,
        source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},input_sha256=hashes,
        detection_parameters_changed=False,physical_shock_absence_established=False,
        causal_claim='Pressure-jump detection gate blocks the inspected early weak-source cells. Physical cause of subthreshold jumps remains unestablished.')
    (output/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['pressure_ratio_threshold','temperature_ratio_threshold','weak_source']},indent=2))


def render(rows,rays,meta,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pmin,tmin=thresholds();fig,axes=plt.subplots(3,2,figsize=(14,12))
    fields=[('pressure','Pressure'),('temperature','P / density'),('log_pressure_jump','Local log pressure jump'),
            ('log_temperature_jump','Local log temperature jump'),('divergence','Velocity divergence')]
    events=sorted({r['event'] for r in rays});colors=plt.get_cmap('viridis')(np.linspace(.05,.9,len(events)))
    for ax,(key,label) in zip(axes.flat,fields):
        for event,color in zip(events,colors):
            data=[r for r in rays if r['event']==event]
            ax.plot([r['radius'] for r in data],[r[key] for r in data],marker='.',color=color,label=f't={data[0]["time"]:.2f}')
        ax.axvspan(0,meta['injection_radius']-1/(2*meta['num_cells']),color='#aaaaaa',alpha=.15)
        ax.axvline(meta['injection_radius'],ls=':',color='#777777')
        ax.set(xlabel='Distance from weaker source along +x',ylabel=label,xlim=(0,.125));ax.grid(alpha=.2)
        if key in ['pressure','temperature']:ax.set_yscale('log')
        if key.startswith('log_'):
            threshold=np.log(pmin if key=='log_pressure_jump' else tmin)
            ax.axhline(threshold,color='#ba3434',ls='--',label='Detection threshold')
        if key=='divergence':ax.axhline(0,c='#555555',ls='--')
    axes[0,0].legend(fontsize=9,ncol=2)
    axes[1,0].legend(handles=[axes[1,0].lines[-1]],labels=['Detection threshold'],fontsize=9)
    ax=axes[-1,1];weak=[r for r in rows if r['source']==1];times=[r['time'] for r in weak]
    for key,label,style in [('compression_and_alignment','Compression + alignment','o-'),('pressure_jump_pass','Also passes pressure jump','s--'),('joint_jump_pass','Also passes both jumps','x:')]:
        ax.plot(times,[r[key] for r in weak],style,label=label)
    ax.set(xlabel='Simulation time',ylabel='Eligible local inward cells',yscale='symlog')
    ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Weaker-source detection audit: the local jump gate is restrictive',fontsize=17,y=.98)
    fig.text(.06,.025,'Profiles: one illustrative +x ray at nearest available y,z centers; these do not represent every surface direction.\n'
        'Gray: discrete injection interior; dotted vertical line: geometric exclusion radius. Red horizontal lines: production M_min=1.3 thresholds.\n'
        'Count panel: all local inward cells outside the discrete injection mask that pass compression and alignment.\n'
        'These are detection-stage local jumps, not final reconstructed shock jumps or proof of a physical shock.',fontsize=10)
    fig.tight_layout(rect=(0,.12,1,.95));fig.savefig(output/'detection_gate_profiles.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/two_bubbles_detection_audit_iwr_n128'))
    parser.add_argument('--reference',type=Path,default=Path('outputs/two_bubbles_unequal_iwr_compgpu1_n128'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/two_bubbles_detection_gate_review_n128'))
    args=parser.parse_args();run(args.root,args.reference,args.output_dir)
