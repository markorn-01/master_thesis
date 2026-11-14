"""Audit saved near-source inward samples without changing detection or tracking."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def probe(surface, radius_multiple=2., alignment_limit=-.5):
    """Geometric inspection population, not a physical shock/source label."""
    p=surface['positions'].astype(float);sources=surface['source_positions']
    delta=p[:,None,:]-sources[None,:,:];r=np.linalg.norm(delta,axis=2)
    nearest=r.argmin(axis=1)
    radial=delta[np.arange(len(p)),nearest]/np.maximum(r[np.arange(len(p)),nearest,None],1e-30)
    n=surface['normals'].astype(float);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
    alignment=np.einsum('ij,ij->i',radial,n)
    local=r[np.arange(len(p)),nearest]<=radius_multiple*float(surface['injection_radius'])
    return [local&(nearest==i)&(alignment<alignment_limit) for i in range(len(sources))]


def write_csv(path,rows):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(root,directory,output):
    report=json.loads((directory/'verification.json').read_text());hashes={}
    for section in ['input_sha256','source_sha256']:
        for name,digest in report[section].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Changed input: {name}')
    def record(path):hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    record(directory/'verification.json');record(directory/'graph.json')
    graph=json.loads((directory/'graph.json').read_text())
    events=[];components=[];sensitivity=[];frames=[]
    for index in range(report['events']):
        paths=[root/'online'/f'surfaces_{index:04d}.npz',root/'online'/f'slice_{index:04d}.npz',directory/f'groups_{index:04d}.npz']
        for path in paths:record(path)
        with np.load(paths[0]) as data,np.load(paths[1]) as bg,np.load(paths[2]) as part:
            s={k:data[k] for k in data.files};p={k:part[k] for k in part.files}
            masks=probe(s)
            for source,mask in enumerate(masks):
                raw=p['raw_labels'][mask];labels=p['labels'][mask]
                groups=np.unique(raw);assert (groups>=0).all()
                for g in groups:
                    full=p['raw_labels']==g;retained=p['component_sizes'][g]>=20
                    assert np.all((p['labels'][full]>=0)==retained)
                    assert int(full.sum())==int(p['component_sizes'][g])
                    components.append(dict(event=index,time=float(s['time']),source=source,raw_component=int(g),
                        full_component_samples=int(full.sum()),probe_samples=int((raw==g).sum()),retained=bool(retained)))
                events.append(dict(event=index,time=float(s['time']),source=source,probe_samples=int(mask.sum()),
                    retained_probe_samples=int((labels>=0).sum()),excluded_probe_samples=int((labels<0).sum()),
                    components=len(groups),largest_component=int(p['component_sizes'][groups].max()) if len(groups) else 0,
                    upstream_overlap=int(s['upstream_inside_any_source'][mask].sum()),
                    downstream_overlap=int(s['downstream_inside_any_source'][mask].sum()),
                    unknown_endpoints=int((~s['sample_positions_valid'][mask]).sum())))
            for radius in [1.5,2.,2.25]:
                for alignment in [0.,-.5,-.8]:
                    weak=probe(s,radius,alignment)[1]
                    sensitivity.append(dict(event=index,time=float(s['time']),radius_multiple=radius,
                        alignment_limit=alignment,probe_samples=int(weak.sum()),
                        retained_probe_samples=int((p['labels'][weak]>=0).sum())))
            if index in [15,16]:
                frames.append(dict(event=index,time=float(s['time']),sources=s['source_positions'],
                    radius=float(s['injection_radius']),dx=float(s['grid_spacing']),z=float(bg['z_coordinate']),
                    density=bg['density'],pressure=bg['pressure'],positions=s['positions'],
                    mask=masks[1],labels=p['labels']))
    # Endpoint populations versus actual saved graph: no inferred ownership used by tracker.
    weak_rows=[r for r in events if r['source']==1]
    first=next((r['event'] for r in weak_rows if r['probe_samples']),None)
    first_retained=next((r['event'] for r in weak_rows if r['retained_probe_samples']),None)
    final_groups=[]
    for g in np.unique(frames[-1]['labels'][frames[-1]['mask']]):
        if g<0:continue
        node=next(n for n in graph['nodes'] if n['event']==16 and n['group']==int(g))
        incoming=[e for e in graph['events'] if node['node_id'] in e['children']]
        final_groups.append(dict(group=int(g),segment_id=node['segment_id'],incoming_events=len(incoming)))
    output.mkdir(parents=True,exist_ok=False)
    write_csv(output/'sample_history.csv',events);write_csv(output/'components.csv',components)
    write_csv(output/'probe_sensitivity.csv',sensitivity)
    render(events,frames,output)
    firsts=[]
    for radius in [1.5,2.,2.25]:
        for angle in [0.,-.5,-.8]:
            rows=[r for r in sensitivity if r['radius_multiple']==radius and r['alignment_limit']==angle]
            firsts.append(dict(radius_multiple=radius,alignment_limit=angle,
                first_event=next((r['event'] for r in rows if r['probe_samples']),None),
                first_retained_event=next((r['event'] for r in rows if r['retained_probe_samples']),None)))
    for name,digest in hashes.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    result=dict(first_weak_probe_event=first,first_weak_retained_event=first_retained,weak_final_groups=final_groups,
        weak_history=weak_rows,probe_sensitivity_first_events=firsts,
        inspection_selection=dict(nearest_source=1,max_radius_in_injection_radii=2.,radial_normal_dot_less_than=-.5),
        selection_is_source_ownership=False,shock_detection_rerun=False,raw_gas_states_available=False,
        tracking_or_thresholds_changed=False,physical_shock_absence_established=False,
        input_sha256=hashes,source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(first_weak_probe_event=first,first_weak_retained_event=first_retained,
        final_groups=final_groups,probe_sensitivity_first_events=firsts),indent=2))


def render(events,frames,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle
    fig=plt.figure(figsize=(13,11))
    grid=fig.add_gridspec(2,2,left=.08,right=.91,bottom=.17,top=.88,hspace=.45,wspace=.3)
    ax=fig.add_subplot(grid[0,:])
    strong=[r for r in events if r['source']==0];weak=[r for r in events if r['source']==1]
    times=[r['time'] for r in weak]
    ax.plot(times,[r['probe_samples'] for r in strong],color='#a7afb8',label='Near stronger source: all probe samples')
    ax.plot(times,[r['probe_samples'] for r in weak],'o-',color='#6c4299',label='Near weaker source: all probe samples')
    ax.plot(times,[r['retained_probe_samples'] for r in weak],'s--',color='#16818b',label='Near weaker source: retained in groups >=20')
    ax.set(yscale='symlog',ylim=(-.2,2500),xlabel='Simulation time',ylabel='Detected probe samples (symlog scale)',
        title='Saved detections appear late near the weaker source; grouping removes additional small pieces')
    ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=9)
    for index in [15,16]:
        row=weak[index];ax.annotate(f'{row["probe_samples"]} total / {row["retained_probe_samples"]} retained',
            (row['time'],row['probe_samples']),xytext=(-130,30 if index==16 else -35),textcoords='offset points',
            arrowprops=dict(arrowstyle='-',color='#555555'),fontsize=10)
    positive=np.concatenate([f['density'].ravel() for f in frames]);norm=LogNorm(positive.min(),positive.max())
    for i,f in enumerate(frames):
        view=fig.add_subplot(grid[1,i]);image=view.imshow(f['density'].T,origin='lower',extent=(0,1,0,1),norm=norm,cmap='magma')
        slab=f['mask']&(np.abs(f['positions'][:,2]-f['z'])<=f['dx'])
        for retained,color,marker,label in [(False,'#00c4e5','x','Below group cutoff'),(True,'#65ee8d','o','Retained group')]:
            mask=slab&((f['labels']>=0)==retained);p=f['positions'][mask]
            view.scatter(p[:,0],p[:,1],c=color,marker=marker,s=30,label=label,linewidths=.8)
        center=f['sources'][1]
        view.add_patch(Circle(center[:2],f['radius'],fill=False,color='white',ls='--',lw=1.2))
        view.add_patch(Circle(center[:2],2*f['radius'],fill=False,color='white',ls=':',lw=.8))
        view.scatter(*center[:2],marker='*',c='white',s=65)
        view.set(xlim=(.5,.81),ylim=(.35,.65),xlabel='x',ylabel='y',aspect='equal',
            title=f't={f["time"]:.2f}: density slice + nearby surface samples\n{int(slab.sum())} displayed / {int(f["mask"].sum())} 3D probe samples',)
        if i==0:view.legend(loc='lower left',fontsize=8)
    colorax=fig.add_axes([.93,.19,.016,.26]);fig.colorbar(image,cax=colorax,label='Density (code units)')
    fig.suptitle('Where does the weaker-source inner history disappear?',fontsize=17,y=.965)
    fig.text(.5,.92,'Inspection only: nearest source, radius <= 2 injection radii, radial normal alignment < -0.5.',ha='center',fontsize=10)
    fig.text(.08,.035,'Top counts use the full 3D catalogs. Lower backgrounds are single density slices at z=0.50390625; markers use |z-z_slice| <= one cell.\n'
        'White dashed circle: injection radius; dotted circle: inspection radius. Geometric selections do not establish shock ownership.\n'
        'No detected sample in this selection does not prove physical absence. Full gas states and shock-zone masks were not saved.\n'
        'No grouping, tracking, physical estimator, or acceptance threshold was changed.',fontsize=10)
    fig.savefig(output/'weak_source_diagnosis.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/two_bubbles_unequal_iwr_compgpu1_n128'))
    parser.add_argument('--directory',type=Path,default=Path('outputs/two_bubbles_unequal_history_v1_n128'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/two_bubbles_weak_source_review_n128'))
    args=parser.parse_args();run(args.root,args.directory,args.output_dir)
