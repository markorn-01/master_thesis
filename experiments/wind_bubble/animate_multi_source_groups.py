"""Checked smooth display, histograms and causal histories from callback catalogs.

Reuses the existing frozen implicit fit and mesh checks. No new detection,
grouping, association, source ownership or temporal interpolation is performed.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.wind_bubble.streaming_group_demo import build_tables
from experiments.wind_bubble.streaming_products import load_event
from experiments.wind_bubble.build_group_history_table import visible_relationships


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_display(physical, labels, nodes):
    """Existing bounded reconstruction plus the existing combined cleanup pass."""
    from experiments.wind_bubble.build_adaptive_sequence import build_event
    from experiments.wind_bubble.finalize_adaptive_sequence import triangle_keys
    from experiments.wind_bubble.finalize_surface_display import remove_defects
    from experiments.wind_bubble.orient_display_mesh import orient_faces
    from experiments.wind_bubble.research_surface_display import measure
    from experiments.wind_bubble.check_adaptive_mesh_times import group_accuracy, display_faces
    v, candidate, faces, groups, approved, status, metrics, details = build_event(physical, labels, nodes)
    details['final_combined_cleanup'] = None
    bad = any(metrics[k] for k in ('self_intersecting', 'nonmanifold_vertices',
                                   'nonmanifold_edges', 'inconsistent_shared_edge_winding'))
    if bad:
        before = Counter(triangle_keys(v, candidate, groups))
        v, candidate, groups, cleanup = remove_defects(v, candidate, groups)
        candidate, orientation = orient_faces(v, candidate)
        if Counter(triangle_keys(v, candidate, groups))-before:
            raise ValueError('Final cleanup added or moved triangle geometry.')
        case = dict(positions=physical['positions'], normals=physical['normals'], labels=labels,
                    dx=float(physical['grid_spacing']))
        metrics = measure(case, v, candidate, groups)
        rows = group_accuracy(physical, labels, nodes, v, candidate, groups)
        faces, approved = display_faces(candidate, groups, rows, metrics)
        # A failed construction cannot be approved by an unrelated cleanup.
        if details['failure'] is not None: faces, approved = np.empty((0,3),int), []
        status = 'mesh_approved' if len(approved)==len(nodes) else 'partial_accuracy_fallback'
        details.update(groups=rows, final_combined_cleanup=dict(cleanup=cleanup, orientation=orientation,
                                                               no_new_or_moved_triangle_geometry=True))
    return v, candidate, faces, groups, approved, status, metrics, details


def build(root, output, resume=False):
    from experiments.wind_bubble.implicit_surface_display import DisplaySettings
    from experiments.wind_bubble.animate_event_history import segment_rows
    meta = json.loads((root/'metadata.json').read_text())
    if meta['status'] != 'complete' or not meta['temporal_tracking']:
        raise ValueError('Require completed callback tracking.')
    sources = [Path(__file__)]+[Path('experiments/wind_bubble')/(name+'.py') for name in
        ('streaming_group_demo','build_adaptive_sequence','adaptive_surface_display',
         'implicit_surface_display','check_adaptive_mesh_times','finalize_surface_display',
         'finalize_adaptive_sequence','orient_display_mesh','research_surface_display')]
    code = {str(p):digest(p) for p in sources}
    inputs = {str(root/'metadata.json'):digest(root/'metadata.json')}
    for i in range(len(meta['actual_times'])):
        for p in (root/'online'/f'surfaces_{i:04d}.npz',root/'online/tracking'/f'groups_{i:04d}.npz',
                  root/'online/tracking'/f'event_{i:04d}.json.gz'):
            inputs[str(p)]=digest(p)
    config=dict(root=str(root), settings=asdict(DisplaySettings()), tolerance_cells=.25,
                target_fraction=.95, max_refinements=2)
    if output.exists():
        if not resume: raise ValueError('Use a fresh output directory or --resume.')
        state=json.loads((output/'metadata.json').read_text())
        if state['source_sha256']!=code or state['input_sha256']!=inputs or state['config']!=config:
            raise ValueError('Cannot resume after input, settings or code changes.')
    else:
        output.mkdir(parents=True)
        build_tables(root,output/'tables',expected_sources=len(meta['source_positions']))
        state=dict(status='running',config=config,input_sha256=inputs,source_sha256=code,completed=[])
    state['status']='running'; (output/'metadata.json').write_text(json.dumps(state,indent=2)+'\n')
    summaries=[]
    for i,time in enumerate(meta['actual_times']):
        mesh_path=output/f'mesh_{i:04d}.npz'; report_path=output/f'event_{i:04d}.json'
        if i in state['completed']:
            saved=json.loads(report_path.read_text())
            if digest(mesh_path)!=saved['mesh_sha256']: raise ValueError('Changed mesh cache.')
            summaries.append(saved['summary']); print(f'Resumed event {i}',flush=True); continue
        with np.load(root/'online'/f'surfaces_{i:04d}.npz') as a: physical={k:a[k].copy() for k in a.files}
        with np.load(root/'online/tracking'/f'groups_{i:04d}.npz') as a: labels=a['labels'].copy()
        record=load_event(root/'online/tracking',i); nodes=record['nodes']
        if float(physical['time'])!=time or record['time']!=time: raise ValueError('Event time differs.')
        print(f'Building event {i}/{len(meta["actual_times"])-1}: {len(labels)} samples, {len(nodes)} groups',flush=True)
        v,candidate,faces,groups,approved,status,metrics,details=checked_display(physical,labels,nodes)
        if len(faces) and (np.ptp(groups[faces],axis=1)!=0).any(): raise ValueError('Display triangle crosses groups.')
        segments=segment_rows(labels,nodes)
        arrays=dict(vertices=v,candidate_faces=candidate,faces=faces,vertex_groups=groups,
                    approved_groups=np.asarray(approved,int),original_labels=labels,original_segments=segments,
                    time=physical['time'],grid_spacing=physical['grid_spacing'],sources=physical['source_positions'])
        for key in ('positions','normals','mach','upstream_inside_any_source','downstream_inside_any_source'):
            arrays['original_'+key]=physical[key]
        np.savez_compressed(mesh_path,**arrays)
        with np.load(mesh_path) as a:
            for k,value in arrays.items(): np.testing.assert_array_equal(a[k],value)
        summary=dict(event=i,time=time,status=status,samples=len(labels),retained_groups=len(nodes),
                     approved_groups=len(approved),displayed_faces=len(faces),
                     fallback_points=int((~np.isin(labels,approved)).sum()),**metrics)
        report=dict(summary=summary,details=details,mesh_sha256=digest(mesh_path))
        report_path.write_text(json.dumps(report,indent=2)+'\n'); summaries.append(summary)
        state['completed'].append(i); (output/'metadata.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({k:summary[k] for k in ('event','status','approved_groups','retained_groups','fallback_points')}),flush=True)
    for p,h in {**inputs,**code}.items():
        if digest(p)!=h: raise ValueError('Input or source changed during display construction.')
    state['status']='complete'; (output/'metadata.json').write_text(json.dumps(state,indent=2)+'\n')
    report=dict(passed=True,events=len(summaries),summaries=summaries,
                fallback_events=[r['event'] for r in summaries if 'fallback' in r['status']],
                physical_samples_changed=False,tracking_decisions_changed=False,temporal_smoothing=False,
                physical_topology_established=False,input_sha256=inputs,source_sha256=code,
                output_sha256={str(p):digest(p) for p in output.rglob('*') if p.is_file() and p.name!='verification.json'})
    # Include the table manifest too; its original simulation joins are independently checked.
    report['output_sha256'][str(output/'tables/verification.json')]=digest(output/'tables/verification.json')
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def segment_palette(count):
    import matplotlib.pyplot as plt
    colors=list(plt.get_cmap('tab20').colors)+list(plt.get_cmap('Dark2').colors)
    if count>len(colors): raise ValueError('More segments than available distinct colors; select a larger explicit palette.')
    return colors[:count]


def render(root, display, output, fps=1., cutaway_z=.75):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from experiments.wind_bubble.check_adaptive_mesh_times import verify_report
    from experiments.wind_bubble.animate_adaptive_sequence import verify_encoded_frames
    if output.exists() or not np.isfinite(fps) or fps<=0: raise ValueError('Use fresh output and positive finite FPS.')
    if not np.isfinite(cutaway_z) or not 0<cutaway_z<=1: raise ValueError('Cutaway must lie in (0,1].')
    hashes={}; checked=verify_report(display/'verification.json',hashes)
    verify_report(display/'tables/verification.json',hashes)
    meta=json.loads((root/'metadata.json').read_text()); tables=display/'tables'
    rows=json.loads((tables/'group_history.json').read_text())
    events=json.loads((tables/'tracking_events.json').read_text())
    support=json.loads((tables/'conditional_support.json').read_text())
    accounting=json.loads((tables/'event_accounting.json').read_text())
    table_report=json.loads((tables/'verification.json').read_text())
    palette=segment_palette(table_report['segments']); lookup={r['node_id']:r for r in rows}
    times=np.asarray(meta['actual_times']); frames=[]; histograms=[]
    for i in range(len(times)):
        with np.load(display/f'mesh_{i:04d}.npz') as a: frames.append({k:a[k].copy() for k in a.files})
        path=root/'online'/f'histograms_{i:04d}.npz'; hashes[str(path)]=digest(path)
        with np.load(path) as a: histograms.append({k:a[k].copy() for k in a.files})
        with np.load(root/'online'/f'surfaces_{i:04d}.npz') as a:
            for field in ('mach','density_jump'):
                keep=a[field+'_valid']; h=histograms[-1]
                np.testing.assert_array_equal(np.histogram(a[field][keep],bins=h[field+'_edges'])[0],h[field+'_counts'])
            np.testing.assert_array_equal(frames[-1]['original_positions'],a['positions'])
        with np.load(root/'online/tracking'/f'groups_{i:04d}.npz') as a:
            np.testing.assert_array_equal(frames[-1]['original_labels'],a['labels'])
        if float(frames[-1]['time'])!=times[i]: raise ValueError('Mesh time mismatch.')
        for field in ('mach','density_jump'):
            np.testing.assert_array_equal(histograms[0][field+'_edges'],histograms[-1][field+'_edges'])
    output.mkdir(parents=True)
    maxima={f:max(1,max(int(h[f+'_counts'].max()) for h in histograms))*1.12 for f in ('mach','density_jump')}
    fig=plt.figure(figsize=(17,10),dpi=95)
    grid=fig.add_gridspec(2,3,left=.055,right=.96,bottom=.20,top=.86,width_ratios=[1.35,1,1],wspace=.30,hspace=.35)
    view=fig.add_subplot(grid[:,0],projection='3d'); ma=fig.add_subplot(grid[0,1]); ra=fig.add_subplot(grid[0,2]); ga=fig.add_subplot(grid[1,1:])
    title=fig.suptitle('',y=.975,fontsize=17); subtitle=fig.text(.5,.925,'',ha='center',fontsize=10)
    note=fig.text(.055,.125,'',fontsize=10,va='top')
    fig.text(.055,.025,'Colors identify geometric continuation segments; combinations start new segments. Labels do not assign physical source ownership.\n'
             'Meshes are display approximations. Gray points retain unapproved or ungrouped samples. No temporal interpolation or hole filling.\n'
             'Histograms use original samples (one count each, not area weights). All axes, bins, camera and colors remain fixed.',fontsize=9)
    headers=[None]*len(frames); visibility=[]

    def update(i):
        mesh=frames[i]; now=times[i]; current=[r for r in rows if r['event']==i]
        prefix=[r for r in rows if r['event']<=i]; accepted,conditional=visible_relationships(events,support,i)
        for ax in (view,ma,ra,ga): ax.clear()
        mapping={r['group']:r['segment_id'] for r in current}
        v,f,vg=mesh['vertices'],mesh['faces'],mesh['vertex_groups']
        shown=f[(v[f,2]<=cutaway_z).all(axis=1)]
        if len(shown):
            colors=np.array([palette[mapping[int(g)]] for g in vg[shown[:,0]]])
            view.add_collection3d(Poly3DCollection(v[shown],facecolors=colors,edgecolors=colors,linewidths=0,
                shade=True,lightsource=matplotlib.colors.LightSource(azdeg=315,altdeg=45)))
        p=mesh['original_positions']; fallback=(p[:,2]<=cutaway_z)&~np.isin(mesh['original_labels'],mesh['approved_groups'])
        view.scatter(*p[fallback].T,color='#858585',s=3,depthshade=False)
        view.scatter(*mesh['sources'].T,marker='*',color='black',s=45,depthshade=False)
        view.set(xlim=(0,1),ylim=(0,1),zlim=(0,1),xlabel='x',ylabel='y',zlabel='z',box_aspect=(1,1,1),
                 title=f'Smooth detected surfaces | z ≤ {cutaway_z:g} cutaway')
        view.view_init(25,-35)
        # Source labels describe the injection setup only, not ownership of nearby groups.
        for s,(pos,strength) in enumerate(zip(mesh['sources'],meta['source_strengths'])):
            view.text(*pos,f' S{s+1}:{strength:g}',fontsize=7,color='black')
        for ax,field,label in ((ma,'mach','Mach'),(ra,'density_jump','Sampled density ratio')):
            h=histograms[i]; ax.stairs(h[field+'_counts'],h[field+'_edges'],color='#087f8c',fill=True,alpha=.65)
            ax.set(xlim=(h[field+'_edges'][0],h[field+'_edges'][-1]),ylim=(0,maxima[field]),xlabel=label,ylabel='Surface-cell count',title='Current '+label.lower())
            ax.grid(alpha=.15)
        for edge in accepted:
            nodes=[lookup[n] for n in edge['parents']+edge['children']]
            cx=np.mean([lookup[edge['parents'][0]]['time'],lookup[edge['children'][0]]['time']]);cy=np.mean([n['segment_id'] for n in nodes])
            for node in nodes:
                ga.plot([node['time'],cx],[node['segment_id'],cy],c='#497b7b' if edge['status']=='shared' else '.6',ls='-' if edge['status']=='shared' else ':',lw=1)
        for edge in conditional:
            a,b=lookup[edge['parent_node']],lookup[edge['child_node']]
            ga.plot([a['time'],b['time']],[a['segment_id'],b['segment_id']],c='#e38a24',ls='--',lw=2)
        for row in prefix:
            ga.scatter(row['time'],row['segment_id'],c=[palette[row['segment_id']]],s=16,zorder=4)
        for row in current:
            ga.scatter(row['time'],row['segment_id'],facecolors='none',edgecolors='black',s=55,zorder=5)
            ga.annotate(f'G{row["segment_id"]:02d}',(row['time'],row['segment_id']),xytext=(5,0),textcoords='offset points',fontsize=7,va='center')
        ga.axvline(now,c='black',lw=.8);ga.set(xlim=(times[0]-.003,times[-1]+.018),ylim=(len(palette)-.5,-.7),xlabel='Simulation time',title='Observed group history | only available relationships')
        ga.set_yticks(range(len(palette)),[f'G{s:02d}' for s in range(len(palette))],fontsize=6);ga.grid(axis='x',alpha=.2)
        ga.legend(handles=[Line2D([],[],c='#497b7b',label='Accepted shared'),Line2D([],[],c='.6',ls=':',label='Gate-specific'),Line2D([],[],c='#e38a24',ls='--',label='Conditional support')],fontsize=8,loc='upper left',bbox_to_anchor=(0,-.18),ncol=3,frameon=False)
        a=accounting[i]; summary=checked['summaries'][i]
        title.set_text(f'{len(meta["source_positions"])} interacting wind sources | t = {now:.2f} | event {i:02d}/{len(times)-1:02d}')
        subtitle.set_text('Unequal strengths: '+', '.join(f'{v:g}' for v in meta['source_strengths'])+' | same total prescribed input as the equal-source case')
        note.set_text(f'Detected samples: {a["detected_samples"]:,} | retained groups: {a["retained_groups"]} | approved mesh groups: {summary["approved_groups"]}/{summary["retained_groups"]}\n'
                      f'Raw-point fallback (full catalog): {summary["fallback_points"]:,} | source-overlap endpoints: upstream {a["all_upstream_source_overlaps"]}, downstream {a["all_downstream_source_overlaps"]}\n'
                      f'Conditional links visible: {len(conditional)}. No future observation is used before it arrives.')
        fig.canvas.draw();headers[i]=np.asarray(fig.canvas.buffer_rgba())[:90,:,:3].copy()

    movie=FuncAnimation(fig,update,frames=len(frames),interval=1000/fps,blit=False)
    path=output/'four_bubble_evolution.gif'; movie.save(path,writer=PillowWriter(fps=fps),dpi=95)
    encoded=verify_encoded_frames(path,headers)
    for i in (0,9,13,14,15,len(times)-1):
        update(i);fig.savefig(output/f'event_{i:04d}.png',dpi=140)
    plt.close(fig)
    for i in range(len(times)):
        accepted,conditional=visible_relationships(events,support,i)
        visibility.append(dict(event=i,visible_event_ids=[e['event_id'] for e in accepted],visible_support_ids=[e['support_id'] for e in conditional],
                               visible_node_ids=[r['node_id'] for r in rows if r['event']<=i]))
    for p,h in hashes.items():
        if digest(p)!=h: raise ValueError('Validated input changed during animation.')
    report=dict(passed=True,**encoded,fps=fps,events=len(frames),per_event_visibility=visibility,
                conditional_first_visible_event=min((e['available_event'] for e in support),default=None),
                physical_samples_changed=False,tracking_changed=False,conditional_support_in_statistics=False,
                same_group_close_sheets_resolved=False,physical_topology_established=False,
                bounds=[[0,1]]*3,camera=[25,-35],cutaway_z=cutaway_z,histogram_y_limits=maxima,
                segment_colors=[list(c) for c in palette],input_sha256=hashes,
                source_sha256={str(Path(__file__)):digest(__file__)},
                output_sha256={str(p):digest(p) for p in output.iterdir()})
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','events','decoded_frames','playback_seconds','conditional_first_visible_event')},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--display-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--build-only',action='store_true')
    parser.add_argument('--render-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--fps',type=float,default=1.)
    parser.add_argument('--cutaway-z',type=float,default=.75)
    args=parser.parse_args()
    if not args.render_only: build(args.root,args.display_dir,args.resume)
    if not args.build_only:
        if args.output_dir is None: parser.error('--output-dir is required for rendering')
        render(args.root,args.display_dir,args.output_dir,args.fps,args.cutaway_z)
