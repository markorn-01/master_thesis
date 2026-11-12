"""Frozen grouping/events and delayed support on a new unequal-wind sequence."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.shock_connectivity import connectivity_partition
from experiments.astrolink.event_tracking import event_tracking
from experiments.astrolink.event_history import assemble_history
from experiments.astrolink.tracklet_fragments import attach_fragments
from experiments.astrolink.check_tracklet_fragments import check_dense
from experiments.wind_bubble.apply_event_tracking import check_candidates,check_selection_dp,event_key
from experiments.wind_bubble.compare_two_bubble_resolutions import load_checked,csv_write
from experiments.wind_bubble.compare_event_history_resolutions import group_statistics
from experiments.wind_bubble.animate_event_history import segment_rows,cutaway_mask


def window_catalogs(data,labels,graph,start):
    """All as-yet-uncontinued groups; all eligible later shared tracklets.

    No inward-orientation, source ownership or manually selected event criterion.
    Empty earlier populations are retained as no-evidence windows.
    """
    lookup={n['node_id']:n for n in graph['nodes']}
    earlier=[e for e in graph['events'] if e['parent_event']==start and e['status']=='shared' and e['kind']=='continuation']
    excluded_a={lookup[e['parents'][0]]['group'] for e in earlier}
    excluded_b={lookup[e['children'][0]]['group'] for e in earlier}
    later=[e for e in graph['events'] if e['parent_event']==start+1 and e['status']=='shared' and e['kind']=='continuation']
    pairs=[(lookup[e['parents'][0]]['group'],lookup[e['children'][0]]['group']) for e in later
           if lookup[e['parents'][0]]['group'] not in excluded_b]
    masks=[(labels[start]>=0)&~np.isin(labels[start],list(excluded_a)),
           np.isin(labels[start+1],[p for p,c in pairs]),np.isin(labels[start+2],[c for p,c in pairs])]
    catalogs=[]
    for offset,mask in enumerate(masks):
        source=data[start+offset]['surface']
        # Match the float64 input preparation used in the frozen equal-wind
        # support experiment; float32 normal normalization alters tiny scores.
        catalogs.append(dict(positions=source['positions'][mask].astype(float),normals=source['normals'][mask].astype(float),
                             labels=labels[start+offset][mask]))
    return catalogs,pairs


def run(root,output):
    meta,data,hashes,readback=load_checked(root)
    if meta.get('source_strengths')!=[1.,.5] or meta['num_cells']!=64:
        raise ValueError('This frozen validation expects the prepared 64³, 1:0.5 wind case.')
    if not json.loads((root/'equivalence.json').read_text())['passed']:raise ValueError('Require passed run equivalence.')
    control_path=Path('outputs/event_tracking_controls_solver_validation/verification.json')
    control=json.loads(control_path.read_text())
    for path,digest in control['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError(f'Controlled source changed: {path}')
    family_report_path=Path('outputs/tracklet_fragment_support_v1/verification.json')
    family_result_path=Path('outputs/tracklet_fragment_support_v1/real_results.json')
    family_report=json.loads(family_report_path.read_text())
    for path,digest in family_report['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError(f'Frozen family source changed: {path}')
    family_parameters=json.loads(family_result_path.read_text())[0]['result']['parameters']
    times=meta['actual_times'];output.mkdir(parents=True,exist_ok=False)
    labels=[];group_counts=[]
    for i,event in enumerate(data):
        surface=event['surface']
        partition=connectivity_partition(surface['positions'],surface['normals'],float(surface['grid_spacing']),
            radius_cells=1.5,max_angle_degrees=60,min_display_size=20)
        labels.append(partition['labels'])
        np.savez_compressed(output/f'groups_{i:04d}.npz',labels=labels[-1],raw_labels=partition['raw_labels'],
                            component_sizes=partition['component_sizes'])
        group_counts.append(dict(event=i,time=times[i],retained_groups=len(np.unique(labels[-1][labels[-1]>=0])),
            retained_samples=int((labels[-1]>=0).sum()),excluded_samples=int((labels[-1]<0).sum()),
            small_components=int((partition['component_sizes']<20).sum())))
    records=[];checks=[];agreements=[]
    for index in range(len(times)-1):
        a,b=[d['surface'] for d in data[index:index+2]];sets=[]
        for gate in [1.5,2.5]:
            result=event_tracking(a['positions'],a['normals'],labels[index],b['positions'],b['normals'],labels[index+1],gate/64)
            for key,value in control['parameters'].items():assert result['parameters'][key]==value
            check=check_candidates(a,b,labels[index],labels[index+1],result);check_selection_dp(result)
            checks.append(dict(parent_event=index,gate_reference_cells=gate,independent_dp_passed=True,**check))
            records.append(dict(parent_event=index,child_event=index+1,gate_reference_cells=gate,result=result))
            sets.append({event_key(e) for e in result['events']})
        agreements.append(dict(parent_event=index,child_event=index+1,event_sets_identical=sets[0]==sets[1],
            common=len(sets[0]&sets[1]),narrow_only=len(sets[0]-sets[1]),wide_only=len(sets[1]-sets[0])))
        print(f'{index:02d}→{index+1:02d}: agreement={sets[0]==sets[1]}; shared={sets[0]&sets[1]}',flush=True)
    groups=[list(map(int,np.unique(l[l>=0]))) for l in labels]
    graph=assemble_history(groups,times,records)
    statistics=[]
    for i,event in enumerate(data):
        nodes=[n for n in graph['nodes'] if n['event']==i]
        stats=group_statistics(event['surface'],labels[i],nodes)
        for node,row in zip(nodes,stats):
            mask=labels[i]==node['group'];s=event['surface']
            row.update(upstream_inside_source=int(s['upstream_inside_any_source'][mask].sum()),
                       downstream_inside_source=int(s['downstream_inside_any_source'][mask].sum()))
            node.update(row);statistics.append(row)
    # Automatic scan of every available triple, without choosing a favorable window.
    windows=[];family_rows=[]
    for start in range(len(times)-2):
        catalogs,pairs=window_catalogs(data,labels,graph,start)
        if not pairs:continue
        for gate in [1.5,2.5]:
            for method,predict in [('static_family',False),('delayed_family',True)]:
                result=attach_fragments(*catalogs,pairs,times[start:start+3],gate/64,predict=predict)
                for key,value in family_parameters.items():
                    if key!='gate':assert result['parameters'][key]==value
                # This run's candidate windows are small enough for dense independent checks.
                check_dense(*catalogs,result,times[start:start+3],predict)
                windows.append(dict(first_event=start,times=times[start:start+3],gate_reference_cells=gate,
                    method=method,earlier_groups=list(map(int,np.unique(catalogs[0]['labels']))),result=result))
                for f in result['families']:
                    family_rows.append(dict(first_event=start,first_time=times[start],gate_reference_cells=gate,
                        method=method,child_group=f['child_group'],parents=','.join(map(str,f['parent_groups'])),
                        fragments=len(f['parent_groups']),coverage=f['child_union_coverage'],accepted=f['accepted']))
    with gzip.open(output/'tracking_results.json.gz','wt') as f:json.dump(records,f)
    (output/'graph.json').write_text(json.dumps(graph,indent=2)+'\n')
    (output/'fragment_windows.json').write_text(json.dumps(windows,indent=2)+'\n')
    csv_write(output/'groups.csv',group_counts);csv_write(output/'group_statistics.csv',statistics)
    csv_write(output/'distance_agreement.csv',agreements)
    if family_rows:csv_write(output/'fragment_families.csv',family_rows)
    frames=[]
    for index,event in enumerate(data):
        nodes=[n for n in graph['nodes'] if n['event']==index];surface=event['surface']
        segments=segment_rows(labels[index],nodes)
        for node in nodes:
            np.testing.assert_array_equal(np.flatnonzero(segments==node['segment_id']),np.flatnonzero(labels[index]==node['group']))
        frames.append(dict(time=times[index],positions=surface['positions'],sources=surface['source_positions'],
            cutaway=cutaway_mask(surface['positions']),segments=segments,nodes=nodes))
    render(graph,frames,group_counts,output)
    equal_graph_path=Path('outputs/two_bubbles_event_history_resolution_n064_n128/n064_graph.json')
    equal_meta_path=Path('outputs/two_bubbles_iwr_compgpu9_n064/metadata.json')
    equal_graph=json.loads(equal_graph_path.read_text());equal_meta=json.loads(equal_meta_path.read_text())
    for key in ['source_positions','injection_radius','wind_terminal_velocity','ambient_density','ambient_pressure','gamma','domain']:
        np.testing.assert_allclose(meta[key],equal_meta[key],atol=1e-12,rtol=0)
    np.testing.assert_allclose(times,equal_meta['actual_times'],atol=1e-7,rtol=0)
    np.testing.assert_allclose(meta['discrete_energy_rates'],np.asarray(equal_meta['discrete_energy_rates'])*[1,.5],rtol=1e-12)
    comparison=compare_summary(graph,equal_graph,times,statistics)
    (output/'equal_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    render_comparison(equal_graph,graph,output)
    for p in [root/'equivalence.json',control_path,family_report_path,family_result_path,equal_graph_path,equal_meta_path]:hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    # Graph JSON readback checks every observation and adjacent-time edge.
    saved=json.loads((output/'graph.json').read_text());lookup={n['node_id']:n for n in saved['nodes']}
    assert len(lookup)==sum(map(len,groups))
    for e in saved['events']:
        assert all(lookup[p]['event']+1==lookup[c]['event'] for p in e['parents'] for c in e['children'])
    from PIL import Image
    with Image.open(output/'unequal_history.gif') as image:
        assert image.n_frames==len(times)
        for index in range(image.n_frames):image.seek(index);image.load()
    summary=dict(readback=readback,events=len(times),group_observations=len(statistics),segments=graph['segment_count'],
        event_counts=dict(Counter(e['status']+'_'+e['kind'] for e in graph['events'])),
        identical_intervals=sum(r['event_sets_identical'] for r in agreements),checks=checks,
        dense_family_checks=len(windows),accepted_family_configurations=sum(r['accepted'] for r in family_rows),
        frozen_grouping=dict(radius_cells=1.5,normal_angle=60,min_samples=20),
        frozen_tracking=control['parameters'],association_gates=[1.5/64,2.5/64],
        input_sha256=hashes,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__)]+[Path('experiments/astrolink')/(s+'.py') for s in
             ['shock_connectivity','event_tracking','event_history','tracklet_fragments']]},
        physical_ancestry_established=False,samples_or_normals_moved=False,decoded_animation_frames=len(times))
    summary['family_input_arithmetic']='float64, matching frozen equal-wind catalog preparation'
    (output/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(comparison=comparison,windows=family_rows,event_counts=summary['event_counts'],
                         identical_intervals=summary['identical_intervals']),indent=2))


def compare_summary(unequal,equal,times,statistics):
    def describe(graph):
        lookup={n['node_id']:n for n in graph['nodes']}
        combinations=[dict(parent_time=times[e['parent_event']],child_time=times[e['child_event']],
            parent_groups=[lookup[p]['group'] for p in e['parents']],child_groups=[lookup[c]['group'] for c in e['children']])
            for e in graph['events'] if e['status']=='shared' and e['kind']=='combination']
        return dict(group_observations=len(graph['nodes']),segments=graph['segment_count'],
                    final_groups=sum(n['event']==len(times)-1 for n in graph['nodes']),shared_combinations=combinations)
    final=[r for r in statistics if r['event']==len(times)-1]
    return dict(equal_64=describe(equal),unequal_64=describe(unequal),
        unequal_first_inward_group_time=next((r['time'] for r in statistics if r['median_nearest_source_alignment']<0),None),
        unequal_final_groups=[{k:r[k] for k in ['group','segment_id','surface_samples','median_mach',
            'median_nearest_source_alignment','upstream_inside_source']} for r in final],
        scope='Same frozen pipeline; different source strengths and total prescribed injection. No physical truth labels.')


def render(graph,frames,counts,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    from matplotlib.colors import to_hex
    palette=[to_hex(c) for c in plt.get_cmap('tab20').colors]
    colors={i:palette[i%len(palette)] for i in range(graph['segment_count'])}
    colors.update({0:'#2369ad',1:'#2e8b57',2:'#93a5b3',3:'#d16c28',4:'#7e43ad'})
    lookup={n['node_id']:n for n in graph['nodes']}
    fig=plt.figure(figsize=(17,8));grid=fig.add_gridspec(1,3,width_ratios=[1,1,1.15],left=.01,right=.985,bottom=.22,top=.83,wspace=.13)
    views=[fig.add_subplot(grid[i],projection='3d') for i in [0,1]];ax=fig.add_subplot(grid[2])
    for e in graph['events']:
        nodes=[lookup[n] for n in e['parents']+e['children']]
        mid=(lookup[e['parents'][0]]['time']+lookup[e['children'][0]]['time'])/2
        level=np.mean([n['segment_id'] for n in nodes])
        for n in nodes:ax.plot([n['time'],mid],[n['segment_id'],level],
            color=colors[n['segment_id']] if e['status']=='shared' else '#c65b30',ls='-' if e['status']=='shared' else '--',lw=1.5)
    for n in graph['nodes']:ax.scatter(n['time'],n['segment_id'],s=20,color=colors[n['segment_id']])
    for start in sorted({e['parent_event'] for e in graph['events'] if e['status']!='shared'}):
        ax.axvspan(frames[start]['time'],frames[start+1]['time'],color='#c65b30',alpha=.1)
    ax.set(xlim=(-.003,frames[-1]['time']+.003),ylim=(graph['segment_count']-.4,-.6),xlabel='Simulation time',title='Detected-group histories')
    ax.set_yticks(range(graph['segment_count']),[f'G{i:02d}' for i in range(graph['segment_count'])]);ax.grid(axis='x',alpha=.2)
    cursor=ax.axvline(0,color='black');current=ax.scatter([],[],s=90,facecolors='none',edgecolors='black',zorder=5)
    title=fig.suptitle('',fontsize=17,y=.96);note=fig.text(.025,.155,'',va='top',fontsize=10)
    fig.text(.5,.885,'Source strengths 1 : 0.5 | fixed colors, camera and bounds | gray: samples below retained-group cutoff',ha='center',fontsize=11)
    fig.text(.025,.035,'Stars: source positions; color identifies continuation segments, not Mach number or source ownership. Cutaway hides z > 0.5.\n'
             'Solid: shared events; dashed: distance-specific alternatives. Conditional family support is saved separately. No physical ancestry is asserted.',fontsize=9)
    def update(index):
        f=frames[index];p=f['positions'];segments=f['segments']
        for view,mask in zip(views,[np.ones(len(p),bool),f['cutaway']]):
            view.clear();gray=mask&(segments<0)
            if gray.any():view.scatter(*p[gray].T,s=3,c='#bcbcbc',alpha=.5,linewidths=0,depthshade=False)
            for n in f['nodes']:
                selected=mask&(segments==n['segment_id'])
                view.scatter(*p[selected].T,c=colors[n['segment_id']],s=4,linewidths=0,depthshade=False)
            view.scatter(*f['sources'].T,c='black',marker='*',s=50,depthshade=False)
            view.set(xlim=(.1,.9),ylim=(.1,.9),zlim=(.1,.9),xlabel='x',ylabel='y',zlabel='z',box_aspect=(1,1,1));view.view_init(25,-65)
        views[0].set_title('Full detected samples');views[1].set_title('Cutaway for inner structures')
        cursor.set_xdata([f['time']]*2);current.set_offsets(np.array([[f['time'],n['segment_id']] for n in f['nodes']]).reshape(-1,2))
        title.set_text(f'Unequal-wind group evolution | t = {f["time"]:.2f}')
        lines=[f'G{n["segment_id"]:02d}: group {n["group"]}, {n["surface_samples"]:,} samples' for n in f['nodes']]
        note.set_text(' | '.join(lines)+f'\nUnretained samples: {counts[index]["excluded_samples"]}' if lines else 'No detected groups')
        return []
    movie=FuncAnimation(fig,update,frames=len(frames),interval=500,blit=False)
    movie.save(output/'unequal_history.gif',writer=PillowWriter(fps=2),dpi=95)
    for index in [8,9,13,14,16]:update(index);fig.savefig(output/f'event_{index:04d}.png',dpi=130)
    plt.close(fig)


def render_comparison(equal,unequal,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,9),sharex=True)
    for ax,graph,title in zip(axes,[equal,unequal],['Equal winds (1 : 1)','Unequal winds (1 : 0.5)']):
        lookup={n['node_id']:n for n in graph['nodes']}
        for e in graph['events']:
            nodes=[lookup[n] for n in e['parents']+e['children']]
            x=(lookup[e['parents'][0]]['time']+lookup[e['children'][0]]['time'])/2
            y=np.mean([n['segment_id'] for n in nodes])
            color='#23818d' if e['kind']=='continuation' else '#8754ae'
            for n in nodes:ax.plot([n['time'],x],[n['segment_id'],y],color=color,ls='-' if e['status']=='shared' else '--')
        ax.scatter([n['time'] for n in graph['nodes']],[n['segment_id'] for n in graph['nodes']],s=25,c='#213d51',zorder=3)
        ax.set(title=title,ylabel='Local continuation segment',ylim=(graph['segment_count']-.3,-.7))
        ax.set_yticks(range(graph['segment_count']));ax.grid(axis='x',alpha=.2)
    axes[-1].set_xlabel('Simulation time (code units)')
    fig.suptitle('64³ equal versus unequal winds: same grouping and tracking settings')
    fig.text(.05,.02,'Green: shared continuation; purple: shared combination. Local segment numbers are independent between runs.\n'
        'A blank connection means ancestry was not established. The unequal case has 75% of the equal case total prescribed injection.\n'
        'Detected-group combination times are sampled events, not exact physical collision times. Inner parent subsets in the equal case have tiny score gaps.',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.95));fig.savefig(output/'equal_unequal_comparison.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('outputs/two_bubbles_unequal_iwr_compgpu9_n064'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();run(args.directory,args.output_dir)
