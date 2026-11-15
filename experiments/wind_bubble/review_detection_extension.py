"""Frozen grouping/tracking on the validated unequal-wind time extension."""
import argparse
from collections import Counter
import csv
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
from experiments.wind_bubble.compare_two_bubble_resolutions import load_checked,csv_write
from experiments.wind_bubble.apply_event_tracking import check_candidates,check_selection_dp,event_key
from experiments.wind_bubble.compare_event_history_resolutions import group_statistics
from experiments.wind_bubble.analyze_unequal_history import window_catalogs
from experiments.wind_bubble.animate_event_history import segment_rows,cutaway_mask
from experiments.wind_bubble.inspect_weak_source import probe


def run(root,prior,output):
    meta,data,hashes,readback=load_checked(root)
    report_path=root/'audit/verification.json';report=json.loads(report_path.read_text())
    assert report['passed'] and report['reference_replay']['events']==17
    assert report['reference_replay']['compared_captured_states']==5
    assert meta['num_cells']==128 and len(data)==21 and meta['source_strengths']==[1.,.5]
    def record(path):hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    record(report_path)
    for name,digest in report['state_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    for name,digest in report['source_sha256'].items():
        path=Path(name.split('/master_thesis/',1)[-1]);assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    old_report=json.loads((prior/'verification.json').read_text())
    for name,digest in old_report['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    record(prior/'verification.json');record(prior/'tracking_results.json.gz');record(prior/'graph.json')
    with gzip.open(prior/'tracking_results.json.gz','rt') as f:records=json.load(f)
    assert len(records)==32
    old_graph=json.loads((prior/'graph.json').read_text())
    output.mkdir(parents=True,exist_ok=False);labels=[];counts=[];checks=[];agreements=[]
    for index,event in enumerate(data):
        s=event['surface']
        p=connectivity_partition(s['positions'],s['normals'],float(s['grid_spacing']),radius_cells=1.5,max_angle_degrees=60,min_display_size=20)
        labels.append(p['labels']);np.savez_compressed(output/f'groups_{index:04d}.npz',labels=p['labels'],raw_labels=p['raw_labels'],component_sizes=p['component_sizes'])
        if index<=16:
            path=prior/f'groups_{index:04d}.npz';record(path)
            with np.load(path) as old:
                for key in ['labels','raw_labels','component_sizes']:np.testing.assert_array_equal(p[key],old[key])
        counts.append(dict(event=index,time=meta['actual_times'][index],retained_groups=len(np.unique(p['labels'][p['labels']>=0])),
            retained_samples=int((p['labels']>=0).sum()),excluded_samples=int((p['labels']<0).sum())))
    for index in range(16,20):
        a,b=[e['surface'] for e in data[index:index+2]]
        for gate in [1.5,2.5]:
            result=event_tracking(a['positions'],a['normals'],labels[index],b['positions'],b['normals'],labels[index+1],gate/64)
            for key,value in old_report['frozen_tracking'].items():assert result['parameters'][key]==value
            check=check_candidates(a,b,labels[index],labels[index+1],result);check_selection_dp(result)
            checks.append(dict(parent_event=index,gate_reference_cells=gate,independent_dp_passed=True,**check))
            records.append(dict(parent_event=index,child_event=index+1,gate_reference_cells=gate,result=result))
        print(f'Checked interval {index}→{index+1}',flush=True)
    for index in range(20):
        sets=[{event_key(e) for e in r['result']['events']} for r in records if r['parent_event']==index]
        assert len(sets)==2
        agreements.append(dict(parent_event=index,child_event=index+1,event_sets_identical=sets[0]==sets[1],
            common=len(sets[0]&sets[1]),narrow_only=len(sets[0]-sets[1]),wide_only=len(sets[1]-sets[0])))
    graph=assemble_history([list(map(int,np.unique(l[l>=0]))) for l in labels],meta['actual_times'],records)
    lookup={n['node_id']:n for n in graph['nodes']}
    for node in old_graph['nodes']:assert lookup[node['node_id']]['segment_id']==node['segment_id']
    stats=[];frames=[];weak_rows=[]
    for index,event in enumerate(data):
        s=event['surface'];nodes=[n for n in graph['nodes'] if n['event']==index]
        for node,row in zip(nodes,group_statistics(s,labels[index],nodes)):
            mask=labels[index]==node['group']
            row.update(upstream_inside_source=int(s['upstream_inside_any_source'][mask].sum()),downstream_inside_source=int(s['downstream_inside_any_source'][mask].sum()))
            node.update(row);stats.append(row)
        weak=probe(s)[1];local=labels[index][weak]
        weak_rows.append(dict(event=index,time=meta['actual_times'][index],probe_samples=int(weak.sum()),
            retained_probe_samples=int((local>=0).sum()),retained_groups=len(np.unique(local[local>=0])),
            upstream_overlap=int(s['upstream_inside_any_source'][weak].sum()),downstream_overlap=int(s['downstream_inside_any_source'][weak].sum())))
        segments=segment_rows(labels[index],nodes)
        frames.append(dict(time=meta['actual_times'][index],positions=s['positions'],sources=s['source_positions'],
            cutaway=cutaway_mask(s['positions']),segments=segments,nodes=nodes))
    # New triples only; previous family records remain unchanged in the original report.
    windows=[]
    for start in range(15,19):
        catalogs,pairs=window_catalogs(data,labels,graph,start)
        if not pairs:continue
        for gate in [1.5,2.5]:
            for predict in [False,True]:
                result=attach_fragments(*catalogs,pairs,meta['actual_times'][start:start+3],gate/64,predict=predict)
                check_dense(*catalogs,result,meta['actual_times'][start:start+3],predict)
                windows.append(dict(first_event=start,gate_reference_cells=gate,predict=predict,result=result))
    with gzip.open(output/'tracking_results.json.gz','wt') as f:json.dump(records,f)
    (output/'graph.json').write_text(json.dumps(graph,indent=2)+'\n')
    (output/'new_family_windows.json').write_text(json.dumps(windows,indent=2)+'\n')
    csv_write(output/'groups.csv',counts);csv_write(output/'group_statistics.csv',stats)
    csv_write(output/'weak_source_history.csv',weak_rows);csv_write(output/'distance_agreement.csv',agreements)
    render_history(graph,output);render_weak(data,labels,graph,weak_rows,output)
    from PIL import Image
    with Image.open(output/'weak_source_evolution.gif') as image:
        assert image.n_frames==21
        for i in range(21):image.seek(i);image.load()
    for name,digest in hashes.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    result=dict(passed=True,readback=readback,unchanged_prefix_groupings=17,unchanged_prefix_segment_ids=True,
        reused_validated_prefix_optimizations=32,new_independent_checks=checks,new_dense_family_checks=len(windows),
        event_counts=dict(Counter(e['status']+'_'+e['kind'] for e in graph['events'])),
        group_observations=len(graph['nodes']),segments=graph['segment_count'],
        new_intervals_agree=sum(a['event_sets_identical'] for a in agreements[16:]),decoded_frames=21,
        weak_source_suffix=weak_rows[16:],input_sha256=hashes,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__)]+[Path(n) for n in old_report['source_sha256']]},
        physics_or_thresholds_changed=False,physical_ancestry_established=False)
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['event_counts','new_intervals_agree','weak_source_suffix','new_dense_family_checks']},indent=2))


def segment_color(segment):
    from matplotlib import colormaps
    overrides={14:'#007299',15:'#b34e00',17:'#743fa3'}
    return overrides.get(segment,colormaps['tab20'].colors[segment%20]) if segment>=0 else '#bbbbbb'


def render_history(graph,output):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(13,8));lookup={n['node_id']:n for n in graph['nodes']}
    for e in graph['events']:
        nodes=[lookup[n] for n in e['parents']+e['children']]
        x=np.mean([lookup[e['parents'][0]]['time'],lookup[e['children'][0]]['time']]);y=np.mean([n['segment_id'] for n in nodes])
        for n in nodes:ax.plot([n['time'],x],[n['segment_id'],y],color='#167f88' if e['kind']=='continuation' else '#a14f94',
            ls='-' if e['status']=='shared' else '--',lw=1.3)
    for n in graph['nodes']:ax.scatter(n['time'],n['segment_id'],c=[segment_color(n['segment_id'])],s=25,zorder=3)
    ax.axvline(.16,c='#777777',ls=':');ax.set(xlabel='Simulation time',ylabel='Local continuation segment',
        title='Unequal-wind group histories through t=0.20',ylim=(graph['segment_count']-.3,-.7))
    ax.set_yticks(range(graph['segment_count']),[f'G{i:02d}' for i in range(graph['segment_count'])]);ax.grid(axis='x',alpha=.2)
    fig.text(.07,.025,'Solid: shared events. Dashed: distance-specific alternatives. Dotted vertical line: previous end time.\n'
        'Segment IDs are geometric histories, not proven physical ancestry. The final weak-source group has no shared predecessor.',fontsize=10)
    fig.tight_layout(rect=(0,.1,1,1));fig.savefig(output/'tracking_history.png',dpi=140);plt.close(fig)


def render_weak(data,labels,graph,rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(15,8));grid=fig.add_gridspec(2,3,height_ratios=[1,.7],left=.06,right=.97,bottom=.15,top=.86,hspace=.25)
    for column,index in enumerate([16,18,20]):
        ax=fig.add_subplot(grid[0,column],projection='3d');s=data[index]['surface'];weak=probe(s)[1]
        nodes=[n for n in graph['nodes'] if n['event']==index]
        for g in np.unique(labels[index][weak]):
            points=s['positions'][weak&(labels[index]==g)]
            segment=next((n['segment_id'] for n in nodes if n['group']==g),-1)
            ax.scatter(*points.T,s=9,color=segment_color(segment),depthshade=False,linewidths=0)
        ax.scatter(.65,.5,.5,c='black',marker='*',s=55)
        ax.set(xlim=(.55,.75),ylim=(.4,.6),zlim=(.4,.6),xlabel='x',ylabel='y',zlabel='z',box_aspect=(1,1,1))
        ax.view_init(25,-65)
        row=rows[index];number=row['retained_groups'];ax.set_title(f't={row["time"]:.2f}: {row["probe_samples"]} samples\n{number} retained '+('group' if number==1 else 'groups'),fontsize=11)
    ax=fig.add_subplot(grid[1,:])
    for key,label,style in [('probe_samples','All near-source inward samples','o-'),('retained_probe_samples','Samples in retained groups','s--')]:
        ax.plot([r['time'] for r in rows],[r[key] for r in rows],style,label=label)
    ax.axvline(.16,c='#777777',ls=':');ax.set(xlabel='Simulation time',ylabel='Samples',xlim=(.08,.205));ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Weaker-source inner detections grow during the unchanged time extension',fontsize=16)
    fig.text(.06,.035,'Original points; same camera and bounds. Only the geometric near-source inward probe is shown; outer surfaces are omitted.\n'
        'Colors identify continuation segments, not established ancestry. Gray samples fall below the 20-sample group cutoff.\n'
        'Probe: nearest weaker source, radius <= 2 injection radii, radial normal alignment < -0.5. No thresholds were changed.',fontsize=10)
    fig.savefig(output/'weak_source_extension.png',dpi=140);plt.close(fig)
    from matplotlib.animation import FuncAnimation,PillowWriter
    fig=plt.figure(figsize=(11,6));view=fig.add_subplot(121,projection='3d');history=fig.add_subplot(122)
    history.plot([r['time'] for r in rows],[r['probe_samples'] for r in rows],label='All probe samples')
    history.plot([r['time'] for r in rows],[r['retained_probe_samples'] for r in rows],ls='--',label='Retained samples')
    history.set(xlim=(0,.20),ylim=(0,520),xlabel='Simulation time',ylabel='Samples');history.legend(fontsize=9);history.grid(alpha=.2)
    cursor=history.axvline(0,color='black');title=fig.suptitle('')
    def update(index):
        view.clear();s=data[index]['surface'];weak=probe(s)[1]
        for g in np.unique(labels[index][weak]):
            points=s['positions'][weak&(labels[index]==g)]
            segment=next((n['segment_id'] for n in graph['nodes'] if n['event']==index and n['group']==g),-1)
            view.scatter(*points.T,s=10,color=segment_color(segment),depthshade=False,linewidths=0)
        view.scatter(.65,.5,.5,c='black',marker='*',s=55)
        view.set(xlim=(.55,.75),ylim=(.4,.6),zlim=(.4,.6),xlabel='x',ylabel='y',zlabel='z',box_aspect=(1,1,1));view.view_init(25,-65)
        cursor.set_xdata([rows[index]['time']]*2);title.set_text(f'Weaker-source inner detections | t={rows[index]["time"]:.2f} | {rows[index]["probe_samples"]} samples')
    fig.text(.05,.035,'Original points. Fixed camera/bounds. Gray: unretained. Colors: continuation segments, not physical ancestry.\nOnly the geometric near-source inward probe is shown. Right: full history; black line marks the current event.',fontsize=9)
    fig.subplots_adjust(bottom=.2,top=.85,wspace=.3)
    movie=FuncAnimation(fig,update,frames=len(rows),interval=500,blit=False)
    movie.save(output/'weak_source_evolution.gif',writer=PillowWriter(fps=2),dpi=110);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/two_bubbles_detection_extension_iwr_n128'))
    parser.add_argument('--prior',type=Path,default=Path('outputs/two_bubbles_unequal_history_v1_n128'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/two_bubbles_detection_extension_review_n128'))
    args=parser.parse_args();run(args.root,args.prior,args.output_dir)
