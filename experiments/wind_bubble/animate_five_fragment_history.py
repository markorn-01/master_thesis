"""Original surface samples beside baseline and experimental ancestry graphs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.wind_bubble.inspect_weak_source import probe
from experiments.wind_bubble.review_detection_extension import segment_color


def run(root,history,experiment,output):
    hashes={}
    def record(path):hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for directory in [history,experiment]:
        report=directory/'verification.json';record(report)
        verified=json.loads(report.read_text())
        for section in ['input_sha256','source_sha256']:
            for name,digest in verified[section].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Changed validated input: {name}')
    graph_paths=[history/'graph.json',experiment/'experimental_graph.json']
    for path in graph_paths:record(path)
    baseline,expanded=[json.loads(path.read_text()) for path in graph_paths]
    old={n['node_id']:n for n in baseline['nodes']};new={n['node_id']:n for n in expanded['nodes']}
    assert set(old)==set(new)
    for name in old:assert old[name]['segment_id']==new[name]['segment_id']
    # All earlier event choices remain exactly those of the baseline.
    assert [e for e in baseline['events'] if e['parent_event']<19]==[e for e in expanded['events'] if e['parent_event']<19]
    frames=[];selected_nodes=set();audit=[]
    for index in range(16,21):
        surface=root/'online'/f'surfaces_{index:04d}.npz';partition=history/f'groups_{index:04d}.npz'
        record(surface);record(partition)
        with np.load(surface) as s,np.load(partition) as grouping:
            mask=probe(s)[1];labels=grouping['labels'][mask];positions=s['positions'][mask].copy()
            segments=np.full(len(labels),-1,int);nodes=[]
            for group in np.unique(labels[labels>=0]):
                node=next(n for n in expanded['nodes'] if n['event']==index and n['group']==int(group))
                selected_nodes.add(node['node_id']);segments[labels==group]=node['segment_id'];nodes.append(node)
                np.testing.assert_array_equal(positions[segments==node['segment_id']],s['positions'][mask&(grouping['labels']==group)])
            assert ((positions>=np.array([.55,.4,.4]))&(positions<=np.array([.75,.6,.6]))).all()
            frame=dict(event=index,time=float(s['time']),positions=positions,segments=segments,nodes=nodes,
                unretained=int((labels<0).sum()),upstream_overlap=int(s['upstream_inside_any_source'][mask].sum()))
            frames.append(frame)
            audit.append(dict(event=index,time=frame['time'],samples=len(positions),unretained=frame['unretained'],
                upstream_overlap=frame['upstream_overlap'],coordinate_sha256=hashlib.sha256(positions.tobytes()).hexdigest()))
    final=next(e for e in expanded['events'] if e['parent_event']==19 and len(e['parents'])==5 and e['status']=='shared')
    assert set(final['parents']+final['children']).issubset(selected_nodes)
    output.mkdir(parents=True,exist_ok=False)
    render(frames,baseline,expanded,selected_nodes,output)
    from PIL import Image
    with Image.open(output/'five_fragment_history.gif') as image:
        assert image.n_frames==len(frames)
        for i in range(image.n_frames):image.seek(i);image.load()
    for name,digest in hashes.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    result=dict(passed=True,unique_events=5,decoded_frames=5,fps=1,frames=audit,
        full_parent_relation=final,original_positions_and_segment_ids_preserved=True,
        previous_events_unchanged=True,original_history_modified=False,
        graph_uses_full_saved_window=True,points_smoothed=False,physical_ancestry_established=False,
        input_sha256=hashes,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path('experiments/wind_bubble/inspect_weak_source.py'),Path('experiments/wind_bubble/review_detection_extension.py')]})
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(events=5,decoded_frames=5,fps=1,parents=len(final['parents']),original_history_modified=False),indent=2))


def render(frames,baseline,expanded,node_ids,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    from matplotlib.lines import Line2D
    segments=sorted({n['segment_id'] for n in expanded['nodes'] if n['node_id'] in node_ids})
    lanes={segment:i for i,segment in enumerate(segments)}
    fig=plt.figure(figsize=(17,8));grid=fig.add_gridspec(1,3,width_ratios=[1.05,1,1],left=.02,right=.985,bottom=.28,top=.8,wspace=.22)
    view=fig.add_subplot(grid[0],projection='3d');axes=[fig.add_subplot(grid[i]) for i in [1,2]]
    cursors=[];current=[]
    for ax,graph,title in zip(axes,[baseline,expanded],['Original: maximum three parents','Experimental: maximum five parents']):
        lookup={n['node_id']:n for n in graph['nodes']}
        for event in graph['events']:
            names=event['parents']+event['children']
            if not set(names).issubset(node_ids):continue
            nodes=[lookup[name] for name in names];x=np.mean([lookup[event['parents'][0]]['time'],lookup[event['children'][0]]['time']])
            y=np.mean([lanes[n['segment_id']] for n in nodes])
            for n in nodes:
                color=segment_color(n['segment_id']) if event['kind']=='continuation' else '#16818b' if event['status']=='shared' else '#b46d2b'
                ax.plot([n['time'],x],[lanes[n['segment_id']],y],color=color,ls='-' if event['status']=='shared' else '--',lw=1.5)
        for name in node_ids:
            n=lookup[name];ax.scatter(n['time'],lanes[n['segment_id']],s=28,color=segment_color(n['segment_id']),zorder=3)
        ax.set(xlim=(.158,.202),ylim=(len(lanes)-.5,-.5),xlabel='Simulation time',title=title)
        ax.set_xticks([.16,.17,.18,.19,.20]);ax.set_yticks(range(len(segments)),[f'G{i:02d}' for i in segments]);ax.grid(axis='x',alpha=.2)
        cursors.append(ax.axvline(.16,color='black',lw=1));current.append(ax.scatter([],[],s=90,facecolors='none',edgecolors='black',zorder=5))
    title=fig.suptitle('',fontsize=17,y=.97);note=fig.text(.03,.18,'',va='top',fontsize=11)
    handles=[Line2D([],[],ls='',marker='o',color=segment_color(s),label=f'G{s:02d}') for s in segments]
    handles.append(Line2D([],[],ls='',marker='o',color='#bbbbbb',label='Unretained'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.9),ncol=7,frameon=False)
    fig.text(.03,.055,'Same detected points and fixed view; only the weaker-source inward probe is shown. Black star: weaker wind source.\n'
        'Graphs show the full saved window; black cursors/rings mark the current state. Solid: shared links; dashed: distance-specific subset.\n'
        'Colors identify continuation segments, not gas particles or proven physical ancestry. No smoothing or original-history changes.',fontsize=10)
    def update(index):
        f=frames[index];view.clear()
        for segment in np.unique(f['segments']):
            p=f['positions'][f['segments']==segment]
            view.scatter(*p.T,s=13,color=segment_color(int(segment)),depthshade=False,linewidths=0)
        view.scatter(.65,.5,.5,c='black',marker='*',s=60)
        view.set(xlim=(.55,.75),ylim=(.4,.6),zlim=(.4,.6),xlabel='x',ylabel='y',zlabel='z',box_aspect=(1,1,1),title='Original detected surface samples')
        view.set_xticks([.55,.65,.75]);view.set_yticks([.4,.5,.6]);view.set_zticks([.4,.5,.6]);view.view_init(25,-65)
        for cursor,ring in zip(cursors,current):
            cursor.set_xdata([f['time']]*2)
            ring.set_offsets(np.asarray([[f['time'],lanes[n['segment_id']]] for n in f['nodes']]).reshape(-1,2))
        title.set_text(f'Five-fragment association beside the actual surfaces | t={f["time"]:.2f}')
        note.set_text(f'{len(f["positions"])} selected samples; {f["unretained"]} below group cutoff; {f["upstream_overlap"]} upstream source overlaps.\n'
            'At 0.19→0.20: the original model has no shared inner link; the experimental model accepts all five parents at both distances.')
    movie=FuncAnimation(fig,update,frames=len(frames),interval=1000,blit=False)
    movie.save(output/'five_fragment_history.gif',writer=PillowWriter(fps=1),dpi=100)
    for index in [2,3,4]:update(index);fig.savefig(output/f'event_{frames[index]["event"]:04d}.png',dpi=140)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/two_bubbles_detection_extension_iwr_n128'))
    parser.add_argument('--history',type=Path,default=Path('outputs/two_bubbles_detection_extension_review_n128'))
    parser.add_argument('--experiment',type=Path,default=Path('outputs/five_fragment_arity_review_n128'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/five_fragment_surface_history_n128'))
    args=parser.parse_args();run(args.root,args.history,args.experiment,args.output_dir)
