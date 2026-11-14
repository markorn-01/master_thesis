"""Focused visual review of the validated 128³ seven-fragment relationship.

Reads saved decisions only. It does not regroup, relink, move, or smooth points.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def run(directory, root, baseline, output):
    report=json.loads((directory/'verification.json').read_text())
    hashes={}
    def record(path):
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    # Validate the completed analysis before making a different display of it.
    for section in ['input_sha256','source_sha256']:
        for name,digest in report[section].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Validated input/source changed: {name}')
    for path in [directory/'verification.json',directory/'fragment_windows.json',
                 directory/'graph.json',directory/'tracking_results.json.gz',baseline/'fragment_windows.json']:
        record(path)
    windows=json.loads((directory/'fragment_windows.json').read_text())
    records=[r for r in windows if r['first_event']==10]
    assert len(records)==4
    selected=next(r for r in records if r['method']=='delayed_family' and r['gate_reference_cells']==1.5)
    family=next(f for f in selected['result']['families'] if f['accepted'])
    parents=family['parent_groups'];child=family['child_group'];later=family['later_group']
    assert len(parents)==7
    for r in records:
        f=next(f for f in r['result']['families'] if f['child_group']==child)
        assert f['accepted'] and f['parent_groups']==parents and f['later_group']==later
    graph=json.loads((directory/'graph.json').read_text());lookup={n['node_id']:n for n in graph['nodes']}
    assert any(e['status']=='shared' and e['kind']=='continuation' and e['parent_event']==11
               and lookup[e['parents'][0]]['group']==child and lookup[e['children'][0]]['group']==later
               for e in graph['events'])
    with gzip.open(directory/'tracking_results.json.gz','rt') as stream:tracking=json.load(stream)
    wide=next(r['result'] for r in tracking if r['parent_event']==10 and r['gate_reference_cells']==2.5)
    narrow=next(r['result'] for r in tracking if r['parent_event']==10 and r['gate_reference_cells']==1.5)
    event=next(e for e in wide['events'] if e['children']==[child])
    assert event['kind']=='combination' and set(event['parents'])<set(parents)
    assert not any(e['children']==[child] for e in narrow['events'])
    gap=wide['selection']['exclusion_gaps'][str(event['candidate_index'])]
    frames=[]
    for index,groups in [(10,parents),(11,[child]),(12,[later])]:
        surface=root/'online'/f'surfaces_{index:04d}.npz';partition=directory/f'groups_{index:04d}.npz'
        record(surface);record(partition)
        with np.load(surface) as s,np.load(partition) as p:
            mask=np.isin(p['labels'],groups)
            assert set(np.unique(p['labels'][mask]))==set(groups)
            positions=s['positions'][mask].copy();labels=p['labels'][mask].copy()
            counts={int(g):int((labels==g).sum()) for g in groups}
            frames.append(dict(event=index,time=float(s['time']),positions=positions,labels=labels,counts=counts,
                source=s['source_positions'][0].copy(),
                upstream_overlap=int(s['upstream_inside_any_source'][mask].sum()),
                downstream_overlap=int(s['downstream_inside_any_source'][mask].sum()),
                positions_sha256=hashlib.sha256(positions.tobytes()).hexdigest()))
    old=json.loads((baseline/'fragment_windows.json').read_text())
    previous=[r for r in old if r['first_event']==13]
    assert len(previous)==4
    coverage=[]
    for gate in [1.5,2.5]:
        row=[]
        for collection in [previous,records]:
            values=[r['result']['families'][0]['child_union_coverage'] for r in collection if r['gate_reference_cells']==gate]
            assert len(values)==2 and values[0]==values[1]  # static == predicted
            row.append(values[0])
        coverage.append(row)
    output.mkdir(parents=True,exist_ok=False)
    render(frames,parents,child,later,event['parents'],gap,np.asarray(coverage),output)
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    result=dict(input_sha256=hashes,source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        frames=[{k:v for k,v in f.items() if k not in ['positions','labels','source']} for f in frames],
        parent_groups=parents,child_group=child,later_group=later,
        wide_gate_adjacent_subset=event['parents'],wide_gate_exclusion_gap=gap,
        narrow_gate_adjacent_inner_event=False,
        coverage_64_128_by_narrow_wide_gate=coverage,
        all_four_family_variants_agree_on_membership=True,static_and_predicted_coverage_identical=True,
        samples_subsampled=False,positions_moved=False,normals_modified=False,
        tracking_graph_modified=False,physical_ancestry_established=False,
        scope='Focused review of the previously validated window; different resolutions use different saved times.')
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['parent_groups','child_group','wide_gate_adjacent_subset','wide_gate_exclusion_gap','coverage_64_128_by_narrow_wide_gate']},indent=2))


def render(frames,parents,child,later,subset,gap,coverage,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors=dict(zip(parents,['#0072b2','#e69f00','#009e73','#cc79a7','#d55e00','#56b4e9','#555555']))
    purple='#7442a3'
    fig=plt.figure(figsize=(17,12))
    grid=fig.add_gridspec(2,3,height_ratios=[1.2,1],left=.05,right=.98,bottom=.23,top=.83,wspace=.28,hspace=.45)
    allpoints=np.concatenate([f['positions'] for f in frames]);center=frames[0]['source']
    half=max(np.abs(allpoints-center).max()*1.12,.085)
    for i,f in enumerate(frames):
        ax=fig.add_subplot(grid[0,i],projection='3d')
        for group in f['counts']:
            p=f['positions'][f['labels']==group]
            ax.scatter(*p.T,s=8,c=colors[group] if i==0 else purple,depthshade=False,linewidths=0)
        ax.scatter(*center,c='black',s=60,marker='*',depthshade=False)
        ax.set(xlim=(center[0]-half,center[0]+half),ylim=(center[1]-half,center[1]+half),
               zlim=(center[2]-half,center[2]+half),xlabel='x',ylabel='y',zlabel='z',box_aspect=(1,1,1))
        ax.view_init(25,-65)
        description='Seven earlier fragments' if i==0 else ('Later inner group' if i==1 else 'Same continuation segment')
        ax.set_title(f'{description} | t={f["time"]:.2f}\n{len(f["positions"]):,} samples; upstream source overlap: {f["upstream_overlap"]}',fontsize=11)
    first=fig.add_subplot(grid[1,0]);second=fig.add_subplot(grid[1,1]);bars=fig.add_subplot(grid[1,2])
    levels={g:float(len(parents)-1-i) for i,g in enumerate(parents)};middle=(len(parents)-1)/2
    for ax in [first,second]:
        for g,y in levels.items():
            ax.scatter(.10,y,c=colors[g],s=35,zorder=3);ax.text(.0992,y,f'g{g}',ha='right',va='center',fontsize=9)
        ax.scatter([.11,.12],[middle,middle],c=purple,s=50,zorder=3)
        ax.plot([.11,.12],[middle,middle],c=purple,lw=2)
        ax.text(.11,middle+.55,f'g{child}',ha='center',fontsize=9)
        ax.text(.12,middle+.55,f'g{later}',ha='center',fontsize=9)
        ax.set(xlim=(.098,.122),ylim=(-1,len(parents)),xticks=[.10,.11,.12],xlabel='Simulation time')
        ax.set_yticks([]);ax.spines[['top','right','left']].set_visible(False)
    for g in subset:first.plot([.10,.11],[levels[g],middle],c='#b35806',ls='--',lw=1.5)
    first.set_title('A. Adjacent-frame tracker',fontsize=12)
    first.text(.5,-.22,'Narrow gate: no inner event\nWide gate: three-parent subset only\nExclusion gap = '+f'{gap:.2e}',ha='center',va='top',transform=first.transAxes,fontsize=10)
    for g,y in levels.items():second.plot([.10,.11],[y,middle],c='#16818b',ls=':',lw=1.6)
    second.set_title('B. Separate conditional family support',fontsize=12)
    second.text(.5,-.22,'All seven fragments support the later group\nDotted links do not replace the tracking graph\nSolid 0.11–0.12 link: shared continuation',ha='center',va='top',transform=second.transAxes,fontsize=10)
    x=np.arange(2)
    for offset,values,color,name in [(-.18,coverage[:,0]*100,'#8c969e','64³: t=0.13 → 0.14'),(.18,coverage[:,1]*100,purple,'128³: t=0.10 → 0.11')]:
        rectangles=bars.bar(x+offset,values,.34,color=color,label=name)
        bars.bar_label(rectangles,labels=[f'{v:.1f}%' for v in values],padding=3,fontsize=10)
    bars.axhline(50,c='#444444',ls='--',lw=1)
    bars.set(xticks=x,xticklabels=['Narrow gate','Wide gate'],ylim=(0,117),ylabel='Later group covered (%)',title='C. Coverage under the unchanged rule')
    bars.text(.98,52,'50% acceptance threshold',ha='right',fontsize=9)
    bars.legend(loc='upper left',bbox_to_anchor=(0,-.16),frameon=False,fontsize=9)
    fig.suptitle('Unequal winds, 128³: reviewing the inner-fragment relationship',fontsize=18,y=.975)
    fig.text(.5,.93,'Top: original detected points, fixed camera and bounds, outer surface omitted to expose the inner structure.',ha='center',fontsize=11)
    handles=[Line2D([],[],marker='o',ls='',color=colors[g],label=f'g{g}: {frames[0]["counts"][g]}') for g in parents]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.90),ncol=7,frameon=False,fontsize=9)
    fig.text(.05,.018,'Fragment colors distinguish earlier geometric groups; purple identifies the later continuation segment, not proven ancestry. Stars: stronger source.\n'
             'Static and predicted family tests give identical membership and coverage. No point is moved or smoothed; no graph edge is added.\n'
             'The resolution comparison uses different detected windows and grouping lengths in physical units; it does not establish tracker convergence.',fontsize=10)
    fig.savefig(output/'inner_fragment_review.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('outputs/two_bubbles_unequal_history_v1_n128'))
    parser.add_argument('--root',type=Path,default=Path('outputs/two_bubbles_unequal_iwr_compgpu1_n128'))
    parser.add_argument('--baseline',type=Path,default=Path('outputs/two_bubbles_unequal_history_v1_n064'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/two_bubbles_unequal_fragment_review_n128'))
    args=parser.parse_args();run(args.directory,args.root,args.baseline,args.output_dir)
