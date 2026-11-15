"""Controlled and saved-data comparison of three- versus five-partner events."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from experiments.astrolink.expanded_event_tracking import expanded_tracking
from experiments.astrolink.controlled_tracking_cases import tracking_cases,evaluate_links
from experiments.astrolink.event_history import assemble_history
from experiments.wind_bubble.apply_event_tracking import check_candidates,check_selection_dp,event_key
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write


def new_cases():
    patches=[np.array([[k+x*.1,y*.1,0.] for x in range(2) for y in range(3)]) for k in range(5)]
    p=np.concatenate(patches);n=np.tile([0.,0,1],(len(p),1));labels=np.repeat(np.arange(5),6)
    def case(name,a,na,la,b,nb,lb,truth):
        return dict(name=name,positions_a=a,normals_a=na,labels_a=la,positions_b=b,normals_b=nb,labels_b=lb,expected_links=truth)
    forward=case('five_disjoint_fragments',p,n,labels,p+[0,0,.02],n,np.full(len(p),10),[(i,10) for i in range(5)])
    result=[forward,case('five_way_separation',forward['positions_b'],n,forward['labels_b'],p,n,labels,[(10,i) for i in range(5)])]
    coincident=np.tile(patches[0],(5,1))
    result.append(case('coincident_parents_ambiguous',coincident,n,labels,patches[0]+[0,0,.02],n[:6],np.full(6,10),[]))
    opposite=n.copy();opposite[labels==4]*=-1
    result.append(case('opposite_normal_fragment',p,opposite,labels,p+[0,0,.02],n,np.full(len(p),10),[(i,10) for i in range(4)]))
    result.append(case('five_fragments_beyond_gate',p,n,labels,p+[0,0,.2],n,np.full(len(p),10),[(i,10) for i in range(5)]))
    take=np.arange(len(p))<25
    result.append(case('small_unique_fragment',p[take],n[take],labels[take],(p+[0,0,.02])[take],n[take],np.full(take.sum(),10),[(i,10) for i in range(5)]))
    return result


def dense_five_audit(a,b,la,lb,result):
    audit=next(r for r in result['higher_arity_audit'] if r['parents']==[2,3,4,5,6] and r['children']==[2])
    parent_rows=np.flatnonzero(np.isin(la,audit['parents']));child_rows=np.flatnonzero(lb==2)
    pa=a['positions'][parent_rows].astype(float);pb=b['positions'][child_rows].astype(float)
    na=a['normals'][parent_rows].astype(float);nb=b['normals'][child_rows].astype(float)
    na/=np.linalg.norm(na,axis=1)[:,None];nb/=np.linalg.norm(nb,axis=1)[:,None]
    distance=cdist(pa,pb);distance[(na@nb.T)<np.cos(np.deg2rad(result['parameters']['normal_angle']))]=np.inf
    good=distance<=result['parameters']['gate'];footprints=[];coverages=[];terms=[]
    for group in audit['parents']:
        hit=good[la[parent_rows]==group];dist=distance[la[parent_rows]==group]
        cover=hit.any(axis=1);fraction=float(cover.mean());footprints.append(hit.any(axis=0))
        coverages.append(fraction);terms.append(fraction*dist.min(axis=1)[cover].mean()/result['parameters']['gate'])
    child=good.any(axis=0);fraction=float(child.mean());coverages.append(fraction)
    terms.append(fraction*distance.min(axis=0)[child].mean()/result['parameters']['gate'])
    unique=[float((f&~np.logical_or.reduce([other for j,other in enumerate(footprints) if j!=i])).mean()) for i,f in enumerate(footprints)]
    np.testing.assert_allclose(coverages,[r['coverage'] for r in audit['node_coverages']],rtol=0,atol=0)
    np.testing.assert_allclose(unique,audit['exclusive_central_fractions'],rtol=0,atol=0)
    np.testing.assert_allclose(sum(coverages)-result['parameters']['distance_weight']*sum(terms),audit['score'],rtol=1e-12)
    return audit


def run(root,history,output):
    verified=json.loads((history/'verification.json').read_text());hashes={}
    for section in ['input_sha256','source_sha256']:
        for name,digest in verified[section].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    def record(path):hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    record(history/'verification.json');record(history/'tracking_results.json.gz')
    with gzip.open(history/'tracking_results.json.gz','rt') as f:saved=json.load(f)
    output.mkdir(parents=True,exist_ok=False);controlled=[];details=[]
    legacy=tracking_cases()
    for case in legacy+new_cases():
        for gate in ([.2,.35] if case['name'] in {c['name'] for c in legacy} else [.04,.07]):
            args=[case[k] for k in ['positions_a','normals_a','labels_a','positions_b','normals_b','labels_b']]
            for cap in [3,5]:
                result=expanded_tracking(*args,gate,max_partners=cap)
                check_candidates(dict(positions=args[0],normals=args[1]),dict(positions=args[3],normals=args[4]),args[2],args[5],result)
                check_selection_dp(result)
                evaluation=evaluate_links(result,case['expected_links'])
                if case['name'] in {c['name'] for c in legacy} and cap==5:
                    baseline=expanded_tracking(*args,gate,max_partners=3)
                    assert result['candidates']==baseline['candidates'] and result['selection']==baseline['selection']
                controlled.append(dict(case=case['name'],gate=gate,max_partners=cap,exact=evaluation['exact_correspondence'],
                    false_links=len(evaluation['false_links']),missed_links=len(evaluation['missed_links'])))
                details.append(dict(case=case['name'],gate=gate,max_partners=cap,evaluation=evaluation,result=result))
    actual=[];summary=[]
    paths=[root/'online/surfaces_0019.npz',root/'online/surfaces_0020.npz',history/'groups_0019.npz',history/'groups_0020.npz']
    for path in paths:record(path)
    with np.load(paths[0]) as a,np.load(paths[1]) as b,np.load(paths[2]) as la,np.load(paths[3]) as lb:
        for gate in [1.5,2.5]:
            for cap in [3,5]:
                result=expanded_tracking(a['positions'],a['normals'],la['labels'],b['positions'],b['normals'],lb['labels'],gate/64,max_partners=cap)
                check=check_candidates(a,b,la['labels'],lb['labels'],result);check_selection_dp(result)
                if cap==3:
                    old=next(r['result'] for r in saved if r['parent_event']==19 and r['gate_reference_cells']==gate)
                    assert result['candidates']==old['candidates'] and result['selection']==old['selection']
                audit=dense_five_audit(a,b,la['labels'],lb['labels'],result) if cap==5 else None
                actual.append(dict(gate_reference_cells=gate,max_partners=cap,result=result,five_parent_dense_audit=audit,independent_check=check))
                weak=[e for e in result['events'] if e['children']==[2]]
                summary.append(dict(gate_reference_cells=gate,max_partners=cap,candidates=len(result['candidates']),
                    accepted_weak_parents=';'.join(','.join(map(str,e['parents'])) for e in weak),
                    five_parent_candidate_admitted=audit['admitted'] if audit else False))
    (output/'controlled_results.json').write_text(json.dumps(details,indent=2)+'\n')
    (output/'real_results.json').write_text(json.dumps(actual,indent=2)+'\n')
    graph_path=history/'graph.json';record(graph_path)
    old_graph=json.loads(graph_path.read_text());records=copy.deepcopy(saved)
    for row in records:
        if row['parent_event']==19:
            row['result']=next(r['result'] for r in actual if r['max_partners']==5 and r['gate_reference_cells']==row['gate_reference_cells'])
    times=[next(n['time'] for n in old_graph['nodes'] if n['event']==i) if i else 0. for i in range(21)]
    groups=[[n['group'] for n in old_graph['nodes'] if n['event']==i] for i in range(21)]
    graph=assemble_history(groups,times,records)
    assert all(n['segment_id']==next(o['segment_id'] for o in old_graph['nodes'] if o['node_id']==n['node_id']) for n in graph['nodes'])
    graph['experimental_scope']='Only 0.19→0.20 evaluated with maximum five partners; original graph unchanged.'
    (output/'experimental_graph.json').write_text(json.dumps(graph,indent=2)+'\n')
    csv_write(output/'controlled_summary.csv',controlled);csv_write(output/'real_summary.csv',summary)
    render(actual,output);render_links(actual,output)
    for name,digest in hashes.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    result=dict(passed=True,controlled_configurations=len(controlled),real_configurations=len(actual),
        independent_dp_checks=len(controlled)+len(actual),dense_full_five_checks=2,baseline_candidates_and_selection_exact=True,
        controlled_summary=controlled,real_summary=summary,input_sha256=hashes,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('experiments/astrolink/expanded_event_tracking.py'),Path('experiments/astrolink/event_tracking.py')]},
        production_tracker_modified=False,original_history_modified=False,physical_ancestry_established=False)
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def render(records,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    ymax=max(25,max(max(r['five_parent_dense_audit']['exclusive_central_fractions'])*100+5 for r in records if r['max_partners']==5))
    for ax,row in zip(axes,[r for r in records if r['max_partners']==5]):
        audit=row['five_parent_dense_audit'];values=np.asarray(audit['exclusive_central_fractions'])*100
        bars=ax.bar([str(g) for g in audit['parents']],values,color='#357a91')
        ax.bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=3)
        ax.axhline(10,c='#b63636',ls='--');ax.set(ylim=(0,ymax),xlabel='Earlier fragment group',
            title=f'Gate {row["gate_reference_cells"]}/64 | full five admitted: {audit["admitted"]}')
        ax.text(.5,-.40,f'Joint child coverage: {audit["node_coverages"][-1]["coverage"]:.1%}\nEvery parent must contribute >=10% uniquely.',ha='center',va='top',transform=ax.transAxes)
    axes[0].set_ylabel('Exclusive contribution to later group (%)')
    fig.suptitle('Can all five fragments satisfy the unchanged event rules?',fontsize=15)
    fig.text(.06,.025,'Exclusive coverage excludes later samples also covered by another parent. Red: unchanged 10% requirement.\n'
        'Offline arity experiment only. Coverage and score support do not establish physical ancestry.',fontsize=10)
    fig.subplots_adjust(left=.08,right=.98,bottom=.4,top=.8,wspace=.08)
    fig.savefig(output/'five_fragment_exclusivity.png',dpi=150);plt.close(fig)


def render_links(records,output):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,6),sharey=True)
    colors=['#147a94','#c16b26','#497a33','#a34985','#6354a4']
    for ax,cap in zip(axes,[3,5]):
        for g,color in zip(range(2,7),colors):
            ax.scatter(.19,6-g,c=color,s=60);ax.text(.189,6-g,f'g{g}',ha='right',va='center')
        ax.scatter(.20,2,c='#743fa3',s=70)
        for row in [r for r in records if r['max_partners']==cap]:
            for e in row['result']['events']:
                if e['children']!=[2]:continue
                for g in e['parents']:
                    ax.plot([.19,.20],[6-g,2],c='#137e88' if cap==5 else '#bd6c27',ls='-' if cap==5 else '--',alpha=.7)
        ax.text(.2005,2,'g2',va='center');ax.set(xlim=(.187,.203),ylim=(-.8,4.8),xticks=[.19,.20],xlabel='Simulation time',title=f'Maximum {cap} parents')
        ax.set_yticks([]);ax.spines[['top','right','left']].set_visible(False)
        text='Narrow: no accepted inner event\nWide: subset [3,4,6]\nNo shared ancestry link' if cap==3 else 'Both distances: all five parents\nCoverage 99.2–100%; gap ~1.1\nShared geometric combination'
        ax.text(.5,-.30,text,ha='center',va='top',transform=ax.transAxes,fontsize=11)
    fig.suptitle('Allowing a full five-fragment event resolves this geometric association',fontsize=15)
    fig.text(.06,.025,'Only the event-size cap changes in the offline experiment. Coverage, normals, score, exclusivity and tie rules are unchanged.\n'
        'Solid links are shared by both distances; dashed links are the old distance-specific subset. No physical merger is asserted.',fontsize=10)
    fig.subplots_adjust(left=.06,right=.97,bottom=.38,top=.82,wspace=.15)
    fig.savefig(output/'association_comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/two_bubbles_detection_extension_iwr_n128'))
    parser.add_argument('--history',type=Path,default=Path('outputs/two_bubbles_detection_extension_review_n128'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/five_fragment_arity_review_n128'))
    args=parser.parse_args();run(args.root,args.history,args.output_dir)
