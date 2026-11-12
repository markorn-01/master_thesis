"""Controlled tests and real three-state windows for delayed fragment support."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.tracklet_fragments import attach_fragments
from experiments.astrolink.event_tracking import event_tracking
from experiments.astrolink.controlled_tracking_cases import evaluate_links
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write


def catalog(points, labels, normals=None):
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    return dict(positions=p, labels=np.asarray(labels, dtype=int),
                normals=np.tile([0., 0., 1.], (len(p), 1)) if normals is None else np.asarray(normals, float))


def cases():
    plane = np.array([[x*.2, y*.2, 0.] for x in range(8) for y in range(8)])
    a = catalog(plane, np.repeat(np.arange(8), 8)); b = catalog(plane+[0,0,.01], [10]*64)
    c = catalog(plane+[0,0,.02], [20]*64)
    result = [dict(name='eight_fragments', a=a,b=b,c=c,continuations=[(10,20)],
                   truth=[(i,10) for i in range(8)], meaning='Eight geometric fragments become one continued surface.')]
    for direction in [1,-1]:
        a = catalog(np.vstack([plane+[0,0,-.06],plane+[0,0,.06]]), [0]*64+[1]*64)
        result.append(dict(name='future_motion_positive' if direction==1 else 'future_motion_negative',
            a=a,b=catalog(plane,[10]*64),c=catalog(plane+[0,0,.06*direction],[20]*64),
            continuations=[(10,20)],truth=[(0 if direction==1 else 1,10)],
            meaning='Subsequent normal motion distinguishes two otherwise equal-distance parents under constant motion.'))
    # Same observed data with opposite stipulated ancestry: cannot solve both.
    for true_parent in [0,1]:
        result.append(dict(name=f'unobservable_ancestry_{true_parent}',
            a=catalog(np.vstack([plane+[0,0,-.02],plane+[0,0,.02]]),[0]*64+[1]*64),
            b=catalog(plane,[10]*64),c=catalog(plane,[20]*64),continuations=[(10,20)],
            truth=[(true_parent,10)],meaning='Identical three-frame geometry, different stipulated ancestry; not identifiable.'))
    result.append(dict(name='coincident_child_families',a=catalog(plane,[0]*64),
        b=catalog(np.vstack([plane,plane]),[10]*64+[11]*64),
        c=catalog(np.vstack([plane,plane]),[20]*64+[21]*64),continuations=[(10,20),(11,21)],
        truth=[],meaning='Two indistinguishable families must remain ambiguous.'))
    n=np.tile([0.,0.,1.],(128,1));n[64:]*=-1
    result.append(dict(name='opposite_normal_distractor',
        a=catalog(np.vstack([plane,plane]),[0]*64+[1]*64,n),
        b=catalog(plane,[10]*64),c=catalog(plane,[20]*64),continuations=[(10,20)],
        truth=[(0,10)],meaning='Opposite-normal sheet must not join the family.'))
    return result


def check_dense(a,b,c,result,times,predict):
    """Independently recompute every normal-gated fragment coverage with cdist."""
    from scipy.spatial.distance import cdist
    p=result['parameters'];gate=p['gate']; ratio=(times[1]-times[0])/(times[2]-times[1])
    normalize=lambda n:n/np.linalg.norm(n,axis=1)[:,None]
    scores={};footprints={}
    for model in result['models']:
        bi=b['labels']==model['child_group'];ci=c['labels']==model['later_group']
        bp=b['positions'][bi];bn=normalize(b['normals'][bi]);cp=c['positions'][ci];cn=normalize(c['normals'][ci])
        distance=cdist(bp,cp);distance[(bn@cn.T)<np.cos(np.deg2rad(p['normal_angle']))]=np.inf
        nearest=np.argmin(distance,axis=1);good=distance[np.arange(len(bp)),nearest]<=gate
        step=float(np.median(np.einsum('ij,ij->i',cp[nearest[good]]-bp[good],bn[good]))) if good.any() else 0.
        np.testing.assert_allclose(step,model['median_normal_displacement'],atol=1e-12)
        assert model['usable']==(min(good.mean(),(distance<=gate).any(axis=0).mean())>=p['coverage'])
        if not model['usable']:continue
        predicted=bp-ratio*step*bn if predict else bp
        distance=cdist(a['positions'],predicted)
        distance[(normalize(a['normals'])@bn.T)<np.cos(np.deg2rad(p['normal_angle']))]=np.inf
        for parent in sorted(set(a['labels'][a['labels']>=0])):
            minima=distance[a['labels']==parent].min(axis=1);valid=minima<=gate
            fraction=valid.mean()
            footprints[int(parent),model['child_group']]=set(np.flatnonzero((distance[a['labels']==parent]<=gate).any(axis=0)))
            if fraction>=p['coverage']:
                score=fraction*(1-p['distance_weight']*minima[valid].mean()/gate)
                scores.setdefault(int(parent),[]).append((score,model['child_group'],fraction))
    for assignment in result['assignments']:
        ranked=sorted(scores.get(assignment['parent_group'],[]),key=lambda x:(-x[0],x[1]))
        if not ranked:assert assignment['status']=='no_support';continue
        gap=ranked[0][0]-(ranked[1][0] if len(ranked)>1 else 0.)
        np.testing.assert_allclose(gap,assignment['score_margin'],atol=1e-12)
        if gap<p['margin']:assert assignment['status']=='ambiguous_family'
        else:assert assignment['child_group']==ranked[0][1]
    for family in result['families']:
        union=set().union(*(footprints[parent,family['child_group']] for parent in family['parent_groups']))
        expected=len(union)/int((b['labels']==family['child_group']).sum())
        np.testing.assert_allclose(expected,family['child_union_coverage'],atol=0,rtol=0)
        assert family['accepted']==(expected>=p['coverage'])


def run(output):
    output.mkdir(parents=True,exist_ok=False);rows=[];details=[];dense_checks=0
    for case in cases():
        for gate in [.05,.07,.09]:
            a,b,c=[case[k] for k in ['a','b','c']]
            baseline=event_tracking(a['positions'],a['normals'],a['labels'],b['positions'],b['normals'],b['labels'],gate)
            methods={'adjacent_event':baseline}
            for name,predict in [('static_family',False),('delayed_family',True)]:
                result=attach_fragments(a,b,c,case['continuations'],[0.,1.,2.],gate,predict=predict)
                check_dense(a,b,c,result,[0.,1.,2.],predict);dense_checks+=1
                # All observations are freshly ordered; no row correspondence can help.
                shuffled=[]
                for data in [a,b,c]:
                    order=np.arange(len(data['labels']))[::-1]
                    shuffled.append({key:value[order] for key,value in data.items()})
                other=attach_fragments(*shuffled,case['continuations'],[0.,1.,2.],gate,predict=predict)
                assert result['links']==other['links']
                methods[name]=result
            for name,result in methods.items():
                metrics=evaluate_links(result,case['truth'])
                rows.append(dict(case=case['name'],gate=gate,method=name,
                    false_links=len(metrics['false_links']),missed_links=len(metrics['missed_links']),
                    exact_correspondence=metrics['exact_correspondence']))
                details.append(dict(case=case['name'],gate=gate,method=name,meaning=case['meaning'],
                                    metrics=metrics,result=result))
    csv_write(output/'controlled_summary.csv',rows)
    (output/'controlled_results.json').write_text(json.dumps(details,indent=2)+'\n')
    meta=json.loads(Path('outputs/two_bubbles_temporal_comparison_n064_n128/metadata.json').read_text())
    real=[];hashes={};audit=[]
    for n,start,history in [(64,13,'outputs/two_bubbles_event_history_resolution_n064_n128/n064_graph.json'),
                            (128,10,'outputs/two_bubbles_event_history_v1_n128/graph.json')]:
        path=Path(history);hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();graph=json.loads(path.read_text())
        root=Path(meta['reference_directory' if n==64 else 'target_directory']);window=[];times=[]
        for index in range(start,start+3):
            source=root/'online'/f'surfaces_{index:04d}.npz'
            partition=Path(meta['comparison_directory'])/f'groups_n{n:03d}_event{index:04d}.npz'
            for p in [source,partition]:hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            with np.load(source) as s,np.load(partition) as g:
                window.append(catalog(s['positions'],g['native_cells_a60_labels'],s['normals']));times.append(float(s['time']))
        nodes={node['node_id']:node for node in graph['nodes']}
        continuations=[(nodes[e['parents'][0]]['group'],nodes[e['children'][0]]['group'])
            for e in graph['events'] if e['parent_event']==start+1 and e['status']=='shared' and e['kind']=='continuation']
        stable=[e for e in graph['events'] if e['parent_event']==start and e['status']=='shared' and e['kind']=='continuation']
        excluded_a={nodes[e['parents'][0]]['group'] for e in stable}
        excluded_b={nodes[e['children'][0]]['group'] for e in stable}
        continuations=[(b,c) for b,c in continuations if b not in excluded_b]
        selections=[(window[0]['labels']>=0)&~np.isin(window[0]['labels'],list(excluded_a)),
                    np.isin(window[1]['labels'],[b for b,c in continuations]),
                    np.isin(window[2]['labels'],[c for b,c in continuations])]
        window=[{key:value[select] for key,value in data.items()} for data,select in zip(window,selections)]
        for gate in [1.5/64,2.5/64]:
            for name,predict in [('static_family',False),('delayed_family',True)]:
                result=attach_fragments(*window,continuations,times,gate,predict=predict)
                check_dense(*window,result,times,predict)
                for row in result['assignments']:
                    if row['child_group'] is not None:
                        assert row['score_margin']>=.1
                real.append(dict(num_cells=n,first_event=start,times=times,gate=gate,method=name,result=result))
                for family in result['families']:
                    audit.append(dict(num_cells=n,first_time=times[0],gate=gate,method=name,
                        child_group=family['child_group'],parent_groups=','.join(map(str,family['parent_groups'])),
                        fragments=len(family['parent_groups']),union_coverage=family['child_union_coverage'],accepted=family['accepted']))
    csv_write(output/'real_families.csv',audit)
    (output/'real_results.json').write_text(json.dumps(real,indent=2)+'\n')
    render_controls(rows,output)
    aggregate=[dict(method=method,configurations=sum(r['method']==method for r in rows),
        exact=sum(r['method']==method and r['exact_correspondence'] for r in rows),
        false_links=sum(r['false_links'] for r in rows if r['method']==method),
        missed_links=sum(r['missed_links'] for r in rows if r['method']==method))
        for method in ['adjacent_event','static_family','delayed_family']]
    sources=[Path(__file__),Path('experiments/astrolink/tracklet_fragments.py')]
    report=dict(controlled=aggregate,dense_checks=dense_checks,shuffle_checks=dense_checks,
        real_windows=2,real_configurations=len(real),real_dense_checks=len(real),input_sha256=hashes,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        original_tracking_outputs_changed=False,physical_ancestry_established=False)
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2));print(json.dumps(audit,indent=2))


def render_controls(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=list(dict.fromkeys(r['case'] for r in rows));methods=['adjacent_event','static_family','delayed_family']
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,field in zip(axes,['false_links','missed_links']):
        matrix=np.array([[sum(r[field] for r in rows if r['case']==case and r['method']==method)
                         for method in methods] for case in cases])
        ax.imshow(matrix,cmap='YlOrRd',vmin=0,vmax=max(1,matrix.max()),aspect='auto')
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center')
        ax.set(xticks=range(3),xticklabels=['Adjacent events','Static family','Delayed family'],
               yticks=range(len(cases)),yticklabels=[c.replace('_',' ') for c in cases],title=field.replace('_',' ').capitalize())
    fig.suptitle('Targeted controls: counts summed over three distance gates')
    fig.text(.03,.02,'Stipulated ancestry is used only for evaluation. Families are geometric support, not proven ancestry.\n'
             'The identical-observation controls expose false physical links if family support is misinterpreted as ancestry.',fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.93));fig.savefig(output/'controlled_comparison.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args().output_dir)
