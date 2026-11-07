"""Compare observation-derived prediction and fixed geometric matching on toys."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.controlled_motion_cases import motion_cases
from experiments.astrolink.controlled_tracking_cases import evaluate_links
from experiments.astrolink.motion_prediction import predict_positions
from experiments.astrolink.support_evidence import support_evidence
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write


def evaluate(case,shuffle=False):
    data={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in case.items()}
    if shuffle:
        rng=np.random.default_rng(42)
        for i in range(3):
            order=rng.permutation(len(data[f'positions_{i}']))
            for prefix in ['positions','normals','labels']:data[f'{prefix}_{i}']=data[f'{prefix}_{i}'][order]
    predicted,estimates,seed=predict_positions(data['positions_0'],data['normals_0'],data['labels_0'],
                                             data['positions_1'],data['normals_1'],data['labels_1'],data['times'])
    records=[]
    for gate in [.08,.2,.35]:
        for method,positions in [('geometry',data['positions_1']),('centroid_prediction',predicted)]:
            result=support_evidence(positions,data['normals_1'],data['labels_1'],data['positions_2'],data['normals_2'],data['labels_2'],gate)
            metrics=evaluate_links(result,case['expected_links'])
            records.append(dict(case=case['name'],method=method,gate=gate,**metrics,estimates=estimates,
                                parents=result['parents'],children=result['children'],links=result['links'],
                                initialized_groups=sum(x['initialized'] for x in estimates),fallback_groups=sum(not x['initialized'] for x in estimates)))
    return records,predicted


def run(output):
    output.mkdir(parents=True,exist_ok=False);records=[];rows=[]
    for case in motion_cases():
        current,predicted=evaluate(case);shuffled,_=evaluate(case,True)
        for a,b in zip(current,shuffled):
            for key in ['accepted_links','false_links','missed_links','parents','children','links','estimates']:assert a[key]==b[key]
            rows.append(dict(case=a['case'],method=a['method'],gate=a['gate'],correct_links=len(a['correct_links']),
                             false_links=len(a['false_links']),missed_links=len(a['missed_links']),
                             initialized_groups=a['initialized_groups'],fallback_groups=a['fallback_groups'],exact_correspondence=a['exact_correspondence']))
        np.savez_compressed(output/f"{case['name']}.npz",**{k:v for k,v in case.items() if isinstance(v,np.ndarray)},predicted_positions=predicted)
        records.extend(current)
    csv_write(output/'summary.csv',rows);(output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
    plot(output,rows)
    sources=[Path(__file__),Path('experiments/astrolink/motion_prediction.py'),Path('experiments/astrolink/controlled_motion_cases.py'),
             Path('experiments/astrolink/controlled_tracking_cases.py'),Path('experiments/astrolink/support_evidence.py'),Path('experiments/astrolink/patch_support.py')]
    meta=dict(cases=len(motion_cases()),configurations=len(records),bootstrap_gate=.2,residual_gates=[.08,.2,.35],shuffle_checks=len(records),
              model='Group centroid constant-velocity translation; normals unchanged; uninitialized groups use geometry fallback.',
              truth_used_for_prediction=False,production_tracker_changed=False,physical_tracker_validated=False,
              source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    for r in rows:
        if r['gate']==.2:print(r)


def plot(output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[c['name'] for c in motion_cases()];fig,axes=plt.subplots(2,2,figsize=(13,9),sharey=True)
    for col,method in enumerate(['geometry','centroid_prediction']):
        for row,field in enumerate(['false_links','missed_links']):
            matrix=np.array([[next(r[field] for r in rows if r['case']==name and r['method']==method and r['gate']==gate) for gate in [.08,.2,.35]] for name in names])
            ax=axes[row,col];ax.imshow(matrix,vmin=0,vmax=2,cmap='YlOrRd',aspect='auto')
            for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center')
            ax.set(xticks=range(3),xticklabels=['.08','.20','.35'],yticks=range(len(names)),yticklabels=[x.replace('_',' ') for x in names],
                   title=f'{method.replace("_"," ")}: {field.replace("_"," ")}',xlabel='Final matching gate (toy units)')
    fig.suptitle('Motion prediction helps some cases and fails when its assumptions break',fontsize=14)
    fig.text(.035,.02,'Velocity from the first two observed catalogs using unique accepted one-to-one geometric bootstrap links (gate .20).\n'
             'Truth is used only for evaluation. Missing initialization uses unshifted geometry. No branching, rotation, deformation or gap bridging.\n'
             'These are selected synthetic examples with supplied groups, not an accuracy estimate for hydro simulations.',fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.95));fig.savefig(output/'motion_prediction_comparison.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);run(p.parse_args().output_dir)
