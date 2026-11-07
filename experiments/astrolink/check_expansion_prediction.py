"""Compare geometric, translation and isotropic-expansion prediction on toys."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.controlled_expansion_cases import expansion_cases
from experiments.astrolink.controlled_tracking_cases import evaluate_links
from experiments.astrolink.motion_prediction import predict_positions
from experiments.astrolink.expansion_prediction import predict_expanding_positions
from experiments.astrolink.support_evidence import support_evidence
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write


def evaluate(case,shuffle=False):
    data={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in case.items()}
    if shuffle:
        rng=np.random.default_rng(42)
        for i in range(3):
            order=rng.permutation(len(data[f'positions_{i}']))
            for field in ['positions','normals','labels']:data[f'{field}_{i}']=data[f'{field}_{i}'][order]
    args=[data[f'{field}_{i}'] for i in [0,1] for field in ['positions','normals','labels']]+[data['times']]
    translation,te,_=predict_positions(*args);expansion,ee,_=predict_expanding_positions(*args)
    records=[]
    for gate in [.08,.2,.35]:
        for method,positions,estimates in [('geometry',data['positions_1'],te),('centroid_translation',translation,te),('isotropic_expansion',expansion,ee)]:
            r=support_evidence(positions,data['normals_1'],data['labels_1'],data['positions_2'],data['normals_2'],data['labels_2'],gate)
            records.append(dict(case=case['name'],method=method,gate=gate,**evaluate_links(r,case['expected_links']),
                                estimates=estimates,parents=r['parents'],children=r['children'],links=r['links']))
    return records,translation,expansion


def run(output):
    output.mkdir(parents=True,exist_ok=False);all_records=[];rows=[];max_shuffle_difference=0.
    for case in expansion_cases():
        records,translation,expansion=evaluate(case);shuffled,_,_=evaluate(case,True)
        for a,b in zip(records,shuffled):
            for key in ['accepted_links','false_links','missed_links','parents','children','links']:assert a[key]==b[key]
            # Moment reductions may differ by roundoff after row permutation.
            for ea,eb in zip(a['estimates'],b['estimates']):
                for key in ['velocity','extent_0','extent_1','extent_rate','scale']:
                    if key in ea and ea[key] is not None:
                        np.testing.assert_allclose(ea[key],eb[key],rtol=1e-12,atol=1e-12)
                        max_shuffle_difference=max(max_shuffle_difference,float(np.max(np.abs(np.asarray(ea[key])-np.asarray(eb[key])))))
            rows.append(dict(case=a['case'],method=a['method'],gate=a['gate'],correct_links=len(a['correct_links']),false_links=len(a['false_links']),
                             missed_links=len(a['missed_links']),exact_correspondence=a['exact_correspondence']))
        np.savez_compressed(output/f"{case['name']}.npz",**{k:v for k,v in case.items() if isinstance(v,np.ndarray)},
                            translation_positions=translation,expansion_positions=expansion)
        all_records.extend(records)
    csv_write(output/'summary.csv',rows);(output/'results.json').write_text(json.dumps(all_records,indent=2)+'\n');plot(output,rows)
    sources=[Path(__file__)]+[Path('experiments/astrolink')/f'{name}.py' for name in
              ['controlled_expansion_cases','controlled_motion_cases','controlled_tracking_cases','motion_prediction','expansion_prediction','support_evidence','patch_support']]
    meta=dict(cases=len(expansion_cases()),configurations=len(rows),bootstrap_gate=.2,gates=[.08,.2,.35],
              shuffle_checks=len(rows),max_shuffle_moment_absolute_difference=max_shuffle_difference,
              motion_model='Centroid plus scalar RMS-extent extrapolation; no future observations or truth initialization.',
              production_tracker_changed=False,physical_tracker_validated=False,
              source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    for r in rows:
        if r['gate']==.2 and r['case'] in ['symmetric_expansion','translating_expanding_sphere','decelerating_expansion','anisotropic_expansion','changing_sample_coverage']:print(r)


def plot(output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[c['name'] for c in expansion_cases()];fig,axes=plt.subplots(2,3,figsize=(16,11),sharey=True)
    for col,method in enumerate(['geometry','centroid_translation','isotropic_expansion']):
        for row,field in enumerate(['false_links','missed_links']):
            m=np.array([[next(r[field] for r in rows if r['case']==name and r['method']==method and r['gate']==gate) for gate in [.08,.2,.35]] for name in names])
            ax=axes[row,col];ax.imshow(m,vmin=0,vmax=2,cmap='YlOrRd',aspect='auto')
            for (i,j),v in np.ndenumerate(m):ax.text(j,i,str(v),ha='center',va='center',fontsize=9)
            ax.set(xticks=range(3),xticklabels=['.08','.20','.35'],yticks=range(len(names)),yticklabels=[n.replace('_',' ') for n in names],
                   title=f'{method.replace("_"," ")}: {field.replace("_"," ")}',xlabel='Final gate (toy units)')
    fig.suptitle('Expansion-aware toy prediction: improvements and remaining failures',fontsize=15)
    fig.text(.035,.02,'All methods use the same observed one-to-one bootstrap (gate .20). Isotropic prediction extrapolates sample centroid and RMS extent.\n'
             'Truth is only for evaluation. Uninitialized motion uses geometry; invalid extent extrapolation uses translation. Normals remain unchanged.\n'
             'Supplied catalog groups and selected synthetic examples: not a physical tracking accuracy estimate or a model of a turbulent box.',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.95));fig.savefig(output/'expansion_prediction_comparison.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True,type=Path);run(p.parse_args().output_dir)
