"""Evaluate fixed pooled-support rules on known synthetic correspondences."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.controlled_tracking_cases import tracking_cases,evaluate_links
from experiments.astrolink.support_evidence import support_evidence
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write


def evaluate(case,gate,shuffle=False):
    a=np.arange(len(case['positions_a']));b=np.arange(len(case['positions_b']))
    if shuffle:
        rng=np.random.default_rng(42);rng.shuffle(a);rng.shuffle(b)
    return support_evidence(case['positions_a'][a],case['normals_a'][a],case['labels_a'][a],
                            case['positions_b'][b],case['normals_b'][b],case['labels_b'][b],gate,
                            max_angle_degrees=60.,min_parent_coverage=.5,min_child_coverage=.5,dominance_margin=.25)


def run(output):
    output.mkdir(parents=True,exist_ok=False);rows=[];records=[];cases=tracking_cases()
    for case in cases:
        arrays={k:v for k,v in case.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(output/f"{case['name']}.npz",**arrays)
        for gate in [.1,.2,.35]:
            result=evaluate(case,gate);metrics=evaluate_links(result,case['expected_links'])
            shuffled=evaluate(case,gate,True);shuffled_metrics=evaluate_links(shuffled,case['expected_links'])
            assert metrics==shuffled_metrics
            # Compare group-level support as well as the accepted pair sets.
            assert result['parents']==shuffled['parents'] and result['children']==shuffled['children'] and result['links']==shuffled['links']
            record=dict(case=case['name'],meaning=case['meaning'],gate=gate,**metrics,
                        parents=result['parents'],children=result['children'],links=result['links'],shuffle_matches=True)
            records.append(record)
            rows.append(dict(case=case['name'],gate=gate,exact_correspondence=metrics['exact_correspondence'],
                             correct_links=len(metrics['correct_links']),false_links=len(metrics['false_links']),
                             missed_links=len(metrics['missed_links']),ambiguous_parents=sum(p['status']=='ambiguous_target' for p in result['parents']),
                             redundant_accepted_links=sum(l['accepted'] and l['redundant_support'] for l in result['links'])))
    csv_write(output/'summary.csv',rows);(output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
    plot(output,rows)
    sources=[Path(__file__),Path('experiments/astrolink/controlled_tracking_cases.py'),Path('experiments/astrolink/support_evidence.py'),Path('experiments/astrolink/patch_support.py')]
    report=dict(cases=len(cases),configurations=len(rows),shuffle_checks=len(rows),gates=[.1,.2,.35],baseline_gate=.2,
                min_parent_coverage=.5,min_child_coverage=.5,dominance_margin=.25,normal_limit_degrees=60,
                truth='Stipulated group correspondences in constructed point catalogs; no hydrodynamic truth claim.',
                known_failures_preserved=True,physical_tracker_validated=False,production_tracker_changed=False,
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (output/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in rows:
        if row['gate']==.2:print(row)


def plot(output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(dict.fromkeys(r['case'] for r in rows));fig,axes=plt.subplots(1,2,figsize=(13,7),sharey=True)
    for ax,field,title in zip(axes,['false_links','missed_links'],['Accepted false links','Missed stipulated links']):
        matrix=np.array([[next(r[field] for r in rows if r['case']==name and r['gate']==g) for g in [.1,.2,.35]] for name in names])
        image=ax.imshow(matrix,cmap='YlOrRd',vmin=0,vmax=2,aspect='auto')
        for (i,j),value in np.ndenumerate(matrix):ax.text(j,i,str(value),ha='center',va='center')
        ax.set(xticks=range(3),xticklabels=['0.10','0.20','0.35'],yticks=range(len(names)),
               yticklabels=[n.replace('_',' ') for n in names],xlabel='Distance gate (toy units)',title=title)
    fig.suptitle('Known synthetic correspondences expose geometric matching limits',fontsize=14)
    fig.text(.035,.025,'Two adjacent catalogs; group labels supplied, not clustered. Gates use arbitrary toy units, not the simulation grid scale.\n'
             'Normal limit 60°; coverage thresholds 0.5; dominance margin 0.25. Split/merge cases describe catalog representation.\n'
             'Zeros in both panels mean exact stipulated correspondence for that case, not a validated physical shock tracker.',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.94));fig.savefig(output/'controlled_tracking_summary.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True,type=Path);run(p.parse_args().output_dir)
