"""Validate advisory labels on existing controls and the bounded collision review."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
from experiments.astrolink.continuation_status import continuation_status,distance_sensitivity
from experiments.astrolink.controlled_tracking_cases import tracking_cases,evaluate_links
from experiments.astrolink.check_controlled_tracking import evaluate
from experiments.wind_bubble.plot_resolution_links import choose_record
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write


def run(directory,review,output):
    archive=directory/'support_records.json.gz';review_csv=review/'link_review.csv'
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [archive,review_csv,review/'verification.json']}
    evidence=json.loads((review/'verification.json').read_text())
    for path,digest in evidence['input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
        hashes[path]=digest
    expected=list(csv.DictReader(review_csv.open()))
    with gzip.open(archive,'rt') as f:records=json.load(f)
    output.mkdir(parents=True,exist_ok=False);parent_rows=[];child_rows=[];control_rows=[];details=[]
    for case in tracking_cases():
        results={gate:evaluate(case,gate) for gate in [.1,.2,.35]}
        tables={gate:continuation_status(result) for gate,result in results.items()}
        warnings=distance_sensitivity(list(tables.values()))
        for gate,table in tables.items():
            result=results[gate];metrics=evaluate_links(result,case['expected_links'])
            # Independent accounting: every matcher-accepted pair must still be reported.
            original={(l['parent_group'],l['child_group']) for l in result['links'] if l['accepted']}
            reported={(p['parent_group'],p['selected_child_group']) for p in table['parents'] if p['matcher_accepted']}
            assert original==reported
            shuffled=continuation_status(evaluate(case,gate,shuffle=True));assert shuffled==table
            counts=Counter(p['advisory_status'] for p in table['parents'])
            control_rows.append(dict(case=case['name'],gate=gate,continuation_candidates=counts['continuation_candidate'],
                ambiguous=counts['ambiguous'],multiple_parent_support=counts['multiple_parent_support'],unmatched=counts['unmatched'],
                unchanged_accepted_pairs=True,false_matcher_links=len(metrics['false_links']),missed_matcher_links=len(metrics['missed_links'])))
            details.append(dict(case=case['name'],gate=gate,table=table,distance_sensitivity=warnings,truth_comparison=metrics))
    for event in [13,14]:
        chosen={gate:choose_record(records,128,'native_cells',event,gate) for gate in [1.5,2.5]}
        tables={gate:continuation_status(record['result']) for gate,record in chosen.items()}
        warnings=distance_sensitivity(list(tables.values()))
        for gate,table in tables.items():
            for parent in table['parents']:
                row=dict(parent_event=event,child_event=event+1,gate_reference_cells=gate,**parent,
                         distance_sensitive=warnings[parent['parent_group']])
                old=next(r for r in expected if int(r['parent_event'])==event and float(r['gate_reference_cells'])==gate and
                         int(r['parent_group'])==parent['parent_group'])
                category='gate_sensitive' if row['distance_sensitive'] else 'multi_parent_support' if parent['advisory_status']=='multiple_parent_support' else 'unique_local_candidate'
                assert category==old['review_category']
                assert parent['matcher_accepted']==(old['accepted']=='True')
                assert parent['exclusive_child_support_samples']==(int(old['exclusive_child_support_samples']) if old['exclusive_child_support_samples'] else None)
                parent_rows.append(row)
            child_rows.extend(dict(parent_event=event,child_event=event+1,gate_reference_cells=gate,
                                   **{**child,'accepted_parent_groups':','.join(map(str,child['accepted_parent_groups']))})
                              for child in table['children'])
    csv_write(output/'parent_status.csv',parent_rows);csv_write(output/'child_status.csv',child_rows);csv_write(output/'control_summary.csv',control_rows)
    (output/'control_details.json').write_text(json.dumps(details,indent=2)+'\n')
    # CSV readback: statuses and Boolean flags must survive serialization.
    reread=list(csv.DictReader((output/'parent_status.csv').open()))
    assert len(reread)==len(parent_rows)==20
    for saved,row in zip(reread,parent_rows):
        assert saved['advisory_status']==row['advisory_status'] and saved['distance_sensitive']==str(row['distance_sensitive'])
        assert saved['matcher_accepted']==str(row['matcher_accepted'])
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    sources=[Path(__file__)]+[Path('experiments/astrolink')/f'{name}.py' for name in
        ['continuation_status','patch_support','support_evidence','controlled_tracking_cases','check_controlled_tracking']]
    report=dict(control_configurations=len(control_rows),control_shuffle_checks=len(control_rows),
        reviewed_parent_rows=len(parent_rows),reviewed_child_rows=len(child_rows),
        parent_status_counts=dict(Counter(p['advisory_status'] for p in parent_rows)),
        distance_sensitive_parent_rows=sum(p['distance_sensitive'] for p in parent_rows),
        bounded_review_reproduced=True,accepted_links_changed=False,persistent_ids_assigned=False,
        physical_mergers_inferred=False,known_false_link_fixed=False,
        input_sha256=hashes,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('sha256')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('outputs/two_bubbles_temporal_comparison_n064_n128'))
    parser.add_argument('--review',type=Path,default=Path('outputs/two_bubbles_collision_link_review_n128'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();run(args.directory,args.review,args.output_dir)
