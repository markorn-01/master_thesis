"""Bounded evidence review of three collision-stage snapshots; no tracking changes."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
from experiments.wind_bubble.plot_resolution_links import choose_record
from experiments.wind_bubble.verify_resolution_associations import enumerate_support
from experiments.wind_bubble.compare_two_bubble_resolutions import csv_write


def review(directory,output):
    archive=directory/'support_records.json.gz';meta_path=directory/'metadata.json'
    meta=json.loads(meta_path.read_text());hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [archive,meta_path]}
    root=Path(meta['target_directory']);partitions=Path(meta['comparison_directory'])
    with gzip.open(archive,'rt') as f:records=json.load(f)
    rows=[];node_rows=[]
    catalogs={};labels={}
    for event in [13,14,15]:
        source=root/'online'/f'surfaces_{event:04d}.npz';partition=partitions/f'groups_n128_event{event:04d}.npz'
        for path in [source,partition]:hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(source) as a:catalogs[event]={k:a[k].copy() for k in ['positions','normals','time']}
        with np.load(partition) as a:labels[event]=a['native_cells_a60_labels'].copy()
        for group in np.unique(labels[event][labels[event]>=0]):
            p=catalogs[event]['positions'][labels[event]==group]
            node_rows.append(dict(event=event,time=float(catalogs[event]['time']),group=int(group),samples=len(p),
                                  centroid_x=float(p[:,0].mean()),centroid_y=float(p[:,1].mean()),centroid_z=float(p[:,2].mean())))
    for event in [13,14]:
        selected={gate:choose_record(records,128,'native_cells',event,gate) for gate in [1.5,2.5]}
        for gate,record in selected.items():
            result=record['result'];config=record['config']
            parents,choices,covered=enumerate_support(catalogs[event],catalogs[event+1],labels[event],labels[event+1],config['gate_code_units'])
            for parent in result['parents']:
                p=parent['parent_group'];target,cov,runner,status=parents[p]
                assert (target,cov,runner,status)==(parent['best_child_group'],parent['parent_coverage'],parent['runner_up_parent_coverage'],parent['status'])
                link=next((x for x in result['links'] if x['parent_group']==p),None)
                child=next(x for x in result['children'] if x['child_group']==target)
                eligible=[q for q,t in choices.items() if t==target]
                union=set().union(*(covered[q] for q in eligible))
                assert sorted(union)==result['supported_child_rows'][str(target)]
                assert len(union)/child['child_samples']==child['union_child_coverage']
                if link:
                    others=set().union(*(covered[q] for q in eligible if q!=p))
                    exclusive=len(covered[p]-others)
                    assert exclusive==link['exclusive_child_support_samples']
                    assert link['redundant_support']==(exclusive==0)
                    assert link['accepted']==child['accepted']
                else:exclusive=None
                other=selected[2.5 if gate==1.5 else 1.5]['result']
                counterpart=next(x for x in other['parents'] if x['parent_group']==p)
                otherlink=next((x for x in other['links'] if x['parent_group']==p),None)
                sensitive=((status,target)!=(counterpart['status'],counterpart['best_child_group']) or
                           bool(link and link['redundant_support'])!=bool(otherlink and otherlink['redundant_support']))
                local='ambiguous_target' if status=='ambiguous_target' else ('redundant_support' if link and link['redundant_support'] else
                      'multi_parent_support' if link and len(eligible)>1 else 'unique_local_candidate' if link and link['accepted'] else 'unsupported')
                category='gate_sensitive' if sensitive else local
                rows.append(dict(parent_event=event,child_event=event+1,gate_reference_cells=gate,
                                 parent_group=p,parent_samples=parent['parent_samples'],best_child_group=target,
                                 best_parent_coverage=cov,runner_up_parent_coverage=runner,local_status=local,
                                 accepted=bool(link and link['accepted']),eligible_parents_of_target=len(eligible),
                                 exclusive_child_support_samples=exclusive,review_category=category,
                                 physical_identity_established=False))
    output.mkdir(parents=True,exist_ok=False);csv_write(output/'link_review.csv',rows);csv_write(output/'snapshot_groups.csv',node_rows)
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    report=dict(snapshots=[13,14,15],comparisons=4,parent_records_checked=len(rows),
                review_categories=dict(Counter(x['review_category'] for x in rows)),
                input_sha256=hashes,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__),Path('experiments/wind_bubble/verify_resolution_associations.py')]},
                original_links_changed=False,physical_truth_labels_available=False,physical_mergers_established=False)
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('sha256')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('outputs/two_bubbles_temporal_comparison_n064_n128'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();review(args.directory,args.output_dir)
