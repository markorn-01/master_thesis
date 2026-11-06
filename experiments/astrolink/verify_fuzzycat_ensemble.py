"""Read back same-event FuzzyCat ensembles, Jaccard scores and weight audits."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.fuzzycat_inputs import hierarchy_memberships,partition_memberships,display_memberships,sample_multiplicities
from experiments.astrolink.fuzzycat_weight_reference import reconstruct_memberships


def verify(directory):
    meta=json.loads((directory/'metadata.json').read_text())
    for name,h in {**meta['input_sha256'],**meta['source_sha256']}.items():
        path=Path(name)
        if not path.exists() and '/experiments/' in name:path=Path('experiments')/name.split('/experiments/',1)[1]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==h,name
    summaries=list(csv.DictReader((directory/'summary.csv').open()))
    assert len(summaries)==38
    compared_pairs=0;selections=0;affected=[]
    for name in sorted({r['result_directory'] for r in summaries}):
        target=directory/name
        realizations=json.loads((target/'realizations.json').read_text());assert len(realizations)==12
        rows=[r for r in summaries if r['result_directory']==name];first=rows[0]
        event=int(first['event_index']);include_root=first['include_root']=='True'
        source=next(Path(p) for p in meta['input_sha256'] if p.endswith(f'event_{event:04d}.npz'))
        with np.load(source) as original,np.load(target/'input_clusters.npz') as bundle:
            expected={}
            for record in realizations:
                prefix=f"{record['config']}_o{record['order_seed']}__"
                if first['encoding']=='hierarchy':
                    groups=hierarchy_memberships(original[prefix+'ordering_catalog'],original[prefix+'intervals'],
                                                 original[prefix+'ids'],int(first['n_points']),include_root)
                else:groups=partition_memberships(original[prefix+'labels'])
                assert record['empty']==(not groups) and record['clusters']==len(groups)
                expected.update({f"{record['sample']:03d}_{key}":v for key,v in groups.items()})
            assert set(expected)==set(bundle.files)
            for key,values in expected.items():np.testing.assert_array_equal(bundle[key],values)
            n_points=int(first['n_points'])
            for row in rows:
                with np.load(target/f"result_s{float(row['min_stability']):g}.npz") as result:
                    files=result['cluster_files'];assert set(map(str,files))=={key+'.npy' for key in expected}
                    if row==rows[0]:
                        for (a,b),jaccard in zip(result['pairs'],result['pair_jaccard']):
                            x,y=expected[str(files[a]).removesuffix('.npy')],expected[str(files[b]).removesuffix('.npy')]
                            intersection=len(np.intersect1d(x,y,assume_unique=True))
                            np.testing.assert_allclose(float(jaccard),intersection/(len(x)+len(y)-intersection),rtol=1e-6,atol=1e-8)
                            compared_pairs+=1
                    counts,reference=sample_multiplicities(files,result['ordering'],result['fuzzy_clusters'],12)
                    np.testing.assert_array_equal(counts,result['sample_multiplicities'])
                    np.testing.assert_array_equal(reference,result['sample_weights_float_reference'])
                    # Library stability is the fraction of represented realizations,
                    # not a calibrated probability of physical correctness.
                    np.testing.assert_allclose((counts>0).sum(axis=0)/12,result['stabilities'],rtol=0,atol=1e-12)
                    assert len(result['stabilities'])==int(row['fuzzy_groups'])
                    assert (result['stabilities']>=float(row['min_stability'])).all()
                    assert (result['intra_jaccard']>=.5).all() and (result['inter_jaccard']<=.5).all()
                    raw,flat,hierarchy=reconstruct_memberships(files,expected,result['ordering'],result['fuzzy_clusters'],
                                                              result['sample_weights_library'],result['stabilities'],12,n_points)
                    for actual,calculated in [(result['memberships'],raw),(result['memberships_flat'],flat),(result['fuzzy_hierarchy'],hierarchy)]:
                        np.testing.assert_allclose(actual,calculated,atol=1e-12,rtol=1e-12)
                    labels,maximum=display_memberships(flat)
                    np.testing.assert_array_equal(labels,result['display_labels'])
                    np.testing.assert_allclose(maximum,result['maximum_membership'],rtol=0,atol=1e-12)
                    assert int((labels>=0).sum())==int(row['display_assigned_samples'])
                    lost=int(((counts>1)&(result['sample_weights_library']==0)).sum())
                    assert lost==int(row['truncated_reciprocal_weight_cases'])
                    if lost:
                        ref_raw,ref_flat,ref_hierarchy=reconstruct_memberships(files,expected,result['ordering'],result['fuzzy_clusters'],
                                                                              reference,result['stabilities'],12,n_points)
                        np.savez_compressed(target/f"float_weight_reference_s{float(row['min_stability']):g}.npz",
                                            memberships=ref_raw,memberships_flat=ref_flat,fuzzy_hierarchy=ref_hierarchy)
                        affected.append(dict(result_directory=name,min_stability=float(row['min_stability']),truncated_weight_cases=lost,
                                             max_absolute_flat_membership_difference=float(np.abs(flat-ref_flat).max()),
                                             interpretation='Counterfactual floating-weight reconstruction with fixed selected clusters; library untouched.'))
                    selections+=1
    reproducer=json.loads((directory/'weight_reproducer.json').read_text())
    assert reproducer['actual_weights']==[[0],[0]] and not reproducer['reciprocal_weight_check_passed']
    result=dict(input_and_mapping_checks_passed=True,jaccard_pairs_checked=compared_pairs,selections_checked=selections,
                memberships_reconstructed_from_saved_library_weights=True,reciprocal_weight_control_passed=False,
                affected_selections=affected,library_modified=False,physical_identity_validation=False,
                audit_source_sha256=hashlib.sha256(Path('experiments/astrolink/fuzzycat_weight_reference.py').read_bytes()).hexdigest(),
                scope='Reproduces saved library behavior, including known integer-weight truncation; not a physical validation.')
    (directory/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    verify(p.parse_args().directory)
