"""Verify saved corrected memberships independently of the adapter execution."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.astrolink.fuzzycat_inputs import sample_multiplicities,display_memberships
from experiments.astrolink.fuzzycat_weight_reference import reconstruct_memberships


def verify(directory):
    meta=json.loads((directory/'metadata.json').read_text())
    for name,h in {**meta['input_sha256'],**meta['source_sha256']}.items():
        path=Path(name)
        if not path.exists() and '/experiments/' in name:path=Path('experiments')/name.split('/experiments/',1)[1]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==h,name
    source=next(Path(p).parent for p in meta['input_sha256'] if Path(p).name=='summary.csv')
    rows=list(csv.DictReader((directory/'summary.csv').open()));assert len(rows)==38
    changed=[]
    for row in rows:
        name=row['result_directory'];filename=f"result_s{float(row['min_stability']):g}.npz"
        with np.load(directory/name/filename) as corrected,np.load(source/name/filename) as original,np.load(source/name/'input_clusters.npz') as bundle:
            clusters={key:bundle[key] for key in bundle.files}
            for key in ['fuzzy_clusters','stabilities','ordering','cluster_files']:
                np.testing.assert_array_equal(corrected[key],original[key])
            counts,weights=sample_multiplicities(original['cluster_files'],original['ordering'],original['fuzzy_clusters'],12)
            np.testing.assert_array_equal(counts,corrected['sample_multiplicities'])
            np.testing.assert_array_equal(weights,corrected['sample_weights'])
            values=reconstruct_memberships(original['cluster_files'],clusters,original['ordering'],original['fuzzy_clusters'],
                                           weights,original['stabilities'],12,original['memberships_flat'].shape[1])
            for key,expected in zip(['memberships','memberships_flat','fuzzy_hierarchy'],values):
                np.testing.assert_allclose(corrected[key],expected,atol=1e-12,rtol=1e-12)
            labels,maximum=display_memberships(values[1])
            np.testing.assert_array_equal(labels,corrected['display_labels'])
            np.testing.assert_allclose(maximum,corrected['maximum_membership'],atol=1e-12,rtol=0)
            diff=np.abs(values[1]-original['memberships_flat'])
            largest=float(diff.max()) if diff.size else 0.
            assert largest==float(row['flat_max_absolute_difference'])
            count=int((diff.max(axis=0)>1e-12).sum()) if len(diff) else 0
            assert count==int(row['changed_samples'])
            assert int((labels>=0).sum())==int(row['new_display_assigned'])
            assert int((maximum>0).sum())==int(row['new_nonzero_samples'])
            assert np.array_equal(labels,original['display_labels'])==(row['display_labels_exact']=='True')
            if largest>1e-12:changed.append(name)
    assert changed==['event0008_o7_hierarchy']
    control=json.loads((directory/'weight_control.json').read_text())
    assert control['experimental_weights']==[[.5],[.5]] and control['original_weights']==[[0],[0]]
    result=dict(readback_checks_passed=True,selections=38,group_selection_and_orderings_unchanged=True,
                independent_float_membership_reference_matches=True,changed_selections=changed,
                corrected_reciprocal_control_passed=True,physical_identity_validation=False,
                scope='Validates the isolated numerical correction on saved cases; does not validate temporal tracking.')
    (directory/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    verify(p.parse_args().directory)
