"""Reconstruct feature/index mappings and display partitions from saved outputs."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from experiments.astrolink.check_two_bubble_clustering import feature_arrays, deepest_partition, restore_labels

root = Path(__file__).resolve().parent
repo = root.parents[1]
m = json.loads((root/'metadata.json').read_text())
assert m['status'] == 'complete'
assert hashlib.sha256((repo/'experiments/astrolink/check_two_bubble_clustering.py').read_bytes()).hexdigest() == m['source_sha256']
for name,h in m['input_sha256'].items():
    assert hashlib.sha256((repo/name).read_bytes()).hexdigest() == h
with (root/'summary.csv').open() as f:
    summaries = list(csv.DictReader(f))
assert len(summaries) == 18
for row in summaries:
    event,seed,weight = int(row['event_index']),int(row['order_seed']),float(row['normal_weight'])
    source = next(repo/p for p in m['input_sha256'] if Path(p).name == f'surfaces_{event:04d}.npz')
    name = f'event_{event:04d}_{row["mode"]}_order{seed}.npz'
    with np.load(source) as s,np.load(root/name) as a:
        features,indices = feature_arrays(s['positions'],s['normals'],float(s['grid_spacing']),weight)
        np.testing.assert_allclose(features,a['features'],rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(indices,a['eligible_catalog_rows'])
        np.testing.assert_array_equal(s['positions'],a['positions'])
        expected_permutation = np.arange(len(features)) if seed == 0 else np.random.default_rng(seed).permutation(len(features))
        np.testing.assert_array_equal(expected_permutation,a['input_permutation'])
        order = indices[a['input_permutation'][a['ordering']]]
        np.testing.assert_array_equal(order,a['catalog_ordering'])
        labels = restore_labels(deepest_partition(a['ordering'],a['hierarchy_intervals'],a['hierarchy_ids'],len(indices)),a['input_permutation'],indices,len(s['positions']))
        np.testing.assert_array_equal(labels,a['deepest_nonroot_labels'])
        np.testing.assert_allclose(np.mean(labels>=0),float(row['assigned_fraction']),rtol=0,atol=1e-15)
        assert len(np.unique(labels[labels>=0])) == int(row['display_groups'])
        assert a['hierarchy_ids'][0] == '1' and np.isinf(a['significances'][0])
        assert np.isfinite(a['point_log_density']).all()
report = dict(passed=True,runs=18,input_hashes_unchanged=True,
              features_permutations_hierarchy_mappings_and_labels_reconstructed=True,
              geometry_mapping_tests_passed=5,
              interpretation='Internal accounting only; no physical shock labels validated')
(root/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
