from pathlib import Path
import csv, hashlib, json
import numpy as np
root = Path('outputs/online_sparse_iwr_compgpu1_n032')
metadata = json.loads((root/'metadata.json').read_text())
remote = json.loads((root/'equivalence.json').read_text())
assert metadata['status'] == 'complete' and metadata['backend'] == 'gpu'
assert remote['passed'] and remote['sparse_surfaces_match'] and remote['final_gas_state_matches']
source_checks = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
                 for name, expected in metadata['source_sha256'].items()}
assert all(source_checks.values()), source_checks
with (root/'online/history.csv').open() as f:
    online_rows = list(csv.DictReader(f))
with (root/'offline_reference/history.csv').open() as f:
    offline_rows = list(csv.DictReader(f))
assert len(online_rows) == len(offline_rows) == remote['compared_events'] == 5
assert len(online_rows[0]) == remote['fields_per_event'] == 41
for actual, reference in zip(online_rows, offline_rows):
    assert actual.keys() == reference.keys()
    for key in actual:
        np.testing.assert_allclose(float(actual[key]), float(reference[key]),rtol=1e-6,atol=1e-8,equal_nan=True)
scalar_exact = all(all(a[k] == b[k] for k in a) for a,b in zip(online_rows,offline_rows))
field_max_abs = {}
catalogs_exact = True
events = []
for i in range(5):
    with np.load(root/f'online/surfaces_{i:04d}.npz',allow_pickle=False) as a, np.load(root/f'offline_reference/surfaces_{i:04d}.npz',allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        assert int(a['schema_version']) == int(b['schema_version']) == 1
        for key in a.files:
            x, y = a[key], b[key]
            if x.dtype.kind == 'f':
                np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-8,equal_nan=True,err_msg=f'{i} {key}')
                catalogs_exact &= np.array_equal(x,y,equal_nan=True)
                finite = np.isfinite(x) & np.isfinite(y)
                field_max_abs[key] = max(field_max_abs.get(key,0.),float(np.max(np.abs(x[finite]-y[finite]),initial=0.)))
            else:
                np.testing.assert_array_equal(x,y,err_msg=f'{i} {key}')
        samples = len(a['positions'])
        assert samples == int(online_rows[i]['surface_cells_outside_injection'])
        assert a['positions'].shape == a['normals'].shape == a['grid_indices'].shape == (samples,3)
        assert np.isfinite(a['positions']).all()
        assert np.all(a['radii'] > float(a['injection_radius']))
        np.testing.assert_allclose(np.linalg.norm(a['positions']-a['domain_center'],axis=-1),a['radii'],rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(a['time'],float(online_rows[i]['time']),rtol=0,atol=1e-8)
        event=dict(event=i,time=float(a['time']),surface_samples=samples,
                   invalid_mach=int((~a['mach_valid']).sum()),
                   online_catalog_bytes=(root/f'online/surfaces_{i:04d}.npz').stat().st_size)
        events.append(event)
    with np.load(root/f'online/histograms_{i:04d}.npz') as a,np.load(root/f'offline_reference/histograms_{i:04d}.npz') as b:
        assert set(a.files)==set(b.files)
        for key in a.files: np.testing.assert_array_equal(a[key],b[key])
    for mode in ('online','offline_reference'):
        with np.load(root/f'{mode}/surfaces_{i:04d}.npz') as cat,np.load(root/f'{mode}/histograms_{i:04d}.npz') as hist:
            for field, valid, edges, counts in [('mach','mach_valid','mach_edges','mach_counts'),('density_jump','density_jump_valid','density_jump_edges','density_jump_counts')]:
                vals=cat[field][cat[valid]]
                vals=vals[np.isfinite(vals)&(vals>0)]
                np.testing.assert_array_equal(np.histogram(vals,bins=hist[edges])[0],hist[counts])
assert not list(root.rglob('*.npy'))
report = dict(passed=True,recorded_source_hashes_match=source_checks,history_rows_exact=scalar_exact,
              histogram_arrays_exact=True,histograms_reconstructed_from_catalogs_exact=True,
              catalog_arrays_exact=bool(catalogs_exact),max_absolute_catalog_differences=field_max_abs,
              remote_final_gas_state_matches=remote['final_gas_state_matches'],
              final_state_rechecked_locally=False,events=events,
              online_catalog_total_bytes=sum(r['online_catalog_bytes'] for r in events))
(root/'local_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
