"""Recheck saved GPU products locally, without a solver or JAX dependency."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

root = Path(__file__).resolve().parent
repo = root.parents[1]
metadata = json.loads((root/'metadata.json').read_text())
remote = json.loads((root/'equivalence.json').read_text())
assert metadata['status'] == 'complete' and metadata['backend'] == 'gpu'
assert remote['passed'] and remote['compared_events'] == 17
for name, expected in metadata['source_sha256'].items():
    path = Path(name)
    if path.is_absolute():
        path = path.relative_to('/export/home/mngo/master_thesis')
    assert hashlib.sha256((repo/path).read_bytes()).hexdigest() == expected, name
np.testing.assert_allclose(metadata['requested_times'], metadata['actual_times'], rtol=0, atol=1e-6)

def rows(mode):
    with (root/mode/'history.csv').open() as stream:
        return list(csv.DictReader(stream))

online = rows('online')
assert online == rows('offline_reference')
assert len(online) == 17
sources = np.asarray(metadata['source_positions'])
radius = metadata['injection_radius']
dx = 1/metadata['num_cells']
clearance = []
for i, row in enumerate(online):
    for prefix in ('surfaces', 'histograms', 'slice'):
        name = f'{prefix}_{i:04d}.npz'
        with np.load(root/'online'/name) as a, np.load(root/'offline_reference'/name) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_equal(a[key], b[key], err_msg=f'{name}: {key}')
            np.testing.assert_allclose(float(a['time']), float(row['time']), rtol=0, atol=1e-7)
    with np.load(root/'online'/f'surfaces_{i:04d}.npz') as s, np.load(root/'online'/f'histograms_{i:04d}.npz') as h:
        assert int(s['schema_version']) == 3
        pos = s['positions']
        distances = np.linalg.norm(pos[:,None,:]-sources[None,:,:], axis=-1)
        np.testing.assert_array_equal(s['source_distances'], distances)
        assert np.isfinite(pos).all() and (distances > radius).all()
        assert len(pos) == int(row['surface_cells_outside_injection'])
        for side in ('upstream', 'downstream'):
            distance = np.linalg.norm(s[f'{side}_positions'][:,None,:]-sources[None,:,:], axis=-1)
            flags = s['sample_positions_valid'][:,None] & (distance <= radius)
            np.testing.assert_array_equal(s[f'{side}_inside_sources'], flags)
            np.testing.assert_array_equal(s[f'{side}_inside_any_source'], flags.any(axis=1))
            assert int(flags.any(axis=1).sum()) == int(row[f'{side}_inside_any_source'])
        assert int((~s['sample_positions_valid']).sum()) == int(row['sample_positions_unknown'])
        for field, name, validity in [('mach','mach','mach_valid'), ('density_jump','density_jump','density_jump_valid')]:
            values = s[field]
            valid = np.isfinite(values) & (values > 0)
            np.testing.assert_array_equal(s[validity], valid)
            counts, _ = np.histogram(values[valid], bins=h[f'{name}_edges'])
            np.testing.assert_array_equal(counts, h[f'{name}_counts'])
            accepted = values[valid]
            percentiles = np.percentile(accepted,[16,50,84]) if len(accepted) else [np.nan]*3
            np.testing.assert_allclose(percentiles, [float(row[f'{name}_{k}']) for k in ('p16','median','p84')], rtol=1e-6, atol=1e-8, equal_nan=True)
        if len(pos):
            distance = float(np.minimum(pos,1-pos).min())
            clearance.append(distance)
            np.testing.assert_allclose(distance, float(row['minimum_surface_boundary_clearance']), rtol=0, atol=0)
    with np.load(root/'online'/f'slice_{i:04d}.npz') as s:
        for key in ('density','pressure'):
            assert s[key].shape == (64,64) and np.isfinite(s[key]).all() and (s[key] > 0).all()
np.testing.assert_array_equal(metadata['injection_cell_counts'], [176,176])
np.testing.assert_allclose(metadata['discrete_energy_rates'][0], metadata['discrete_energy_rates'][1], rtol=0, atol=0)
report = dict(passed=True, source_hashes_match=True, events=17, history_fields=len(online[0]),
              saved_online_offline_csv_and_npz_exact=True,
              histograms_percentiles_source_selection_and_endpoint_flags_recomputed=True,
              minimum_surface_boundary_clearance=min(clearance),
              minimum_surface_boundary_clearance_cells=min(clearance)/dx,
              final_surface_samples=int(online[-1]['surface_cells_outside_injection']),
              remote_final_gas_state_matches=remote['final_gas_state_matches'],
              full_states_rechecked_locally=False,
              validation_scope='GPU workflow equivalence and local readback, not physical convergence')
(root/'local_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
