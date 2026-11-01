"""Verify imported temporal products from the repository root, using NumPy only."""
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
assert all(remote[key] for key in ('passed', 'tracking_matches', 'energy_matches',
                                  'offline_batch_energy_matches', 'sparse_surfaces_match',
                                  'final_gas_state_matches'))
source_checks = {name: hashlib.sha256((repo/name).read_bytes()).hexdigest() == expected
                 for name, expected in metadata['source_sha256'].items()}
assert all(source_checks.values()), source_checks
np.testing.assert_array_equal(metadata['requested_times'], metadata['actual_times'])

def read_rows(mode, name):
    with (root/mode/f'{name}.csv').open() as stream:
        return list(csv.DictReader(stream))

event_count = len(metadata['actual_times'])
assert remote['compared_events'] == event_count
csv_exact = {}
for name, expected_count in [('history', event_count), ('tracks', 2*event_count), ('energy', event_count)]:
    a, b = read_rows('online', name), read_rows('offline_reference', name)
    assert len(a) == len(b) == expected_count
    csv_exact[name] = a == b
    for x, y in zip(a, b):
        assert x.keys() == y.keys()
        for key in x:
            try:
                first, second = float(x[key]), float(y[key])
            except ValueError:
                assert x[key] == y[key], (name, key, x[key], y[key])
            else:
                np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-8, equal_nan=True)

histories = read_rows('online', 'history')
tracks = read_rows('online', 'tracks')
energies = read_rows('online', 'energy')
assert len(histories[0]) == remote['fields_per_event'] == 41
counts = []
sparse_exact = True
for i in range(event_count):
    with np.load(root/f'online/surfaces_{i:04d}.npz',allow_pickle=False) as a, np.load(root/f'offline_reference/surfaces_{i:04d}.npz',allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        counts.append(len(a['positions']))
        assert counts[-1] == int(histories[i]['surface_cells_outside_injection'])
        assert a['positions'].shape == a['normals'].shape == a['grid_indices'].shape == (counts[-1], 3)
        assert np.all(a['radii'] > float(a['injection_radius']))
        for key in a.files:
            x, y = a[key], b[key]
            if x.dtype.kind == 'f':
                np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-8,equal_nan=True)
                sparse_exact &= np.array_equal(x,y,equal_nan=True)
            else:
                np.testing.assert_array_equal(x,y)
    with np.load(root/f'online/histograms_{i:04d}.npz') as a, np.load(root/f'offline_reference/histograms_{i:04d}.npz') as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key],b[key])
    for mode in ('online', 'offline_reference'):
        with np.load(root/f'{mode}/surfaces_{i:04d}.npz') as cat, np.load(root/f'{mode}/histograms_{i:04d}.npz') as hist:
            for field, valid, edges, counts_key in [('mach','mach_valid','mach_edges','mach_counts'),('density_jump','density_jump_valid','density_jump_edges','density_jump_counts')]:
                values = cat[field][cat[valid]]
                values = values[np.isfinite(values) & (values > 0)]
                np.testing.assert_array_equal(np.histogram(values,bins=hist[edges])[0],hist[counts_key])

# Check the saved energy CSV independently of the streaming writer's objects.
for kind in ('reverse', 'forward'):
    previous_time, previous_rate, total = None, np.nan, np.nan
    duration = 0.
    for history, row in zip(histories, energies):
        time, rate = float(history['time']), float(history[f'{kind}_dissipation_rate'])
        valid_interval = False
        if np.isfinite(rate):
            if not np.isfinite(total):
                total = 0.
            elif np.isfinite(previous_rate):
                total += .5*(previous_rate+rate)*(time-previous_time)
                duration += time-previous_time
                valid_interval = True
        np.testing.assert_allclose(float(row[f'{kind}_cumulative_dissipated_energy']),total,
                                   rtol=1e-6,atol=1e-8,equal_nan=True)
        np.testing.assert_allclose(float(row[f'{kind}_integrated_duration']), duration,rtol=1e-6,atol=1e-8)
        assert row[f'{kind}_integration_interval_valid'] == str(valid_interval)
        previous_time, previous_rate = time, rate
for row in energies:
    values = np.array([float(row[f'{kind}_cumulative_dissipated_energy']) for kind in ('reverse','forward')])
    expected = np.nansum(values) if np.isfinite(values).any() else np.nan
    np.testing.assert_allclose(float(row['combined_cumulative_dissipated_energy']),expected,
                               rtol=1e-6,atol=1e-8,equal_nan=True)
assert not list(root.rglob('*.npy'))
report = dict(passed=True,source_hashes_match=source_checks,
              online_offline_csv_values_exact=csv_exact,catalog_arrays_exact=bool(sparse_exact),
              histogram_arrays_exact=True,histograms_reconstructed_from_catalogs_exact=True,
              energy_recomputed_from_history_matches=True,compared_events=event_count,track_rows=2*event_count,
              surface_samples=counts,
              forward_statuses=[r['track_status'] for r in tracks if r['shock_kind']=='forward'],
              reverse_statuses=[r['track_status'] for r in tracks if r['shock_kind']=='reverse'],
              final_forward_cumulative_energy=float(energies[-1]['forward_cumulative_dissipated_energy']),
              forward_first_power_time=float(energies[-1]['forward_first_power_detection_time']),
              forward_integrated_duration=float(energies[-1]['forward_integrated_duration']),
              remote_final_gas_state_matches=remote['final_gas_state_matches'],
              final_state_rechecked_locally=False)
report['band_summaries'] = {}
for kind in ('reverse', 'forward'):
    detected = [r for r in tracks if r['shock_kind'] == kind and r['detected'] == 'True']
    report['band_summaries'][kind] = dict(
        detected_events=len(detected),
        first_detection_time=float(detected[0]['time']) if detected else None,
        final_radius=float(detected[-1]['radius_median']) if detected else None,
        all_detected_continuity_ok=all(r['continuity_ok'] == 'True' for r in detected),
        final_cumulative_energy=float(energies[-1][f'{kind}_cumulative_dissipated_energy']),
        first_power_time=float(energies[-1][f'{kind}_first_power_detection_time']),
        integrated_duration=float(energies[-1][f'{kind}_integrated_duration']))
(root/'local_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
