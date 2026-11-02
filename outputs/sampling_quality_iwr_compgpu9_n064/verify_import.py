"""Check flagged GPU output, accounting and unchanged legacy products."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
root=Path(__file__).resolve().parent
repo=root.parents[1]
run=root
metadata=json.loads((run/'metadata.json').read_text())
assert metadata['status']=='complete' and metadata['backend']=='gpu'
for name,expected in metadata['source_sha256'].items():
    assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==expected,name

def rows(folder,name):
    with (folder/f'{name}.csv').open() as stream: return list(csv.DictReader(stream))

np.testing.assert_array_equal(metadata['requested_times'],metadata['actual_times'])
remote=json.loads((root/'equivalence.json').read_text())
assert all(remote[k] for k in ('passed','sampling_quality_matches','sparse_surfaces_match','tracking_matches','energy_matches','offline_batch_energy_matches','final_gas_state_matches'))
event_count=len(metadata['actual_times'])
before=repo/'outputs/online_temporal_iwr_compgpu3_n064_cadence21/online'
online=run/'online';offline=run/'offline_reference'
legacy={}
for name in ('history','tracks','energy'):
    actual=rows(online,name)
    assert actual==rows(offline,name)
    legacy[name]=actual==rows(before,name)
    assert legacy[name],name
quality=rows(online,'sampling_quality')
assert quality==rows(offline,'sampling_quality')
tracks=rows(online,'tracks')
assert len(quality)==event_count
for index,row in enumerate(quality):
    with np.load(online/f'surfaces_{index:04d}.npz') as cat, np.load(offline/f'surfaces_{index:04d}.npz') as other:
        assert int(cat['schema_version'])==2
        for key in cat.files: np.testing.assert_equal(cat[key],other[key])
        for side in ('upstream','downstream'):
            radii=np.linalg.norm(cat[f'{side}_positions']-.5,axis=1)
            np.testing.assert_allclose(cat[f'{side}_radii'],radii,rtol=0,atol=0,equal_nan=True)
            np.testing.assert_array_equal(cat[f'{side}_inside_injection'],cat['sample_positions_valid'] & (radii<=.0625))
        separation=float(next(r['separation_radius'] for r in tracks if int(r['event_index'])==index))
        reverse=cat['radii']<=separation if np.isfinite(separation) else np.zeros(len(cat['radii']),bool)
        for name,band in [('all',np.ones(len(reverse),bool)),('reverse',reverse),('forward',~reverse)]:
            known=band & cat['sample_positions_valid']
            assert int(row[f'{name}_surface_samples'])==band.sum()
            assert int(row[f'{name}_sample_positions_known'])==known.sum()
            assert int(row[f'{name}_sample_positions_unknown'])==(band & ~known).sum()
            assert int(row[f'{name}_sampling_valid'])==(band & cat['sampling_valid']).sum()
            for side in ('upstream','downstream'):
                assert int(row[f'{name}_{side}_inside_injection'])==(known & cat[f'{side}_inside_injection']).sum()
    with np.load(online/f'histograms_{index:04d}.npz') as a, np.load(before/f'histograms_{index:04d}.npz') as b:
        for key in a.files: np.testing.assert_array_equal(a[key],b[key])
report=dict(passed=True,backend='gpu',quality_rows_exact_online_offline=True,quality_events=event_count,
            sparse_arrays_exact_online_offline=True,geometry_and_quality_counts_recomputed=True,
            existing_history_tracks_energy_csv_exact=legacy,existing_histograms_exact=True,
            source_hashes_match_current=True,requested_actual_times_match=True,
            remote_final_gas_state_matches=remote['final_gas_state_matches'],full_states_rechecked=False)
(root/'local_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
