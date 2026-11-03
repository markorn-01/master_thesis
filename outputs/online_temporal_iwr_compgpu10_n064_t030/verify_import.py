"""Read back copied t=0.30 diagnostics and animations; no simulation required."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from experiments.wind_bubble.animate_shock_distributions import load_distribution_events
from experiments.wind_bubble.animate_shock_histories import load_histories

run = Path(__file__).resolve().parent
repo = run.parents[1]
metadata = json.loads((run / 'metadata.json').read_text())
assert metadata['status'] == 'complete' and metadata['backend'] == 'gpu'
for name, expected in metadata['source_sha256'].items():
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == expected, name
np.testing.assert_allclose(metadata['requested_times'], metadata['actual_times'], rtol=1e-6, atol=1e-8)
remote = json.loads((run / 'equivalence.json').read_text())
assert remote['passed'] and remote['compared_events'] == 31

def rows(name, mode='online'):
    with (run / mode / f'{name}.csv').open() as stream:
        return list(csv.DictReader(stream))

for name in ('history', 'tracks', 'energy', 'sampling_quality'):
    assert rows(name) == rows(name, 'offline_reference'), name
clearance = []
for index in range(31):
    for prefix in ('surfaces', 'histograms'):
        name = f'{prefix}_{index:04d}.npz'
        with np.load(run / 'online' / name) as a, np.load(run / 'offline_reference' / name) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_equal(a[key], b[key])
            if prefix == 'surfaces' and len(a['positions']):
                clearance.append(float(np.minimum(a['positions'], 1-a['positions']).min()))
assert len(load_distribution_events(run / 'online')) == 31
load_histories(run / 'online')  # Independently checks finite-endpoint energy accumulation.
animations = {}
for kind, gif_name in [('distributions', 'shock_distributions.gif'),
                       ('histories', 'shock_histories.gif'), ('evolution', 'shock_evolution.gif')]:
    folder = repo / 'outputs' / f'single_bubble_{kind}_iwr_compgpu10_n064_t030'
    rendering = json.loads((folder / 'render_metadata.json').read_text())
    # Paths in remote rendering records are relative to the repository root.
    for name, expected in rendering['input_sha256'].items():
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == expected, name
    durations = []
    with Image.open(folder / gif_name) as gif:
        assert gif.n_frames == 31
        for i in range(gif.n_frames):
            gif.seek(i)
            gif.load()
            durations.append(gif.info['duration'])
    assert durations == [500]*31
    animations[kind] = dict(decoded_frames=31, total_duration_ms=sum(durations), input_hashes_match=True)
    (folder / 'verification.json').write_text(json.dumps(dict(passed=True, **animations[kind]), indent=2)+'\n')
tracks = rows('tracks')
report = dict(passed=True, source_hashes_match=True, saved_online_offline_products_exact=True,
              histogram_reconstruction_and_energy_checks_passed=True,
              minimum_surface_boundary_clearance=min(clearance),
              minimum_surface_boundary_clearance_cells=64*min(clearance),
              tracking_status_counts={kind: dict(Counter(r['track_status'] for r in tracks if r['shock_kind']==kind))
                                      for kind in ('forward', 'reverse')},
              final_sampling_quality=rows('sampling_quality')[-1], animations=animations,
              remote_final_gas_state_matches=remote['final_gas_state_matches'], full_states_rechecked_locally=False)
(run / 'local_checks.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
