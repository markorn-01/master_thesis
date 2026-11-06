"""Read back contact-study archives and check their mappings and summaries."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score
from experiments.astrolink.shock_connectivity import connectivity_partition


def verify(directory):
    metadata = json.loads((directory / 'metadata.json').read_text())
    for name, digest in metadata['input_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    # Source paths may be absolute in the original machine's provenance record.
    for name, digest in metadata['source_sha256'].items():
        path = Path(name)
        if not path.exists() and '/experiments/' in name:
            path = Path('experiments') / name.split('/experiments/', 1)[1]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    rows = list(csv.DictReader((directory / 'summary.csv').open()))
    stability = list(csv.DictReader((directory / 'order_stability.csv').open()))
    assert len(rows) == 336 and len(stability) == 336
    assert sum(r['algorithm'] == 'astrolink' for r in rows) == 252
    warnings = []
    for event in metadata['events']:
        source = next(Path(p) for p in metadata['input_sha256'] if p.endswith(f'surfaces_{event:04d}.npz'))
        with np.load(source) as original, np.load(directory / f'event_{event:04d}.npz') as saved:
            p, n = saved['positions'], saved['normals']
            np.testing.assert_array_equal(p, original['positions'])
            np.testing.assert_array_equal(n, original['normals'])
            np.testing.assert_array_equal(saved['sources'], original['source_positions'])
            assert float(saved['time']) == float(original['time'])
            eligible = saved['eligible_rows']
            for row in (r for r in rows if int(r['event_index']) == event):
                prefix = f"{row['config']}_o{row['order_seed']}__"
                labels = saved[prefix + 'labels']
                assert len(labels) == len(p)
                ids, counts = np.unique(labels[eligible][labels[eligible] >= 0], return_counts=True)
                assert len(ids) == int(row['display_groups'])
                assert np.mean(labels[eligible] >= 0) == float(row['assigned_fraction'])
                assert max(counts, default=0) / len(eligible) == float(row['largest_display_group_fraction'])
                if row['algorithm'] == 'astrolink':
                    ordering = saved[prefix + 'ordering_catalog']
                    np.testing.assert_array_equal(np.sort(ordering), eligible)
                    reconstructed = np.full(len(p), -2, dtype=int)
                    reconstructed[eligible] = -1
                    intervals, hierarchy_ids = saved[prefix + 'intervals'], saved[prefix + 'ids']
                    # Same declared deepest-descendant display rule; original row IDs.
                    for j in sorted(range(len(hierarchy_ids)), key=lambda j: len(str(hierarchy_ids[j]))):
                        start, end = map(int, intervals[j])
                        assert 0 <= start <= end <= len(eligible)
                        if str(hierarchy_ids[j]) != '1':
                            reconstructed[ordering[start:end]] = j
                    np.testing.assert_array_equal(reconstructed, labels)
                    message = str(saved[prefix + 'messages'])
                    assert message == row['library_messages']
                    assert ('failed to converge' in message) == (row['model_fit_warning'] == 'True')
                    if row['model_fit_warning'] == 'True':
                        warnings.append(dict(event=event, config=row['config'], order_seed=int(row['order_seed'])))
                else:
                    seed = int(row['order_seed'])
                    perm = np.arange(len(p)) if seed == 0 else np.random.default_rng(seed).permutation(len(p))
                    graph = connectivity_partition(p[perm], n[perm], float(saved['grid_spacing']),
                                                   float(row['radius_cells']), float(row['angle_degrees']), 20)
                    np.testing.assert_array_equal(graph['labels'], labels[perm])
                    np.testing.assert_array_equal(graph['raw_labels'], saved[prefix + 'raw_labels'][perm])
                    assert len(graph['edges']) == int(row['edges'])
                    assert len(graph['component_sizes']) == int(row['raw_graph_components'])
            for row in (r for r in stability if int(r['event_index']) == event):
                a = saved[f"{row['config']}_o{row['order_a']}__labels"][eligible]
                b = saved[f"{row['config']}_o{row['order_b']}__labels"][eligible]
                joint = (a >= 0) & (b >= 0)
                assert adjusted_rand_score(a, b) == float(row['adjusted_rand_all'])
                assert np.mean(joint) == float(row['jointly_assigned_fraction'])
                expected = adjusted_rand_score(a[joint], b[joint]) if joint.sum() >= 2 else np.nan
                np.testing.assert_equal(expected, float(row['adjusted_rand_joint']))
    result = dict(passed=True, checked_runs=len(rows), checked_order_comparisons=len(stability),
                  inputs_and_analysis_sources_unchanged=True, graph_partitions_recomputed=True,
                  astrolink_display_mappings_reconstructed=True, model_fit_warnings=warnings,
                  scope='Data integrity and partition bookkeeping; not physical validation or a fresh AstroLink rerun.')
    (directory / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    verify(parser.parse_args().directory)
