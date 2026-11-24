import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.wind_bubble.compare_front_motion_resolution import (
    load_checked_run, physical_distances, validate_matching_setup,
)


class FrontMotionResolutionComparisonTests(unittest.TestCase):
    def test_endpoint_distance_control_is_equal_in_physical_units(self):
        np.testing.assert_array_equal(physical_distances(128)/128, physical_distances(256)/256)

    def test_setup_check_rejects_changed_source_budget_or_schedule(self):
        base = dict(source_positions=[[.375, .375, .5]], source_strengths=[1.],
            injection_radius=.0625, gamma=5/3, solver_dt_max=.01,
            requested_times=[0., .01], target_y=.375, target_z=.5,
            discrete_mass_rates=[.01], discrete_energy_rates=[.5], source_sha256={})
        self.assertEqual(validate_matching_setup(base, copy.deepcopy(base)), [])
        for key, value in (('discrete_energy_rates', [.4]), ('requested_times', [0., .02])):
            other = copy.deepcopy(base); other[key] = value
            with self.assertRaises(AssertionError):
                validate_matching_setup(base, other)

    def test_hydrodynamic_code_changes_cannot_be_called_only_resolution_changes(self):
        base = dict(source_positions=[], source_strengths=[], injection_radius=.0625,
            gamma=5/3, solver_dt_max=.01, requested_times=[0., .01], target_y=.375,
            target_z=.5, discrete_mass_rates=[.01], discrete_energy_rates=[.5],
            source_sha256={'astronomix/time_stepping/time_integration.py': 'old'})
        other = copy.deepcopy(base)
        other['source_sha256']['astronomix/time_stepping/time_integration.py'] = 'new'
        with self.assertRaisesRegex(ValueError, 'Hydrodynamic source fingerprints differ'):
            validate_matching_setup(base, other)

    def make_archive(self, root):
        (root/'strips').mkdir()
        q = np.ones((5, 16, 4, 4), dtype=np.float32)
        hashes = {}
        for i, time in enumerate((0., .01, .02)):
            p = root/'strips'/f'strip_{i:04d}.npz'
            np.savez_compressed(p, primitive=q, time=time)
            hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        meta = dict(status='complete', num_cells=16, actual_times=[0., .01, .02],
            requested_times=[0., .01, .02], stage_names=[], strip_sha256=hashes,
            target_y=.375, target_z=.5,
            transverse_y=(.375+np.array([-1.5, -.5, .5, 1.5])/16).tolist(),
            transverse_z=(.5+np.array([-1.5, -.5, .5, 1.5])/16).tolist())
        (root/'metadata.json').write_text(json.dumps(meta))

    def test_changed_copied_strip_is_rejected_before_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.make_archive(root)
            result = load_checked_run(root)
            self.assertEqual(result['lines'].shape, (3, 5, 16))
            path = root/'strips/strip_0001.npz'
            path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError, 'Strip hash mismatch'):
                load_checked_run(root)

    def test_unexpected_or_duplicate_capture_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.make_archive(root)
            (root/'strips/strip_0001_copy.npz').write_bytes((root/'strips/strip_0001.npz').read_bytes())
            with self.assertRaisesRegex(ValueError, 'file set differs'):
                load_checked_run(root)


if __name__ == '__main__':
    unittest.main()
