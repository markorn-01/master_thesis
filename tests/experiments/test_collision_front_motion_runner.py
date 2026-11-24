import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace

import numpy as np

from experiments.wind_bubble.run_collision_front_motion import (
    REFERENCE, STAGES, StripWriter, experiment_plan, strip_indices,
)
from experiments.wind_bubble.review_collision_front_motion import persistent_runs, production_comparison


class CollisionMotionRunnerTests(unittest.TestCase):
    def test_original_source_budgets_are_preserved_at_supported_grids(self):
        for n in (32, 64, 128, 256):
            reference, budget, times, _ = experiment_plan(REFERENCE, n, 321, .16)
            self.assertEqual(times.shape, (321,))
            np.testing.assert_allclose(budget['discrete_energy_rates'], reference['discrete_energy_rates'], rtol=1.e-12)
            np.testing.assert_allclose(budget['discrete_mass_rates'], reference['discrete_mass_rates'], rtol=1.e-12)
            self.assertEqual(times[160], np.float32(.08))
            self.assertEqual(times[240], np.float32(.12))
            for target in (.375, .5):
                coords = (strip_indices(n, target)+.5)/n
                self.assertLess(coords[0], target-1/n)
                self.assertGreater(coords[-1], target+1/n)

    def test_writer_checks_order_shape_and_domain_validity(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = StripWriter(Path(tmp)/'strips', [0., .01], 16)
            state = np.ones((5, 16, 4, 4), dtype=np.float32)
            stages = np.zeros((len(STAGES), 16, 4, 4), dtype=np.uint8)
            writer.receive(0., state, stages, True)
            self.assertEqual(len(writer.hashes), 1)
            with self.assertRaises(ValueError):
                writer.receive(0., state, stages, True)
            with self.assertRaises(ValueError):
                writer.receive(.01, state, stages, False)
            writer.receive(.01, state, stages, True)
            self.assertEqual(len(writer.actual), 2)

    def test_persistence_requires_consecutive_events_and_cannot_bridge_a_gap(self):
        rows = [dict(track=1, region='central_left', event=i, time=.06+.001*i, accepted=True)
                for i in (0, 1, 2, 4, 5, 6)]
        self.assertEqual(persistent_runs(rows, 'accepted'), [])
        rows += [dict(track=1, region='central_left', event=i, time=.06+.001*i, accepted=True)
                 for i in (7, 8)]
        self.assertEqual(len(persistent_runs(rows, 'accepted')), 1)

    def test_production_support_uses_the_requested_event_and_3d_distance(self):
        n = 32; x = (np.arange(n)+.5)/n
        y, z = (strip_indices(n, target)+.5 for target in (.375, .5))
        y, z = y/n, z/n
        masks = np.zeros((2, len(STAGES), n, 4, 4), bool)
        masks[1, STAGES.index('shock_surface'), 15, 1, 1] = True
        rows = [dict(event=i, x=x[15], compatible_10pct=True) for i in (0, 1)]
        production_comparison(rows, masks, x, y, z, STAGES)
        self.assertTrue(rows[0]['compatible_without_nearby_surface_10pct'])
        self.assertFalse(rows[1]['compatible_without_nearby_surface_10pct'])

    def test_analyze_only_refuses_partial_integration(self):
        from experiments.wind_bubble.review_collision_front_motion import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'metadata.json').write_text(json.dumps(dict(status='running')))
            with self.assertRaisesRegex(ValueError, 'completed strip simulation'):
                run(root)

    def test_analyze_only_does_not_reintegrate_or_allocate_a_new_problem(self):
        from experiments.wind_bubble.run_collision_front_motion import run
        root = Path('saved-simulation')
        output = Path('fresh-review')
        with patch('experiments.wind_bubble.review_collision_front_motion.run', return_value={'reviewed': True}) as review, \
             patch('experiments.wind_bubble.run_collision_front_motion.time_integration') as integrate, \
             patch('experiments.wind_bubble.run_collision_front_motion.build_problem') as build:
            result = run(Namespace(analyze_only=True, output_dir=root, analysis_dir=output))
        review.assert_called_once_with(root, output)
        integrate.assert_not_called(); build.assert_not_called()
        self.assertTrue(result['reviewed'])


if __name__ == '__main__':
    unittest.main()
