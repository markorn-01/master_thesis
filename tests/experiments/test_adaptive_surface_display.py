"""Check extraction accuracy on translated/rotated sheets not in the suite."""
import unittest

import numpy as np

from experiments.wind_bubble.adaptive_surface_display import refine_display, distances_to_mesh
from experiments.wind_bubble.implicit_surface_display import implicit_samples
from experiments.wind_bubble.finalize_surface_display import remove_defects


class AdaptiveDisplayTests(unittest.TestCase):
    def test_shifted_and_rotated_planar_boundaries_meet_fixed_accuracy(self):
        base = np.array([[x, y, 0.] for x in range(5) for y in range(5)])
        angle = np.deg2rad(27)
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                             [-np.sin(angle), 0., np.cos(angle)]])
        for shift, matrix in [([.17, .29, .11], np.eye(3)), ([.13, .23, .19], rotation)]:
            with self.subTest(shift=shift):
                p = base@matrix.T+shift; n = np.tile(matrix@[0., 0., 1.], (len(p), 1))
                labels = np.zeros(len(p), int)
                v, f, g, _ = implicit_samples(p, n, labels, 1.)
                v, f, g, _ = remove_defects(v, f, g)
                v, f, g, report = refine_display(p, n, labels, 1., v, f, g)
                self.assertTrue(report['all_group_accuracy_targets_met'])
                self.assertGreaterEqual(float((distances_to_mesh(p, v, f) <= .25).mean()), .95)
                normal_offsets = (v-shift)@(matrix@[0., 0., 1.])
                np.testing.assert_allclose(normal_offsets, 0., atol=2e-6)


if __name__ == '__main__':
    unittest.main()
