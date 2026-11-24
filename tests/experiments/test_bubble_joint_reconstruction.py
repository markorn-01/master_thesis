import unittest

import numpy as np

from experiments.wind_bubble.check_bubble_joint_reconstruction import (
    allowed_cells, choose_candidate, guarded_window, interpretation,
)


SOURCES = np.array([[.375, .375, .5], [.625, .375, .5], [.5, .625, .375]])


class BubbleJointReconstructionTests(unittest.TestCase):
    def test_native_central_ray_has_only_eight_guarded_cells_per_side(self):
        for region, index in [('central_left', 62), ('central_right', 65)]:
            mask = allowed_cells(128, .37109375, .49609375, SOURCES, .0625, region)
            self.assertEqual(int(mask.sum()), 8)
            window, available = guarded_window(index, mask)
            self.assertIsNone(window)
            self.assertEqual(available, 8)

    def test_minimum_length_is_a_prototype_requirement_not_physical_no_shock(self):
        mask = allowed_cells(256, .375-1/512, .5-1/512, SOURCES, .0625, 'central_left')
        window, available = guarded_window(124, mask)
        self.assertEqual(available, 16)
        self.assertEqual(window, (112, 128))

    def test_full_cell_source_guard_excludes_a_cell_even_when_its_center_is_outside(self):
        n = 128; dx = 1/n
        # Shift the source so that its left spherical edge lies inside cell 39,
        # although that cell center is outside the sphere.
        sources = SOURCES.copy(); sources[0, 0] -= .002
        center = (39+.5)*dx
        self.assertGreater(abs(center-sources[0, 0]), .0625)
        mask = allowed_cells(n, .375, .5, sources, .0625, 'outer_left')
        self.assertFalse(mask[39])
        self.assertTrue(mask[38])

    def test_window_clamps_without_crossing_source_or_domain_boundary(self):
        mask = allowed_cells(128, .37109375, .49609375, SOURCES, .0625, 'outer_left')
        for candidate in (3, 21, 38):
            window, available = guarded_window(candidate, mask)
            self.assertEqual(available, 39)
            first, stop = window
            self.assertEqual(stop-first, 16)
            self.assertTrue(mask[first:stop].all())
            self.assertLessEqual(first, candidate)
            self.assertLess(candidate, stop)
        window, available = guarded_window(44, mask)
        self.assertIsNone(window)
        self.assertEqual(available, 0)

    def test_geometry_selection_uses_outermost_control_and_nearest_central_front(self):
        ridges = [dict(index=i, x=(i+.5)/128) for i in (20, 30, 56, 62, 65, 71, 98, 106)]
        expected = dict(outer_left=20, outer_right=106, central_left=62, central_right=65)
        for region, index in expected.items():
            mask = allowed_cells(128, .37109375, .49609375, SOURCES, .0625, region)
            self.assertEqual(choose_candidate(ridges, mask, region)['index'], index)

    def test_failed_control_and_unavailable_windows_never_become_absence_proof(self):
        rows = [dict(region='outer_left', fit_attempted=True,
            joint_reconstruction_supported=False, shock_model_supported=False, status='model_mismatch'),
            dict(region='central_left', fit_attempted=False,
                joint_reconstruction_supported=False, shock_model_supported=False, status='insufficient_guarded_cells')]
        result = interpretation(rows)
        self.assertFalse(result['outer_positive_control_passes'])
        self.assertFalse(result['physical_shock_absence_established'])
        self.assertFalse(result['missing_bubble_fronts_recovered'])
        self.assertFalse(result['production_promotion_allowed'])
        self.assertEqual(result['central_short_windows'], 1)

    def test_isolated_shock_can_supply_control_without_inventing_contact(self):
        rows = [dict(region='outer_left', fit_attempted=True,
            joint_reconstruction_supported=False, shock_model_supported=True, status='shock')]
        result = interpretation(rows)
        self.assertTrue(result['outer_positive_control_passes'])
        self.assertEqual(result['outer_supported_joint_pairs'], 0)
        self.assertEqual(result['outer_supported_shock_models'], 1)
        self.assertFalse(result['missing_bubble_fronts_recovered'])


if __name__ == '__main__':
    unittest.main()
