import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.wind_bubble.compare_collision_jump_screening import (
    SOURCES, crop_layout, native_axis_indices, pressure_features, region_summary,
    validate_archive, sha)
from experiments.wind_bubble.classify_saved_collision_frames import (
    classification, choose_observation, feature_evidence)


class CollisionJumpComparisonTests(unittest.TestCase):
    def test_failed_outer_controls_never_produce_a_positive_or_negative_physical_verdict(self):
        self.assertEqual(classification(False, True, True, True, False), 'ambiguous')
        self.assertEqual(classification(False, True, True, False, True), 'ambiguous')
        self.assertEqual(classification(True, True, True, True, False), 'resolved shock')
        self.assertEqual(classification(True, True, True, False, True), 'rarefaction/contact-compatible')

    def test_unmatched_evolution_or_unbracketed_time_forces_ambiguity(self):
        self.assertEqual(classification(True, False, True, True, False), 'ambiguous')
        self.assertEqual(classification(True, True, False, True, False), 'ambiguous')

    def test_clean_outer_controls_use_outermost_features_not_most_favorable_jump(self):
        observations = [dict(event=80, x=x, region='outer_control') for x in (.2,.3,.7,.8)]
        self.assertEqual(choose_observation(observations, [], 80, 'left', outer=True)['x'], .2)
        self.assertEqual(choose_observation(observations, [], 80, 'right', outer=True)['x'], .8)

    def test_final_time_is_not_assigned_a_one_sided_motion_fit(self):
        track = [dict(event=i,time=i*.001,x=.48-i*.0001,track=0) for i in range(20)]
        observation = dict(event=19,track=0,x=track[-1]['x'],normal_x=1.)
        result = feature_evidence(observation, [], [track], np.arange(20)*.001, 1/128)
        self.assertFalse(result['bracketed'])
        self.assertIsNone(result['measured_x_speed'])
        self.assertIsNone(result['shock_frame_mach_lower'])

    def test_crop_retains_exact_original_centers_and_grid_spacing(self):
        for n in (64, 128):
            crop, target, centers = crop_layout(n, 24)
            axis = (np.arange(n)+.5)/n
            full = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1)
            np.testing.assert_array_equal(full[crop][target], centers)
            np.testing.assert_allclose(np.diff(centers[:, 0, 0, 0]), 1/n, rtol=0, atol=0)
            j, k = native_axis_indices(centers, n)
            self.assertEqual(centers[0, j, k, 2], .5+.5/n)
            self.assertLessEqual(abs(centers[0, j, k, 1]-.375), .5/n)
        with self.assertRaises(ValueError):
            crop_layout(64, -1)

    def test_separate_compression_alignment_passes_do_not_create_joint_candidates(self):
        shape = (2, 2, 2); zero = np.zeros(shape, bool)
        keys = ('compression', 'gradient_alignment', 'baseline_zone', 'transition_zone',
            'baseline_exported', 'transition_exported', 'transition_raw_surface',
            'transition_mach_pass', 'transition_entropy_pass', 'transition_sampling_valid',
            'transition_upstream_inside_source', 'transition_downstream_inside_source')
        fields = {k:zero.copy() for k in keys}
        fields['compression'][0, 0, 0] = True
        fields['gradient_alignment'][1, 0, 0] = True
        result = region_summary(fields, np.ones(shape, bool))
        self.assertEqual(result['compression'], 1)
        self.assertEqual(result['aligned'], 1)
        self.assertEqual(result['compressive_and_aligned'], 0)

    def test_added_removed_and_source_overlap_are_sample_counts(self):
        shape = (2, 2, 2); zero = np.zeros(shape, bool)
        keys = ('compression', 'gradient_alignment', 'baseline_zone', 'transition_zone',
            'baseline_exported', 'transition_exported', 'transition_raw_surface',
            'transition_mach_pass', 'transition_entropy_pass', 'transition_sampling_valid',
            'transition_upstream_inside_source', 'transition_downstream_inside_source')
        fields = {k:zero.copy() for k in keys}
        fields['baseline_exported'].flat[[0, 1]] = True
        fields['transition_exported'].flat[[1, 2, 3]] = True
        fields['transition_upstream_inside_source'].flat[2] = True
        result = region_summary(fields, np.ones(shape, bool))
        self.assertEqual(result['added_surface_cells'], 2)
        self.assertEqual(result['removed_surface_cells'], 1)
        self.assertEqual(result['added_with_upstream_inside_source'], 1)

    def test_pressure_features_are_retained_even_without_any_shock_claim(self):
        n = 128; _, _, centers = crop_layout(n, 24); shape = centers.shape[:-1]
        x = centers[..., 0]
        p = 1+3*np.exp(-((x-.5)/.025)**2)
        state = np.zeros((5,)+shape); state[0] = 1.; state[4] = p
        fields = dict(centers=centers, state=state)
        for name in ('compression', 'gradient_alignment', 'old_jump', 'extra_jump',
                     'baseline_zone', 'transition_zone', 'baseline_exported', 'transition_exported'):
            fields[name] = np.zeros(shape, bool)
        for name in ('gradient_dot','temperature_gradient_x','density_gradient_x','normal_x'):
            fields[name] = np.zeros(shape)
        rows = pressure_features(fields, n)
        self.assertEqual({r['side'] for r in rows}, {'left', 'right'})
        self.assertFalse(any(r['transition_exported_at_feature'] for r in rows))
        self.assertFalse(any('shock_certified' in r for r in rows))

    def test_saved_state_fingerprint_is_required_but_source_changes_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'audit').mkdir(); (root/'online').mkdir()
            state_path = root/'audit/state_0004.npy'
            np.save(state_path, np.ones((5,64,64,64), np.float32))
            source = root/'detector.py'; source.write_text('current detector\n')
            meta = dict(status='complete', backend='gpu', num_cells=64, source_positions=SOURCES,
                source_strengths=[1.,1.,1.], gamma=5/3, injection_radius=.0625)
            report = dict(passed=True, captured_events=[4], state_sha256={str(state_path):sha(state_path)},
                source_sha256={str(source):'0'*64})
            (root/'metadata.json').write_text(json.dumps(meta))
            (root/'audit/verification.json').write_text(json.dumps(report))
            np.savez(root/'audit/collision_crop_0004.npz', test=True)
            np.savez(root/'online/surfaces_0004.npz', test=True)
            _, inputs, changes = validate_archive(root, events=(4,))
            self.assertIn(str(state_path), inputs)
            self.assertIn(str(source), changes)
            np.save(state_path, np.full((5,64,64,64), 2., np.float32))
            with self.assertRaisesRegex(ValueError, 'changed'):
                validate_archive(root, events=(4,))


if __name__ == '__main__':
    unittest.main()
