"""Temporal behavior and equivalence to the established offline energy rules."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from experiments.wind_bubble.online_temporal import (
    RadialShockTracker, CumulativeEnergy, TemporalDiagnostics, compare_records,
)
from experiments.wind_bubble.run_single_bubble import _cumulative_trapezoid_over_detections


def samples_at(radius=None, reverse_radius=None):
    radii = np.array([]) if radius is None else np.full(16, radius)
    signs = np.ones(len(radii))
    if reverse_radius is not None:
        radii = np.concatenate([np.full(16, reverse_radius), radii])
        signs = np.concatenate([-np.ones(16), signs])
    positions = np.full((len(radii), 3), .5)
    positions[:, 0] += radii
    normals = np.zeros_like(positions)
    normals[:, 0] = signs
    return dict(positions=positions, normals=normals, radii=radii,
                mach=np.full(len(radii), 5.), outside_injection=radii > .0625)


class OnlineTrackingTests(unittest.TestCase):
    def test_constant_speed_with_irregular_analysis_times(self):
        tracker = RadialShockTracker(.01, .0625)
        rows = [tracker.update(t, samples_at(.2+.3*t))[1] for t in [0., .05, .11, .25]]
        self.assertEqual(rows[0]['track_status'], 'initialized')
        self.assertTrue(np.isnan(rows[0]['radial_velocity']))
        for row in rows[1:]:
            self.assertEqual(row['track_status'], 'continued')
            self.assertTrue(row['continuity_ok'])
            self.assertAlmostEqual(row['radial_velocity'], .3)
        self.assertAlmostEqual(rows[-1]['prediction_residual'], 0.)
        self.assertAlmostEqual(rows[-1]['elapsed_time'], .14)

    def test_missing_detection_reacquires_same_candidate_using_elapsed_time(self):
        tracker = RadialShockTracker(.01, .0625)
        radii = [None, .2, .21, None, .23]
        rows = [tracker.update(t, samples_at(r))[1] for t, r in enumerate(radii)]
        self.assertEqual([r['track_status'] for r in rows],
                         ['not_yet_detected', 'initialized', 'continued', 'missing', 'reacquired'])
        self.assertEqual({r['track_id'] for r in rows}, {'forward_shock'})
        self.assertAlmostEqual(rows[-1]['elapsed_time'], 2.)
        self.assertAlmostEqual(rows[-1]['radial_velocity'], .01)
        self.assertTrue(rows[-1]['continuity_ok'])

    def test_two_bands_retain_order_orientation_and_separate_identities(self):
        tracker = RadialShockTracker(.01, .0625)
        reverse, forward = tracker.update(.1, samples_at(.3, .1))
        self.assertEqual(reverse['track_id'], 'reverse_shock')
        self.assertEqual(forward['track_id'], 'forward_shock')
        self.assertTrue(reverse['two_bands_detected'])
        self.assertLess(reverse['median_radial_alignment'], 0)
        self.assertGreater(forward['median_radial_alignment'], 0)
        self.assertTrue(reverse['normal_orientation_ok'])
        self.assertTrue(forward['normal_orientation_ok'])
        self.assertLess(reverse['radius_median'], forward['radius_median'])

    def test_discontinuity_is_flagged_but_preserves_offline_state_update(self):
        tracker = RadialShockTracker(.01, .0625)
        tracker.update(0., samples_at(.2))
        tracker.update(1., samples_at(.21))
        row = tracker.update(2., samples_at(.4))[1]
        self.assertEqual(row['track_status'], 'discontinuous')
        self.assertFalse(row['continuity_ok'])
        self.assertAlmostEqual(tracker.states['forward']['last_radius'], .4)
        self.assertEqual(row['track_id'], 'forward_shock')
        with self.assertRaises(ValueError):
            tracker.update(2., samples_at(.4))


class OnlineEnergyTests(unittest.TestCase):
    def test_stream_matches_offline_trapezoids_including_gaps_and_zero_power(self):
        times = np.array([0., .1, .3, .4, .7, 1., 1.2, 1.4])
        rates = np.array([np.nan, 2., 4., np.nan, 6., 0., np.nan, 8.])
        expected, valid = _cumulative_trapezoid_over_detections(times, rates)
        accumulator = CumulativeEnergy()
        rows = [accumulator.update(t, p) for t, p in zip(times, rates)]
        np.testing.assert_array_equal([r['integration_interval_valid'] for r in rows], valid)
        np.testing.assert_allclose([r['cumulative_dissipated_energy'] for r in rows], expected,
                                   rtol=0, atol=0, equal_nan=True)
        self.assertAlmostEqual(rows[-1]['integrated_duration'], .5)
        self.assertAlmostEqual(rows[4]['cumulative_dissipated_energy'], .6)  # No bridge over .4.
        self.assertFalse(rows[-1]['integration_interval_valid'])  # Reappearance adds no interval.
        with self.assertRaises(ValueError):
            accumulator.update(1.4, 1.)

    def test_no_detections_never_become_zero_energy(self):
        accumulator = CumulativeEnergy()
        for t in (0., .1, .2):
            row = accumulator.update(t, np.nan)
            self.assertTrue(np.isnan(row['cumulative_dissipated_energy']))
            self.assertFalse(row['integration_interval_valid'])
            self.assertEqual(row['integrated_duration'], 0.)

    def test_combined_energy_sums_per_band_not_partially_available_total_power(self):
        with tempfile.TemporaryDirectory() as directory:
            temporal = TemporalDiagnostics(Path(directory), .01, .0625, integrate_energy=True)
            powers = [(2., 4.), (np.nan, 4.), (2., 4.), (2., 4.)]
            try:
                for t, (reverse, forward) in enumerate(powers):
                    temporal.receive(t, {}, {'reverse_dissipation_rate': reverse,
                                             'forward_dissipation_rate': forward})
                row = temporal.energy_rows[-1]
                self.assertEqual(row['reverse_cumulative_dissipated_energy'], 2.)
                self.assertEqual(row['forward_cumulative_dissipated_energy'], 12.)
                self.assertEqual(row['combined_cumulative_dissipated_energy'], 14.)
                self.assertFalse((Path(directory)/'tracks.csv').exists())
            finally:
                temporal.close()

    def test_auxiliary_comparison_detects_status_and_energy_mismatches(self):
        row = dict(time=.2, track_id='forward_shock', continuity_ok=True, energy=.1)
        compare_records([row], [row.copy()], 'test')
        for changes in ({'track_id': 'reverse_shock'}, {'continuity_ok': False}, {'energy': .2}):
            with self.assertRaises(AssertionError):
                compare_records([row], [{**row, **changes}], 'test')


if __name__ == '__main__':
    unittest.main()
