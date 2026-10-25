"""Real small-grid tests of callback scheduling and state-time consistency."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from astronomix import (
    CARTESIAN, SimulationConfig, SimulationParams, construct_primitive_state,
    finalize_config, get_registered_variables, time_integration,
)
from astronomix.option_classes.simulation_config import SnapshotSettings
from astronomix.time_stepping._utils import _unpad
from experiments.shock.check_callback_timing import run_probe


class CallbackTimingTests(unittest.TestCase):
    def assert_times(self, report):
        self.assertTrue(report['callback_times_match'], report)
        self.assertTrue(report['callback_final_at_t_end'], report)
        self.assertEqual(report['final_state_shape'], [3, 16])
        # Existing padded callback API is deliberately preserved.
        for record in report['observed_callbacks']:
            self.assertEqual(record['shape'], [3, 20])

    def test_regular_large_steps_do_not_skip_events(self):
        self.assert_times(run_probe('regular'))

    def test_regular_smaller_steps_hit_targets(self):
        self.assert_times(run_probe('regular', pressure=0.01))

    def test_explicit_early_time_advances_clock(self):
        self.assert_times(run_probe('early', requested_times=[0., .05, .31, .5]))

    def test_explicit_late_times(self):
        self.assert_times(run_probe('late', requested_times=[0., .2, .35, .5]))

    def test_initial_time_need_not_be_requested(self):
        self.assert_times(run_probe('no_initial', requested_times=[.1, .2, .35, .5]))

    def test_final_callback_when_final_not_listed(self):
        self.assert_times(run_probe('no_final', requested_times=[0., .05, .31]))

    def test_nonzero_start(self):
        self.assert_times(run_probe('restart', t_start=.2))

    def test_fixed_regular_aligned_schedule(self):
        self.assert_times(run_probe('fixed', fixed=True))

    def test_fixed_explicit_aligned_schedule(self):
        self.assert_times(run_probe('fixed', fixed=True, requested_times=[0., .05, .3, .5]))

    def test_fixed_nonzero_start(self):
        self.assert_times(run_probe('fixed', fixed=True, t_start=.2))

    def test_fixed_unaligned_schedule_rejected(self):
        with self.assertRaisesRegex(ValueError, 'align'):
            run_probe('fixed', fixed=True, requested_times=[0., .05, .31, .5])

    def test_invalid_explicit_times_rejected(self):
        for targets in ([], [.2, .1], [0., .2, .2], [-.1, .5], [0., .6], [0., np.nan]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                run_probe('invalid', requested_times=targets)

    def test_stored_snapshot_precedence_preserved(self):
        report = run_probe('both', requested_times=[0., .2, .35, .5], stored=True)
        self.assertEqual(report['observed_callbacks'], [])
        np.testing.assert_allclose(report['stored_times'], [0., .2, .35, .5], atol=1e-6)

    def test_nonuniform_state_matches_offline_at_explicit_times(self):
        targets = jnp.asarray([0., .05, .13, .3])
        config = SimulationConfig(
            geometry=CARTESIAN, dimensionality=1, num_cells=16, box_size=1.,
            num_snapshots=4, return_snapshots=True, use_specific_snapshot_timepoints=True,
            snapshot_settings=SnapshotSettings(return_states=True, return_final_state=True),
            print_elapsed_time=False, progress_bar=False,
        )
        variables = get_registered_variables(config)
        x = (jnp.arange(16) + .5) / 16
        state = construct_primitive_state(
            config=config, registered_variables=variables,
            density=1 + .1*jnp.sin(2*jnp.pi*x), velocity_x=jnp.full(16, .2),
            gas_pressure=jnp.full(16, .01),
        )
        config = finalize_config(config, state.shape)
        params = SimulationParams(t_end=.3, snapshot_timepoints=targets)
        offline = time_integration(state, config, params, variables)
        captured = []

        def collect(time, physical):
            captured.append((float(time), np.asarray(physical).copy()))

        callback_config = config._replace(return_snapshots=False, activate_snapshot_callback=True)

        def callback(time, padded, registry):
            jax.debug.callback(collect, time, _unpad(padded, callback_config), ordered=True)

        online_final = time_integration(state, callback_config, params, variables,
                                        snapshot_callable=callback)
        jax.block_until_ready(online_final)
        jax.effects_barrier()
        np.testing.assert_allclose([r[0] for r in captured], np.asarray(offline.time_points), atol=1e-6)
        np.testing.assert_array_equal(np.stack([r[1] for r in captured]), np.asarray(offline.states))
        np.testing.assert_array_equal(np.asarray(online_final), np.asarray(offline.final_state))
        # Nonuniform flow actually evolves; this is not just a constant-state check.
        self.assertGreater(np.max(np.abs(captured[-1][1] - np.asarray(state))), 1e-5)


if __name__ == '__main__':
    unittest.main()
