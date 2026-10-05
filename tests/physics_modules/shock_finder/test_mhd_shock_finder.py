import argparse
import unittest

import numpy as np

from astronomix._physics_modules._shock_finder.mhd_shock_finder import (
    find_fast_mhd_shocks,
)
from experiments.wind_bubble.run_single_bubble import build_problem


class TestMHDShockFinder(unittest.TestCase):
    def test_planar_perpendicular_fast_shock_is_detected(self):
        args = argparse.Namespace(
            num_cells=16,
            num_snapshots=3,
            t_end=0.01,
            num_injection_cells=2,
            plasma_beta=1.0,
            output_dir=None,
        )
        state, config, _, registered_variables, helper_data = build_problem(args)
        downstream = (slice(None, 8), slice(None), slice(None))
        upstream = (slice(8, None), slice(None), slice(None))

        state = state.at[
            (registered_variables.density_index,) + downstream
        ].set(2.0)
        state = state.at[
            (registered_variables.density_index,) + upstream
        ].set(1.0)
        state = state.at[
            (registered_variables.pressure_index,) + downstream
        ].set(6.0)
        state = state.at[
            (registered_variables.pressure_index,) + upstream
        ].set(0.6)
        state = state.at[
            (registered_variables.velocity_index.x,) + downstream
        ].set(-1.0)
        state = state.at[
            (registered_variables.velocity_index.x,) + upstream
        ].set(-3.0)
        state = state.at[
            (registered_variables.magnetic_index.x,) + (slice(None),) * 3
        ].set(0.0)
        state = state.at[
            (registered_variables.magnetic_index.y,) + downstream
        ].set(1.0)
        state = state.at[
            (registered_variables.magnetic_index.y,) + upstream
        ].set(0.5)
        state = state.at[
            (registered_variables.magnetic_index.z,) + (slice(None),) * 3
        ].set(0.0)

        result = find_fast_mhd_shocks(
            state,
            config,
            registered_variables,
            helper_data,
        )

        surface = np.asarray(result.shock_surface_cells)
        fast_shock = np.asarray(result.fast_shock_cells)
        self.assertGreater(np.count_nonzero(surface), 0)
        self.assertGreater(np.count_nonzero(fast_shock), 0)
        self.assertTrue(np.all(fast_shock <= surface))
        self.assertAlmostEqual(
            float(np.nanmedian(np.asarray(result.field_obliquity_degrees))),
            90.0,
            places=4,
        )
        self.assertGreater(
            float(np.nanmedian(np.asarray(result.upstream_fast_mach))),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
