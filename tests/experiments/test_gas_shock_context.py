import unittest
import numpy as np
from experiments.wind_bubble.plot_gas_shock_context import prepare


class GasContext(unittest.TestCase):
    def test_temperature_preserves_grid_order(self):
        rho=np.array([[1.,2,4],[2,4,8]])
        pressure=np.array([[2.,6,16],[10,24,56]])
        t,rows=prepare(rho,pressure,np.empty((0,3)),.5,.1)
        np.testing.assert_array_equal(t,[[2,3,4],[5,6,7]])
        self.assertEqual(t.shape,rho.shape);self.assertEqual(len(rows),0)

    def test_overlay_is_a_half_cell_slab(self):
        p=np.array([[0,0,.375],[0,0,.5],[0,0,.625],[0,0,.626]])
        _,rows=prepare(np.ones((2,2)),np.ones((2,2)),p,.5,.25)
        np.testing.assert_array_equal(rows,[0,1,2])

    def test_invalid_gas_is_not_silently_masked(self):
        for value in [0.,-1.,np.nan]:
            with self.assertRaises(ValueError):prepare(np.full((2,2),value),np.ones((2,2)),np.empty((0,3)),.5,.1)

    def test_mismatched_fields(self):
        with self.assertRaises(ValueError):prepare(np.ones((2,2)),np.ones((2,3)),np.empty((0,3)),.5,.1)
