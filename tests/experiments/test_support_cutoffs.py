import unittest
import numpy as np
from experiments.astrolink.check_support_robustness import size_filter


class SizeFilterTests(unittest.TestCase):
    def test_cutoff_is_inclusive_and_invalid_geometry_stays_distinct(self):
        np.testing.assert_array_equal(size_filter([0,0,1,-2],2),[0,0,-1,-2])

    def test_cutoff_one_retains_all_raw_components(self):
        np.testing.assert_array_equal(size_filter([0,0,5,-2],1),[0,0,5,-2])

    def test_empty_and_invalid_cutoff(self):
        self.assertEqual(len(size_filter([],1)),0)
        with self.assertRaises(ValueError):size_filter([0],0)


if __name__=='__main__':unittest.main()
