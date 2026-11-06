import unittest
import numpy as np
from experiments.astrolink.fuzzycat_float_weights import floating_sample_weights,experimental_float_weight_class


class FloatWeightTests(unittest.TestCase):
    def test_fractional_weights_and_missing_realization(self):
        which,w=floating_sample_weights(np.arange(3),[[0,3]],[0,0,1],3)
        np.testing.assert_array_equal(which,[0,0,0])
        np.testing.assert_allclose(w,[[.5],[1.],[0.]])
        self.assertTrue(np.issubdtype(w.dtype,np.floating))

    def test_original_index_mapping_and_multiple_groups(self):
        which,w=floating_sample_weights([2,0,3,1],[[0,2],[2,4]],[0,1,0,1],2)
        np.testing.assert_array_equal(which,[0,1,0,1])
        np.testing.assert_allclose(w,[[.5,0],[0,.5]])

    def test_empty_selection(self):
        which,w=floating_sample_weights([1,0],np.empty((0,2),int),[0,1],2)
        np.testing.assert_array_equal(which,[-1,-1]);self.assertEqual(w.shape,(2,0))

    def test_subclass_does_not_patch_base_class(self):
        try:from fuzzycat import FuzzyCat
        except ImportError:self.skipTest('Optional fuzzycategories dependency is not installed.')
        cls=experimental_float_weight_class()
        arguments=(np.arange(4,dtype=np.uint32),np.array([[0,4]],dtype=np.uint32),np.array([0,0,1,1],dtype=np.uint32),2)
        _,old=FuzzyCat._setupHierarchyInformation_njit(*arguments)
        _,new=cls._setupHierarchyInformation_njit(*arguments)
        np.testing.assert_array_equal(old,[[0],[0]])
        np.testing.assert_array_equal(new,[[.5],[.5]])
        self.assertIsNot(FuzzyCat._setupHierarchyInformation_njit,cls._setupHierarchyInformation_njit)


if __name__=='__main__':unittest.main()
