import unittest
import numpy as np
from experiments.astrolink.fuzzycat_weight_reference import reconstruct_memberships


class WeightReferenceTests(unittest.TestCase):
    def test_fractional_and_integer_weight_counterfactuals(self):
        names=np.array(['000_1-1.npy','000_1-2.npy','001_1-1.npy','001_1-2.npy'])
        clusters={name[:-4]:np.array([0,1] if '-1.npy' in name else [2,3]) for name in names}
        raw,flat,_=reconstruct_memberships(names,clusters,np.arange(4),[[0,4]],np.array([[.5],[.5]]),[1.],2,4)
        np.testing.assert_allclose(flat,[[.5,.5,.5,.5]])
        _,truncated,_=reconstruct_memberships(names,clusters,np.arange(4),[[0,4]],np.zeros((2,1),int),[1.],2,4)
        np.testing.assert_array_equal(truncated,np.zeros((1,4)))

    def test_hierarchy_correction_and_sample_prefix(self):
        names=np.array(['000_1-1.npy','000_1-1-1.npy'])
        clusters={'000_1-1':np.array([0,1,2]),'000_1-1-1':np.array([1])}
        raw,flat,hierarchy=reconstruct_memberships(names,clusters,np.arange(2),[[0,1],[1,2]],np.ones((1,2)),[1.,1.],1,3)
        np.testing.assert_array_equal(flat,[[1,0,1],[0,1,0]])
        self.assertEqual(hierarchy[0,1],1)


if __name__=='__main__':unittest.main()
