import unittest
import numpy as np
from experiments.astrolink.fuzzycat_inputs import hierarchy_memberships,partition_memberships,display_memberships,sample_multiplicities


class FuzzyInputTests(unittest.TestCase):
    def test_original_catalog_mapping_and_nested_memberships(self):
        r=hierarchy_memberships([5,1,3,0],[[0,4],[0,3],[1,2]],['1','1-1','1-1-1'],6)
        self.assertNotIn('1',r)
        np.testing.assert_array_equal(r['1-1'],[1,3,5])
        np.testing.assert_array_equal(r['1-1-1'],[1])

    def test_root_only_realization_exports_no_clusters(self):
        self.assertEqual(hierarchy_memberships([0,1],[[0,2]],['1'],2),{})

    def test_root_inclusion_is_explicit_control(self):
        np.testing.assert_array_equal(hierarchy_memberships([1,0],[[0,2]],['1'],2,True)['1'],[0,1])

    def test_partition_is_disjoint_and_excludes_unassigned_and_invalid(self):
        r=partition_memberships(np.array([4,4,-1,8,-2]))
        np.testing.assert_array_equal(r['1-1'],[0,1]);np.testing.assert_array_equal(r['1-2'],[3])

    def test_invalid_indices_and_intervals_rejected(self):
        for ordering in [[0,0],[0,3]]:
            with self.assertRaises(ValueError):hierarchy_memberships(ordering,[[0,2]],['1'],2)
        with self.assertRaises(ValueError):hierarchy_memberships([0],[[0,2]],['1'],1)

    def test_display_preserves_low_membership_and_ties(self):
        labels,maximum=display_memberships([[.8,.3,.5],[.1,.2,.5]])
        np.testing.assert_array_equal(labels,[0,-1,-3])
        np.testing.assert_allclose(maximum,[.8,.3,.5])

    def test_empty_fuzzy_output_has_no_display_groups(self):
        labels,maximum=display_memberships(np.empty((0,3)))
        np.testing.assert_array_equal(labels,[-1,-1,-1]);np.testing.assert_array_equal(maximum,[0,0,0])

    def test_reciprocal_weight_reference_keeps_fractional_values(self):
        counts,weights=sample_multiplicities(['000_1-1.npy','000_1-2.npy','001_1-1.npy','001_1-2.npy'],
                                            np.arange(4),[[0,4]],2)
        np.testing.assert_array_equal(counts,[[2],[2]])
        np.testing.assert_allclose(weights,[[.5],[.5]])


class FuzzyCatSmokeTest(unittest.TestCase):
    def test_repeated_disjoint_partition_recovers_memberships(self):
        try:from fuzzycat import FuzzyCat
        except ImportError:self.skipTest('Optional fuzzycategories dependency is not installed.')
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory(prefix='thesis-fuzzycat-test-') as directory:
            path=Path(directory)/'Clusters';path.mkdir()
            for sample in range(4):
                for j,rows in enumerate([np.arange(4,dtype=np.int64),np.arange(4,8,dtype=np.int64)]):
                    np.save(path/f'{sample:03d}_1-{j+1}.npy',rows)
            c=FuzzyCat(4,8,directoryName=directory);c.run()
            self.assertEqual(c.memberships_flat.shape,(2,8))
            np.testing.assert_allclose(c.stabilities,[1,1])
            self.assertEqual({tuple(np.flatnonzero(row)) for row in c.memberships_flat},{(0,1,2,3),(4,5,6,7)})


if __name__=='__main__':unittest.main()
