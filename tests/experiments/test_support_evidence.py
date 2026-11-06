import unittest
import numpy as np
from experiments.astrolink.support_evidence import support_evidence,controlled_examples


class SupportEvidenceTests(unittest.TestCase):
    def test_near_parallel_control_exposes_known_geometric_failure(self):
        c=controlled_examples()[0]
        self.assertEqual(c['unexpected_parent_groups'],[1])
        self.assertEqual(c['redundant_accepted_parents'],2)

    def test_far_and_opposite_controls_rejected(self):
        for c in controlled_examples()[1:3]:
            self.assertEqual(c['accepted_parent_groups'],[0])
            self.assertEqual(c['unexpected_parent_groups'],[])

    def test_complementary_patches_have_exclusive_support(self):
        c=controlled_examples()[3]
        self.assertEqual(c['accepted_parent_groups'],[0,1])
        self.assertEqual(c['redundant_accepted_parents'],0)

    def test_evidence_does_not_change_acceptance(self):
        p=np.array([[0.,0.,0.],[1.,0.,0.]])
        r=support_evidence(p,np.tile([0.,0.,1.],(2,1)),[0,1],p,np.tile([0.,0.,1.],(2,1)),[2,2],.2)
        self.assertTrue(all(l['accepted'] for l in r['links']))
        self.assertTrue(all(l['exclusive_child_support_samples']==1 for l in r['links']))

    def test_normals_are_not_modified(self):
        p=np.array([[0.,0.,0.]])
        n=np.array([[0.,0.,3.]])
        support_evidence(p,n,[0],p,n,[1],.2)
        np.testing.assert_array_equal(n,[[0.,0.,3.]])


if __name__=='__main__':unittest.main()
