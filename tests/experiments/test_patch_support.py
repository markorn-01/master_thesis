import unittest
import numpy as np
from experiments.astrolink.patch_support import pooled_support


class PatchSupportTests(unittest.TestCase):
    def run_case(self,pa,la,pb,lb,**kw):
        return pooled_support(pa,np.tile([0,0,1.],(len(pa),1)),la,
                              pb,np.tile([0,0,1.],(len(pb),1)),lb,.2,**kw)

    def test_disjoint_patches_collectively_cover_child(self):
        p=np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0.]])
        r=self.run_case(p,[0,1,2,3],p,[4]*4,min_child_coverage=.75)
        self.assertEqual(sum(x['accepted'] for x in r['links']),4)
        child=r['children'][0]
        self.assertEqual(child['largest_single_child_coverage'],.25)
        self.assertEqual(child['union_child_coverage'],1)

    def test_overlapping_support_not_summed(self):
        r=self.run_case([[0,0,0],[0,0,0]],[0,1],[[0,0,0],[2,0,0],[3,0,0],[4,0,0]],[5]*4)
        c=r['children'][0]
        self.assertEqual(c['summed_child_support_counts'],2)
        self.assertEqual(c['union_covered_child_samples'],1)
        self.assertFalse(c['accepted'])

    def test_competing_targets_not_arbitrarily_assigned(self):
        r=self.run_case([[0,0,0]],[0],[[0,0,0],[0,0,0]],[1,2])
        self.assertEqual(r['parents'][0]['status'],'ambiguous_target')
        self.assertEqual(r['links'],[])

    def test_unrelated_distant_patch_has_no_link(self):
        r=self.run_case([[0,0,0],[10,0,0]],[0,1],[[0,0,0]],[2])
        self.assertEqual([x['parent_group'] for x in r['links']],[0])

    def test_opposite_normals_rejected(self):
        r=pooled_support([[0,0,0]],[[0,0,1]],[0],[[0,0,0]],[[0,0,-1]],[1],.2)
        self.assertEqual(r['links'],[])

    def test_row_order_and_group_numbers_do_not_control_support(self):
        p=np.array([[0,0,0],[1,0,0],[2,0,0.]])
        a=self.run_case(p,[0,1,2],p,[4]*3)
        b=self.run_case(p[::-1],[2,1,0],p[[1,0,2]],[4]*3)
        self.assertEqual(a['parents'],b['parents'])
        self.assertEqual(a['children'],b['children'])

    def test_empty_inputs_and_threshold_validation(self):
        r=self.run_case(np.empty((0,3)),[],[[0,0,0]],[0])
        self.assertEqual(r['children'][0]['union_child_coverage'],0)
        with self.assertRaises(ValueError):self.run_case([[0,0,0]],[0],[[0,0,0]],[0],dominance_margin=0)


if __name__=='__main__':unittest.main()
