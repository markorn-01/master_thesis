import unittest
import numpy as np
from experiments.astrolink.surface_associations import association_scores, accepted_links


class AssociationTests(unittest.TestCase):
    def score(self, pa, la, pb, lb, gate=.2):
        return association_scores(pa, np.tile([0,0,1.], (len(pa),1)), la,
                                  pb, np.tile([0,0,1.], (len(pb),1)), lb, gate)

    def test_two_parents_can_join_one_child(self):
        p = np.array([[0,0,0],[0,1,0],[2,0,0],[2,1,0.]])
        scores = self.score(p, [0,0,1,1], p+[.1,0,0], [4,4,4,4])
        links = accepted_links(scores,.4)
        self.assertEqual(len(links),2)
        self.assertTrue(all(x['parent_coverage']==1 and x['child_coverage']==.5 for x in links))
        self.assertEqual(len(accepted_links(scores,.6)),0)

    def test_one_parent_can_have_two_children(self):
        p = np.array([[0,0,0],[0,1,0],[2,0,0],[2,1,0.]])
        self.assertEqual(len(accepted_links(self.score(p,[0]*4,p,[1,1,2,2]),.4)),2)

    def test_counts_points_once_not_neighbor_pairs(self):
        s = self.score([[0,0,0]], [0], [[0,0,0],[.1,0,0]], [1,1])[0]
        self.assertEqual(s['covered_parent_samples'],1)
        self.assertEqual(s['covered_child_samples'],2)
        self.assertEqual(s['parent_coverage'],1)

    def test_displacement_and_opposite_normals_reject_links(self):
        self.assertEqual(accepted_links(self.score([[0,0,0]],[0],[[1,0,0]],[0]),.25),[])
        scores = association_scores([[0,0,0]],[[0,0,1]],[0],[[0,0,0]],[[0,0,-1]],[0],.2)
        self.assertEqual(accepted_links(scores,.25),[])

    def test_row_reordering_and_label_names_do_not_define_correspondence(self):
        p = np.array([[0,0,0],[0,1,0],[2,0,0],[2,1,0.]])
        original = self.score(p,[0,0,1,1],p,[3,3,4,4])
        shuffled = self.score(p[::-1],[1,1,0,0],p[[1,3,0,2]],[3,4,3,4])
        self.assertEqual(original,shuffled)

    def test_empty_or_excluded_groups_have_no_links(self):
        self.assertEqual(self.score(np.empty((0,3)),[],[[0,0,0]],[0]),[])
        self.assertEqual(self.score([[0,0,0]],[-1],[[0,0,0]],[0]),[])

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError): accepted_links([],0)
        with self.assertRaises(ValueError):
            association_scores([[0,0,0]],[[0,0,0]],[0],[[0,0,0]],[[0,0,1]],[0],1)


if __name__ == '__main__': unittest.main()
