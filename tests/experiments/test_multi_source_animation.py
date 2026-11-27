"""Four-source display adapters preserve callback identities and visibility."""
import json
from pathlib import Path
import unittest

import numpy as np

from tests.experiments import test_streaming_group_demo as fixtures
from experiments.wind_bubble.streaming_group_demo import build_tables
from experiments.wind_bubble.animate_multi_source_groups import segment_palette
from experiments.wind_bubble.build_group_history_table import visible_relationships


class FourSourceTables(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.StreamingGroupDemo()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_four_source_adapter_preserves_callback_segments(self):
        case=self.fixture
        case.metadata['source_positions'].append([.5,.5,.7])
        case.write_metadata()
        for i in range(3):
            path=case.root/'online'/f'surfaces_{i:04d}.npz'
            with np.load(path) as a: data={k:a[k].copy() for k in a.files}
            data['source_positions']=np.asarray(case.metadata['source_positions'])
            np.savez_compressed(path,**data)
        report=build_tables(case.root,case.directory/'four_tables',expected_sources=4)
        self.assertEqual(report['segments'],1)
        self.assertEqual(report['group_observations'],3)
        self.assertTrue(report['accepted_segment_assignments_exact'])
        with self.assertRaisesRegex(ValueError,'3-source'):
            build_tables(case.root,case.directory/'wrong_default')


class DisplayIdentityTests(unittest.TestCase):
    def test_25_segments_have_unique_colors_without_modulo_reuse(self):
        palette=segment_palette(25)
        self.assertEqual(len(set(map(tuple,palette))),25)
        self.assertEqual(palette,segment_palette(25))
        with self.assertRaises(ValueError): segment_palette(29)

    def test_conditional_link_is_hidden_until_the_next_observation(self):
        accepted=[dict(event_id='h1',available_event=14)]
        support=[dict(support_id='s1',available_event=15,parent_event=13,child_event=14)]
        self.assertEqual(visible_relationships(accepted,support,13),([],[]))
        self.assertEqual(visible_relationships(accepted,support,14),(accepted,[]))
        self.assertEqual(visible_relationships(accepted,support,15),(accepted,support))


if __name__=='__main__': unittest.main()
