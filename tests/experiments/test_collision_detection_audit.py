import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import numpy as np

from experiments.shock.run_planar_shock_benchmark import build_planar_shock_state
from experiments.wind_bubble.audit_collision_detection import (
    GATES, REFERENCE, check_reference, collision_fields, region_counts, run, validate_fields,
)


class CollisionDetectionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state, cls.config, cls.registry, cls.helper, _, _ = build_planar_shock_state(16, 5., 3.)
        cls.analyze = staticmethod(jax.jit(lambda s:collision_fields(s,cls.config,cls.registry,cls.helper)))

    def test_known_compressive_shock_is_retained_and_failures_accounted(self):
        fields = {k:np.asarray(v) for k,v in self.analyze(self.state).items()}
        validate_fields(fields)
        self.assertTrue(fields['shock_zone'].any())
        self.assertTrue(fields['shock_surface'].any())
        roi = np.ones(fields['shock_zone'].shape, bool)
        row = region_counts(fields, roi, fields['shock_surface'])
        self.assertEqual(row['cells'],16**3)
        self.assertEqual(row['pass_pattern_1111'],int(fields['shock_zone'].sum()))
        self.assertEqual(sum(row[f'pass_pattern_{i:04b}'] for i in range(16)),row['cells'])

    def test_isobaric_contact_rejected_by_alignment_despite_compression(self):
        # Preserve a converging velocity profile, but replace P with a constant.
        # This separates the contact filter from the absence of compression.
        contact = self.state.at[self.registry.pressure_index].set(1.)
        fields = {k:np.asarray(v) for k,v in self.analyze(contact).items()}
        validate_fields(fields)
        changing = np.asarray(self.state[self.registry.density_index])
        roi = (changing>1.01)&(changing<3.5)
        self.assertTrue(roi.any())
        self.assertTrue((fields['compression']&roi).any())
        self.assertTrue((fields['gradient_dot'][roi]<0).all())
        self.assertFalse((fields['gradient_alignment']&roi).any())
        self.assertFalse(fields['shock_zone'].any())

    def test_wrong_reference_geometry_rejected_before_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'reference'
            root.mkdir()
            (root/'metadata.json').write_text(json.dumps(dict(status='complete',num_cells=64,
                source_positions=[[.35,.5,.5],[.65,.5,.5]])))
            args=argparse.Namespace(reference_run=root, output_dir=Path(tmp)/'new',
                capture_events=[4],audit_only=False,require_gpu=False)
            with patch('experiments.wind_bubble.audit_collision_detection.run_simulation') as simulation:
                with self.assertRaisesRegex(ValueError,'three-source geometry'):run(args)
                simulation.assert_not_called()
            self.assertFalse(args.output_dir.exists())

    def test_actual_reference_inputs_still_match_fingerprints(self):
        if not REFERENCE.exists():
            self.skipTest('Saved GPU reference not present on this checkout.')
        meta, fingerprints = check_reference(REFERENCE)
        self.assertEqual(len(meta['requested_times']),17)
        self.assertIn(str(REFERENCE/'online/slice_0016.npz'),fingerprints)


if __name__=='__main__':
    unittest.main()
