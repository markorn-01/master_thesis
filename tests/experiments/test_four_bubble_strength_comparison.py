"""Guard against interpreting a confounded pair as a strength-only experiment."""
import copy
import json
from pathlib import Path
import unittest

from experiments.wind_bubble.compare_four_bubble_strengths import validate_pair


class StrengthComparisonTests(unittest.TestCase):
    def setUp(self):
        self.equal = json.loads(Path('outputs/four_bubbles_iwr_compgpu2_n064/metadata.json').read_text())
        self.unequal = copy.deepcopy(self.equal)
        strengths = [1.5, 1., .75, .75]
        self.unequal['source_strengths'] = strengths
        for key in ('discrete_mass_rates', 'discrete_energy_rates', 'nominal_wind_luminosities'):
            self.unequal[key] = [v*s for v,s in zip(self.equal[key],strengths)]

    def test_redistribution_preserves_matched_pair(self):
        validate_pair(self.equal, self.unequal)

    def test_rejects_change_in_geometry_schedule_or_injection(self):
        for field in ('source_positions', 'actual_times', 'solver_dt_max', 'wind_terminal_velocity'):
            target = copy.deepcopy(self.unequal)
            if field == 'source_positions': target[field][0][0] += .01
            elif field == 'actual_times': target[field][3] += .001
            else: target[field] *= 1.01
            with self.subTest(field=field), self.assertRaises(AssertionError):
                validate_pair(self.equal, target)
        target = copy.deepcopy(self.unequal); target['injection_prescription'] = 'changed'
        with self.assertRaises(ValueError): validate_pair(self.equal, target)

    def test_rejects_different_total_even_with_same_strength_metadata(self):
        for field in ('discrete_mass_rates', 'discrete_energy_rates', 'nominal_wind_luminosities'):
            target=copy.deepcopy(self.unequal); target[field][0] *= 1.1
            with self.subTest(field=field), self.assertRaises(AssertionError):
                validate_pair(self.equal, target)

    def test_rejects_incomplete_run_and_wrong_strengths(self):
        target=copy.deepcopy(self.unequal); target['status']='running'
        with self.assertRaises(ValueError): validate_pair(self.equal,target)
        target=copy.deepcopy(self.unequal); target['source_strengths']=[1.5,.75,1,.75]
        with self.assertRaises(AssertionError): validate_pair(self.equal,target)


if __name__ == '__main__': unittest.main()
