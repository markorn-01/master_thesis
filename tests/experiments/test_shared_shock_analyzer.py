"""Extraction regression against the original committed offline analyzer."""
import unittest

from experiments.shock.analyze_state import analyze_shock_state
from experiments.shock.check_analyzer_equivalence import compare_case, load_baseline
from experiments.shock.run_planar_shock_benchmark import build_planar_shock_state


class SharedShockAnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = build_planar_shock_state(16, 5., 3., (1., 1., 1.))

    def test_all_fields_match_original_oblique_shock(self):
        state, config, variables, helper, _, _ = self.problem
        _, report = compare_case("oblique_planar", state, config, variables,
                                 helper, load_baseline())
        self.assertTrue(report["exact_match"])

    def test_rejects_padded_state(self):
        state, config, variables, helper, _, _ = self.problem
        with self.assertRaisesRegex(ValueError, "unpadded"):
            analyze_shock_state(state[:, :-1], config, variables, helper)

    def test_rejects_mhd(self):
        state, config, variables, helper, _, _ = self.problem
        with self.assertRaisesRegex(ValueError, "hydrodynamics"):
            analyze_shock_state(state, config._replace(mhd=True), variables, helper)


if __name__ == "__main__":
    unittest.main()
