"""CPU checks of the split cost model against the evidence files it reads (no hardware)."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import split_cost_model as model  # noqa: E402


class SplitCostModelTests(unittest.TestCase):
    def setUp(self):
        self.cal = model.calibrate()

    def test_reads_measured_inputs(self):
        gpu = model.read_gpu_points()
        self.assertEqual(sorted(gpu), [10240, 14336])
        self.assertAlmostEqual(gpu[14336], 20.222 * 0.980, places=6)
        self.assertAlmostEqual(self.cal["t_npu_measured"], 11.098, delta=0.001)
        self.assertAlmostEqual(self.cal["ffn_measured"], 12.601, delta=0.002)
        self.assertEqual(sorted({s["npu_channels"] for s in self.cal["sweep"]}), [4352, 6144, 6400, 6656])

    def test_phases_explain_the_measured_prefill(self):
        predicted = model.LAYER_CHUNKS * (self.cal["outside"] + self.cal["ffn_measured"])
        self.assertLess(abs(predicted / self.cal["eval"][("merged_ffn", 1024)] - 1), 0.01)  # documents the observed 0.1%

    def test_npu_bound_scenarios_reproduce_the_calibration_point(self):
        for name, level in model.scenarios(self.cal).items():
            self.assertAlmostEqual(model.predict(self.cal, level, model.P_MEASURED)["T"], self.cal["ffn_measured"], places=9, msg=name)

    def test_fits_have_physical_signs(self):
        self.assertGreater(self.cal["b_npu"], 0)
        self.assertGreater(self.cal["b_gpu"], 0)
        self.assertGreater(self.cal["gpu_upper"], self.cal["gpu_isolated"])

    def test_candidates_respect_the_contract(self):
        self.assertTrue(all(p % 256 == 0 and 0 < p <= 8192 for p in model.VALID_P + model.TABLE_P))
        for level in model.scenarios(self.cal).values():
            self.assertIn(model.best(self.cal, level)["p"], model.VALID_P)

    def test_report_states_the_decision_corrections_and_gaps(self):
        text = model.report(self.cal)
        for phrase in ("Do not build other split ratios", "Corrections to the previous version", "estimated",
                       "What the data cannot determine", "heteromosaic_mdown"):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
