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
        self.assertAlmostEqual(self.cal["path_concurrent"], 32.46, delta=0.01)
        self.assertEqual(self.cal["p_measured"], 6400)
        ffn = self.cal["profile"]["ffn"]
        self.assertAlmostEqual(ffn["hetero_serial-detail-sync"]["per_layer_chunk_ms"], (437.1 + 49.4 + 422.2) / 64, delta=0.05)

    def test_every_scenario_reproduces_the_calibration_point(self):
        for name in model.SCENARIOS:
            m = model.scenario_model(self.cal, name)
            self.assertAlmostEqual(m["predict"](6400)["code_ordered"], self.cal["path_concurrent"], places=9)
            self.assertGreater(m["b_n"], 0)

    def test_candidates_respect_the_contract(self):
        self.assertTrue(all(p % 256 == 0 and 0 < p <= 8192 for p in model.VALID_P + model.CANDIDATES))
        best = model.best(model.scenario_model(self.cal, "central"), "code_ordered")
        self.assertIn(best["p"], model.VALID_P)

    def test_measured_evaluation_is_read_and_close_to_the_smoke_prediction(self):
        ev = self.cal["evaluation"]
        self.assertAlmostEqual(ev[("merged_ffn", 1024)], 2558.706, places=2)
        extra = (ev[("merged_ffn", 1024)] - ev[("gpu_original", 1024)]) / 64
        predicted = self.cal["path_concurrent"] - model._hm_total(self.cal)
        self.assertLess(abs(predicted / extra - 1), 0.15)  # documents the observed 7%; not a tolerance on results

    def test_report_labels_estimates_and_gaps(self):
        text = model.report(self.cal)
        for phrase in ("estimated", "What the data cannot determine", "Experiment on the AUP box", "Main B"):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
