"""CPU checks of the split cost model against the evidence files it reads (no hardware)."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heteromosaic_records as records  # noqa: E402
import make_gateup_config  # noqa: E402
import split_cost_model as model  # noqa: E402


class GateUpConfigTests(unittest.TestCase):
    def test_moves_only_gate_and_up_at_1024_tokens(self):
        entry = lambda layer: dict(use=True, layer=layer, npuM=256, fw_path="shipped/")  # noqa: E731
        config = dict(kernels_gemm_chunked=[dict(prompt_len=1024, kernels=[entry("q"), entry("up"), entry("gate"), entry("down")]),
                                            dict(prompt_len=2048, kernels=[entry("gate")])])
        self.assertEqual(make_gateup_config.enable_all_rows(config), 2)
        rows = {e["layer"]: e["npuM"] for e in config["kernels_gemm_chunked"][0]["kernels"]}
        self.assertEqual(rows, dict(q=256, up=512, gate=512, down=256))
        self.assertEqual(config["kernels_gemm_chunked"][1]["kernels"][0]["npuM"], 256)


class HeteromosaicRecordsTests(unittest.TestCase):
    def test_reads_records_and_leaves_out_instrumented_runs(self):
        found = {r["source"]: r for r in records.records()}
        self.assertAlmostEqual(found["2026-11-02-fix-step3c/head-to-head"]["median_ms"], 1317.6, delta=0.1)
        self.assertEqual(found["2026-11-02-inflight2-repeat/repeat"]["inflight"], 2)
        self.assertEqual(found["2026-11-02-inflight2-repeat/repeat"]["samples"], 30)
        self.assertFalse([s for s in found if "diag" in s or "trace" in s])


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
