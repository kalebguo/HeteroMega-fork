"""CPU tests for the NPU error probes. Synthetic records are fixtures, never measurements.

    python3 -m unittest discover -s tools/npu_error_probe -p 'test_*.py' -v
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "emulation"))
import path_logits as pl  # noqa: E402
import probe_cases as pc  # noqa: E402
import probe_report as report  # noqa: E402
import token_agreement as ta  # noqa: E402
import w4a16_npu_emulation as emu  # noqa: E402


def fixture(name, series, k, e_npu, e_gpu=0.01, n=2048):
    stats = lambda e: dict(mean_abs=e, max_abs=10 * e, rms=e, max_row=0, max_col=0, relative_l2=e, finite=True)
    return dict(name=name, series=series, k=k, n=n, status="ok", npu=stats(e_npu), gpu=stats(e_gpu),
                npu_vs_gpu_max_abs=10 * e_npu, sweep_allclose_rtol0p2_atol0p2=True)


def sign_meta(ratio_sign=(2, 2, 2), ratio_meta=(2, 2, 2, 2)):
    rows = [fixture(f"sign_{a}", "sign", 2048, 0.01 * r) for a, r in zip(("positive", "negative", "mixed"), ratio_sign)]
    keys = (("equal", "eight"), ("equal", "random"), ("random", "eight"), ("random", "random"))
    rows += [fixture(f"meta_scales_{s}_zeros_{z}", "metadata", 2048, 0.01 * r) for (s, z), r in zip(keys, ratio_meta)]
    return rows


class CaseTests(unittest.TestCase):
    def test_cases_order_and_blocked_reasons(self):
        names = [c["name"] for c in pc.cases()]
        self.assertEqual(names[:4], ["k1536_n2048", "k2048_n2048", "k10240_n2048", "k11264_n2048"])
        self.assertEqual([c["series"] for c in pc.cases()][-7:], ["sign"] * 3 + ["metadata"] * 4)
        blocked = {(b["k"], b["n"]): b["reason"] for b in pc.BLOCKED}
        self.assertIn("128-element quantization group", blocked[(64, 4096)])
        self.assertIn((256, 4096), blocked)

    @unittest.skipUnless((pc.ROOT / "hw_bins/npu2").is_dir(), "hw_bins not in this checkout")
    def test_every_runnable_case_has_a_native_m_split_artifact_and_blocked_ones_do_not(self):
        for case in pc.cases():
            fw = pc.artifact(pc.ROOT, 512, case["k"], case["n"])
            self.assertIsNotNone(fw, case["name"])
            self.assertTrue(fw.endswith("bf16_int4AWQ_bf16_M/"))
        for b in pc.BLOCKED:
            self.assertIsNone(pc.artifact(pc.ROOT, 512, b["k"], b["n"]), b)

    def test_inputs_share_weights_across_sign_and_reference_is_fp32(self):
        cases = {c["name"]: c for c in pc.cases()}
        (pos, wp), (neg, wn), (mix, wm) = (pc.make_inputs(cases[f"sign_{a}"]) for a in ("positive", "negative", "mixed"))
        self.assertTrue(np.array_equal(wp.packed, wn.packed) and np.array_equal(wp.scales, wm.scales))
        self.assertTrue(np.array_equal(pos, -neg) and pos.min() >= 0 and mix.min() < 0 < mix.max())
        _, eq = pc.make_inputs(cases["meta_scales_equal_zeros_eight"])
        self.assertTrue(np.all(eq.scales == 1 / 16) and np.all(eq.zeros == 8))
        small = dict(cases["k1536_n2048"], k=256, n=128)
        a, w = pc.make_inputs(small)
        expected = pc.reference(a, w)
        self.assertEqual(expected.dtype, np.float32)
        q = emu.unpack_int4(w.packed)
        exact = emu.exact_projection(a, q, w.scales, w.zeros)
        self.assertLess(np.max(np.abs(expected - exact)), 1e-5)
        stats = pc.compare(emu.bf16(expected), expected, expected)
        self.assertEqual(stats["npu_vs_gpu_max_abs"], np.max(np.abs(emu.bf16(expected) - expected)))
        self.assertTrue(stats["sweep_allclose_rtol0p2_atol0p2"])

    def test_native_config_is_all_rows_m_split(self):
        case = pc.cases()[1]
        kernel = pc.native_config(case, "hw_bins/npu2/8192x2048x2048/bf16_int4AWQ_bf16_M/")["kernels_gemm"][0]
        self.assertEqual((kernel["forM"], kernel["npuM"], kernel["npuK"], kernel["npuN"]), (512, 512, 2048, 2048))

    def test_sweep_dry_run_passes_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(HERE / "probe_sweep.py"), "--dry-run", "--out", tmp],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sum("--run-case" in line for line in result.stdout.splitlines()), len(pc.cases()))
            self.assertEqual(result.stdout.count("BLOCKED"), len(pc.BLOCKED))
            self.assertFalse(Path(tmp, "probe.jsonl").exists())


class ReportTests(unittest.TestCase):
    def k_series(self, ratios):
        return [fixture(f"k{k}_n2048", "k_n2048", k, 0.01 * r * (k / 1536) ** 0.5, 0.01 * (k / 1536) ** 0.5)
                for k, r in zip((1536, 2048, 10240, 11264), ratios)]

    def test_growing_excess_is_accumulation(self):
        found, details = report.verdicts(self.k_series((1.2, 1.5, 6, 7)) + sign_meta())
        self.assertTrue(any(f.startswith("accumulation") for f in found), found)
        self.assertGreater(details["k_n2048"]["ratio_slope"], report.SLOPE)

    def test_flat_excess_is_per_element(self):
        found, _ = report.verdicts(self.k_series((3, 3, 3, 3)) + sign_meta())
        self.assertTrue(any(f.startswith("per-element") for f in found), found)
        self.assertFalse(any(f.startswith("accumulation") for f in found), found)

    def test_sign_and_metadata_rules(self):
        found, details = report.verdicts(self.k_series((3, 3, 3, 3)) + sign_meta((1.2, 5, 2), (1, 4, 1.1, 5)))
        self.assertTrue(any(f.startswith("signed-input") for f in found), found)
        meta = [f for f in found if f.startswith("quantization metadata")]
        self.assertEqual(len(meta), 1)
        self.assertIn("zero points", meta[0])
        self.assertNotIn("scales,", meta[0])

    def test_sign_rule_uses_exact_negation_not_mixed_sign(self):
        rows = self.k_series((3, 3, 3, 3)) + sign_meta((2, 2, 9), (2, 2, 2, 2))
        found, details = report.verdicts(rows)
        self.assertFalse(any(f.startswith("signed-input") for f in found), found)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "outputs").mkdir()
            out = np.float32([[0.5, -1.25], [2.0, 0.0]])
            np.save(Path(tmp, "outputs", "sign_positive.npy"), out)
            np.save(Path(tmp, "outputs", "sign_negative.npy"), -out)
            found, details = report.verdicts(rows, tmp)
            self.assertTrue(details["sign"]["outputs_exact_negation"])
            np.save(Path(tmp, "outputs", "sign_negative.npy"), -out + np.float32([[0, 0.0078125], [0, 0]]))
            found, details = report.verdicts(rows, tmp)
            self.assertFalse(details["sign"]["outputs_exact_negation"])
            self.assertTrue(any(f.startswith("signed-input") for f in found), found)

    def test_no_npu_specific_error_and_undetermined(self):
        found, details = report.verdicts(self.k_series((1, 1.1, 1, 1.2)) + sign_meta((1, 1, 1), (1, 1, 1, 1)))
        self.assertIn("no NPU-specific error at these inputs (all r <= 1.5)", found)
        found, details = report.verdicts([fixture("k1536_n2048", "k_n2048", 1536, 0.02)])
        self.assertIn("not determined", details["k_n2048"])
        self.assertEqual(details["sign"], "not determined")

    def test_report_cli_prints_table_and_blocked_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "probe.jsonl")
            lines = self.k_series((1, 2, 6, 7)) + [dict(pc.BLOCKED[0], status="blocked")]
            path.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
            result = subprocess.run([sys.executable, str(HERE / "probe_report.py"), str(path), "--json", str(Path(tmp, "v.json"))],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("SUPPORTED: accumulation", result.stdout)
            self.assertIn("blocked", result.stdout)
            self.assertIn("hardware", json.loads(Path(tmp, "v.json").read_text()))


class TokenAgreementTests(unittest.TestCase):
    def test_agreement_and_first_disagreement(self):
        self.assertEqual(ta.agreement([1, 2, 3, 4], [1, 2, 9, 4])["first_disagreement"], 2)
        self.assertEqual(ta.agreement([1, 2, 3, 4], [1, 2, 9, 4])["agree"], 3)
        self.assertIsNone(ta.agreement([5, 6], [5, 6])["first_disagreement"])

    def test_nll_perplexity_and_logit_stats(self):
        logits = np.log(np.array([0.5, 0.25, 0.25]))
        self.assertAlmostEqual(ta.nll(logits, 0), np.log(2))
        self.assertAlmostEqual(ta.perplexity([np.log(2), np.log(4)])["perplexity"], np.sqrt(8))
        stats = ta.logit_stats([0, 1, 3, 2, 0, 0], [0, 1, 2.5, 2, 0, 0])
        self.assertEqual((stats["max_abs_diff"], stats["argmax_equal"], stats["top5_overlap"]), (0.5, True, 5))

    def test_dry_run_and_argument_guards(self):
        script = str(HERE / "token_agreement.py")
        result = subprocess.run([sys.executable, script, "--dry-run"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("perplexity", result.stdout)
        bad = subprocess.run([sys.executable, script, "--dry-run", "--prompt-tokens", "768"], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        missing = subprocess.run([sys.executable, script], capture_output=True, text=True)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("--merged-ffn-config", missing.stderr)


class PathLogitsTests(unittest.TestCase):
    def test_compare_reports_each_run_against_the_first_gpu_process(self):
        logits = np.array([2.0, 1.0, 0.0, -1.0, -2.0, -3.0], dtype=np.float32)
        def save(directory, stem, values, tokens, nlls=None):
            extra = {} if nlls is None else dict(nlls=np.asarray(nlls, np.float64))
            np.savez(Path(directory) / f"{stem}.npz", meta=json.dumps(dict(run=stem)), logits=values, logits_repeat=values,
                     tokens=np.asarray(tokens, np.int64), **extra)
        with tempfile.TemporaryDirectory() as directory:
            save(directory, "gpu_original-512-r0", logits, [1, 2, 3], [1.0, 2.0])
            save(directory, "heteromosaic-512-r0", logits + np.array([0, 0.5, 0, 0, 0, 0], np.float32), [1, 2, 4], [1.5, 2.0])
            rows = {r["run"]: r for r in pl.compare(directory)}
        reference, native = rows["gpu_original-512-r0"], rows["heteromosaic-512-r0"]
        self.assertEqual(reference["step1"]["max_abs_diff"], 0.0)
        self.assertEqual(reference["step1_kl_nats"], 0.0)
        self.assertEqual(native["step1"]["max_abs_diff"], 0.5)
        self.assertAlmostEqual(native["step1_relative_l2"], 0.5 / np.linalg.norm(logits), places=6)
        self.assertGreater(native["step1_kl_nats"], 0.0)
        self.assertEqual((native["greedy"]["agree"], native["greedy"]["first_disagreement"]), (2, 2))
        self.assertAlmostEqual(native["mean_abs_nll_difference"], 0.25)

    def test_scored_stats_compare_every_position(self):
        reference = np.array([[3.0, 1.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float32)
        other = np.array([[3.0, 1.0, 0.0], [3.0, 0.0, 1.0]], dtype=np.float32)
        stats = pl.scored_stats(other, reference)
        self.assertEqual((stats["positions"], stats["argmax_agreement"]), (2, 0.5))
        self.assertEqual(pl.scored_stats(reference, reference)["mean_kl_nats"], 0.0)
        self.assertGreater(stats["max_kl_nats"], stats["median_kl_nats"])


class EmulationTests(unittest.TestCase):
    """Secondary CPU emulation (estimates only)."""

    def test_bfp16_block_shares_exponent_and_rounds_half_even(self):
        # Block max 1.0 sets the step 2^-6: 1/128 is half a step (ties to even -> 0), 3/256 rounds up.
        out = emu.bfp16_ebs8(np.float32([1.0, 1 / 128, 3 / 256, 0, 0, 0, 0, -0.5]))
        self.assertTrue(np.array_equal(out, np.float32([1.0, 0.0, 0.015625, 0, 0, 0, 0, -0.5])))

    def test_all_flags_off_equals_gpu_like_and_partials_add_error(self):
        rng = np.random.default_rng(1)
        a = emu.bf16(rng.normal(size=(32, 512)).astype(np.float32))
        q, z = rng.integers(0, 16, (64, 512)), rng.integers(0, 16, (64, 4))
        s = emu.bf16(rng.uniform(0.01, 0.1, (64, 4)).astype(np.float32))
        off = emu.npu_projection(a, q, s, z, bfp16_a=False, bfp16_b=False, bf16_partials=False, k8_rounding="exact")
        self.assertTrue(np.array_equal(off, emu.gpu_like_projection(a, q, s, z)))
        exact = emu.exact_projection(a, q, s, z)
        deployed = emu.npu_projection(a, q, s, z)
        self.assertGreater(np.abs(deployed - exact).mean(), np.abs(off - exact).mean())

    def test_sigmoid_table_matches_kernel_comment_formula(self):
        slope, intercept = emu.sigmoid_table()
        mid = -16 + (np.arange(128) + 0.5) / 4
        sig = 1 / (1 + np.exp(-mid))
        self.assertTrue(np.array_equal(emu.bf16((sig * (1 - sig)).astype(np.float32)), slope))
        self.assertLess(np.max(np.abs(sig - slope.astype(np.float64) * mid - intercept)), 1e-6)


if __name__ == "__main__":
    unittest.main()
