"""CPU checks; synthetic timings are test fixtures, never hardware measurements."""
import importlib.util
import contextlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
import subprocess
import sys

spec = importlib.util.spec_from_file_location("evaluation", Path(__file__).with_name("evaluate_prefill.py"))
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
spec = importlib.util.spec_from_file_location("model_runner", Path(__file__).with_name("run_model_ffn.py"))
model_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_runner)


class EvaluationTests(unittest.TestCase):
    def test_native_runner_uses_selected_input_and_keeps_timer_boundaries(self):
        def generate(ids, max_new_tokens, temperature):
            result = MagicMock()
            result.__getitem__.side_effect = lambda index: types.SimpleNamespace(
                tolist=lambda: ([11, 12] + [77] * max_new_tokens)[index[1]])
            return result
        torch = MagicMock(__version__="CPU fixture")
        torch.no_grad = contextlib.nullcontext
        torch.cuda.get_device_name.return_value = "CPU fixture"
        torch.cuda.max_memory_allocated.return_value = 0
        torch.tensor.return_value.cpu.return_value.numpy.return_value.tobytes.return_value = b"fixture"
        frontend = MagicMock()
        frontend.load_config_with_comments.return_value = {"chunking": True}
        frontend.LLaMA3W4A16Model.return_value.generate.side_effect = generate
        modules = {"torch": torch, "unified_llm_w4a16_hetero_libtorch": types.SimpleNamespace(__file__="fixture.so"),
                   "llama3_8b_w4a16_model": frontend}
        with tempfile.TemporaryDirectory() as tmp:
            ref, out = Path(tmp) / "ref.json", Path(tmp) / "out.json"
            ref.write_text(json.dumps(dict(input_ids=[[11, 12, 13, 14]], generated_ids=[77, 78])))
            argv = ["runner", "--model", tmp, "--ref", str(ref), "--out", str(out), "--prompt-tokens", "2",
                    "--heterogeneity", "hetero", "--reps", "1", "--new-tokens", "2"]
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), \
                    patch.object(model_runner, "stub_missing_modules"), patch.object(model_runner, "merge_shards"):
                model_runner.main()
            result = json.loads(out.read_text())
            config = json.loads(Path(result["config"]).read_text())
        self.assertEqual((config["heterogeneity"], config["prompt_len"], config["chunking"]), ("hetero", 2, True))
        self.assertEqual(torch.tensor.call_args.args[0], [[11, 12]])
        self.assertEqual(result["reps"][0]["prefill_ids"], [77])
        self.assertIsNone(result["matches_reference"])
        self.assertEqual(torch.cuda.synchronize.call_count, 6)

    def test_case_order_and_complete_gpu_control(self):
        args = types.SimpleNamespace(prompt_tokens=[512], rounds=2, methods=["gpu_original", "gpu_ffn_all"],
                                     out=Path("out"), ref=Path("ref"), build="build-model", config="config",
                                     reps=5, new_tokens=64, dummy=True, model=None)
        with patch.dict(evaluation.os.environ, HETEROMEGA_GPU_FFN_SPLIT="7168,14336"):
            cases = list(evaluation.cases(args))
        self.assertEqual([c[0] for c in cases], ["gpu_original", "gpu_ffn_all", "gpu_ffn_all", "gpu_original"])
        self.assertEqual([c[4]["HETEROMEGA_GPU_FFN"] for c in cases], ["0", "1", "1", "0"])
        self.assertTrue(all("HETEROMEGA_GPU_FFN_SPLIT" not in c[4] for c in cases))
        args.methods = ["heteromosaic"]
        command = next(evaluation.cases(args))[3]
        self.assertEqual(command[command.index("--heterogeneity") + 1], "hetero")

    def test_long_prompt_cannot_silently_reuse_short_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(Path(evaluation.__file__)), "--out", tmp,
                                     "--prompt-tokens", "2048", "--dry-run"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("reference has 1024 tokens", result.stderr)

    def test_summary_excludes_warmup_and_distinguishes_prompt_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for method, length, times in [("gpu_original", 512, [1, 3, 2]), ("gpu_ffn_all", 512, [0.5, 1, 1.5]), ("gpu_original", 1024, [9])]:
                r = dict(method=method, prompt_tokens=length, warmup_s=999, warmup_ids=[7], matches_reference=None,
                         gpu_ffn_active=True, reps=[dict(prefill_s=t, prefill_ids=[7], generated_ids=[7, 8]) for t in times])
                (root / f"{method}-{length}.result.json").write_text(json.dumps(r))
            rows = evaluation.summarize(root)
            new = next(r for r in rows if r["method"] == "gpu_ffn_all")
            self.assertEqual((new["samples"], new["prefill_median_ms"], new["prefill_min_ms"], new["prefill_max_ms"], new["speedup_vs_gpu"]), (3, 1000, 500, 1500, 2))
            self.assertTrue((root / "summary.csv").exists())

    def run_fixture_runner(self, tmp, counter_values, events, prompt_tokens="2"):
        """Runs run_model_ffn.main against CPU mocks; events records synchronize/generate/counter order."""
        def generate(ids, max_new_tokens, temperature):
            events.append("generate")
            result = MagicMock()
            result.__getitem__.side_effect = lambda index: types.SimpleNamespace(
                tolist=lambda: ([11, 12] + [77] * max_new_tokens)[index[1]])
            return result
        values = iter(counter_values)

        def counter():
            events.append("count")
            return next(values)
        torch = MagicMock(__version__="CPU fixture")
        torch.no_grad = contextlib.nullcontext
        torch.cuda.synchronize.side_effect = lambda: events.append("sync")
        torch.cuda.get_device_name.return_value = "CPU fixture"
        torch.cuda.max_memory_allocated.return_value = 0
        torch.tensor.return_value.cpu.return_value.numpy.return_value.tobytes.return_value = b"fixture"
        frontend = MagicMock()
        frontend.load_config_with_comments.return_value = {
            "chunking": True, "gpu_chunking": {"gpu_chunk_size": [2048], "gpu_chunking_inflight": 1},
            "kernels_gemm_chunked": [{"prompt_len": 2, "hetero_inflight": 1}, {"prompt_len": 1024, "hetero_inflight": 1}]}
        frontend.LLaMA3W4A16Model.return_value.generate.side_effect = generate
        frontend.LLaMA3W4A16Model.return_value.model.merged_ffn_completed_dispatches.side_effect = counter
        modules = {"torch": torch, "unified_llm_w4a16_hetero_libtorch": types.SimpleNamespace(__file__="fixture.so"),
                   "llama3_8b_w4a16_model": frontend}
        ref, out = Path(tmp) / "ref.json", Path(tmp) / "out.json"
        ref.write_text(json.dumps(dict(input_ids=[[11, 12, 13, 14]], generated_ids=[77, 78])))
        argv = ["runner", "--model", tmp, "--ref", str(ref), "--out", str(out), "--prompt-tokens", prompt_tokens,
                "--heterogeneity", "gpu", "--reps", "2", "--new-tokens", "2", "--gpu-chunk-size", "512", "--inflight", "2"]
        with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), \
                patch.object(model_runner, "stub_missing_modules"), patch.object(model_runner, "merge_shards"), \
                patch.dict(model_runner.os.environ, HETEROMEGA_FFN_CONFIG="/fixture/deployment.json"):
            model_runner.main()
        result = json.loads(out.read_text())
        return result, json.loads(Path(result["config"]).read_text())

    def test_runner_counts_merged_completions_outside_the_timer(self):
        events = []
        # Cumulative counter values read before/after the warmup and two generate(1)/generate(2) pairs.
        values = [0, 64, 64, 128, 128, 192, 192, 256, 256, 288]
        with tempfile.TemporaryDirectory() as tmp:
            result, config = self.run_fixture_runner(tmp, values, events)
        self.assertEqual(result["warmup_merged_ffn_completions"], 64)
        self.assertEqual([(p["prefill_merged_ffn_completions"], p["generate_merged_ffn_completions"]) for p in result["reps"]],
                         [(64, 64), (64, 32)])
        self.assertEqual(result["merged_ffn_config_env"], "/fixture/deployment.json")
        # Every timed call is count, sync, generate, sync, count: the counter never enters the timed window.
        self.assertEqual(events, ["count", "sync", "generate", "sync", "count"] * 5)
        self.assertEqual(config["gpu_chunking"], {"gpu_chunk_size": 512, "gpu_chunking_inflight": 2})
        self.assertEqual([g["hetero_inflight"] for g in config["kernels_gemm_chunked"]], [2, 1])

    def test_only_merged_methods_receive_the_candidate_switch(self):
        methods = ["gpu_original", "merged_ffn", "gpu_ffn_all", "heteromosaic", "merged_ffn_hetero"]
        args = types.SimpleNamespace(prompt_tokens=[1024], rounds=2, methods=methods, out=Path("out"), ref=Path("ref"),
                                     build="build-model", config="config", reps=5, new_tokens=64, dummy=True, model=None,
                                     merged_ffn_config=Path("/abs/deployment.json"), gpu_chunk_size=512, inflight=None)
        inherited = dict(HETEROMEGA_FFN_CONFIG="/inherited.json", HETEROMEGA_GPU_FFN="1", HETEROMEGA_GPU_FFN_SPLIT="7168,14336")
        with patch.dict(evaluation.os.environ, inherited):
            evaluation.os.environ.pop("HETEROMOSAIC_ROOT", None)
            cases = list(evaluation.cases(args))
        self.assertEqual([c[0] for c in cases], methods + list(reversed(methods)))
        for method, _, _, cmd, env, _ in cases:
            merged = method.startswith("merged_ffn")
            self.assertEqual(env.get("HETEROMEGA_FFN_CONFIG"), "/abs/deployment.json" if merged else None, method)
            self.assertNotIn("HETEROMEGA_GPU_FFN_SPLIT", env)
            self.assertEqual(env["HETEROMEGA_GPU_FFN"], "1" if method == "gpu_ffn_all" else "0")
            hetero = method in ("heteromosaic", "merged_ffn_hetero")
            self.assertEqual(cmd[cmd.index("--heterogeneity") + 1], "hetero" if hetero else "gpu")
            self.assertEqual(env.get("HETEROMOSAIC_ROOT"), str(evaluation.ROOT) if hetero else None)
            self.assertEqual(cmd[cmd.index("--gpu-chunk-size") + 1], "512")
        self.assertEqual(evaluation.expected_merged("merged_ffn", 1024), 64)
        self.assertEqual(evaluation.expected_merged("merged_ffn_hetero", 512), 32)
        self.assertEqual(evaluation.expected_merged("gpu_original", 1024), 0)

    def write_result(self, root, method, times, counts, tokens=(7,), length=1024, name=None):
        """counts: per-pass merged completions [warmup, rep0 generate(1), rep0 generate(N), ...] or None."""
        reps = []
        for i, t in enumerate(times):
            rep = dict(prefill_s=t, prefill_ids=[7], generated_ids=list(tokens))
            if counts is not None:
                rep.update(prefill_merged_ffn_completions=counts[1 + 2 * i], generate_merged_ffn_completions=counts[2 + 2 * i])
            reps.append(rep)
        r = dict(method=method, prompt_tokens=length, warmup_ids=[7], matches_reference=None, gpu_ffn_active=False, reps=reps)
        if counts is not None:
            r["warmup_merged_ffn_completions"] = counts[0]
        (root / f"{name or method}-{length}.result.json").write_text(json.dumps(r))

    def test_merged_verdict_and_speedup_require_exact_completion_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_result(root, "gpu_original", [2, 2], [0] * 5)
            self.write_result(root, "heteromosaic", [1.6, 1.6], [0] * 5)
            for counts, verdict in [([64] * 5, "active"), ([64, 64, 64, 32, 64], "wrong_count"), (None, "unrecorded"),
                                    ([0] * 5, "wrong_count")]:
                self.write_result(root, "merged_ffn", [1, 1], counts)
                self.write_result(root, "merged_ffn_hetero", [0.8, 0.8], counts)
                rows = {r["method"]: r for r in evaluation.summarize(root)}
                for method in ("merged_ffn", "merged_ffn_hetero"):
                    row = rows[method]
                    self.assertEqual(row["merged_ffn_verdict"], verdict)
                    self.assertEqual(row["merged_ffn_expected_per_pass"], 64)
                    if verdict == "active":
                        self.assertEqual((row["merged_ffn_passes"], row["merged_ffn_completions"], row["merged_ffn_wrong_passes"]),
                                         (5, 320, 0))
                    else:
                        self.assertEqual((row["speedup_vs_gpu"], row["speedup_vs_heteromosaic"]), ("", ""))
                self.assertEqual(rows["gpu_original"]["merged_ffn_verdict"], "clean")
            self.write_result(root, "merged_ffn", [1, 1], [64] * 5)
            self.write_result(root, "merged_ffn_hetero", [0.8, 0.8], [64] * 5)
            rows = {r["method"]: r for r in evaluation.summarize(root)}
            self.assertEqual((rows["merged_ffn"]["speedup_vs_gpu"], rows["merged_ffn_hetero"]["speedup_vs_heteromosaic"]), (2, 2))
            self.assertEqual(rows["merged_ffn"]["npu_execution"], "merged_ffn_counted")
            self.assertEqual(rows["merged_ffn_hetero"]["npu_execution"], "unverified")
            # A candidate with different greedy tokens keeps its median but loses the speedup claim.
            self.write_result(root, "merged_ffn", [1, 1], [64] * 5, tokens=(8,))
            row = next(r for r in evaluation.summarize(root) if r["method"] == "merged_ffn")
            self.assertEqual((row["prefill_median_ms"], row["speedup_vs_gpu"], row["tokens_match_gpu"]), (1000, "", False))
            with (root / "summary.csv").open() as f:
                header = f.readline().strip().split(",")
            for column in ("merged_ffn_expected_per_pass", "merged_ffn_completions", "merged_ffn_wrong_passes", "merged_ffn_verdict"):
                self.assertIn(column, header)

    def test_contaminated_control_suppresses_speedups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_result(root, "gpu_original", [2, 2], [0, 0, 0, 64, 0])
            self.write_result(root, "merged_ffn", [1, 1], [64] * 5)
            rows = {r["method"]: r for r in evaluation.summarize(root)}
            self.assertEqual(rows["gpu_original"]["merged_ffn_verdict"], "contaminated")
            self.assertEqual((rows["gpu_original"]["speedup_vs_gpu"], rows["merged_ffn"]["speedup_vs_gpu"]), ("", ""))

    def test_mdown_method_alone_receives_the_mdown_config(self):
        args = types.SimpleNamespace(prompt_tokens=[1024], rounds=1, methods=["heteromosaic", "heteromosaic_mdown"], out=Path("out"),
                                     ref=Path("ref.json"), build="build", config="shipped.json5", mdown_config="mdown.json",
                                     reps=1, new_tokens=1, dummy=True, merged_ffn_config=None)
        configs = {method: cmd[cmd.index("--config") + 1] for method, _, _, cmd, _, _ in evaluation.cases(args)}
        self.assertEqual(configs, {"heteromosaic": "shipped.json5", "heteromosaic_mdown": "mdown.json"})

    def test_gateup_and_p8192_methods_alone_receive_their_configs(self):
        args = types.SimpleNamespace(prompt_tokens=[1024], rounds=1, out=Path("out"), ref=Path("ref.json"), build="build",
                                     methods=["heteromosaic", "heteromosaic_gateup", "merged_ffn", "merged_ffn_p8192"],
                                     config="shipped.json5", mdown_config=None, gateup_config="gateup.json", reps=1, new_tokens=1,
                                     dummy=True, merged_ffn_config=Path("p6400.json"), p8192_ffn_config=Path("p8192.json"))
        cases = list(evaluation.cases(args))
        self.assertEqual({method: cmd[cmd.index("--config") + 1] for method, _, _, cmd, _, _ in cases},
                         {"heteromosaic": "shipped.json5", "heteromosaic_gateup": "gateup.json",
                          "merged_ffn": "shipped.json5", "merged_ffn_p8192": "shipped.json5"})
        self.assertEqual({method: env.get("HETEROMEGA_FFN_CONFIG") for method, _, _, _, env, _ in cases},
                         {"heteromosaic": None, "heteromosaic_gateup": None, "merged_ffn": "p6400.json", "merged_ffn_p8192": "p8192.json"})

    def test_p8192_method_refuses_another_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"
            artifact.mkdir()
            (artifact / "final.xclbin.json").write_text(json.dumps(dict(m=512, hidden=4096, weight_n=14336, channels=6400)))
            config = Path(tmp) / "deployment.json"
            config.write_text(json.dumps(dict(artifact_dir="artifact")))
            result = subprocess.run([sys.executable, str(Path(evaluation.__file__)), "--out", tmp, "--model", tmp, "--methods", "merged_ffn_p8192",
                                     "--p8192-ffn-config", str(config), "--gpu-chunk-size", "512"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("8192 NPU channels", result.stderr)

    def test_merged_cli_guards(self):
        def dry(*extra):
            return subprocess.run([sys.executable, str(Path(evaluation.__file__)), "--out", tmp, "--dry-run", *extra],
                                  capture_output=True, text=True)
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"
            artifact.mkdir()
            (artifact / "final.xclbin.json").write_text(json.dumps(dict(m=512, hidden=4096, weight_n=14336, channels=6400)))
            config = Path(tmp) / "deployment.json"
            config.write_text(json.dumps(dict(artifact_dir="artifact")))
            self.assertIn("--merged-ffn-config", dry("--methods", "merged_ffn", "--gpu-chunk-size", "512").stderr)
            self.assertIn("multiples of 512", dry("--methods", "merged_ffn_hetero", "--merged-ffn-config", str(config),
                                                  "--prompt-tokens", "768").stderr)
            self.assertIn("--gpu-chunk-size 512", dry("--methods", "gpu_original", "merged_ffn",
                                                     "--merged-ffn-config", str(config)).stderr)
            result = dry("--methods", "gpu_original", "merged_ffn", "--merged-ffn-config", str(config),
                         "--gpu-chunk-size", "512", "--prompt-tokens", "512", "1024")
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.splitlines()
            self.assertEqual(len(lines), 8)
            for line in lines:
                self.assertEqual(f"HETEROMEGA_FFN_CONFIG={config}" in line, line.startswith("merged_ffn"), line)
                self.assertIn("--gpu-chunk-size 512", line)
            (artifact / "final.xclbin.json").write_text(json.dumps(dict(m=1024, hidden=4096, weight_n=14336)))
            result = subprocess.run([sys.executable, str(Path(evaluation.__file__)), "--out", tmp, "--methods", "merged_ffn",
                                     "--merged-ffn-config", str(config), "--gpu-chunk-size", "512", "--dummy"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("M512/H4096/I14336", result.stderr)

    def test_mismatch_or_inactive_ffn_has_no_speedup_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = dict(method="gpu_original", prompt_tokens=512, warmup_ids=[7], matches_reference=True,
                        gpu_ffn_active=False, reps=[dict(prefill_s=2, prefill_ids=[7], generated_ids=[7])])
            (root / "base.result.json").write_text(json.dumps(base))
            for active, tokens, reference in [(True, [8], True), (False, [7], True), (True, [7], False)]:
                new = dict(base, method="gpu_ffn_all", gpu_ffn_active=active, matches_reference=reference,
                           reps=[dict(prefill_s=1, prefill_ids=[7], generated_ids=tokens)])
                (root / "new.result.json").write_text(json.dumps(new))
                self.assertEqual(evaluation.summarize(root)[0]["speedup_vs_gpu"], "")


if __name__ == "__main__":
    unittest.main()
