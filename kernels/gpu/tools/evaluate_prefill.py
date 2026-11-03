#!/usr/bin/env python3
"""Batch run_model_ffn.py without changing its synchronized generate(1) timer."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
LAYERS = 32  # Llama 3.1 8B decoder layers; one merged-FFN dispatch per layer per M512 chunk
MERGED_M = 512  # rows per dispatch of the tested merged artifact (M512/H4096/I14336/P6400)
# method: (heterogeneity, HETEROMEGA_GPU_FFN, merged GPU+NPU FFN via HETEROMEGA_FFN_CONFIG)
METHODS = {"gpu_original": ("gpu", "0", False), "gpu_ffn_all": ("gpu", "1", False), "heteromosaic": ("hetero", "0", False),
           "merged_ffn": ("gpu", "0", True), "merged_ffn_hetero": ("hetero", "0", True),
           # heteromosaic with half of down on the NPU: same code, --mdown-config enables the native M-split down entries.
           "heteromosaic_mdown": ("hetero", "0", False),
           # heteromosaic with every gate/up row on the NPU: same code, --gateup-config from tools/split_model/make_gateup_config.py.
           "heteromosaic_gateup": ("hetero", "0", False),
           # merged_ffn on a second artifact with 8192 NPU channels: same code, --p8192-ffn-config.
           "merged_ffn_p8192": ("gpu", "0", True)}
# Never inherited: the candidate switch, and PR #1's partial-channel mode (it omits the NPU output).
CLEARED_ENV = ("HETEROMEGA_FFN_CONFIG", "HETEROMEGA_GPU_FFN_SPLIT")
SHOWN_ENV = ("HETEROMEGA_GPU_FFN", "HETEROMEGA_FFN_CONFIG", "HETEROMEGA_FFN_FINITE_CHECK", "HETEROMEGA_FFN_TIMERS", "HETEROMOSAIC_ROOT")


def is_merged(method):
    return METHODS.get(method, (None, None, False))[2]


def expected_merged(method, length):
    """Merged-FFN completions expected in one prefill pass (one generate call)."""
    return LAYERS * (length // MERGED_M) if is_merged(method) else 0


def merged_config(method, args):
    return args.p8192_ffn_config if method == "merged_ffn_p8192" else args.merged_ffn_config


def native_config(method, args):
    if method == "heteromosaic_mdown":
        return args.mdown_config
    return args.gateup_config if method == "heteromosaic_gateup" else args.config


def method_env(method, args):
    backend, gpu_ffn, merged = METHODS[method]
    env = dict(os.environ, HETEROMEGA_GPU_FFN=gpu_ffn)
    for name in CLEARED_ENV:
        env.pop(name, None)
    if merged:
        env["HETEROMEGA_FFN_CONFIG"] = str(merged_config(method, args))
    if backend == "hetero":
        # The native loader otherwise looks in the original author's checkout and silently falls back to the GPU.
        env.setdefault("HETEROMOSAIC_ROOT", str(ROOT))
    return env


def cases(args):
    for length in args.prompt_tokens:
        for round_id in range(args.rounds):
            order = args.methods if round_id % 2 == 0 else list(reversed(args.methods))
            for method in order:
                backend = METHODS[method][0]
                out = args.out / f"{method}-{length}-r{round_id}.result.json"
                cmd = [sys.executable, "-u", str(Path(__file__).with_name("run_model_ffn.py")),
                       "--ref", str(args.ref), "--out", str(out), "--build", args.build,
                       "--config", native_config(method, args),
                       "--heterogeneity", backend, "--prompt-tokens", str(length),
                       "--reps", str(args.reps), "--new-tokens", str(args.new_tokens)]
                # Applied to every method so the compared jobs share one chunk schedule.
                if getattr(args, "gpu_chunk_size", None):
                    cmd += ["--gpu-chunk-size", str(args.gpu_chunk_size)]
                if getattr(args, "inflight", None):
                    cmd += ["--inflight", str(args.inflight)]
                cmd += ["--dummy"] if args.dummy else ["--model", args.model or "MODEL_DIR"]
                yield method, length, round_id, cmd, method_env(method, args), out


def check_deployment(path):
    """Returns (problem or None, artifact manifest or None) for a deployment.json."""
    try:
        settings = json.loads(path.read_text())
        manifest = json.loads((path.parent / settings["artifact_dir"] / "final.xclbin.json").read_text())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"cannot read the merged FFN artifact manifest through {path}: {exc}", None
    shape = tuple(manifest.get(key) for key in ("m", "hidden", "weight_n"))
    if shape != (MERGED_M, 4096, 14336):
        return f"expected an M{MERGED_M}/H4096/I14336 merged artifact, got m/hidden/weight_n = {shape}", manifest
    return None, manifest


def signatures(runs):
    return {(tuple(r["warmup_ids"]), tuple(p["prefill_ids"]), tuple(p["generated_ids"]))
            for r in runs for p in r["reps"]}


def pass_counts(run):
    """Merged-FFN completions per prefill pass: the warmup and every generate(1)/generate(N) call."""
    counts = [run.get("warmup_merged_ffn_completions")]
    for p in run["reps"]:
        counts += [p.get("prefill_merged_ffn_completions"), p.get("generate_merged_ffn_completions")]
    return counts


def merged_check(method, length, runs):
    """Returns (expected per pass, passes, total completions, wrong passes, verdict)."""
    expected = expected_merged(method, length)
    counts = [c for r in runs for c in pass_counts(r)]
    if not counts or any(c is None for c in counts):
        return expected, len(counts), "", "", "unrecorded"
    wrong = sum(c != expected for c in counts)
    if is_merged(method):
        verdict = "wrong_count" if wrong else ("active" if expected > 0 else "inactive")
    else:
        verdict = "contaminated" if wrong else "clean"
    return expected, len(counts), sum(counts), wrong, verdict


def summarize(directory):
    groups = {}
    for path in sorted(directory.glob("*.result.json")):
        r = json.loads(path.read_text())
        groups.setdefault((r["method"], r["prompt_tokens"]), []).append(r)
    checks = {key: merged_check(key[0], key[1], runs) for key, runs in groups.items()}

    def comparable(key, base_key):
        """Speedup is shown only for a verified candidate against a clean baseline with identical tokens."""
        runs, baseline = groups[key], groups.get(base_key, [])
        if not baseline:
            return None, False
        same = len(signatures(runs) | signatures(baseline)) == 1
        verdict, base_verdict = checks[key][-1], checks[base_key][-1]
        merged_ok = verdict == "active" if is_merged(key[0]) else verdict != "contaminated"
        return same, same and merged_ok and base_verdict != "contaminated"

    def median_ms(runs):
        return statistics.median(p["prefill_s"] * 1000 for r in runs for p in r["reps"])

    rows = []
    for (method, length), runs in sorted(groups.items()):
        times = [p["prefill_s"] * 1000 for r in runs for p in r["reps"]]
        active = all(r["gpu_ffn_active"] for r in runs) if method == "gpu_ffn_all" else None
        refs = [r["matches_reference"] for r in runs if r["matches_reference"] is not None]
        reference_match = all(refs) if refs else None
        valid = active is not False and reference_match is not False
        speedups, same_tokens = {}, {}
        for base in ("gpu_original", "heteromosaic"):
            same_tokens[base], ok = comparable((method, length), (base, length))
            speedups[base] = median_ms(groups[(base, length)]) / statistics.median(times) if ok and valid else ""
        expected, passes, completions, wrong, verdict = checks[(method, length)]
        backend = METHODS.get(method, ("gpu",))[0]
        npu = "not_requested" if backend == "gpu" else "unverified"
        if is_merged(method) and backend == "gpu":
            npu = "merged_ffn_counted" if verdict == "active" else "unverified"
        rows.append(dict(method=method, prompt_tokens=length, dummy=any(r.get("dummy", False) for r in runs), samples=len(times),
                         prefill_median_ms=statistics.median(times), prefill_min_ms=min(times), prefill_max_ms=max(times),
                         speedup_vs_gpu=speedups["gpu_original"], tokens_match_gpu=same_tokens["gpu_original"],
                         speedup_vs_heteromosaic=speedups["heteromosaic"], tokens_match_heteromosaic=same_tokens["heteromosaic"],
                         matches_reference=reference_match, gpu_ffn_active=active,
                         merged_ffn_expected_per_pass=expected, merged_ffn_passes=passes, merged_ffn_completions=completions,
                         merged_ffn_wrong_passes=wrong, merged_ffn_verdict=verdict, npu_execution=npu))
    if rows:
        with (directory / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model")
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--ref", type=Path, default=ROOT / "docs/runs/2026-10-12-aup/gpu-model-result.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--build", default="build-model")
    p.add_argument("--config", default="py/unified_llm_w4a16/configs/configs_strixP_llama3_8b.json5")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=["gpu_original", "gpu_ffn_all"])
    p.add_argument("--mdown-config", help="configuration of heteromosaic_mdown, from tools/split_model/make_mdown_config.py")
    p.add_argument("--gateup-config", help="configuration of heteromosaic_gateup, from tools/split_model/make_gateup_config.py")
    p.add_argument("--p8192-ffn-config", type=Path, help="deployment.json of merged_ffn_p8192; its artifact must have 8192 NPU channels")
    p.add_argument("--merged-ffn-config", type=Path,
                   help="deployment.json for merged_ffn/merged_ffn_hetero; exported as HETEROMEGA_FFN_CONFIG for those jobs only")
    p.add_argument("--gpu-chunk-size", type=int, help="GPU-backend prefill chunk rows for every method (merged_ffn needs 512)")
    p.add_argument("--inflight", type=int, help="inflight chunk slots for every method (GPU and heterogeneous schedules)")
    p.add_argument("--prompt-tokens", nargs="+", type=int, default=[1024])
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    args.out, args.ref = args.out.resolve(), args.ref.resolve()
    if args.summarize:
        print(json.dumps(summarize(args.out), indent=2))
        return
    available = len(json.loads(args.ref.read_text())["input_ids"][0])
    if any(n < 1 or n > available for n in args.prompt_tokens):
        p.error(f"reference has {available} tokens; provide a longer --ref for longer prompts")
    if min(args.rounds, args.reps, args.new_tokens) < 1:
        p.error("rounds, reps and new-tokens must be positive")
    if any(v is not None and v < 1 for v in (args.gpu_chunk_size, args.inflight)):
        p.error("--gpu-chunk-size and --inflight must be positive")
    if not args.model and not args.dummy and not args.dry_run:
        p.error("--model is required without --dummy")
    if "heteromosaic_mdown" in args.methods and not args.mdown_config:
        p.error("heteromosaic_mdown needs --mdown-config")
    if "heteromosaic_gateup" in args.methods and not args.gateup_config:
        p.error("heteromosaic_gateup needs --gateup-config")
    manifests = {}
    merged_methods = [m for m in args.methods if is_merged(m)]
    if merged_methods:
        if "merged_ffn_p8192" in merged_methods and args.p8192_ffn_config is None:
            p.error("merged_ffn_p8192 needs --p8192-ffn-config deployment.json")
        if any(m != "merged_ffn_p8192" for m in merged_methods) and args.merged_ffn_config is None:
            p.error("merged methods need --merged-ffn-config deployment.json")
        if any(n % MERGED_M for n in args.prompt_tokens):
            p.error(f"merged methods need prompt lengths that are multiples of {MERGED_M}; "
                    f"other rows fall back to the original FFN and the completion count would not cover them")
        if {"merged_ffn", "merged_ffn_p8192"} & set(merged_methods) and args.gpu_chunk_size != MERGED_M:
            p.error(f"merged_ffn runs GPU attention with the gpu_chunking schedule: pass --gpu-chunk-size {MERGED_M} "
                    f"(applied to every method, so all methods share one chunk schedule)")
        for name in ("merged_ffn_config", "p8192_ffn_config"):
            if getattr(args, name) is not None:
                setattr(args, name, getattr(args, name).resolve())
        for method in merged_methods:
            problem, manifests[method] = check_deployment(merged_config(method, args))
            if method == "merged_ffn_p8192" and manifests[method] is not None and manifests[method].get("channels") != 8192:
                problem = problem or f"merged_ffn_p8192 needs 8192 NPU channels, got {manifests[method].get('channels')}"
            if problem and not args.dry_run:
                p.error(problem)
            if problem:
                print(f"dry run note: {problem}", flush=True)
    for method, length, round_id, cmd, env, out in cases(args):
        shown = " ".join(f"{k}={env[k]}" for k in SHOWN_ENV if k in env)
        print(f"{method} length={length} round={round_id}: [{shown}] {shlex.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        args.out.mkdir(parents=True, exist_ok=True)
        log = out.with_suffix(".log")
        with log.open("w") as f:
            subprocess.run(cmd, env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
        record = json.loads(out.read_text())
        text = log.read_text()
        record.update(method=method, round=round_id,
                      gpu_ffn_active="HeteroMega GPU FFN: 32 layers, 14336 prefill" in text,
                      merged_ffn_configured="HeteroMega merged FFN configured:" in text,
                      merged_ffn_expected_per_pass=expected_merged(method, length))
        if manifests.get(method) is not None:
            record.update(merged_ffn_config=str(merged_config(method, args)), merged_ffn_npu_channels=manifests[method].get("channels"))
        out.write_text(json.dumps(record, indent=2))
    if not args.dry_run:
        print(json.dumps(summarize(args.out), indent=2))


if __name__ == "__main__":
    main()
