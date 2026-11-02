#!/usr/bin/env python3
"""Run the NPU W4A16 arithmetic emulation experiments (CPU only, NumPy).

Experiment A reproduces the exact inputs of the October 31 isolated C++ smoke
(contract.random_weights seeds 12832/12833 and its seeded normal inputs; the input
SHA256 is checked against the retained cpp-smoke.json) and compares the emulated
NPU P6400 branch error with the error measured on AUP for the deployed artifact.

Experiment B emulates the first original M-sweep shape (M=N=K=2048, sweep input
distributions from src/microbenchmarks/tensor_parallel_gemm_npu_gpu_sweepM.cpp) to
test whether intermediate accumulation/rounding reproduces the observed maximum
GPU/NPU difference of 1.5. That difference came from precompiled hw_bins/npu2
binaries without source here; the deployed kernel's arithmetic is the hypothesis.

Every tolerance is the existing one: contract.error_metrics defaults (rtol 0.03,
atol 0.05, relative L2 0.02) and the sweep's allclose(rtol=0.2, atol=0.2).
Nothing here is a hardware measurement; all emulated numbers are estimates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from w4a16_npu_emulation import (DEPLOYED, bf16, bfp16_ebs8, exact_projection,  # noqa: E402
                                 gpu_like_projection, npu_ffn_branch, npu_projection)
from experiments.ffn_partition.contract import (error_metrics, random_weights,  # noqa: E402
                                                reference_full, split_weights)

SMOKE = ROOT / "docs/runs/2026-10-31-ffn-deploy/deployment-results/cpp-smoke.json"
SWEEP_RTOL = SWEEP_ATOL = 0.2  # tensor_parallel_gemm_npu_gpu_sweepM.cpp:40-41
MEASURED_SWEEP = {  # docs/runs/2026-10-13-compat/evidence/*.log, max_abs_diff NPU split vs GPU
    "random": 1.5, "exact_scale": 1.25, "constant_weight": 0.25, "exact_matrix": 0.0}

ABLATIONS = {  # name: (projection arithmetic, SiLU)
    "deployed": (dict(DEPLOYED), "npu"),
    "deployed_exact_silu": (dict(DEPLOYED), "exact"),
    "fp32_partials": (dict(DEPLOYED, bf16_partials=False), "npu"),
    "bf16_partials_only": (dict(bfp16_a=False, bfp16_b=False, bf16_partials=True), "npu"),
    "bfp16_a_only": (dict(bfp16_a=True, bfp16_b=False, bf16_partials=False), "npu"),
    "bfp16_b_only": (dict(bfp16_a=False, bfp16_b=True, bf16_partials=False), "npu"),
    "silu_lut_only": (dict(bfp16_a=False, bfp16_b=False, bf16_partials=False), "npu"),
    "deployed_renormalize": (dict(DEPLOYED, overflow="renormalize"), "npu"),
    "deployed_9bit_mantissa": (dict(DEPLOYED, mantissa_bits=9), "npu"),
}


def summarize(metrics):
    keys = ("passed", "max_abs_error", "relative_l2_error", "rms_error", "reference_max_abs", "max_error_limit")
    return {k: metrics.get(k) for k in keys}


def worst(actual, expected, other=None, count=5):
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    flat = np.argsort(delta, axis=None)[::-1][:count]
    rows = []
    for index in flat:
        r, c = np.unravel_index(index, delta.shape)
        row = dict(row=int(r), col=int(c), emulated=float(actual[r, c]), reference=float(expected[r, c]),
                   abs_error=float(delta[r, c]))
        if other is not None:
            row["gpu_like"] = float(other[r, c])
        rows.append(row)
    return rows


def experiment_smoke(args, failing):
    p = 6400
    measured = [c["npu"] for c in json.loads(SMOKE.read_text())["checks"] if c["npu"]["reference_max_abs"]]
    measured_by_refmax = {}
    for m in measured:
        measured_by_refmax.setdefault(round(m["reference_max_abs"], 6), summarize(m))
    rng = np.random.default_rng(12833)
    inputs = [bf16(rng.normal(size=(512, 4096)).astype(np.float32)) for _ in range(2)]
    shas = [hashlib.sha256(x.tobytes()).hexdigest() for x in inputs]
    if shas != json.loads(SMOKE.read_text())["input_sha256"][:2]:
        raise RuntimeError("reconstructed smoke inputs do not match the retained input SHA256")
    results = dict(scope="NPU branch P6400 of the isolated smoke; reference = contract.reference_full on the NPU slice",
                   input_sha256_match=True, cases=[])
    layers = [0] if args.quick else [0, 1]
    for layer in layers:
        branch = split_weights(random_weights(4096, 14336, 12832 + layer), p)["npu"]
        for index, x in enumerate(inputs if not args.quick else inputs[:1]):
            reference = reference_full(x, branch)
            refmax = round(float(np.max(np.abs(reference))), 6)
            case = dict(layer=layer, input=index, reference_max_abs=refmax,
                        measured_aup=measured_by_refmax.get(refmax), emulated={})
            names = ["deployed"] if (layer, index) != (0, 0) else list(ABLATIONS)
            for name in names:
                arith, silu = ABLATIONS[name]
                order = "fp32" if name == "deployed" else "exact"
                start = time.perf_counter()
                out = npu_ffn_branch(x, branch, silu=silu, k8_rounding=order, **arith)
                metrics = error_metrics(out, reference)
                case["emulated"][name] = dict(summarize(metrics), k8_rounding=order,
                                              cpu_seconds=round(time.perf_counter() - start, 1))
                if not metrics["passed"]:
                    failing.append(dict(experiment="smoke_p6400", layer=layer, input=index, config=name,
                                        check="contract.error_metrics(rtol=0.03, atol=0.05, l2_limit=0.02)",
                                        metrics=summarize(metrics), worst_elements=worst(out, reference)))
                print(f"smoke layer={layer} input={index} {name}: {case['emulated'][name]}", flush=True)
            results["cases"].append(case)
    return results


def sweep_inputs(variant, rows, seed):
    """Distributions of tensor_parallel_gemm_npu_gpu_sweepM.cpp:308-326 and the evidence variants."""
    rng = np.random.default_rng(seed)
    k = n = 2048
    if variant == "exact_matrix":
        a = np.full((rows, k), 1 / 16, np.float32)
    else:
        a = bf16(bf16(rng.random((rows, k), dtype=np.float32)) * np.float32(0.1))
    q = rng.integers(0, 16, (n, k)) if variant == "random" or variant == "exact_scale" else np.full((n, k), 9)
    scales = (bf16(bf16(rng.random((n, k // 128), dtype=np.float32)) * np.float32(0.1)) if variant == "random"
              else np.full((n, k // 128), 0.0625, np.float32))
    zeros = rng.integers(0, 16, (n, k // 128)) if variant in ("random", "exact_scale") else np.full((n, k // 128), 8)
    return a, q, scales, zeros


def experiment_sweep(args, failing):
    rows = 64 if args.quick else 256
    seeds = [0] if args.quick else [0, 1, 2]
    configs = {k: ABLATIONS[k][0] for k in ("deployed", "fp32_partials", "bf16_partials_only",
                                            "bfp16_a_only", "bfp16_b_only")}
    results = dict(scope=f"M={rows} NPU rows x N=2048 x K=2048 (smallest measured npuM=256 also gave 1.5)",
                   measured_max_abs_diff=MEASURED_SWEEP, cases=[])
    for variant in MEASURED_SWEEP:
        for seed in seeds:
            a, q, scales, zeros = sweep_inputs(variant, rows, seed)
            gpu = gpu_like_projection(a, q, scales, zeros)
            exact = exact_projection(a, q, scales, zeros)
            case = dict(variant=variant, seed=seed, reference_abs_max=float(np.max(np.abs(exact))),
                        gpu_like_vs_exact_max=float(np.max(np.abs(gpu - exact))), configs={})
            for name, arith in configs.items():
                order = "fp32" if name == "deployed" else "exact"
                out = npu_projection(a, q, scales, zeros, k8_rounding=order, **arith)
                diff = np.abs(out.astype(np.float64) - gpu)
                allclose = bool(np.all(diff <= SWEEP_ATOL + SWEEP_RTOL * np.abs(gpu)))
                case["configs"][name] = dict(max_abs_diff_vs_gpu_like=float(diff.max()),
                                             last_row_max_error_vs_exact=float(np.max(np.abs(out[-1] - exact[-1]))),
                                             sweep_allclose=allclose, k8_rounding=order)
                if not allclose:
                    failing.append(dict(experiment="sweep_2048", variant=variant, seed=seed, config=name,
                                        check="allclose(rtol=0.2, atol=0.2) vs GPU-like output",
                                        worst_elements=worst(out, gpu, count=5)))
            print(f"sweep {variant} seed={seed}: {json.dumps(case['configs'])}", flush=True)
            results["cases"].append(case)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--quick", action="store_true", help="small subset for a fast CPU check")
    ap.add_argument("--only", choices=("smoke", "sweep"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    failing, report = [], dict(numpy=np.__version__, quick=args.quick,
                               label="estimated (CPU emulation); not a hardware measurement")
    if args.only in (None, "sweep"):
        report["sweep"] = experiment_sweep(args, failing)
    if args.only in (None, "smoke"):
        report["smoke"] = experiment_smoke(args, failing)
    suffix = "-quick" if args.quick else ""
    (args.out / f"emulation-results{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / f"failing-examples{suffix}.json").write_text(json.dumps(failing, indent=2) + "\n")
    print(f"wrote {args.out}; {len(failing)} failing configurations retained")


if __name__ == "__main__":
    main()
