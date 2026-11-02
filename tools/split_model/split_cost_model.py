#!/usr/bin/env python3
"""Split-ratio cost model for the merged GPU+NPU FFN (M=512, H=4096, I=14336). ESTIMATES ONLY.

    python3 tools/split_model/split_cost_model.py              # prints the table
    python3 tools/split_model/split_cost_model.py --report tools/split_model/REPORT.md

Per layer and 512-row chunk, NPU channels p, GPU channels I - p:
    requested form: T(p) = max(t_npu(p), t_gpu(I-p)) + t_sum + t_staging
    code-ordered:   T(p) = max(t_staging + t_npu(p), t_gpu(I-p)) + t_sum
The second follows merged_ffn.cpp:244-292: input staging, scans and BO copies run on the
host while the GPU branch executes, so they lengthen only the NPU side.

Every number is read from the evidence files in SOURCES. One measured NPU point exists
(P6400), so t_npu(p) = a_n + b_n * p needs an assumed fixed cost a_n. The staging part
that has no measurement (host finite scans) and the GPU slowdown under concurrent NPU work
are explicit scenario parameters, never fitted to a hoped-for answer.
"""
import argparse
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
I = 14336
VALID_P = list(range(256, 8192 + 1, 256))  # include/heteromega/merged_ffn_contract.hpp:95-99
CANDIDATES = [512, 1024, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 4096, 5120, 6400, 8192]
SOURCES = {
    "gpu": "kernels/gpu/README.md",  # bench_ffn_ablation M=512 rows: hm total and fused ratio (measured 2026-10-17)
    "smoke": "docs/runs/2026-10-31-ffn-deploy/deployment-results/cpp-smoke.json",  # merged P6400 calls (measured)
    "timeline": "experiments/ffn_partition/results/timeline_m512_p512_array8.json",  # D2H/H2D/reduction GPU events
    "bo_io": "experiments/ffn_partition/results/bottleneck_analysis.json",  # earlier design's host BO IO medians
    "artifact": "docs/runs/2026-10-31-ffn-deploy/deployment-results/artifact/final.xclbin.json",
    "profile": "docs/runs/2026-10-14-profile/evidence/analysis.json",  # in-model stages and native NPU waits (measured)
}
# Native gate/up NPU calls in the Oct-14 profile: instruction indices with 256x4096x14336 work. Inferred from the
# four equal ~3.24 ms completion waits and their gate/up readiness pattern; the README defers to the kernel map.
NATIVE_GATE_UP_INSTRUCTIONS = ("2", "3", "7", "8")
# name: (finite-scan ms for two 4 MiB scans, NPU fixed cost a_n ms, GPU slowdown under NPU load)
SCENARIOS = {"optimistic": (1.0, 0.5, 1.00), "central": (2.0, 1.0, 1.05), "pessimistic": (4.0, 3.0, 1.10)}


def read_gpu_points(root=ROOT):
    """{channels: fused 3-launch GPU FFN ms} from the README's M=512 ablation rows."""
    points = {}
    for line in (root / SOURCES["gpu"]).read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 7 and cells[0] == "512" and cells[1].isdigit():
            fused = float(re.match(r"[\d.]+", cells[3]).group())
            points[int(cells[1])] = float(cells[6]) * fused
    return points


def read_smoke(root=ROOT):
    """Median complete-call wall ms of the P6400 merged FFN by (layer, slot, serial), warmups excluded."""
    groups = {}
    for s in json.loads((root / SOURCES["smoke"]).read_text())["samples"]:
        if not s["warmup"]:
            groups.setdefault((s["layer"], s["slot"], s["serial"]), []).append(s["complete_host_wall_us"] / 1000)
    return {k: statistics.median(v) for k, v in groups.items()}


def read_staging(root=ROOT):
    events = json.loads((root / SOURCES["timeline"]).read_text())["stats"]["gpu_event_us"]
    rows = json.loads((root / SOURCES["bo_io"]).read_text())["rows"]
    bo = [r["median_ms"]["host_bo_io_outside_dispatch"] for r in rows if r["m"] == 512 and r["columns_max"] == 8]
    return dict(d2h_ms=events["input_d2h"]["median_us"] / 1000, h2d_ms=events["output_h2d"]["median_us"] / 1000,
                sum_ms=events["reduction"]["median_us"] / 1000, bo_io_ms=statistics.median(bo), bo_io_range=(min(bo), max(bo)))


def read_profile(root=ROOT):
    """Per layer-chunk FFN stage ms (synchronized diagnostic) and the native NPU GEMM rate."""
    data = json.loads((root / SOURCES["profile"]).read_text())
    ffn = {}
    for config in ("gpu_matched-detail-sync", "hetero_serial-detail-sync"):
        stages = data["stages"][config]
        keys = [k for k in stages if k.startswith("prefill/FFN_") and k != "prefill/FFN_norm"]
        ffn[config] = dict(stages={k: stages[k]["ms"]["median"] for k in keys},
                           per_layer_chunk_ms=sum(stages[k]["ms"]["median"] for k in keys) / 64)
    waits = data["npu_phases"]["hetero_serial-detail-sync"]["prefill"]["by_instruction"]
    wait_ms = statistics.median(waits[i]["completion_wait"]["median"] for i in NATIVE_GATE_UP_INSTRUCTIONS) / 1000
    macs_per_ms = 256 * 4096 * 14336 / wait_ms
    # Merged NPU work per channel at 512 rows: gate + up + down = 3 * 512 * 4096 MACs.
    return dict(ffn=ffn, native_wait_ms=wait_ms, native_tflops=2 * macs_per_ms / 1e9,
                native_b_n=3 * 512 * 4096 / macs_per_ms)


def calibrate(root=ROOT):
    gpu = read_gpu_points(root)
    (c1, t1), (c2, t2) = sorted(gpu.items())
    b_g = (t2 - t1) / (c2 - c1)
    smoke = read_smoke(root)
    staging = read_staging(root)
    p_measured = json.loads((root / SOURCES["artifact"]).read_text())["channels"]
    return dict(gpu_points=gpu, a_g=t1 - b_g * c1, b_g=b_g, smoke=smoke, staging=staging, p_measured=p_measured,
                profile=read_profile(root),
                path_concurrent=smoke[(0, 0, False)], path_serial=smoke[(0, 0, True)],
                path_concurrent_slot1=smoke.get((1, 1, False)))


def scenario_model(cal, name, b_n_override=None):
    scans, a_n, slowdown = SCENARIOS["central" if b_n_override else name]
    st = cal["staging"]
    t_staging = st["d2h_ms"] + st["h2d_ms"] + st["bo_io_ms"] + scans
    t_sum = st["sum_ms"]
    # NPU-bound calibration point: concurrent call = staging + t_npu(P6400) + sum (code-ordered form).
    t_npu_measured = cal["path_concurrent"] - t_staging - t_sum
    b_n = b_n_override or (t_npu_measured - a_n) / cal["p_measured"]

    def t_gpu(c):
        return 0.0 if c == 0 else slowdown * (cal["a_g"] + cal["b_g"] * c)

    def t_npu(p):
        return 0.0 if p == 0 else a_n + b_n * p

    def predict(p):
        requested = max(t_npu(p), t_gpu(I - p)) + t_sum + t_staging
        code = max(t_staging + t_npu(p), t_gpu(I - p)) + t_sum
        return dict(p=p, t_npu=t_npu(p), t_gpu=t_gpu(I - p), requested=requested, code_ordered=code)

    return dict(name=name, t_staging=t_staging, t_sum=t_sum, t_npu_measured=t_npu_measured, a_n=a_n, b_n=b_n,
                scans=scans, slowdown=slowdown, predict=predict, gpu_only=cal["a_g"] + cal["b_g"] * I)


def best(model, key):
    return min((model["predict"](p) for p in VALID_P), key=lambda r: r[key])


def report(cal):
    models = {n: scenario_model(cal, n) for n in SCENARIOS}
    c = models["central"]
    prof = cal["profile"]
    native = scenario_model(cal, "what-if: NPU at native rate", prof["native_b_n"])
    lines = ["# Split-ratio cost model: merged GPU+NPU FFN, M=512, H=4096, I=14336",
             "", "**All numbers are estimated** (model output) unless marked measured. Generated by "
             "`python3 tools/split_model/split_cost_model.py --report tools/split_model/REPORT.md`.", "",
             "## Inputs read from evidence files", "",
             "| Quantity | Value | Source | Status |", "| --- | --- | --- | --- |"]
    for ch, t in sorted(cal["gpu_points"].items()):
        lines.append(f"| GPU fused 3-launch FFN, {ch} channels | {t:.3f} ms | `{SOURCES['gpu']}` (hm total × fused ratio) | measured, isolated GPU, 2026-10-17 |")
    st = cal["staging"]
    lines += [f"| Merged FFN call P{cal['p_measured']}, concurrent (layer 0, stream 0) | {cal['path_concurrent']:.2f} ms | `{SOURCES['smoke']}` | measured, synthetic weights, n=7 |",
              f"| Same, serial (GPU branch then NPU) | {cal['path_serial']:.2f} ms | same | measured |",
              f"| Same, concurrent, layer 1 / stream 1 (no overlap observed) | {cal['path_concurrent_slot1']:.2f} ms | same | measured, unexplained |",
              f"| D2H of X / H2D of Y_npu (4 MiB, GPU events) | {st['d2h_ms']:.3f} / {st['h2d_ms']:.3f} ms | `{SOURCES['timeline']}` | measured, earlier 4-launch design |",
              f"| Boundary sum (add of two [512,4096] BF16) | {st['sum_ms']:.3f} ms | same | measured |",
              f"| Host BO write+read outside dispatch | {st['bo_io_ms']:.3f} ms (range {st['bo_io_range'][0]:.3f}-{st['bo_io_range'][1]:.3f}) | `{SOURCES['bo_io']}` (M512, 8 columns) | measured, earlier design |",
              "", "Derived (estimated): GPU branch at 7936 channels from the smoke, serial - concurrent = "
              f"{cal['path_serial'] - cal['path_concurrent']:.2f} ms, vs the linear GPU model "
              f"{cal['a_g'] + cal['b_g'] * (I - cal['p_measured']):.2f} ms. GPU model "
              f"t_gpu(c) = {cal['a_g']:.3f} + {cal['b_g'] * 1000:.4f} µs × c "
              f"(≈ {3 * 2 * 512 * 4096 / cal['b_g'] / 1e9:.1f} TFLOPS), interpolated in [10240, 14336], extrapolated below.", "",
              "## Scenarios (parameters the data cannot determine)", "",
              "| Scenario | Two finite scans (ms) | NPU fixed cost a_n (ms) | GPU slowdown under NPU | t_staging (ms) | t_npu(P6400) (ms) | b_n (µs/channel) | NPU TFLOPS |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in models.values():
        lines.append(f"| {m['name']} | {m['scans']:.1f} | {m['a_n']:.1f} | {m['slowdown']:.2f} | {m['t_staging']:.2f} | "
                     f"{m['t_npu_measured']:.2f} | {m['b_n'] * 1000:.3f} | {3 * 2 * 512 * 4096 / m['b_n'] / 1e9:.2f} |")
    lines += ["", f"## Predicted FFN time per layer-chunk (central scenario; GPU-only fused FFN = {c['gpu_only']:.2f} ms)", "",
              "| NPU channels p | Merged artifact / build | t_npu(p) | t_gpu(I-p) | T requested form | T code-ordered | vs GPU-only (code-ordered) | code-ordered, optimistic-pessimistic |",
              "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for p in CANDIDATES:
        r = c["predict"](p)
        lo, hi = sorted(models[n]["predict"](p)["code_ordered"] for n in SCENARIOS)[::len(SCENARIOS) - 1]
        art = "**exists** (Oct-31 artifact)" if p == cal["p_measured"] else "IRON build needed (valid: p % 256 == 0)"
        lines.append(f"| {p} | {art} | {r['t_npu']:.2f} | {r['t_gpu']:.2f} | {r['requested']:.2f} | {r['code_ordered']:.2f} | "
                     f"{(1 - r['code_ordered'] / c['gpu_only']) * 100:+.1f}% | {lo:.2f}-{hi:.2f} |")
    lines += ["", "| Scenario | Best p (requested form) | T | Best p (code-ordered) | T | Gain vs GPU-only (code-ordered) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for m in list(models.values()) + [native]:
        a, b = best(m, "requested"), best(m, "code_ordered")
        lines.append(f"| {m['name']} | {a['p']} | {a['requested']:.2f} | {b['p']} | {b['code_ordered']:.2f} | "
                     f"{(1 - b['code_ordered'] / m['gpu_only']) * 100:.1f}% |")
    per_chunk = cal["path_concurrent"] - c["gpu_only"]
    gpu_model = prof["ffn"]["gpu_matched-detail-sync"]["per_layer_chunk_ms"]
    main_b = prof["ffn"]["hetero_serial-detail-sync"]["per_layer_chunk_ms"]
    best_central = best(c, "code_ordered")
    lines += ["", "## Against the original heterogeneous FFN (Main B)", "",
              f"- Measured in-model FFN per layer-chunk (`{SOURCES['profile']}`, synchronized diagnostic, gate/up + "
              f"SiLU/multiply + down incl. residual, / 64): GPU-only {gpu_model:.2f} ms; original heterogeneous "
              f"M-split (Main B) **{main_b:.2f} ms**.",
              f"- Merged FFN best predicted (code-ordered): {best_central['code_ordered']:.2f} ms at p={best_central['p']} "
              "(central), range in the table above. **Under these estimates no split of the current merged graph beats "
              "Main B's FFN**; it can beat the GPU-only FFN.",
              f"- Why: the native NPU GEMM's measured completion wait for a 256x4096x14336 gate/up call is "
              f"{prof['native_wait_ms']:.2f} ms (instructions {', '.join(NATIVE_GATE_UP_INSTRUCTIONS)}, inferred mapping), "
              f"about {prof['native_tflops']:.1f} TFLOPS, vs about "
              f"{3 * 2 * 512 * 4096 / models['optimistic']['b_n'] / 1e9:.1f}-{3 * 2 * 512 * 4096 / models['pessimistic']['b_n'] / 1e9:.1f} "
              "TFLOPS estimated for the merged graph. The merged graph's NPU work is roughly 3x slower per FLOP.",
              f"- What-if (estimated): at the native rate (b_n = {prof['native_b_n'] * 1000:.3f} µs/channel, central "
              f"staging), the best split is p={best(native, 'code_ordered')['p']} with "
              f"{best(native, 'code_ordered')['code_ordered']:.2f} ms. The NPU graph's efficiency, more than the split "
              "ratio, decides whether persistent channel partitioning can beat Main B.",
              "", "## Consistency checks", "",
              f"- At P{cal['p_measured']} the model is NPU-bound: per layer-chunk about {per_chunk:.1f} ms slower than "
              f"GPU-only, i.e. about {32 * per_chunk / 1000:.2f} s per 512-token and {64 * per_chunk / 1000:.2f} s per "
              "1024-token prefill (estimated). The local session reports `eval-main` medians of about 0.9 → 1.3 s "
              "(512) and 1.8 → 2.5 s (1024) for gpu_original → merged_ffn. These are peer-reported, not yet in this "
              "repository, and not verified here.",
              "- `npu/ffn/BFP16_ADAPTATION.md` documents 19.1-20.3% lower isolated FFN latency than GPU-only at "
              "P2560-2816 for an earlier two-launch BFP16 design. Its evidence directory is not in this checkout, so it is "
              "not used for fitting; it is consistent with the code-ordered optimum here.", "",
              "## What the data cannot determine", "",
              "1. NPU scaling with p: one merged-graph point (P6400) cannot separate fixed cost from per-channel "
              "cost, or show non-linearity from column/tile granularity (P/8 per column, N32 pairs).",
              "2. GPU branch below 10240 channels: extrapolated; small-N GEMM efficiency is unmeasured.",
              "3. Staging split: the C++ path's D2H, finite scans, BO write/read and H2D are not timed separately; "
              "BO IO and copies come from the earlier Python design; scan cost is a scenario parameter.",
              "4. Interference: the GPU slowdown under concurrent NPU DRAM traffic is a scenario parameter "
              "(the smoke's 12.3 ms vs the 11.2 ms isolated model suggests a few percent, confounded by harness differences).",
              "5. Whether the requested or the code-ordered form holds: they differ by where staging sits; "
              "the code says code-ordered, which is unverified on hardware.",
              "6. Model-level effects (attention, the other chunk, decode fallback) are outside this per-FFN model.", "",
              "## Experiment on the AUP box to fill the gaps", "",
              "1. GPU curve, isolated: `./build/gpu/bin/bench_gpu_ffn_slice --m 512 --s 6144,7936,8192,10240,11264,11776,12032,12288,14336 --prefix`.",
              "2. NPU points: for p in 2560, 3072 and 4096, `python -m experiments.ffn_deploy.prepare --build --iron-python "
              "/home/mliraie/mlir-aie/ironenv/bin/python --artifact-dir \"$PWD/artifacts/deploy-m512-h4096-p$p\" --m 512 "
              "--hidden 4096 --intermediate 14336 --npu-channels $p --output deployment-p$p.json`. Record compile time; "
              "it is unknown.",
              "3. Per p: `python -m experiments.ffn_deploy.smoke --config deployment-p$p.json --iterations 3 --output ...`. "
              "Concurrent call = staging + t_npu(p) + sum; serial − concurrent ≈ t_gpu(I−p). Three p values fit a_n and "
              "b_n directly.",
              "4. Per p: `evaluate_prefill.py --methods gpu_original merged_ffn --merged-ffn-config \"$PWD/deployment-p$p.json\" "
              "--gpu-chunk-size 512 --prompt-tokens 512 1024` in a fresh output directory, compared with P6400.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, help="write the markdown report here")
    args = ap.parse_args()
    text = report(calibrate())
    if args.report:
        args.report.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
