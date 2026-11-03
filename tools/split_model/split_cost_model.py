#!/usr/bin/env python3
"""Split-ratio cost model for the merged GPU+NPU FFN (M=512, H=4096, I=14336), refit on 2026-11-02.

    python3 tools/split_model/split_cost_model.py              # prints the report
    python3 tools/split_model/split_cost_model.py --report tools/split_model/REPORT.md

Per layer and 512-row chunk, NPU channels p, GPU channels I - p, code order (merged_ffn.cpp) and measured phases:
    T(p) = max(host + t_npu(p), t_gpu(I - p)) + tail + other
    prefill(p) = 64 * (since_previous_call + wait_for_attention + T(p))
host is D2H, scans and BO work; tail is H2D and the sum; other is what the phase medians leave out at P6400.

t_npu and t_gpu take their slopes from the four measured split points of the Oct 28 sweep. t_npu passes through the
in-model NPU wait at P6400. The in-model GPU branch has no timer of its own, only an upper bound (it finished before
the NPU path), so its level at 7936 channels is a scenario parameter. Predictions are estimates; inputs are measured.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
I = 14336
LAYER_CHUNKS = 64  # 32 layers x 2 chunks of 512 rows at 1024 tokens
P_MEASURED = 6400
FLOPS_PER_CHANNEL = 3 * 2 * 512 * 4096  # gate, up and down over 512 rows and 4096 hidden channels
VALID_P = list(range(256, 8192 + 1, 256))  # include/heteromega/merged_ffn_contract.hpp:95-99
TABLE_P = [4352, 5120, 5632, 5888, 6144, 6400, 6656, 6912, 7168, 8192]
SOURCES = {
    "phases": "docs/runs/2026-11-02-fix-step2/model-phases.json",  # in-model phases, -O3, sampled check (measured)
    "smoke": "docs/runs/2026-11-02-fix-step2/smoke-phases.json",  # GPU branch alone in the serial smoke (measured)
    "sweep": "tools/split_model/oct28_partition_points.json",  # four split points from the Oct 28 harness (measured)
    "gpu": "kernels/gpu/README.md",  # isolated fused GPU FFN at 10240 and 14336 channels (measured 2026-10-17)
    "profile": "docs/runs/2026-10-14-profile/evidence/analysis.json",  # native NPU waits and stages (measured)
    "first_eval": "docs/runs/2026-11-02-eval-merged-gpu/summary.csv",  # no -O, full scans (measured)
    "eval": "docs/runs/2026-11-02-fix-step2b/eval-sampled/summary.csv",  # headline gpu_original vs merged_ffn (measured)
    "h2": "docs/runs/2026-11-02-fix-step3-h2/summary.csv",  # heteromosaic vs merged_ffn_hetero (measured)
    "mdown": "docs/runs/2026-11-02-fix-step4-mdown/summary.csv",  # heteromosaic vs M-split down (measured)
    "head": "docs/runs/2026-11-02-fix-step3c/head-to-head/summary.csv",  # heteromosaic vs merged_ffn (measured)
}
NATIVE_GATE_UP_INSTRUCTIONS = ("2", "3", "7", "8")  # 256x4096x14336 work, inferred; see kernels/gpu/README.md
# Measured in docs/runs/2026-11-02-fix-step3c/README.md (per-call medians from the timed ioctl traces).
CONTEXT_CHANGE_MS = {"heteromosaic_mdown": (2.85, 2.93), "merged_ffn_hetero": (2.93, 2.04)}


def read_gpu_points(root=ROOT):
    """{channels: fused 3-launch GPU FFN ms} from the README's M=512 ablation rows."""
    points = {}
    for line in (root / SOURCES["gpu"]).read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 7 and cells[0] == "512" and cells[1].isdigit():
            points[int(cells[1])] = float(cells[6]) * float(cells[3].split()[0].rstrip("x"))
    return points


def read_summary(root, key):
    """{(method, prompt_tokens): prefill median ms} from one measured evaluation summary."""
    with (root / SOURCES[key]).open() as f:
        return {(r["method"], int(r["prompt_tokens"])): float(r["prefill_median_ms"]) for r in csv.DictReader(f)}


def read_native(root=ROOT):
    data = json.loads((root / SOURCES["profile"]).read_text())
    waits = data["npu_phases"]["hetero_serial-detail-sync"]["prefill"]["by_instruction"]
    wait_ms = statistics.median(waits[i]["completion_wait"]["median"] for i in NATIVE_GATE_UP_INSTRUCTIONS) / 1000
    return dict(wait_ms=wait_ms, tflops=2 * 256 * 4096 * 14336 / wait_ms / 1e9)


def line_slope(xs, ys):
    """Least-squares slope and intercept."""
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return slope, my - slope * mx


def calibrate(root=ROOT):
    phases = json.loads((root / SOURCES["phases"]).read_text())["all"]
    smoke = json.loads((root / SOURCES["smoke"]).read_text())["groups"]
    sweep = json.loads((root / SOURCES["sweep"]).read_text())["points"]
    b_npu, a_npu = line_slope([s["npu_channels"] for s in sweep], [s["npu_wait_concurrent_ms"] for s in sweep])
    b_gpu, a_gpu = line_slope([I - s["npu_channels"] for s in sweep], [s["gpu_branch_serial_ms"] for s in sweep])
    host = sum(phases[k + "_ms"] for k in ("d2h", "input_scan", "bo_write", "bo_sync", "run_start", "y_sync", "y_read", "output_scan"))
    tail = phases["h2d_ms"] + phases["sum_ms"]
    t_npu_measured = phases["run_wait_ms"]
    ffn_measured = phases["total_ms"] - phases["input_wait_ms"]
    gpu_points = read_gpu_points(root)
    (c1, t1), (c2, t2) = sorted(gpu_points.items())
    isolated_slope = (t2 - t1) / (c2 - c1)
    return dict(phases=phases, sweep=sweep, b_npu=b_npu, a_npu=a_npu, b_gpu=b_gpu, a_gpu=a_gpu, host=host, tail=tail,
                t_npu_measured=t_npu_measured, ffn_measured=ffn_measured, other=ffn_measured - host - t_npu_measured - tail,
                outside=phases["since_prev_end_ms"] + phases["input_wait_ms"],
                gpu_upper=host + t_npu_measured, gpu_isolated=t1 + isolated_slope * (I - P_MEASURED - c1),
                gpu_smoke_serial=smoke["slot0_serial"]["serial_gpu_ms"], gpu_points=gpu_points, isolated_slope=isolated_slope,
                native=read_native(root),
                first_eval=read_summary(root, "first_eval"), eval=read_summary(root, "eval"), h2=read_summary(root, "h2"),
                mdown=read_summary(root, "mdown"), head=read_summary(root, "head"))


def scenarios(cal):
    """Level of the in-model GPU branch at 7936 channels: its measured upper bound, its isolated value, the midpoint."""
    high, low = cal["gpu_upper"], cal["gpu_isolated"]
    return {"upper bound": high, "midpoint": (high + low) / 2, "isolated": low}


def predict(cal, gpu_at_7936, p):
    t_npu = cal["t_npu_measured"] + cal["b_npu"] * (p - P_MEASURED)
    t_gpu = gpu_at_7936 + cal["b_gpu"] * (I - p - (I - P_MEASURED))
    t = max(cal["host"] + t_npu, t_gpu) + cal["tail"] + cal["other"]
    return dict(p=p, t_npu=t_npu, t_gpu=t_gpu, T=t, prefill=LAYER_CHUNKS * (cal["outside"] + t))


def best(cal, gpu_at_7936, candidates=VALID_P):
    return min((predict(cal, gpu_at_7936, p) for p in candidates), key=lambda r: r["T"])


def report(cal):
    scen = scenarios(cal)
    ph = cal["phases"]
    ev, h2, md, hd, first = cal["eval"], cal["h2"], cal["mdown"], cal["head"], cal["first_eval"]
    tflops = FLOPS_PER_CHANNEL / cal["b_npu"] / 1e9
    gains = {n: predict(cal, v, P_MEASURED)["prefill"] - best(cal, v)["prefill"] for n, v in scen.items()}
    gain_max = max(gains.values())
    prefill_6400 = LAYER_CHUNKS * (cal["outside"] + cal["ffn_measured"])
    lines = ["# Split-ratio cost model: merged GPU+NPU FFN, M=512, H=4096, I=14336", "",
             "Refit on 2026-11-02 from in-model phase timers and the four measured split points of the Oct 28 sweep. "
             "Inputs are measured; everything in the prediction table is estimated. Generated by "
             "`python3 tools/split_model/split_cost_model.py --report tools/split_model/REPORT.md`.", "",
             "The decision below was derived at one inflight chunk. The measured two-inflight results in "
             "`docs/runs/2026-11-02-inflight2-*/` are not folded into this report yet.", "",
             "## Decision", "",
             f"**Do not build other split ratios for prefill latency.** P{P_MEASURED} is within "
             f"{gain_max:.0f} ms of the best valid split in every scenario ({gain_max / prefill_6400 * 100:.1f}% of a "
             f"{prefill_6400:.0f} ms prefill; estimated). The measured sweep shows no winner either: P6144 at "
             f"{cal['sweep'][1]['concurrent_ms']:.2f} ms and P6656 at {cal['sweep'][5]['concurrent_ms']:.2f} ms sit inside the "
             f"spread of the three P6400 blocks ({min(s['concurrent_ms'] for s in cal['sweep'][2:5]):.2f}-"
             f"{max(s['concurrent_ms'] for s in cal['sweep'][2:5]):.2f} ms).", "",
             "What decides the result is the NPU context change, not the split: see the prefill table below.", "",
             "## Corrections to the previous version of this report", "",
             "| Previous claim | Status | Evidence (measured) |", "| --- | --- | --- |",
             f"| Merged FFN call takes 32.46 ms at P{P_MEASURED} and is NPU-bound by far | Wrong input: no `-O`, and two unoptimized "
             f"scans of about 10 ms each | {cal['ffn_measured']:.2f} ms with `-O3` and the `sampled` check |",
             f"| Merged graph is about 3x slower per FLOP than native (2.8-3.4 TFLOPS) | Wrong: the scans were charged to the NPU | "
             f"{tflops:.1f} TFLOPS from the sweep slope; native {cal['native']['tflops']:.1f} TFLOPS |",
             "| Best split is p = 2560-2816 | Wrong: followed from the two errors above | "
             f"balanced at about p = {P_MEASURED} |",
             f"| No split beats Main B's FFN | Superseded: the 14.20 ms was a synchronized diagnostic | "
             f"`merged_ffn` {hd[('merged_ffn', 1024)]:.1f} ms vs `heteromosaic` {hd[('heteromosaic', 1024)]:.1f} ms per whole prefill |",
             "| Layer 1 / stream 1 loses overlap, unexplained | Fixed by waiting for input readiness before the D2H copy | "
             "`docs/runs/2026-11-02-fix-step2b/README.md`; cause is a hypothesis |", "",
             "## Whole prefill at 1024 tokens (measured, synchronized `generate(1)`)", "",
             "| Configuration | Attention | FFN | NPU contexts | Prefill median ms | Paired against |", "| --- | --- | --- | ---: | ---: | --- |",
             f"| `gpu_original` | GPU | GPU | 0 | {ev[('gpu_original', 1024)]:.1f} | `merged_ffn` {ev[('merged_ffn', 1024)]:.1f} |",
             f"| `merged_ffn`, no `-O`, full scans (first evaluation) | GPU | merged | 1 | {first[('merged_ffn', 1024)]:.1f} | `gpu_original` {first[('gpu_original', 1024)]:.1f} |",
             f"| `merged_ffn`, `-O3`, `sampled` | GPU | merged | 1 | {hd[('merged_ffn', 1024)]:.1f} | `heteromosaic` {hd[('heteromosaic', 1024)]:.1f} |",
             f"| `heteromosaic` | native NPU | native gate/up | 2 | {h2[('heteromosaic', 1024)]:.1f} | `merged_ffn_hetero` {h2[('merged_ffn_hetero', 1024)]:.1f} |",
             f"| `merged_ffn_hetero` | native NPU | merged | 3 | {h2[('merged_ffn_hetero', 1024)]:.1f} | `heteromosaic` {h2[('heteromosaic', 1024)]:.1f} |",
             f"| `heteromosaic_mdown` | native NPU | native gate/up/down | 3 | {md[('heteromosaic_mdown', 1024)]:.1f} | `heteromosaic` {md[('heteromosaic', 1024)]:.1f} |", "",
             "Context-change arithmetic from `docs/runs/2026-11-02-fix-step3c/README.md` (estimated):", ""]
    for method, summary in (("heteromosaic_mdown", md), ("merged_ffn_hetero", h2)):
        back, forward = CONTEXT_CHANGE_MS[method]
        cost = (back + forward) * LAYER_CHUNKS
        lines.append(f"- `{method}`: {back:.2f} + {forward:.2f} ms per layer-chunk, {cost:.0f} ms per prefill. Without it: "
                     f"{summary[(method, 1024)] - cost:.0f} ms.")
    lines += ["", "No artifact exists that removes these changes; whether one PDI can hold both workloads is unknown. "
              "These are latency estimates only. `heteromosaic_mdown` used the artifact family `16384x14336x4096/bf16_int4AWQ_bf16_M`, and its "
              "latency and context changes are measured for that family. It also fails numerically "
              "(`docs/runs/2026-11-02-fix-step5b-sampled/README.md`), but that verdict is suspect: no shipped entry uses this artifact, "
              "and whether the failure comes from it or from native down is not established.", "",
              "## Inputs", "",
              "| Quantity | Value | Source | Status |", "| --- | --- | --- | --- |",
              f"| NPU wait in the model at P{P_MEASURED} | {cal['t_npu_measured']:.3f} ms | `{SOURCES['phases']}` | measured, 640 calls |",
              f"| Host path: D2H, scans, BO write/sync/start, y sync/read | {cal['host']:.3f} ms | same | measured |",
              f"| Tail: H2D and sum | {cal['tail']:.3f} ms | same | measured |",
              f"| Other: FFN proper minus the three rows above | {cal['other']:.3f} ms | same | measured; medians are not additive |",
              f"| FFN proper at P{P_MEASURED} | {cal['ffn_measured']:.3f} ms | same | measured |",
              f"| Between FFNs: host enqueue and GPU attention | {cal['outside']:.3f} ms | same | measured |",
              f"| GPU branch finished before the NPU path; wait after it | {ph['gpu_wait_ms']:.3f} ms | same | measured; so in-model t_gpu(7936) <= {cal['gpu_upper']:.2f} ms |",
              f"| GPU branch alone, serial smoke | {cal['gpu_smoke_serial']:.2f} ms | `{SOURCES['smoke']}` | measured, synthetic weights |",
              f"| Fused GPU FFN, isolated, linear in channels, at 7936 | {cal['gpu_isolated']:.2f} ms | `{SOURCES['gpu']}` | estimated from two isolated points |",
              f"| NPU slope b_npu | {cal['b_npu'] * 1000:.3f} µs/channel, {tflops:.1f} TFLOPS; fitted fixed cost {cal['a_npu']:.2f} ms | `{SOURCES['sweep']}` | fitted to 6 measured blocks at 4 splits |",
              f"| GPU slope b_gpu | {cal['b_gpu'] * 1000:.3f} µs/channel (isolated: {cal['isolated_slope'] * 1000:.3f}) | same | fitted to the same blocks |", "",
              "## Measured split sweep (Oct 28 Python harness, one layer, synthetic weights)", "",
              "| p | Samples per mode | Complete FFN, concurrent ms | NPU wait ms | GPU branch, serial ms |", "| ---: | ---: | ---: | ---: | ---: |"]
    for s in cal["sweep"]:
        lines.append(f"| {s['npu_channels']} | {s['samples_per_mode']} | {s['concurrent_ms']:.3f} | {s['npu_wait_concurrent_ms']:.3f} | {s['gpu_branch_serial_ms']:.3f} |")
    names = list(scen)
    lines += ["", "This harness's complete FFN is about 1.3-2.0 ms above the C++ path's at P6400. Its P4352 point is the only one far from balance.", "",
              "## Predicted FFN per layer-chunk and prefill by split (estimated)", "",
              "Scenario values are the in-model GPU branch at 7936 channels: " +
              ", ".join(f"{n} {v:.2f} ms" for n, v in scen.items()) + ".", "",
              "| p | t_npu(p) | t_gpu(I-p), " + names[0] + " | T(p), " + " | T(p), ".join(names) + " | Prefill, " + " | Prefill, ".join(names) + " |",
              "| ---: | ---: | ---: | " + "---: | " * (2 * len(names))]
    for p in TABLE_P:
        rows = [predict(cal, scen[n], p) for n in names]
        lines.append(f"| {p} | {rows[0]['t_npu']:.2f} | {rows[0]['t_gpu']:.2f} | " + " | ".join(f"{r['T']:.2f}" for r in rows) + " | " +
                     " | ".join(f"{r['prefill']:.0f}" for r in rows) + " |")
    lines += ["", "| Scenario | Best valid p | T | Prefill | Gain vs P6400 |", "| --- | ---: | ---: | ---: | ---: |"]
    for n in names:
        b = best(cal, scen[n])
        lines.append(f"| {n} | {b['p']} | {b['T']:.2f} | {b['prefill']:.0f} | {gains[n]:.0f} ms |")
    lines += ["", "## Consistency checks (measured against the model)", "",
              f"- Phases at P{P_MEASURED}: {LAYER_CHUNKS} x ({cal['outside']:.3f} + {cal['ffn_measured']:.3f}) = {prefill_6400:.1f} ms; "
              f"paired evaluation {ev[('merged_ffn', 1024)]:.1f} ms.",
              f"- Slope check: P4352 predicts a GPU-bound FFN of about {predict(cal, cal['gpu_upper'], 4352)['t_gpu']:.1f} ms; the sweep's concurrent "
              f"P4352 is {cal['sweep'][0]['concurrent_ms']:.2f} ms with a {cal['sweep'][0]['gpu_branch_serial_ms']:.2f} ms GPU branch, so it is GPU-bound too.",
              "", "## What the data cannot determine", "",
              "1. The in-model GPU branch has no end timer, only the upper bound above. A completion timestamp would fix its level.",
              "2. The sweep's NPU waits are noisy: three P6400 blocks range over "
              f"{max(s['npu_wait_concurrent_ms'] for s in cal['sweep'][2:5]) - min(s['npu_wait_concurrent_ms'] for s in cal['sweep'][2:5]):.2f} ms, "
              f"which is more than the {cal['sweep'][5]['npu_wait_concurrent_ms'] - cal['sweep'][1]['npu_wait_concurrent_ms']:.2f} ms between P6144 and P6656.",
              "3. With overlapping attention (two inflight chunks) the balance could move toward more NPU channels. Two inflight "
              "chunks gain nothing today (`docs/runs/2026-11-02-fix-step2b/README.md`), so this model does not cover it.",
              "4. This model is latency only. Numerics are in `docs/runs/2026-11-02-fix-step5-numerics/` and "
              "`docs/runs/2026-11-02-fix-step5b-sampled/`.", ""]
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
