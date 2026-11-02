#!/usr/bin/env python3
"""Tabulate NPU error-probe lines and state which hypothesis the data supports.

    python3 tools/npu_error_probe/probe_report.py RESULT_DIR/probe.jsonl [--json OUT.json]
    python3 tools/npu_error_probe/probe_report.py --emulate OUT.jsonl     # CPU predictions, no hardware

Quantities per case: mean absolute error of the NPU (e_npu) and of the GPU control
(e_gpu) against the FP32 reference, on identical inputs and packed bytes, and their
ratio r = e_npu / e_gpu. The GPU control applies the same BF16 weight rounding and a
single BF16 output rounding, so r isolates NPU-specific arithmetic; raw errors also grow
with K simply because outputs grow with K.

Decision rule (thresholds fixed here before any hardware data; never tuned to a run):

1. Accumulation precision / partial-sum rounding: in a K series, the slope of
   log(r) against log(K) is >= 0.25, i.e. the NPU excess error grows with K.
   The raw slope of log(e_npu) against log(K) is printed as well: >= 0.75 is
   "about linear in K"; random-walk accumulation of rounded partials typically gives
   0.5-1.0 raw while r still rises.
2. Per-element dequantization / operand format: |slope of log(r)| < 0.25 (flat in K
   relative to the control) while median r > 1.5.
3. Signed-input packing / sign handling: all-negative activations are the exact
   negation of the all-positive ones (same weights), so sign-symmetric arithmetic
   (round-to-nearest-even, block formats) gives NPU(-a) == -NPU(a) bit for bit and
   equal errors. The rule fires when the saved outputs are not exact negations or
   r(negative)/r(positive) is outside [0.8, 1.25]. Mixed-sign r is printed but not
   used: its outputs are near zero, so the GPU control's error shrinks and r rises
   even for sign-symmetric arithmetic (CPU predictions show this).
4. Quantization metadata (scale or zero-point addressing): in the metadata series,
   r with random scales and random zeros is > 2 x r with equal scales and zeros of 8.
   The single-factor cases name the factor (scales or zeros) whose r is > 2 x baseline.
5. No NPU-specific error: every r <= 1.5.
Rules are evaluated independently; several can hold. A series with fewer than two
successful cases is "not determined". Blocked cases (no native artifact) are listed.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SLOPE = 0.25
LINEAR_RAW_SLOPE = 0.75
RATIO_FLAT = 1.5
FACTOR = 2.0
SIGN_LOW, SIGN_HIGH = 0.8, 1.25


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def ratio(record):
    e_gpu = record["gpu"]["mean_abs"]
    return record["npu"]["mean_abs"] / e_gpu if e_gpu > 0 else float("inf")


def slope(points):
    """Least-squares slope of log(y) vs log(x)."""
    x, y = np.log([p[0] for p in points]), np.log([p[1] for p in points])
    return float(np.polyfit(x, y, 1)[0])


def verdicts(records, directory=None):
    ok = [r for r in records if r.get("status") == "ok"]
    by_name = {r["name"]: r for r in ok}
    found, details = [], {}
    k_slopes = {}
    for series in ("k_n2048", "k_n4096"):
        points = sorted((r["k"], r) for r in ok if r["series"] == series)
        if len(points) < 2:
            details[series] = "not determined (fewer than two successful cases)"
            continue
        ratio_slope = slope([(k, ratio(r)) for k, r in points])
        raw_slope = slope([(k, r["npu"]["mean_abs"]) for k, r in points])
        k_slopes[series] = ratio_slope
        details[series] = dict(ratio_slope=ratio_slope, raw_npu_slope=raw_slope,
                               raw_about_linear=raw_slope >= LINEAR_RAW_SLOPE,
                               ratios={k: ratio(r) for k, r in points})
    if any(s >= SLOPE for s in k_slopes.values()):
        found.append("accumulation precision / partial-sum rounding (NPU excess error grows with K)")
    flat = [s for s in k_slopes.values() if abs(s) < SLOPE]
    ratios = [ratio(r) for r in ok]
    if flat and ratios and float(np.median(ratios)) > RATIO_FLAT:
        found.append("per-element dequantization / operand format (excess flat in K, median r > 1.5)")
    positive, negative, mixed = (by_name.get(f"sign_{a}") for a in ("positive", "negative", "mixed"))
    if positive and negative:
        balance = ratio(negative) / ratio(positive)
        mirror = None
        files = [Path(directory or ".", "outputs", f"sign_{a}.npy") for a in ("positive", "negative")]
        if all(f.is_file() for f in files):
            mirror = bool(np.array_equal(np.load(files[1]), -np.load(files[0])))
        details["sign"] = dict(r_positive=ratio(positive), r_negative=ratio(negative), negative_over_positive=balance,
                               outputs_exact_negation=mirror, r_mixed_informational=ratio(mixed) if mixed else None)
        if mirror is False or not SIGN_LOW <= balance <= SIGN_HIGH:
            found.append("signed-input packing / sign handling (negative inputs are not handled symmetrically)")
    else:
        details["sign"] = "not determined"
    meta = {key: by_name.get(f"meta_scales_{key[0]}_zeros_{key[1]}")
            for key in (("equal", "eight"), ("equal", "random"), ("random", "eight"), ("random", "random"))}
    if all(meta.values()):
        base = ratio(meta[("equal", "eight")])
        relative = {f"scales_{s}_zeros_{z}": ratio(r) / base for (s, z), r in meta.items()}
        details["metadata"] = relative
        if relative["scales_random_zeros_random"] > FACTOR:
            factors = [name for name, key in (("zero points", "scales_equal_zeros_random"),
                                              ("scales", "scales_random_zeros_eight")) if relative[key] > FACTOR]
            found.append("quantization metadata addressing (random metadata raises error; factor: "
                         + (", ".join(factors) if factors else "only the combination") + ")")
    else:
        details["metadata"] = "not determined"
    if ratios and max(ratios) <= RATIO_FLAT:
        found.append("no NPU-specific error at these inputs (all r <= 1.5)")
    return found or ["inconclusive under the fixed thresholds"], details


def table(records):
    head = f"{'case':34} {'K':>6} {'N':>5} {'status':8} {'e_npu':>9} {'e_gpu':>9} {'r':>6} {'max npu':>8} {'max|npu-gpu|':>12} {'allclose':>8}"
    rows = [head, "-" * len(head)]
    for r in records:
        if r.get("status") != "ok":
            label = r.get("name", f"K{r.get('k')}_N{r.get('n')}")
            rows.append(f"{label:34} {r.get('k', ''):>6} {r.get('n', ''):>5} {r.get('status', '?'):8} "
                        + (r.get("reason") or r.get("error") or "")[:90])
            continue
        rows.append(f"{r['name']:34} {r['k']:>6} {r['n']:>5} {'ok':8} {r['npu']['mean_abs']:9.5f} "
                    f"{r['gpu']['mean_abs']:9.5f} {ratio(r):6.2f} {r['npu']['max_abs']:8.4f} "
                    f"{r['npu_vs_gpu_max_abs']:12.4f} {str(r['sweep_allclose_rtol0p2_atol0p2']):>8}")
    return "\n".join(rows)


def emulate(path):
    """CPU predictions (estimated) for every case under two arithmetic hypotheses."""
    import probe_cases as pc
    sys.path.insert(0, str(HERE / "emulation"))
    from w4a16_npu_emulation import gpu_like_projection, npu_projection
    hypotheses = {"emulated_bf16_partials_bfp16": dict(bfp16_a=True, bfp16_b=True, bf16_partials=True),
                  "emulated_fp32_partials_bfp16": dict(bfp16_a=True, bfp16_b=True, bf16_partials=False),
                  "emulated_bf16_partials_every_k16": dict(bfp16_a=True, bfp16_b=True, bf16_partials=True,
                                                           partial_interval=16)}
    with Path(path).open("w") as f:
        for case in pc.cases():
            a, weight = pc.make_inputs(case)
            expected = pc.reference(a, weight)
            from reference import unpack_int4
            q = unpack_int4(weight.packed)
            gpu = gpu_like_projection(a, q, weight.scales, weight.zeros)
            for name, arith in hypotheses.items():
                npu = npu_projection(a, q, weight.scales, weight.zeros, k8_rounding="exact", **arith)
                record = dict(case, **pc.compare(npu, gpu, expected), status="ok", hypothesis=name,
                              label="estimated: CPU emulation of an assumed arithmetic, not a hardware measurement")
                f.write(json.dumps(record) + "\n")
                print(f"{case['name']} {name}: r={ratio(record):.2f} max|npu-gpu|={record['npu_vs_gpu_max_abs']:.4f}",
                      flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", type=Path, nargs="?")
    ap.add_argument("--json", type=Path, help="also write the verdicts as JSON")
    ap.add_argument("--emulate", type=Path, metavar="OUT_JSONL", help="write CPU predictions instead of reading data")
    args = ap.parse_args()
    if args.emulate:
        emulate(args.emulate)
        return
    if args.jsonl is None:
        ap.error("give a probe.jsonl or --emulate OUT_JSONL")
    records = load(args.jsonl)
    groups = {}
    for r in records:
        groups.setdefault(r.get("hypothesis", "hardware"), []).append(r)
    result = {}
    for source, items in groups.items():
        found, details = verdicts(items, args.jsonl.parent)
        result[source] = dict(supported=found, details=details)
        print(f"== {source}\n{table(items)}\n")
        for line in found:
            print(f"SUPPORTED: {line}")
        print(json.dumps(details, indent=2, default=str))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
