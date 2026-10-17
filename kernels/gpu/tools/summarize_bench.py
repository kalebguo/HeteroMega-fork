#!/usr/bin/env python3
"""Summarize repeated bench_ffn_ablation runs: the spread over rounds, and each variant against hm.

Usage: python3 summarize_bench.py ablation.csv [more.csv ...] [--metric total_ms]

Lines that are not CSV rows of the header (for example the GEMM autotune messages) are skipped.
Ratios are taken per round (same round, same M and s), then summarized, so that slow drift between rounds cancels.
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict


def read_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            lines = [line for line in f if line.strip()]
        header = None
        for i, line in enumerate(lines):
            if line.startswith("round,"):
                header = i
                break
        if header is None:
            sys.exit(f"{path}: no 'round,...' header")
        names = lines[header].strip().split(",")
        for line in lines[header + 1:]:
            parts = line.strip().split(",")
            if len(parts) != len(names) or not parts[0].isdigit():
                continue
            rows.append(dict(zip(names, parts)))
    return rows


def spread(values):
    med = statistics.median(values)
    return med, min(values), max(values), (max(values) - min(values)) / med * 100 if med else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--metric", default="total_ms")
    args = ap.parse_args()
    rows = read_rows(args.csv)
    if not rows:
        sys.exit("no rows")

    groups = defaultdict(list)
    by_round = {}
    for r in rows:
        key = (r["variant"], int(r["M"]), int(r["s"]))
        groups[key].append(r)
        by_round[(r["round"],) + key] = float(r[args.metric])

    print(f"metric: {args.metric}")
    print("variant,M,s,rounds,median,min,max,spread_pct,sclk_MHz_median,sclk_MHz_min,power_W,temp_C_max")
    for key in sorted(groups, key=lambda k: (k[2], k[1], k[0])):
        g = groups[key]
        med, lo, hi, pct = spread([float(r[args.metric]) for r in g])
        sclk = statistics.median(float(r["sclk_MHz_median"]) for r in g)
        sclk_min = min(float(r["sclk_MHz_min"]) for r in g)
        power = statistics.median(float(r["power_W"]) for r in g)
        temp = max(float(r["temp_C"]) for r in g)
        print(f"{key[0]},{key[1]},{key[2]},{len(g)},{med:.4f},{lo:.4f},{hi:.4f},{pct:.1f},{sclk:.0f},{sclk_min:.0f},{power:.1f},{temp:.0f}")

    print()
    print("variant / hm per round (< 1 is faster than HeteroMosaic's order)")
    print("variant,M,s,rounds,median_ratio,min_ratio,max_ratio")
    for key in sorted(groups, key=lambda k: (k[2], k[1], k[0])):
        variant, m, s = key
        if variant == "hm":
            continue
        ratios = []
        for r in groups[key]:
            base = by_round.get((r["round"], "hm", m, s))
            if base:
                ratios.append(float(r[args.metric]) / base)
        if ratios:
            med, lo, hi, _ = spread(ratios)
            print(f"{variant},{m},{s},{len(ratios)},{med:.3f},{lo:.3f},{hi:.3f}")


if __name__ == "__main__":
    main()
