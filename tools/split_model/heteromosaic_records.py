#!/usr/bin/env python3
"""Compare the latest measured runs with every archived heteromosaic prefill record, without running heteromosaic again.

    python3 tools/split_model/heteromosaic_records.py --latest LABEL=SUMMARY_CSV:METHOD [--latest ...] --out OUT.csv

Records are the heteromosaic rows of docs/runs/**/summary.csv at the prompt length, from runs without timers or tracers. They
are not paired with the latest runs: heteromosaic's records spread more than its process spread within one job.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHOD = "heteromosaic"
INSTRUMENTED = ("diag", "trace")  # these runs carry a tracer, so they are not latency records


def inflight_of(summary, method, tokens):
    """The run's --inflight; runs that did not pass it used the shipped one inflight chunk."""
    result = summary.parent / f"{method}-{tokens}-r0.result.json"
    return (json.loads(result.read_text()).get("inflight") or 1) if result.exists() else 1


def read_row(summary, method, tokens):
    with Path(summary).open() as f:
        for row in csv.DictReader(f):
            if row["method"] == method and int(row["prompt_tokens"]) == tokens and row.get("dummy", "False") == "False":
                return dict(median_ms=float(row["prefill_median_ms"]), min_ms=float(row["prefill_min_ms"]),
                            max_ms=float(row["prefill_max_ms"]), samples=int(row["samples"]))
    return None


def records(root=ROOT, tokens=1024):
    found = []
    for summary in sorted((root / "docs/runs").glob("**/summary.csv")):
        source = str(summary.parent.relative_to(root / "docs/runs"))
        row = None if any(word in source for word in INSTRUMENTED) else read_row(summary, METHOD, tokens)
        if row:
            found.append(dict(source=source, inflight=inflight_of(summary, METHOD, tokens), **row))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latest", action="append", required=True, help="LABEL=SUMMARY_CSV:METHOD")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for spec in args.latest:
        label, rest = spec.split("=", 1)
        summary, method = rest.rsplit(":", 1)
        latest = read_row(ROOT / summary, method, args.tokens)
        for record in records(tokens=args.tokens):
            rows.append(dict(latest=label, latest_median_ms=round(latest["median_ms"], 1), latest_samples=latest["samples"],
                             record_source=record["source"], record_inflight=record["inflight"],
                             record_median_ms=round(record["median_ms"], 1), record_samples=record["samples"],
                             gain_ms=round(record["median_ms"] - latest["median_ms"], 1),
                             speedup=round(record["median_ms"] / latest["median_ms"], 3)))
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(*row.values(), sep=",")


if __name__ == "__main__":
    main()
