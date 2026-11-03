#!/usr/bin/env python3
"""Median NPU completion wait by (context, previous context) from timed-ioctl CSVs of tools/aup-kit/jobs/step3c.

A submit is paired with the next timeline wait on the same thread. A pair whose context differs from the previous
submit's context is the first call after a context change. Setup and warmup calls are included in the counts.
"""
import csv
import json
from pathlib import Path
import statistics
import sys


def pairs(path):
    open_submit, result = {}, []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["operation"] == "submit":
                open_submit[row["tid"]] = row
            elif row["operation"] == "timeline_wait" and row["tid"] in open_submit and row["ret"] == "0":
                submit = open_submit.pop(row["tid"])
                result.append((int(submit["start_ns"]), int(submit["ctx"]), (int(row["end_ns"]) - int(row["start_ns"])) / 1e6))
    return sorted(result)


def summarize(path):
    calls = pairs(path)
    groups = {}
    for (_, previous, _), (_, ctx, wait) in zip(calls, calls[1:]):
        groups.setdefault((previous, ctx), []).append(wait)
    return [dict(previous_ctx=p, ctx=c, calls=len(v), median_wait_ms=statistics.median(v)) for (p, c), v in sorted(groups.items())]


def main():
    report = {}
    for path in sys.argv[1:]:
        if Path(path).stat().st_size > 100:
            report[Path(path).name] = summarize(path)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
