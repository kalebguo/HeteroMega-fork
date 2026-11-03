#!/usr/bin/env python3
"""NPU waits per prefill from timed-ioctl CSVs of jobs/inflight2-diag: 288 calls per prefill, the last 10 prefills of a run.

Those runs use --new-tokens 1, so after the warmup every submit belongs to a prefill. "union" is the host's time waiting for
any NPU call in the prefill, and "span" is first submit to last completion. union close to span means the NPU limits the
prefill; a clear gap means the GPU or the host does. At one inflight chunk the calls are in order, so positions in a
layer-chunk can be read directly (chunk 0: 4 calls per layer, chunk 1: 5), as in docs/runs/2026-11-02-fix-step3c.
"""
import csv
import json
from pathlib import Path
import statistics
import sys

PER_PREFILL = 288
PREFILLS = 10
LAYERS = 32
CHUNK0_CALLS = 4


def pairs(path):
    open_submit, result = {}, []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["operation"] == "submit":
                open_submit[row["tid"]] = row
            elif row["operation"] == "timeline_wait" and row["tid"] in open_submit and row["ret"] == "0":
                submit = open_submit.pop(row["tid"])
                result.append((int(submit["start_ns"]), int(row["start_ns"]), int(row["end_ns"])))
    return sorted(result)


def union_ms(calls):
    total, end = 0, None
    for _, wait_start, wait_end in sorted(calls, key=lambda c: c[1]):
        if end is None or wait_start > end:
            total += wait_end - wait_start
            end = wait_end
        elif wait_end > end:
            total += wait_end - end
            end = wait_end
    return total / 1e6


def positions(calls, per_layer):
    return [statistics.median((c[2] - c[1]) / 1e6 for c in calls[i::per_layer]) for i in range(per_layer)]


def summarize(path):
    calls = pairs(path)
    if len(calls) < PER_PREFILL * PREFILLS:
        return dict(calls=len(calls), error="fewer calls than expected")
    calls = calls[-PER_PREFILL * PREFILLS:]
    prefills = [calls[i:i + PER_PREFILL] for i in range(0, len(calls), PER_PREFILL)]
    spans = [(max(c[2] for c in p) - p[0][0]) / 1e6 for p in prefills]
    row = dict(calls=len(calls), span_ms=statistics.median(spans), union_wait_ms=statistics.median(union_ms(p) for p in prefills),
               summed_wait_ms=statistics.median(sum(c[2] - c[1] for c in p) / 1e6 for p in prefills),
               call_median_ms=statistics.median((c[2] - c[1]) / 1e6 for c in calls))
    first = LAYERS * CHUNK0_CALLS
    chunk0 = [c for p in prefills for c in p[:first]]
    chunk1 = [c for p in prefills for c in p[first:]]
    row["in_order_chunk0_ms"] = positions(chunk0, CHUNK0_CALLS)
    row["in_order_chunk1_ms"] = positions(chunk1, CHUNK0_CALLS + 1)
    return row


def main():
    print(json.dumps({Path(path).name: summarize(path) for path in sys.argv[1:] if Path(path).stat().st_size > 100}, indent=1))


if __name__ == "__main__":
    main()
