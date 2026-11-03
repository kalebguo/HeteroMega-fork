#!/usr/bin/env python3
"""NPU busy time per prefill from the timed-ioctl CSVs of this job: submit to completion, the union over the 288 calls of each of
the last 10 prefills. Calls over LONG_MS are the 512-row gate and up calls of config 3.

    python3 tools/aup-kit/jobs/inflight2-diag/npu_busy.py ioctl-*.csv.gz > npu-busy.json
"""
import gzip
import json
import statistics
import sys

from trace_busy import PER_PREFILL, PREFILLS, pairs_from

LONG_MS = 5.0


def summarize(calls):
    rows = []
    for start in range(len(calls) - PER_PREFILL * PREFILLS, len(calls), PER_PREFILL):
        prefill = calls[start:start + PER_PREFILL]
        busy, end = 0, None
        for submit, _, done in prefill:
            begin = submit if end is None or submit > end else end
            if done > begin:
                busy += done - begin
                end = done
        span = (max(c[2] for c in prefill) - prefill[0][0]) / 1e6
        durations = [(done - submit) / 1e6 for submit, _, done in prefill]
        rows.append(dict(span_ms=span, busy_ms=busy / 1e6, idle_ms=span - busy / 1e6, busy_percent=100 * busy / 1e6 / span,
                         long_calls=sum(d > LONG_MS for d in durations), long_calls_ms=sum(d for d in durations if d > LONG_MS)))
    return {key: statistics.median(row[key] for row in rows) for key in rows[0]}


def main():
    print(json.dumps({path.rsplit("/", 1)[-1]: summarize(pairs_from(gzip.open(path, "rt", newline=""))) for path in sys.argv[1:]}, indent=1))


if __name__ == "__main__":
    main()
