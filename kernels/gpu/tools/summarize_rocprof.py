#!/usr/bin/env python3
"""Per-kernel averages of rocprofv3 counter CSVs (--pmc ... --output-format csv).

Usage: python3 summarize_rocprof.py <dir> [<dir> ...]
  Reads every *counter_collection.csv under the directories. Prints one row per (kernel, counter): the mean value
  per dispatch, the dispatch count, and the kernel's grid, workgroup, VGPR and LDS sizes.

Example (GEMM counters of one fused FFN at M = 512, s = 10240):
  rocprofv3 --pmc SQ_INSTS_VALU SQ_INSTS_LDS SQ_WAVES -d prof/a -o run --output-format csv -- \
      build/gpu/bin/bench_ffn_ablation --variants fused --m 512 --s 10240 --rounds 1 --iters 3 --warmup 1
  python3 kernels/gpu/tools/summarize_rocprof.py prof
"""
import csv
import pathlib
import re
import statistics
import sys
from collections import defaultdict


def short_name(name):
    # Drop the argument list; keep template arguments.
    depth = 0
    for i, ch in enumerate(name):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            return name[:i]
    return name


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    values = defaultdict(list)
    shape = {}
    for d in sys.argv[1:]:
        for path in pathlib.Path(d).rglob("*counter_collection.csv"):
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    kernel = short_name(r["Kernel_Name"])
                    values[(kernel, r["Counter_Name"])].append(float(r["Counter_Value"]))
                    shape[kernel] = (r.get("Grid_Size", ""), r.get("Workgroup_Size", ""), r.get("VGPR_Count", ""),
                                     r.get("LDS_Block_Size", ""))
    if not values:
        sys.exit("no counter_collection.csv rows found")
    print("kernel,counter,mean_per_dispatch,dispatches,grid,workgroup,vgpr,lds_bytes")
    for (kernel, counter) in sorted(values):
        v = values[(kernel, counter)]
        grid, wg, vgpr, lds = shape[kernel]
        print(f"{re.sub(',', ';', kernel)},{counter},{statistics.mean(v):.0f},{len(v)},{grid},{wg},{vgpr},{lds}")


if __name__ == "__main__":
    main()
