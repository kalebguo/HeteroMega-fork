#!/usr/bin/env python3
"""Instruction mix of the loops in AMDGPU kernels, from an llvm-objdump disassembly.

Usage:
  /opt/rocm/bin/roc-obj -d -o isa build/gpu/bin/bench_ffn_ablation   # writes isa/*.s (one per code object)
  python3 kernels/gpu/tools/isa_loop_stats.py isa/<file>.s [kernel-name-substring]

A loop is the address range from the target of a backward branch to the branch. For each loop of each matching
kernel it prints the static instruction counts per class, for one pass through the loop body:
  wmma   v_wmma_*                      (matrix instructions)
  valu   other v_*                     (for W4A16: unpacking, scaling, address math)
  lds    ds_*                          vmem  global_/buffer_/flat_/scratch_*
  smem   s_load_* / s_buffer_load_*    salu  other s_* (without s_waitcnt, s_nop and branches)
  wait   s_waitcnt*                    nop   s_nop
Nested loops are listed separately; an outer loop's counts include its inner loops once.
"""
import re
import sys
from collections import Counter

FUNC = re.compile(r"^([0-9a-f]+) <(.+)>:\s*$")
INSN = re.compile(r"^\s+([a-z][a-z0-9_]*)\b(.*?)//\s*([0-9A-Fa-f]+):")
TARGET = re.compile(r"<(.+)\+0x([0-9a-f]+)>")


def classify(m):
    if m.startswith("v_wmma"):
        return "wmma"
    if m.startswith("v_"):
        return "valu"
    if m.startswith("ds_"):
        return "lds"
    if m.startswith(("global_", "buffer_", "flat_", "scratch_")):
        return "vmem"
    if m.startswith(("s_load", "s_buffer_load")):
        return "smem"
    if m.startswith("s_waitcnt"):
        return "wait"
    if m == "s_nop":
        return "nop"
    if m.startswith(("s_branch", "s_cbranch")):
        return "branch"
    if m.startswith("s_"):
        return "salu"
    return "other"


def parse(path):
    funcs = {}
    name = None
    start = 0
    with open(path, errors="replace") as f:
        for line in f:
            fm = FUNC.match(line)
            if fm:
                start, name = int(fm.group(1), 16), fm.group(2)
                funcs[name] = (start, [])
                continue
            im = INSN.match(line)
            if im and name is not None:
                addr = int(im.group(3), 16)
                target = None
                if im.group(1).startswith(("s_branch", "s_cbranch")):
                    tm = TARGET.search(line)
                    if tm:
                        target = start + int(tm.group(2), 16)
                funcs[name][1].append((addr, im.group(1), target))
    return funcs


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    want = sys.argv[2] if len(sys.argv) > 2 else ""
    classes = ["wmma", "valu", "lds", "vmem", "smem", "salu", "wait", "nop", "branch"]
    print("kernel,loop_start,loop_end,insns," + ",".join(classes) + ",valu_per_wmma")
    for name, (_, insns) in parse(sys.argv[1]).items():
        if want not in name:
            continue
        label = name.replace(",", ";")
        total = Counter(classify(m) for _, m, _ in insns)
        print(f"{label},whole,kernel,{len(insns)}," + ",".join(str(total[c]) for c in classes) + ",")
        for addr, _, target in insns:
            if target is None or target > addr:
                continue
            body = [m for a, m, _ in insns if target <= a <= addr]
            c = Counter(classify(m) for m in body)
            ratio = f"{c['valu'] / c['wmma']:.2f}" if c["wmma"] else ""
            print(f"{label},{target:x},{addr:x},{len(body)}," + ",".join(str(c[k]) for k in classes) + f",{ratio}")


if __name__ == "__main__":
    main()
