#!/usr/bin/env bash
# One pass for gate, up and the activation against the three kernels, in the GPU FFN bench, with uncached weights, with the NPU idle and busy.
# Run gemm-build first. bg gemm-fuse -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/fuse.sh'
set -uo pipefail
export SRC=/notebooks/hm-gemm
source /notebooks/hm-gemm/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/gemm-fuse
mkdir -p "$OUT"
J=tools/aup-kit/jobs/gemm-eval
g++ -std=c++17 -O2 -Iinclude -I/opt/xilinx/xrt/include "$J/npu_load.cpp" -L/opt/xilinx/xrt/lib -lxrt_coreutil -o "$OUT/npu_load" -lpthread || exit 1
BENCH=build/gpu/bin/bench_ffn_ablation
export HETEROMEGA_GEMM_UNCACHED=1
measure() { # label
    "$BENCH" --m 512 --s 6144,7168,7680 --variants fast,fuse --rounds 3 --iters 30 --warmup 5 --clock-warmup 1 2>"$OUT/$1.err" > "$OUT/$1.csv"
    python3 - "$OUT/$1.csv" <<'PY'
import csv, statistics as st, sys
rows = [r for r in csv.DictReader(line for line in open(sys.argv[1]) if ',' in line)]
for s in sorted({int(r['s']) for r in rows}):
    for v in ('fast', 'fuse'):
        g = [r for r in rows if int(r['s']) == s and r['variant'] == v]
        print(f"  {v} s={s}: total {st.median(float(r['total_ms']) for r in g):.3f} gemm1 {st.median(float(r['gemm1_ms']) for r in g):.3f} act {st.median(float(r['act_ms']) for r in g):.3f} down {st.median(float(r['down_ms']) for r in g):.3f}")
PY
}
P8192=$RESULTS/inflight2-prep/deployment-p8192.json
for round in 1 2; do
    step "round $round: NPU idle"; measure off$round
    step "round $round: NPU loop, P8192"
    "$OUT/npu_load" "$P8192" 0 > "$OUT/p8192$round-npu.log" 2>&1 &
    NPU=$!
    sleep 4
    measure p8192$round
    kill "$NPU"; wait "$NPU" 2>/dev/null
    tail -1 "$OUT/p8192$round-npu.log" | cut -c1-80
done
step "done"
