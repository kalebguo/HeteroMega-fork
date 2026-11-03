#!/usr/bin/env bash
# What the NPU does to the GPU, with no model: GPU probes and the GPU FFN bench with the NPU idle, and with the NPU artifact in a loop.
# bg gemm-interfere -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/interfere.sh'
set -uo pipefail
export SRC=/notebooks/hm-gemm
source /notebooks/hm-gemm/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/gemm-interfere
mkdir -p "$OUT"
J=tools/aup-kit/jobs/gemm-eval
step "build the NPU load and the GPU probes"
g++ -std=c++17 -O2 -Iinclude -I/opt/xilinx/xrt/include "$J/npu_load.cpp" -L/opt/xilinx/xrt/lib -lxrt_coreutil -o "$OUT/npu_load" -lpthread || exit 1
"$ROCM_PATH/lib/llvm/bin/clang++" -x hip --offload-arch=gfx1150 -O3 -isystem "$ROCM_PATH/include" --hip-path="$ROCM_PATH" --rocm-path="$ROCM_PATH" \
  "$J/gpu_probe.cu" -L"$ROCM_PATH/lib" -lamdhip64 -Wl,-rpath,"$ROCM_PATH/lib" -o "$OUT/gpu_probe" 2>&1 | grep -E 'error' | head || true
[ -x "$OUT/gpu_probe" ] || exit 1
BENCH=$(find build/gpu -name bench_ffn_ablation -type f | head -1)
measure() { # label
    python3 "$J/sensors.py" "$OUT/$1-sensors.csv" &
    local sensors=$!
    "$OUT/gpu_probe" 8
    "$BENCH" --m 512 --s 6144,7680 --variants fast --rounds 3 --iters 30 --warmup 5 --clock-warmup 1 2>/dev/null > "$OUT/$1-bench.csv"
    kill "$sensors" 2>/dev/null
    wait "$sensors" 2>/dev/null
    python3 - "$OUT/$1-bench.csv" "$OUT/$1-sensors.csv" <<'PY'
import csv, statistics as st, sys
rows = [r for r in csv.DictReader(line for line in open(sys.argv[1]) if ',' in line)]
for s in sorted({r['s'] for r in rows}):
    group = [r for r in rows if r['s'] == s]
    print(f"  bench fast s={s}: total {st.median(float(r['total_ms']) for r in group):.2f} ms, gemm1 {st.median(float(r['gemm1_ms']) for r in group):.2f}, "
          f"act {st.median(float(r['act_ms']) for r in group):.2f}, down {st.median(float(r['down_ms']) for r in group):.2f}")
samples = [r for r in csv.DictReader(open(sys.argv[2])) if r['busy'] and int(r['busy']) >= 90 and r['freq']]
print(f"  sensors, GPU busy: n={len(samples)} freq1 {st.median(float(r['freq']) / 1e6 for r in samples):.0f} MHz, "
      f"power {st.median(float(r['power']) for r in samples if r['power']):.1f} W, CPU cores {st.median(float(r['cores']) for r in samples):.1f}")
PY
}
load() { # config label
    "$OUT/npu_load" "$1" > "$OUT/$2-npu.log" 2>&1 &
    NPU=$!
    sleep 4
    grep -q ready "$OUT/$2-npu.log" || { echo "NPU load did not start"; tail -2 "$OUT/$2-npu.log" | cut -c1-200; }
}
step "device before"; snapshot
for round in 1 2; do
    step "round $round: NPU idle"
    measure off$round
    step "round $round: NPU loop, P6400"
    load "$PWD/deployment.json" p6400$round
    measure p6400$round
    kill "$NPU"; wait "$NPU" 2>/dev/null
    tail -1 "$OUT/p6400$round-npu.log" | cut -c1-80
    step "round $round: NPU loop, P8192"
    load "$RESULTS/inflight2-prep/deployment-p8192.json" p8192$round
    measure p8192$round
    kill "$NPU"; wait "$NPU" 2>/dev/null
    tail -1 "$OUT/p8192$round-npu.log" | cut -c1-80
done
step "device after"; snapshot
step "done"
