#!/usr/bin/env bash
# Which part of the GPU FFN the NPU slows down: memory probes, and the bench at sizes whose weights fit the GPU cache or do not. Sensors run during the bench only.
# bg gemm-interfere2 -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/interfere2.sh'
set -uo pipefail
export SRC=/notebooks/hm-gemm
source /notebooks/hm-gemm/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/gemm-interfere2
mkdir -p "$OUT"
J=tools/aup-kit/jobs/gemm-eval
step "build the NPU load and the GPU probes"
g++ -std=c++17 -O2 -Iinclude -I/opt/xilinx/xrt/include "$J/npu_load.cpp" -L/opt/xilinx/xrt/lib -lxrt_coreutil -o "$OUT/npu_load" -lpthread || exit 1
rm -f "$OUT/gpu_probe"
"$ROCM_PATH/lib/llvm/bin/clang++" -x hip --offload-arch=gfx1150 -O3 -isystem "$ROCM_PATH/include" --hip-path="$ROCM_PATH" --rocm-path="$ROCM_PATH" \
  "$J/gpu_probe.cu" -L"$ROCM_PATH/lib" -lamdhip64 -Wl,-rpath,"$ROCM_PATH/lib" -o "$OUT/gpu_probe" 2>&1 | grep -E 'error' | head || true
[ -x "$OUT/gpu_probe" ] || exit 1
BENCH=$(find build/gpu -name bench_ffn_ablation -type f | head -1)
measure() { # label
    "$OUT/gpu_probe" 5
    python3 "$J/sensors.py" "$OUT/$1-sensors.csv" &
    local sensors=$!
    "$BENCH" --m 512,128 --s 256,1024,4096,7680 --variants fast --rounds 3 --iters 30 --warmup 5 --clock-warmup 1 2>/dev/null > "$OUT/$1-bench.csv"
    kill "$sensors" 2>/dev/null
    wait "$sensors" 2>/dev/null
    python3 - "$OUT/$1-bench.csv" "$OUT/$1-sensors.csv" <<'PY'
import csv, statistics as st, sys
rows = [r for r in csv.DictReader(line for line in open(sys.argv[1]) if ',' in line)]
for m in sorted({int(r['M']) for r in rows}, reverse=True):
    for s in sorted({int(r['s']) for r in rows}):
        g = [r for r in rows if int(r['s']) == s and int(r['M']) == m]
        print(f"  bench M={m} s={s}: total {st.median(float(r['total_ms']) for r in g):.3f} gemm1 {st.median(float(r['gemm1_ms']) for r in g):.3f} act {st.median(float(r['act_ms']) for r in g):.3f} down {st.median(float(r['down_ms']) for r in g):.3f}")
samples = [r for r in csv.DictReader(open(sys.argv[2])) if r['busy'] and int(r['busy']) >= 90 and r['freq']]
print(f"  sensors in the bench, GPU busy: n={len(samples)} freq1 {st.median(float(r['freq']) / 1e6 for r in samples):.0f} MHz, "
      f"power {st.median(float(r['power']) for r in samples if r['power']):.1f} W")
PY
}
load() { # config label idle_ms
    "$OUT/npu_load" "$1" "$3" > "$OUT/$2-npu.log" 2>&1 &
    NPU=$!
    sleep 4
    grep -q ready "$OUT/$2-npu.log" || { echo "NPU load did not start"; tail -2 "$OUT/$2-npu.log" | cut -c1-200; }
}
stop_load() { kill "$NPU"; wait "$NPU" 2>/dev/null; tail -1 "$OUT/$1-npu.log" | cut -c1-80; }
P8192=$RESULTS/inflight2-prep/deployment-p8192.json
step "device before"; snapshot
for round in 1 2; do
    step "round $round: NPU idle"; measure off$round
    step "round $round: NPU loop, P6400"; load "$PWD/deployment.json" p6400$round 0; measure p6400$round; stop_load p6400$round
    step "round $round: NPU loop, P8192"; load "$P8192" p8192$round 0; measure p8192$round; stop_load p8192$round
    step "round $round: NPU P8192 with 5 ms idle after each run"; load "$P8192" duty$round 5; measure duty$round; stop_load duty$round
done
step "device after"; snapshot
step "done"
