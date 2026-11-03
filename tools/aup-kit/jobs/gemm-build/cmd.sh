#!/usr/bin/env bash
# Builds kernels/gpu in the separate tree /notebooks/hm-gemm, then runs the given tools from build/gpu/bin.
# /notebooks/hm-fix/build and /notebooks/hm-eval-main/build stay untouched as evidence.
#   cmd.sh [test|bench ARGS...]
set -uo pipefail
export SRC=/notebooks/hm-gemm
source /notebooks/hm-gemm/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/gemm
mkdir -p "$OUT"

step "configure kernels/gpu"
cmake -S kernels/gpu -B build/gpu 2>&1 | grep -E 'ROCm:|HIP architectures|Error|error' | head -10
step "build"
cmake --build build/gpu -j4 2>&1 | tee "$OUT/build.log" | grep -E 'error:|Error |Built target (test_gpu_ffn|bench_ffn_ablation)' | head -40
step "new kernel: instruction mix per kernel"
export PATH=$ROCM_PATH/lib/llvm/bin:$PATH
rm -rf "$OUT/isa"; mkdir -p "$OUT/isa"
(cd "$OUT/isa" && $ROCM_PATH/bin/roc-obj -d -t gfx1150 "$SRC/build/gpu/bin/bench_ffn_ablation" >/dev/null 2>&1
 f=$(grep -l prepare_input_kernel ./*.s | head -1); cp "$f" ../fast.s
 grep -nE '^[0-9a-f]+ <.*gemm_w4a16_kernel' ../fast.s | cut -c1-110
 echo "scratch ops in the file: $(grep -c scratch_ ../fast.s)   wmma: $(grep -c v_wmma ../fast.s)")
[ -x build/gpu/bin/test_gpu_ffn ] || { echo "build failed"; exit 1; }

case "${1:-}" in
  test)
    step "test_gpu_ffn"
    ./build/gpu/bin/test_gpu_ffn 2>&1 | tee "$OUT/test_gpu_ffn.log" | grep -E 'FAIL|ALL PASS|FAILURES' | head -40
    shift
    if [ $# -gt 0 ]; then
        step "bench_ffn_ablation $*"
        ./build/gpu/bin/bench_ffn_ablation "$@" 2>&1 | tee "$OUT/bench-$(date +%H%M).csv" | cut -c1-110
    fi
    ;;
  bench)
    shift
    step "bench_ffn_ablation $*"
    ./build/gpu/bin/bench_ffn_ablation "$@" 2>&1 | tee "$OUT/bench-$(date +%H%M).csv" | cut -c1-110
    ;;
esac
