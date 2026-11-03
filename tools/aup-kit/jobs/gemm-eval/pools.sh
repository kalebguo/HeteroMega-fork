#!/usr/bin/env bash
# Builds and runs pools.cpp. bg gemm-pools -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/pools.sh'
set -uo pipefail
export SRC=/notebooks/hm-gemm
source /notebooks/hm-gemm/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/gemm-abl
mkdir -p "$OUT"
J=tools/aup-kit/jobs/gemm-eval
"$ROCM_PATH/lib/llvm/bin/clang++" -x hip --offload-arch=gfx1150 -O2 -std=c++17 -isystem "$ROCM_PATH/include" --hip-path="$ROCM_PATH" --rocm-path="$ROCM_PATH" \
  "$J/pools.cpp" -L"$ROCM_PATH/lib" -lamdhip64 -lhsa-runtime64 -Wl,-rpath,"$ROCM_PATH/lib" -o "$OUT/pools" 2>&1 | grep -E 'error' | head
"$OUT/pools" 2>&1 | tail -60
env | grep -E '^(HIP_|HSA_|GPU_|ROC_)' | cut -c1-80
cat /proc/cmdline | tr ' ' '\n' | grep -E 'iommu|amdgpu|amd_' 
