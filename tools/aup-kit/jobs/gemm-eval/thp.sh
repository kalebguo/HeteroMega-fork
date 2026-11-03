#!/usr/bin/env bash
# Does the NPU disturb the GPU less when its buffers use 2 MB pages? bg gemm-thp -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/thp.sh'
set -uo pipefail
export SRC=/notebooks/hm-gemm
source /notebooks/hm-gemm/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/gemm-thp
mkdir -p "$OUT"
J=tools/aup-kit/jobs/gemm-eval
step "build"
g++ -std=c++17 -O2 -Iinclude -I/opt/xilinx/xrt/include "$J/npu_load.cpp" -L/opt/xilinx/xrt/lib -lxrt_coreutil -o "$OUT/npu_load" -lpthread || exit 1
"$ROCM_PATH/lib/llvm/bin/clang++" -x hip --offload-arch=gfx1150 -O3 -std=c++17 -isystem "$ROCM_PATH/include" --hip-path="$ROCM_PATH" --rocm-path="$ROCM_PATH" \
  "$J/gemm_abl.cu" -L"$ROCM_PATH/lib" -lamdhip64 -Wl,-rpath,"$ROCM_PATH/lib" -o "$OUT/gemm_abl" 2>&1 | grep -E 'error' | head
"$ROCM_PATH/lib/llvm/bin/clang++" -x hip --offload-arch=gfx1150 -O3 -isystem "$ROCM_PATH/include" --hip-path="$ROCM_PATH" --rocm-path="$ROCM_PATH" \
  "$J/gpu_probe.cu" -L"$ROCM_PATH/lib" -lamdhip64 -Wl,-rpath,"$ROCM_PATH/lib" -o "$OUT/gpu_probe" 2>&1 | grep -E 'error' | head
cat /sys/kernel/mm/transparent_hugepage/shmem_enabled | sed 's/^/shmem_enabled /'
measure() { "$OUT/gpu_probe" 3 | grep -E 'read_dram|read_random|write_dram'; "$OUT/gemm_abl" 8; }
P8192=$RESULTS/inflight2-prep/deployment-p8192.json
for round in 1 2 3; do
    step "round $round: NPU idle"; measure
    step "round $round: NPU loop, normal buffers"
    "$OUT/npu_load" "$P8192" 0 > "$OUT/normal$round.log" 2>&1 &
    NPU=$!
    sleep 4
    grep -E 'AnonHuge|Rss|rror' "$OUT/normal$round.log" | cut -c1-120
    measure
    kill "$NPU"; wait "$NPU" 2>/dev/null
    tail -1 "$OUT/normal$round.log" | cut -c1-80
    step "round $round: NPU loop, 2 MB pages"
    NPU_LOAD_THP=1 "$OUT/npu_load" "$P8192" 0 > "$OUT/thp$round.log" 2>&1 &
    NPU=$!
    sleep 4
    grep -E 'AnonHuge|Rss|rror|what' "$OUT/thp$round.log" | cut -c1-160
    if grep -q ready "$OUT/thp$round.log"; then
        measure
        kill "$NPU"; wait "$NPU" 2>/dev/null
        tail -1 "$OUT/thp$round.log" | cut -c1-80
    else
        kill "$NPU" 2>/dev/null; tail -3 "$OUT/thp$round.log" | cut -c1-200
        break
    fi
done
step "done"
