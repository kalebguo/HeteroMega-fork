#!/usr/bin/env bash
# Runs gemm_abl with the NPU idle and with the NPU in a loop. bg gemm-abl -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/abl.sh'
set -uo pipefail
export SRC=/notebooks/hm-gemm
source /notebooks/hm-gemm/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/gemm-abl
mkdir -p "$OUT"
J=tools/aup-kit/jobs/gemm-eval
step "build"
g++ -std=c++17 -O2 -Iinclude -I/opt/xilinx/xrt/include "$J/npu_load.cpp" -L/opt/xilinx/xrt/lib -lxrt_coreutil -o "$OUT/npu_load" -lpthread || exit 1
rm -f "$OUT/gemm_abl"
"$ROCM_PATH/lib/llvm/bin/clang++" -x hip --offload-arch=gfx1150 -O3 -std=c++17 -isystem "$ROCM_PATH/include" --hip-path="$ROCM_PATH" --rocm-path="$ROCM_PATH" \
  "$J/gemm_abl.cu" -L"$ROCM_PATH/lib" -lamdhip64 -Wl,-rpath,"$ROCM_PATH/lib" -o "$OUT/gemm_abl" 2>&1 | grep -E 'error' | head
[ -x "$OUT/gemm_abl" ] || exit 1
P8192=$RESULTS/inflight2-prep/deployment-p8192.json
for f in mem_info_vram_total mem_info_vram_used mem_info_gtt_total mem_info_gtt_used; do echo "$f $(cat /sys/class/drm/card*/device/$f 2>/dev/null | tr '\n' ' ')"; done
grep -E 'always|madvise|never' /sys/kernel/mm/transparent_hugepage/enabled | sed 's/^/THP /'
grep -E 'HugePages_Total|Hugepagesize|AnonHugePages' /proc/meminfo
for round in 1 2; do
    step "round $round: NPU idle"; "$OUT/gemm_abl" 12
    step "round $round: NPU loop, P8192"
    "$OUT/npu_load" "$P8192" 0 > "$OUT/npu$round.log" 2>&1 &
    NPU=$!
    sleep 4
    "$OUT/gemm_abl" 12
    kill "$NPU"; wait "$NPU" 2>/dev/null
done
step "done"
