#!/usr/bin/env bash
# Attention kernel against the torch attention at the three best splits, after the GPU slices stopped being held twice. Build, then 3 passes of the 6 cases.
# bg gemm-ab5 -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/ab5.sh'
set -uo pipefail
J=/notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval
RESULTS=${RESULTS:-/notebooks/hm-results/2026-11-02}
mkdir -p "$RESULTS/gemm-ab5"
bash "$J/build_model.sh" || exit $?
(
    gtt=$(ls /sys/class/drm/card*/device/mem_info_gtt_used | head -1)
    while :; do
        echo "$(date -u +%H:%M:%S.%N | cut -c1-10),$(($(cat "$gtt") / 1048576)),$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)"
        sleep 0.5
    done
) > "$RESULTS/gemm-ab5/gtt.csv" &
sampler=$!
status=0
for pass in 1 2 3; do
    bash "$J/ab.sh" "gemm-ab5/pass$pass" 1 3 \
      p8192-unc:p8192:2:HETEROMEGA_GEMM_UNCACHED=1 p8192-sdpa:p8192:2:HETEROMEGA_GEMM_UNCACHED=1,HETEROMEGA_SDPA=fast \
      p6400-unc:p6400:2:HETEROMEGA_GEMM_UNCACHED=1 p6400-sdpa:p6400:2:HETEROMEGA_GEMM_UNCACHED=1,HETEROMEGA_SDPA=fast \
      p7168-unc:p7168:2:HETEROMEGA_GEMM_UNCACHED=1 p7168-sdpa:p7168:2:HETEROMEGA_GEMM_UNCACHED=1,HETEROMEGA_SDPA=fast \
      2>&1 | stdbuf -oL grep -E 'median|FAILED|rror' | cut -c1-200 || status=1
done
kill "$sampler" 2>/dev/null
echo "peak GTT MB, lowest available MB: $(cut -d, -f2 "$RESULTS/gemm-ab5/gtt.csv" | sort -n | tail -1), $(cut -d, -f3 "$RESULTS/gemm-ab5/gtt.csv" | sort -n | head -1)"
exit $status
