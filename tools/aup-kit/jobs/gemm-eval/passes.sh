#!/usr/bin/env bash
# Build the model library, then run ab.sh several times with GTT and host memory sampled, with live lines per case.
# bg gemm-abN -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/passes.sh OUT_NAME PASSES CASE...'. CASE as in ab.sh, with samples of 1 round and 3 reps.
set -uo pipefail
J=/notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval
RESULTS=${RESULTS:-/notebooks/hm-results/2026-11-02}
OUT_NAME=$1
PASSES=$2
shift 2
mkdir -p "$RESULTS/$OUT_NAME"
bash "$J/build_model.sh" || exit $?
(
    gtt=$(ls /sys/class/drm/card*/device/mem_info_gtt_used | head -1)
    while :; do
        echo "$(date -u +%H:%M:%S.%N | cut -c1-10),$(($(cat "$gtt") / 1048576)),$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)"
        sleep 0.5
    done
) > "$RESULTS/$OUT_NAME/gtt.csv" &
sampler=$!
status=0
for pass in $(seq 1 "$PASSES"); do
    bash "$J/ab.sh" "$OUT_NAME/pass$pass" 1 3 "$@" 2>&1 | grep --line-buffered -E 'median|FAILED|rror' || status=1
done
kill "$sampler" 2>/dev/null
echo "peak GTT MB, lowest available MB: $(cut -d, -f2 "$RESULTS/$OUT_NAME/gtt.csv" | sort -n | tail -1), $(cut -d, -f3 "$RESULTS/$OUT_NAME/gtt.csv" | sort -n | head -1)"
exit $status
