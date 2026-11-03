#!/usr/bin/env bash
# Where the time goes at the three best cases, with GPU timestamps per stage (HETEROMEGA_STAGE_EVENTS), phase timers and GPU sensors.
# Each run is one process, apart from the timing jobs because markers and timers add waits.
# bg gemm-stages -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/stages.sh OUT_NAME RUN...'
# RUN is LABEL:METHOD:INFLIGHT[:serial], METHOD is p6400, p7168, p8192 or gpu.
set -uo pipefail
export SRC=/notebooks/hm-gemm
source /notebooks/hm-gemm/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/$1
shift
mkdir -p "$OUT"
export HETEROMEGA_FFN_FINITE_CHECK=sampled HETEROMEGA_GEMM=${HETEROMEGA_GEMM-fast}
config() {
    case "$1" in
        p6400) echo "$PWD/deployment.json" ;;
        p7168) echo "$RESULTS/gemm-splits/deployment-p7168.json" ;;
        p8192) echo "$RESULTS/inflight2-prep/deployment-p8192.json" ;;
    esac
}
step "device before"; snapshot
for run in "$@"; do
    IFS=: read -r label method inflight mode <<< "$run"
    step "$label"
    D=$OUT/$label
    rm -rf "$D" "$D-stages" "$D-timers"
    python3 tools/aup-kit/jobs/gemm-eval/sensors.py "$D-sensors.csv" &
    sensors=$!
    if [ "$method" = gpu ]; then
        HETEROMEGA_STAGE_EVENTS=$D-stages python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$D" \
          --methods gpu_original --gpu-chunk-size 512 --inflight "$inflight" --prompt-tokens 1024 --rounds 1 2>&1 | tee "$D.log" | cut -c1-160 | tail -1
    else
        [ "${mode:-}" = serial ] && export HETEROMEGA_FFN_SERIAL_GPU=1 || unset HETEROMEGA_FFN_SERIAL_GPU
        HETEROMEGA_STAGE_EVENTS=$D-stages HETEROMEGA_FFN_TIMERS=$D-timers python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build \
          --out "$D" --methods merged_ffn --merged-ffn-config "$(config "$method")" --gpu-chunk-size 512 --inflight "$inflight" \
          --prompt-tokens 1024 --rounds 1 2>&1 | tee "$D.log" | cut -c1-160 | tail -1
        python -m experiments.ffn_deploy.phase_report "$D-timers" --skip 64 --output "$D-phases.json" > /dev/null
    fi
    kill "$sensors" 2>/dev/null
    wait "$sensors" 2>/dev/null
    cut -d, -f1,2,4,5,6 "$D/summary.csv" | tail -1
done
unset HETEROMEGA_FFN_SERIAL_GPU
step "device after"; snapshot
tar czf "$OUT.tgz" -C "$RESULTS" --exclude='*.result.*' --exclude='*.log' "$(basename "$OUT")"
step "done"
