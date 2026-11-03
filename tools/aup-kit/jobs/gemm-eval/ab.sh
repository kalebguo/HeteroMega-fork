#!/usr/bin/env bash
# Compare switches of merged_ffn with gemm_w4a16 by plain prefill timing (no timers, no markers), one line per case, with native calls counted.
# bg gemm-ab -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/ab.sh OUT_NAME ROUNDS REPS CASE...'
# CASE is LABEL:METHOD:INFLIGHT[:NAME=VALUE,NAME=VALUE]. METHOD is p6400, p7168, p8192 or gpu. A case's variables hold for that case only.
set -uo pipefail
export SRC=/notebooks/hm-gemm
source /notebooks/hm-gemm/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/$1
ROUNDS=$2
REPS=$3
shift 3
mkdir -p "$OUT"
export HETEROMEGA_FFN_FINITE_CHECK=sampled HETEROMEGA_GEMM=fast
status=0
config() {
    case "$1" in
        p6400) echo "$PWD/deployment.json" ;;
        p7168) echo "$RESULTS/gemm-splits/deployment-p7168.json" ;;
        p8192) echo "$RESULTS/inflight2-prep/deployment-p8192.json" ;;
    esac
}
step "device before"; snapshot
for run in "$@"; do
    IFS=: read -r label method inflight vars <<< "$run"
    D=$OUT/$label
    rm -rf "$D"
    (
        IFS=, read -ra pairs <<< "${vars:-}"
        for pair in "${pairs[@]:-}"; do [ -n "$pair" ] && export "$pair"; done
        if [ "$method" = gpu ]; then
            python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$D" --methods gpu_original --gpu-chunk-size 512 \
              --inflight "$inflight" --prompt-tokens 1024 --rounds "$ROUNDS" --reps "$REPS" > "$D.log" 2>&1
        else
            LD_PRELOAD=$COUNT_IOCTLS python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$D" --methods merged_ffn \
              --merged-ffn-config "$(config "$method")" --gpu-chunk-size 512 --inflight "$inflight" --prompt-tokens 1024 \
              --rounds "$ROUNDS" --reps "$REPS" > "$D.log" 2>&1
        fi
    ) || { status=1; echo "$label: FAILED"; tail -3 "$D"/*.log 2>/dev/null | cut -c1-240; continue; }
    median=$(tail -1 "$D/summary.csv" | cut -d, -f5,6,7 | cut -c1-30)
    calls=$(grep -h IOCTL_COUNTS "$D"/*.log 2>/dev/null | grep -o 'exec_ok=[0-9]*' | sort -u | tr '\n' ' ')
    fallbacks=$(cat "$D"/*.log 2>/dev/null | grep -c UNCACHED_FALLBACK)
    sdpa=$(cat "$D"/*.log 2>/dev/null | grep -c 'sdpa_bf16: first launch')
    echo "$(date -u +%H:%M:%S) $label: median,min,max $median native $calls fallbacks=$fallbacks sdpa_lines=$sdpa"
done
step "device after"; snapshot
step "done status=$status"
exit $status
