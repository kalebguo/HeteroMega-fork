#!/usr/bin/env bash
# Two-inflight comparison, timing. Every method alternates in one job (A B C D D C B A), with 10 samples per cell and native
# calls counted. No timers and no power sampling in this job.
# bg inflight2-timing -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/inflight2-timing/cmd.sh'
set -eo pipefail
source /notebooks/hm-fix/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/inflight2-timing
PREP=$RESULTS/inflight2-prep
mkdir -p "$OUT"
export HETEROMEGA_FFN_FINITE_CHECK=sampled
CONFIGS=(--merged-ffn-config "$PWD/deployment.json" --p8192-ffn-config "$PREP/deployment-p8192.json" --gateup-config "$PREP/gateup.config.json" --gpu-chunk-size 512)
evaluate() { # name, then evaluate_prefill.py arguments
    name=$1
    shift
    step "$name"
    LD_PRELOAD=$COUNT_IOCTLS python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$OUT/$name" "${CONFIGS[@]}" "$@" \
      2>&1 | tee "$OUT/$name.log" | cut -c1-160 | tail -1
    cut -d, -f1,2,4,5,6,7,8,9,10,11,12,18 "$OUT/$name/summary.csv"
    grep -H IOCTL_COUNTS "$OUT/$name"/*.log | sed "s|$OUT/$name/||" | cut -c1-170
}
step "device before"; snapshot
evaluate inflight2-1024 --methods heteromosaic heteromosaic_gateup merged_ffn merged_ffn_p8192 --inflight 2 --prompt-tokens 1024
evaluate inflight1-1024 --methods heteromosaic heteromosaic_gateup merged_ffn merged_ffn_p8192 --inflight 1 --prompt-tokens 1024
evaluate inflight1-512 --methods gpu_original merged_ffn merged_ffn_p8192 --inflight 1 --prompt-tokens 512
step "device after"; snapshot
step "done"
