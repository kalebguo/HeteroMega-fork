#!/usr/bin/env bash
# merged_ffn with gemm_w4a16 end to end: P6400 and P8192, inflight 1 and 2, 1024 tokens. The baselines are the archived records
# (tools/split_model/heteromosaic_records.py), not rerun here. HETEROMEGA_GEMM=fast selects the new kernel in the merged FFN and in attention.
# bg gemm-eval -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/cmd.sh [rounds] [reps]'
set -eo pipefail
export SRC=/notebooks/hm-gemm
source /notebooks/hm-gemm/tools/aup-kit/jobs/fix-env.sh
ROUNDS=${1:-2}
REPS=${2:-5}
OUT=$RESULTS/gemm-eval
PREP=$RESULTS/inflight2-prep
mkdir -p "$OUT"
export HETEROMEGA_FFN_FINITE_CHECK=sampled HETEROMEGA_GEMM=fast
step "device before"; snapshot
for inflight in 1 2; do
    step "inflight $inflight"
    D=$OUT/i$inflight
    LD_PRELOAD=$COUNT_IOCTLS python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$D" \
      --merged-ffn-config "$PWD/deployment.json" --p8192-ffn-config "$PREP/deployment-p8192.json" --gpu-chunk-size 512 \
      --methods merged_ffn merged_ffn_p8192 --inflight "$inflight" --prompt-tokens 1024 --rounds "$ROUNDS" --reps "$REPS" \
      2>&1 | tee "$D.log" | cut -c1-160 | tail -1
    cut -d, -f1,2,3,4,5,6,9,10,11,18 "$D/summary.csv"
    grep -H IOCTL_COUNTS "$D"/*.log | sed "s|$D/||" | cut -c1-150
done
step "device after"; snapshot
step "done"
