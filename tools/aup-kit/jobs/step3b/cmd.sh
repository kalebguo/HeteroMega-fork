#!/usr/bin/env bash
# Why is merged_ffn_hetero slower than both heteromosaic and merged_ffn? Phase timers of the merged FFN with native NPU
# attention at 1024 tokens, one process. Diagnostic timing.
# bg step3b -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/step3b/cmd.sh'
set -eo pipefail
source /notebooks/hm-fix/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/fix-step3b
mkdir -p "$OUT"
HETEROMEGA_FFN_FINITE_CHECK=sampled HETEROMEGA_FFN_TIMERS=$OUT/timers LD_PRELOAD=$COUNT_IOCTLS python3 kernels/gpu/tools/evaluate_prefill.py \
  --model "$MODEL" --build build --out "$OUT/model" --methods merged_ffn_hetero --merged-ffn-config "$PWD/deployment.json" \
  --prompt-tokens 1024 --rounds 1 2>&1 | tee "$OUT/model.log" | cut -c1-160 | tail -1
python -m experiments.ffn_deploy.phase_report "$OUT/timers" --skip 64 --output "$OUT/phases.json" | tail -30
cut -d, -f1,2,5,6,7,18 "$OUT/model/summary.csv"
grep -H IOCTL_COUNTS "$OUT"/model/*.log | sed "s|$OUT/||"
step "done"
