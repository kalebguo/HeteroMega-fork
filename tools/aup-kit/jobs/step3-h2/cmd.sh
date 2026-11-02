#!/usr/bin/env bash
# Step 3 of merged-ffn-fix (H2): original heterogeneous attention, heteromosaic vs merged_ffn_hetero, on one timer, with
# the ioctl counter proving native NPU calls in this new directory.
# bg step3-h2 -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/step3-h2/cmd.sh'
set -eo pipefail
source /notebooks/hm-fix/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/fix-step3-h2
step "device before"; snapshot
HETEROMEGA_FFN_FINITE_CHECK=sampled LD_PRELOAD=$COUNT_IOCTLS python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build \
  --out "$OUT" --methods heteromosaic merged_ffn_hetero --merged-ffn-config "$PWD/deployment.json" --prompt-tokens 1024 \
  2>&1 | tee "$OUT.log" | grep -vE '^\s*$' | cut -c1-200 | tail -6
echo "--- summary.csv"; cat "$OUT/summary.csv"
step "native calls per process"; grep -H IOCTL_COUNTS "$OUT"/*.log | sed "s|$OUT/||"
grep -l "fallback" "$OUT"/*.log | sed "s|$OUT/||" || true
step "device after"; snapshot
step "done"
