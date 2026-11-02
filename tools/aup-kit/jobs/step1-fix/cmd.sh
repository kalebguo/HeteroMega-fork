#!/usr/bin/env bash
# Step 1 of merged-ffn-fix: -O3 with the full finite scans vs -O3 with sampled scans. C++ smoke, then the paired
# evaluation in fresh directories. bg step1-fix -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/step1-fix/cmd.sh'
set -eo pipefail
source /notebooks/hm-fix/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/fix-step1
mkdir -p "$OUT"
step "device before"; snapshot

for mode in full sampled; do
    step "C++ smoke, finite_check=$mode (expect 30 dispatches)"
    HETEROMEGA_FFN_FINITE_CHECK=$mode python -m experiments.ffn_deploy.smoke --config deployment.json \
      --iterations 3 --output "$OUT/smoke-$mode.json" 2>&1 | tee "$OUT/smoke-$mode.log" | tail -12
done

for mode in full sampled; do
    step "paired evaluation, finite_check=$mode"
    HETEROMEGA_FFN_FINITE_CHECK=$mode python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build \
      --out "$OUT/eval-$mode" --methods gpu_original merged_ffn --merged-ffn-config "$PWD/deployment.json" \
      --gpu-chunk-size 512 --prompt-tokens 512 1024 \
      2>&1 | tee "$OUT/eval-$mode.log" | grep -vE '^\s*$' | cut -c1-200 | tail -6
    echo "--- summary.csv"; cat "$OUT/eval-$mode/summary.csv"
    step "device after $mode"; snapshot
done
step "done"
