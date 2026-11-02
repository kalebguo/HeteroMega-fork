#!/usr/bin/env bash
# Step 2 of merged-ffn-fix: per-phase timers of MergedFfn::forward. Smoke (synthetic layers, two slots), then the model
# path at 1024 tokens with GPU attention, one process. Diagnostic timing: compare only phases, not the prefill median.
# bg step2-phases -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/step2-phases/cmd.sh'
set -eo pipefail
source /notebooks/hm-fix/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/fix-step2
mkdir -p "$OUT"
export HETEROMEGA_FFN_FINITE_CHECK=sampled
step "device before"; snapshot

step "smoke with timers"
HETEROMEGA_FFN_TIMERS=$OUT/smoke-timers python -m experiments.ffn_deploy.smoke --config deployment.json \
  --iterations 3 --output "$OUT/smoke.json" 2>&1 | tee "$OUT/smoke.log" | tail -6
python -m experiments.ffn_deploy.phase_report "$OUT/smoke-timers" --skip 4 --output "$OUT/smoke-phases.json" | tail -60

step "model path with timers, 1024 tokens"
HETEROMEGA_FFN_TIMERS=$OUT/model-timers python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build \
  --out "$OUT/model" --methods merged_ffn --merged-ffn-config "$PWD/deployment.json" \
  --gpu-chunk-size 512 --prompt-tokens 1024 --rounds 1 2>&1 | tee "$OUT/model.log" | cut -c1-200 | tail -4
python -m experiments.ffn_deploy.phase_report "$OUT/model-timers" --skip 64 --output "$OUT/model-phases.json" | tail -60
step "device after"; snapshot
step "done"
