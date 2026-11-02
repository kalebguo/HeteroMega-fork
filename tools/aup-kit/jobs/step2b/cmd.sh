#!/usr/bin/env bash
# Follow-up of step 2 of merged-ffn-fix, three questions:
# A. Does the unconditional wait for input readiness restore GPU/NPU overlap on the smoke's second stream, without timers?
#    Does the headline number hold at this commit?
# B. Does the missing overlap affect the model path with two inflight chunks?
# C. Attribution: where did the 19 ms per call of the unoptimized build go? Same code without -O3, with timers.
# bg step2b -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/step2b/cmd.sh'
set -eo pipefail
JOBS=/notebooks/hm-fix/tools/aup-kit/jobs
source $JOBS/fix-env.sh
OUT=$RESULTS/fix-step2b
mkdir -p "$OUT"
export HETEROMEGA_FFN_FINITE_CHECK=sampled
step "device before"; snapshot

step "A. smoke, no timers"
python -m experiments.ffn_deploy.smoke --config deployment.json --iterations 3 --output "$OUT/smoke-sampled.json" \
  2>&1 | tee "$OUT/smoke-sampled.log" | tail -3
step "A. paired evaluation"
python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$OUT/eval-sampled" \
  --methods gpu_original merged_ffn --merged-ffn-config "$PWD/deployment.json" --gpu-chunk-size 512 --prompt-tokens 512 1024 \
  2>&1 | tee "$OUT/eval-sampled.log" | cut -c1-160 | tail -1
cut -d, -f1,2,5,6,7,8,9,12,18 "$OUT/eval-sampled/summary.csv"

step "B. two inflight chunks, no timers"
python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$OUT/eval-inflight2" \
  --methods gpu_original merged_ffn --merged-ffn-config "$PWD/deployment.json" --gpu-chunk-size 512 --inflight 2 --prompt-tokens 1024 \
  2>&1 | tee "$OUT/eval-inflight2.log" | cut -c1-160 | tail -1
cut -d, -f1,2,5,6,7,8,9,12,18 "$OUT/eval-inflight2/summary.csv"
step "B. two inflight chunks, timers"
HETEROMEGA_FFN_TIMERS=$OUT/inflight2-timers python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build \
  --out "$OUT/inflight2-model" --methods merged_ffn --merged-ffn-config "$PWD/deployment.json" --gpu-chunk-size 512 --inflight 2 \
  --prompt-tokens 1024 --rounds 1 2>&1 | tee "$OUT/inflight2-model.log" | cut -c1-160 | tail -1
python -m experiments.ffn_deploy.phase_report "$OUT/inflight2-timers" --skip 64 --output "$OUT/inflight2-phases.json" | tail -5

step "C. build the same commit without -O3 in a separate tree"
SRC=/notebooks/hm-fix-noopt NOOPT=1 bash $JOBS/build-fix/cmd.sh 2>&1 | tail -30 | grep -E "optimization|Built target unified|O3|error" || true
(
    export SRC=/notebooks/hm-fix-noopt
    source $JOBS/fix-env.sh
    export HETEROMEGA_FFN_FINITE_CHECK=full
    step "C. no -O: smoke, full check, timers"
    HETEROMEGA_FFN_TIMERS=$OUT/noopt-smoke-timers python -m experiments.ffn_deploy.smoke --config deployment.json --iterations 3 \
      --output "$OUT/noopt-smoke.json" 2>&1 | tee "$OUT/noopt-smoke.log" | tail -3
    python -m experiments.ffn_deploy.phase_report "$OUT/noopt-smoke-timers" --skip 4 --output "$OUT/noopt-smoke-phases.json" | tail -3
    step "C. no -O: model, full check, timers"
    HETEROMEGA_FFN_TIMERS=$OUT/noopt-model-timers python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build \
      --out "$OUT/noopt-model" --methods merged_ffn --merged-ffn-config "$PWD/deployment.json" --gpu-chunk-size 512 \
      --prompt-tokens 1024 --rounds 1 2>&1 | tee "$OUT/noopt-model.log" | cut -c1-160 | tail -1
    python -m experiments.ffn_deploy.phase_report "$OUT/noopt-model-timers" --skip 64 --output "$OUT/noopt-model-phases.json" | tail -3
)
source $JOBS/fix-env.sh
step "C. -O3: model, full check, timers (same commit)"
HETEROMEGA_FFN_FINITE_CHECK=full HETEROMEGA_FFN_TIMERS=$OUT/o3full-model-timers python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build \
  --out "$OUT/o3full-model" --methods merged_ffn --merged-ffn-config "$PWD/deployment.json" --gpu-chunk-size 512 \
  --prompt-tokens 1024 --rounds 1 2>&1 | tee "$OUT/o3full-model.log" | cut -c1-160 | tail -1
python -m experiments.ffn_deploy.phase_report "$OUT/o3full-model-timers" --skip 64 --output "$OUT/o3full-model-phases.json" | tail -3
step "device after"; snapshot
step "done"
