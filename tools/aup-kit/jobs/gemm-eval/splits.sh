#!/usr/bin/env bash
# merged_ffn with gemm_w4a16 at other split points: builds each NPU artifact once into a new directory, smoke tests it, then times
# inflight 1 and 2 at 1024 tokens, with native calls counted. The P6400 and P8192 artifacts and configs stay as they are.
# bg gemm-splits -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/splits.sh ROUNDS REPS P [P...]'
# P must be a multiple of 256 and at most 8192 (experiments/ffn_deploy/prepare.py).
set -uo pipefail
export SRC=/notebooks/hm-gemm
source /notebooks/hm-gemm/tools/aup-kit/jobs/fix-env.sh
ROUNDS=$1
REPS=$2
shift 2
OUT=$RESULTS/gemm-splits
mkdir -p "$OUT"
export HETEROMEGA_FFN_FINITE_CHECK=sampled HETEROMEGA_GEMM=fast
status=0
step "device before"; snapshot
for p in "$@"; do
    cfg=$OUT/deployment-p$p.json
    if [ ! -e "$cfg" ]; then
        step "P$p: build"
        python -m experiments.ffn_deploy.prepare --build --iron-python /home/mliraie/mlir-aie/ironenv/bin/python \
          --artifact-dir "$OUT/artifact-p$p" --m 512 --hidden 4096 --intermediate 14336 --npu-channels "$p" \
          --output "$cfg" 2>&1 | tail -3 || status=$?
        step "P$p: built"
        [ -e "$cfg" ] || { echo "P$p: no config"; status=1; continue; }
        python -m experiments.ffn_deploy.smoke --config "$cfg" --iterations 3 --output "$OUT/smoke-p$p.json" \
          2>&1 | tee "$OUT/smoke-p$p.log" | tail -2 | cut -c1-200 || status=$?
    fi
    for inflight in 1 2; do
        step "P$p inflight $inflight"
        D=$OUT/p$p-i$inflight
        LD_PRELOAD=$COUNT_IOCTLS python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$D" \
          --merged-ffn-config "$cfg" --gpu-chunk-size 512 --methods merged_ffn --inflight "$inflight" --prompt-tokens 1024 \
          --rounds "$ROUNDS" --reps "$REPS" 2>&1 | tee "$D.log" | cut -c1-160 | tail -1 || status=$?
        cut -d, -f1,2,3,4,5,6,18 "$D/summary.csv"
        grep -H IOCTL_COUNTS "$D"/*.log | sed "s|$D/||" | cut -c1-150
    done
done
step "device after"; snapshot
step "done status=$status"
exit $status
