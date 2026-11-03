#!/usr/bin/env bash
# Numerics of gemm_w4a16 end to end. The model-sampled prompts of step 5b, against gpu_original with the old kernel, one fresh process per
# run. gpu_fast isolates the kernel's own error: it is gpu_original with HETEROMEGA_GEMM=fast. The rows of the other methods are archived.
# bg gemm-numerics -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/numerics.sh'
set -uo pipefail
export SRC=/notebooks/hm-gemm
source /notebooks/hm-gemm/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/gemm-numerics
PREP=$RESULTS/inflight2-prep
PROMPTS=$RESULTS/fix-step5b-sampled
mkdir -p "$OUT"
export HETEROMEGA_FFN_FINITE_CHECK=sampled
status=0
step "device before"; snapshot
for seed in 1 2; do
    DIR=$OUT/seed$seed
    mkdir -p "$DIR"
    path() { # gemm(old|fast) method name [extra arguments]
        gemm=$1
        method=$2
        name=$3
        shift 3
        echo "--- $name ($gemm)"
        if [ "$gemm" = fast ]; then export HETEROMEGA_GEMM=fast; else unset HETEROMEGA_GEMM; fi
        LD_PRELOAD=$COUNT_IOCTLS python3 tools/npu_error_probe/path_logits.py --method "$method" --model "$MODEL" --build build \
          --merged-ffn-config "$PWD/deployment.json" --ref "$PROMPTS/sampled-$seed.json" --prompt-tokens 1024 --out "$DIR/$name.npz" "$@" \
          > "$DIR/$name.log" 2>&1 || status=$?
        grep -E 'IOCTL_COUNTS' "$DIR/$name.log" | cut -c1-200 || true
        tail -1 "$DIR/$name.log" | grep -E 'rror|Traceback' | cut -c1-240 || true
    }
    step "seed $seed"
    path old gpu_original gpu_original-1024-r0
    path fast gpu_original gpu_fast-1024-r0
    path fast merged_ffn merged_p6400_fast-1024-r0
    path fast merged_ffn_p8192 merged_p8192_fast-1024-r0 --merged-ffn-config "$PREP/deployment-p8192.json"
    path old gpu_original gpu_original-1024-r1
    step "seed $seed: comparison"
    python3 tools/npu_error_probe/path_logits.py --compare "$DIR" --json "$OUT/comparison-$seed.json" 2>&1 | cut -c1-260 || status=$?
    # The per-position logits are 262 MB per run and only the comparison needs them.
    rm -f "${DIR:?}"/*.npz
done
step "device after"; snapshot
step "done status=$status"
exit $status
