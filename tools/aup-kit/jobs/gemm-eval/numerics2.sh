#!/usr/bin/env bash
# Numerics with every switch of this kernel work on. The model-sampled prompts of step 5b, against gpu_original with the old kernels, one fresh process per run.
# gpu_sdpa isolates the attention kernel: gpu_original with the fast GEMM and the fast attention.
# bg gemm-numerics2 -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/numerics2.sh OUT_NAME [EXTRA_ENV=VALUE ...]'
set -uo pipefail
export SRC=/notebooks/hm-gemm
source /notebooks/hm-gemm/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/$1
shift
PROMPTS=$RESULTS/fix-step5b-sampled
mkdir -p "$OUT"
export HETEROMEGA_FFN_FINITE_CHECK=sampled
status=0
config() {
    case "$1" in
        p6400) echo "$PWD/deployment.json" ;;
        p7168) echo "$RESULTS/gemm-splits/deployment-p7168.json" ;;
        p8192) echo "$RESULTS/inflight2-prep/deployment-p8192.json" ;;
    esac
}
ALL="HETEROMEGA_GEMM=fast HETEROMEGA_GEMM_UNCACHED=1 HETEROMEGA_SDPA=fast HETEROMEGA_SDPA_TO_O=1 HETEROMEGA_FFN_FUSE=1 HETEROMEGA_FFN_SLOT_BO=2 $*"
step "device before"; snapshot
for seed in 1 2; do
    DIR=$OUT/seed$seed
    mkdir -p "$DIR"
    path() { # method name split(or -) env...
        method=$1
        name=$2
        split=$3
        shift 3
        echo "--- $name"
        (
            export "$@" LD_PRELOAD=$COUNT_IOCTLS
            extra=()
            [ "$split" != - ] && extra=(--merged-ffn-config "$(config "$split")")
            python3 tools/npu_error_probe/path_logits.py --method "$method" --model "$MODEL" --build build --ref "$PROMPTS/sampled-$seed.json" --prompt-tokens 1024 \
              --out "$DIR/$name.npz" "${extra[@]}" > "$DIR/$name.log" 2>&1
        ) || status=$?
        grep -E 'IOCTL_COUNTS' "$DIR/$name.log" | cut -c1-120 || true
        grep -c UNCACHED_FALLBACK "$DIR/$name.log" | grep -v '^0$' | sed 's/^/uncached fallbacks: /'
        tail -1 "$DIR/$name.log" | grep -E 'rror|Traceback' | cut -c1-240 || true
    }
    step "seed $seed"
    path gpu_original gpu_original-1024-r0 - NOTHING=1
    path gpu_original gpu_fast-1024-r0 - HETEROMEGA_GEMM=fast
    path gpu_original gpu_sdpa-1024-r0 - HETEROMEGA_GEMM=fast HETEROMEGA_SDPA=fast
    # shellcheck disable=SC2086
    path merged_ffn merged_p8192_all-1024-r0 p8192 $ALL
    # shellcheck disable=SC2086
    path merged_ffn merged_p7168_all-1024-r0 p7168 $ALL
    # shellcheck disable=SC2086
    path merged_ffn merged_p6400_all-1024-r0 p6400 $ALL
    path gpu_original gpu_original-1024-r1 - NOTHING=1
    step "seed $seed: comparison"
    python3 tools/npu_error_probe/path_logits.py --compare "$DIR" --json "$OUT/comparison-$seed.json" 2>&1 | cut -c1-260 || status=$?
    rm -f "${DIR:?}"/*.npz
done
step "device after"; snapshot
step "done status=$status"
exit $status
