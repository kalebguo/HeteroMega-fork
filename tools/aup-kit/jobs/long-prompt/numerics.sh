#!/usr/bin/env bash
# Logits at 2048 and 4096 tokens against gpu_original on the sampled prompt, one fresh process per run. 256 scored positions at 2048; none at 4096 because the prompt ends at 4160.
# gpu_chunk* show whether the chunk size alone moves the GPU-only output. heteromosaic runs both shipped 2048-token settings and the shipped 4096-token setting.
# merged4096: merged_ffn at 4096 tokens, with the switches that fit in GTT. seed2: 2048 tokens on a second sampled prompt.
# bg long-numerics -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/long-prompt/numerics.sh all|merged4096|seed2'
set -uo pipefail
export SRC=/notebooks/hm-gemm
source /notebooks/hm-gemm/tools/aup-kit/jobs/fix-env.sh
STAGE=$1
OUT=$RESULTS/long-prompt/numerics
REF=$RESULTS/long-prompt/sampled-4160.json
SCHED=$RESULTS/long-prompt/config-scheduled.json5
ALL=(HETEROMEGA_GEMM=fast HETEROMEGA_GEMM_UNCACHED=1 HETEROMEGA_SDPA=fast HETEROMEGA_SDPA_TO_O=1 HETEROMEGA_FFN_FUSE=1 HETEROMEGA_FFN_SLOT_BO=2 HETEROMEGA_RESIDUAL_NORM=1)
NO_UNCACHED=(HETEROMEGA_GEMM=fast HETEROMEGA_SDPA=fast HETEROMEGA_SDPA_TO_O=1 HETEROMEGA_FFN_FUSE=1 HETEROMEGA_FFN_SLOT_BO=2 HETEROMEGA_RESIDUAL_NORM=1)
export HETEROMEGA_FFN_FINITE_CHECK=sampled
status=0
if [ "$STAGE" = seed2 ]; then
    OUT=$RESULTS/long-prompt/numerics-seed2
    REF=$RESULTS/long-prompt/sampled-2304-seed2.json
fi
mkdir -p "$OUT/2048" "$OUT/4096"
config() {
    case "$1" in
        p6400) echo "$PWD/deployment.json" ;;
        p7168) echo "$RESULTS/gemm-splits/deployment-p7168.json" ;;
        p8192) echo "$RESULTS/inflight2-prep/deployment-p8192.json" ;;
    esac
}
run() { # STEM METHOD LENGTH CHUNK SPLIT(or -) INFLIGHT CONFIG(or -) [NAME=VALUE ...]
    local stem=$1 method=$2 length=$3 chunk=$4 split=$5 inflight=$6 cfg=$7
    shift 7
    local D=$OUT/$length capacity=$((length + 256))
    [ "$length" = 4096 ] && capacity=4160
    (
        export LD_PRELOAD=$COUNT_IOCTLS
        for pair in "$@"; do export "$pair"; done
        extra=()
        [ "$split" != - ] && extra+=(--merged-ffn-config "$(config "$split")")
        [ "$cfg" != - ] && extra+=(--config "$cfg")
        timeout -k 30 900 python3 tools/npu_error_probe/path_logits.py --method "$method" --model "$MODEL" --build build --ref "$REF" --prompt-tokens "$length" --chunk "$chunk" \
          --inflight "$inflight" --max-seq-len "$capacity" --ppl-tokens 256 --out "$D/$stem.npz" "${extra[@]}" > "$D/$stem.log" 2>&1
    ) || { status=1; echo "$stem: FAILED"; tail -2 "$D/$stem.log" | cut -c1-240; }
    echo "$(date -u +%H:%M:%S) $stem $(grep -h IOCTL_COUNTS "$D/$stem.log" | grep -o 'exec_ok=[0-9]*' | tail -1) fallbacks=$(grep -c UNCACHED_FALLBACK "$D/$stem.log")"
}
step "device before"; snapshot
compare() { # LENGTH JSON_NAME
    python3 tools/npu_error_probe/path_logits.py --compare "$OUT/$1" --json "$OUT/$2" 2>&1 | cut -c1-260 || status=1
    rm -f "$OUT/$1"/*.npz
}
case "$STAGE" in
    all)
        step "2048"
        run gpu_original-2048-r0 gpu_original 2048 512 - 2 -
        run gpu_chunk1024-2048-r0 gpu_original 2048 1024 - 2 -
        run hetero1024-2048-r0 heteromosaic 2048 1024 - 2 -
        run heterosched-2048-r0 heteromosaic 2048 512 - 4 "$SCHED"
        run merged_p8192_all-2048-r0 merged_ffn 2048 512 p8192 2 - "${ALL[@]}"
        run merged_p7168_all-2048-r0 merged_ffn 2048 512 p7168 2 - "${ALL[@]}"
        run merged_p6400_all-2048-r0 merged_ffn 2048 512 p6400 2 - "${ALL[@]}"
        run gpu_original-2048-r1 gpu_original 2048 512 - 2 -
        compare 2048 comparison-2048.json
        step "4096"
        run gpu_original-4096-r0 gpu_original 4096 512 - 2 -
        run gpu_chunk2048-4096-r0 gpu_original 4096 2048 - 2 -
        run hetero2048-4096-r0 heteromosaic 4096 2048 - 2 -
        compare 4096 comparison-4096.json
        ;;
    merged4096)
        step "4096, merged_ffn"
        run gpu_original-4096-r0 gpu_original 4096 512 - 2 -
        run merged_p8192-4096-r0 merged_ffn 4096 512 p8192 2 - "${NO_UNCACHED[@]}"
        run merged_p7168-4096-r0 merged_ffn 4096 512 p7168 2 - "${NO_UNCACHED[@]}"
        run merged_p6400-4096-r0 merged_ffn 4096 512 p6400 2 - "${NO_UNCACHED[@]}"
        compare 4096 comparison-4096-merged.json
        ;;
    seed2)
        step "sample 2304 tokens, seed 2"
        [ -e "$REF" ] || python3 tools/aup-kit/jobs/long-prompt/sample_ref.py "$REF" "$MODEL" 2304 "$RESULTS/fix-step5b-sampled/sampled-2.json" 2 > "$OUT/sample.log" 2>&1 || { echo "sampling failed"; tail -4 "$OUT/sample.log" | cut -c1-240; exit 1; }
        grep -E '^\{"tokens|tokens equal' "$OUT/sample.log" | cut -c1-200
        step "2048"
        run gpu_original-2048-r0 gpu_original 2048 512 - 2 -
        run hetero1024-2048-r0 heteromosaic 2048 1024 - 2 -
        run heterosched-2048-r0 heteromosaic 2048 512 - 4 "$SCHED"
        run merged_p8192_all-2048-r0 merged_ffn 2048 512 p8192 2 - "${ALL[@]}"
        compare 2048 comparison-2048.json
        ;;
    *) echo "unknown stage $STAGE"; exit 2 ;;
esac
step "device after"; snapshot
step "done status=$status"
exit $status
