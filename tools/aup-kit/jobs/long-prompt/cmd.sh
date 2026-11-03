#!/usr/bin/env bash
# Prefill at 2048 and 4096 tokens: merged_ffn with every kept switch against gpu_original and heteromosaic with the old kernels, GTT sampled per case.
# heteromosaic uses its shipped group of each length: 1024-row chunks at 2048 tokens, 2048-row chunks at 4096 tokens, two inflight chunks. merged_ffn stays at M=512.
# bg long-probe -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/long-prompt/cmd.sh probe'
set -uo pipefail
export SRC=/notebooks/hm-gemm
source /notebooks/hm-gemm/tools/aup-kit/jobs/fix-env.sh
STAGE=$1
OUT=$RESULTS/long-prompt
REF=$OUT/sampled-4160.json
ALL=(HETEROMEGA_GEMM=fast HETEROMEGA_GEMM_UNCACHED=1 HETEROMEGA_SDPA=fast HETEROMEGA_SDPA_TO_O=1 HETEROMEGA_FFN_FUSE=1 HETEROMEGA_FFN_SLOT_BO=2 HETEROMEGA_RESIDUAL_NORM=1)
# The cases below drop the switches that hold memory: uncached weights copies, and the V tiles of sdpa_bf16.
NO_UNCACHED=(HETEROMEGA_GEMM=fast HETEROMEGA_SDPA=fast HETEROMEGA_SDPA_TO_O=1 HETEROMEGA_FFN_FUSE=1 HETEROMEGA_FFN_SLOT_BO=2 HETEROMEGA_RESIDUAL_NORM=1)
NO_SDPA=(HETEROMEGA_GEMM=fast HETEROMEGA_GEMM_UNCACHED=1 HETEROMEGA_FFN_FUSE=1 HETEROMEGA_FFN_SLOT_BO=2 HETEROMEGA_RESIDUAL_NORM=1)
NEITHER=(HETEROMEGA_GEMM=fast HETEROMEGA_FFN_FUSE=1 HETEROMEGA_FFN_SLOT_BO=2 HETEROMEGA_RESIDUAL_NORM=1)
SCHED=$OUT/config-scheduled.json5
export HETEROMEGA_FFN_FINITE_CHECK=sampled
mkdir -p "$OUT/$STAGE"
status=0
config() {
    case "$1" in
        p6400) echo "$PWD/deployment.json" ;;
        p7168) echo "$RESULTS/gemm-splits/deployment-p7168.json" ;;
        p8192) echo "$RESULTS/inflight2-prep/deployment-p8192.json" ;;
    esac
}
run_case() { # LABEL METHOD LENGTH CHUNK ROUNDS REPS [NAME=VALUE ...], two inflight chunks unless INFLIGHT is set, shipped config unless CONFIG is set
    local label=$1 method=$2 length=$3 chunk=$4 rounds=$5 reps=$6
    shift 6
    local D=$OUT/$STAGE/$label
    rm -rf "$D"
    mkdir -p "$D"
    local gtt
    gtt=$(ls /sys/class/drm/card*/device/mem_info_gtt_used | head -1)
    (
        while :; do
            echo "$(($(cat "$gtt") / 1048576)),$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)"
            sleep 0.5
        done
    ) > "$D.gtt.csv" &
    local sampler=$!
    (
        for pair in "$@"; do export "$pair"; done
        common=(--model "$MODEL" --build build --out "$D" --ref "$REF" --gpu-chunk-size "$chunk" --inflight "${INFLIGHT:-2}" --prompt-tokens "$length" --rounds "$rounds" --reps "$reps" ${CONFIG:+--config "$CONFIG"})
        case "$method" in
            gpu) timeout -k 30 1500 python3 kernels/gpu/tools/evaluate_prefill.py "${common[@]}" --methods gpu_original ;;
            hetero) LD_PRELOAD=$COUNT_IOCTLS timeout -k 30 1500 python3 kernels/gpu/tools/evaluate_prefill.py "${common[@]}" --methods heteromosaic ;;
            *) LD_PRELOAD=$COUNT_IOCTLS timeout -k 30 1500 python3 kernels/gpu/tools/evaluate_prefill.py "${common[@]}" --methods merged_ffn --merged-ffn-config "$(config "$method")" ;;
        esac
    ) > "$D.log" 2>&1
    local result=$?
    kill "$sampler" 2>/dev/null
    local peak avail calls fallbacks
    peak=$(cut -d, -f1 "$D.gtt.csv" | sort -n | tail -1)
    avail=$(cut -d, -f2 "$D.gtt.csv" | sort -n | head -1)
    calls=$(grep -h IOCTL_COUNTS "$D"/*.log 2>/dev/null | grep -o 'exec_ok=[0-9]*' | sort -u | tr '\n' ' ')
    fallbacks=$(cat "$D"/*.log 2>/dev/null | grep -c UNCACHED_FALLBACK)
    if [ "$result" -ne 0 ] || [ ! -s "$D/summary.csv" ]; then
        status=1
        echo "$(date -u +%H:%M:%S) $label: FAILED exit=$result peakGTT=$peak lowestAvail=$avail native $calls fallbacks=$fallbacks"
        tail -4 "$D.log" 2>/dev/null | cut -c1-260
        grep -hiE 'out of memory|bad_alloc|hipError|failed|error' "$D"/*.log 2>/dev/null | sort | uniq -c | sort -rn | head -5 | cut -c1-220
        return
    fi
    echo "$(date -u +%H:%M:%S) $label: median,min,max $(tail -1 "$D/summary.csv" | cut -d, -f5,6,7 | cut -c1-32) peakGTT=$peak lowestAvail=$avail native $calls fallbacks=$fallbacks"
}
sample() {
    [ -e "$REF" ] && return
    step "sample 4160 tokens"
    python3 tools/aup-kit/jobs/long-prompt/sample_ref.py "$REF" "$MODEL" 4160 "$RESULTS/fix-step5b-sampled/sampled-1.json" > "$OUT/sample.log" 2>&1 || { echo "sampling failed"; tail -4 "$OUT/sample.log" | cut -c1-240; exit 1; }
    grep -E '^\{"tokens|tokens equal' "$OUT/sample.log" | cut -c1-200
    step "sampled"
}
step "device before"; snapshot
sample
case "$STAGE" in
    probe)
        run_case gpu-2048 gpu 2048 512 1 1
        run_case hetero-2048 hetero 2048 1024 1 1
        run_case p8192-2048 p8192 2048 512 1 1 "${ALL[@]}"
        run_case gpu-4096 gpu 4096 512 1 1
        run_case hetero-4096 hetero 4096 2048 1 1
        run_case p8192-4096 p8192 4096 512 1 1 "${ALL[@]}"
        ;;
    probe4096)
        run_case p8192-no-uncached-4096 p8192 4096 512 1 1 "${NO_UNCACHED[@]}"
        run_case p8192-no-sdpa-4096 p8192 4096 512 1 1 "${NO_SDPA[@]}"
        run_case p8192-neither-4096 p8192 4096 512 1 1 "${NEITHER[@]}"
        ;;
    timing2048)
        # heteromosaic ships two 2048-token settings: 1024-row chunks at two inflight, and four 512-row chunks at four inflight. Both run.
        sed 's/"chunking_scheduled": false/"chunking_scheduled": true/' py/unified_llm_w4a16/configs/configs_strixP_llama3_8b.json5 > "$SCHED"
        run_case gpu-2048 gpu 2048 512 1 3
        for r in 0 1 2; do
            run_case p8192-2048-r$r p8192 2048 512 1 3 "${ALL[@]}"
            INFLIGHT=4 CONFIG=$SCHED run_case hetero-sched-2048-r$r hetero 2048 512 1 3
            run_case p7168-2048-r$r p7168 2048 512 1 3 "${ALL[@]}"
            run_case hetero-2048-r$r hetero 2048 1024 1 3
            run_case p6400-2048-r$r p6400 2048 512 1 3 "${ALL[@]}"
            run_case p8192-1024-r$r p8192 1024 512 1 3 "${ALL[@]}"
        done
        ;;
    timing4096)
        # All switches do not fit in GTT at 4096 tokens, measured in stage probe. NO_UNCACHED fits.
        run_case gpu-4096 gpu 4096 512 1 3
        for r in 0 1 2; do
            run_case p8192-4096-r$r p8192 4096 512 1 3 "${NO_UNCACHED[@]}"
            run_case hetero-4096-r$r hetero 4096 2048 1 3
            run_case p7168-4096-r$r p7168 4096 512 1 3 "${NO_UNCACHED[@]}"
            run_case p6400-4096-r$r p6400 4096 512 1 3 "${NO_UNCACHED[@]}"
        done
        ;;
    *) echo "unknown stage $STAGE"; exit 2 ;;
esac
step "device after"; snapshot
step "done status=$status"
exit $status
