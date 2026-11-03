#!/usr/bin/env bash
# Where the time of merged_ffn with gemm_w4a16 goes, apart from the timing jobs because timers change latency: merged phase timers per
# split and inflight, and gpu_original with the new kernel, which gives the whole GPU load with no NPU.
# bg gemm-diag -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/diag.sh CONFIG:LABEL [CONFIG:LABEL...]'
set -uo pipefail
export SRC=/notebooks/hm-gemm
source /notebooks/hm-gemm/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/gemm-diag
mkdir -p "$OUT"
export HETEROMEGA_FFN_FINITE_CHECK=sampled HETEROMEGA_GEMM=fast
step "device before"; snapshot
for pair in "$@"; do
    cfg=${pair%%:*}
    label=${pair##*:}
    for inflight in 1 2; do
        name=$label-i$inflight
        step "timers: $name"
        HETEROMEGA_FFN_TIMERS=$OUT/$name-timers python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$OUT/$name" \
          --methods merged_ffn --merged-ffn-config "$cfg" --gpu-chunk-size 512 --inflight "$inflight" --prompt-tokens 1024 --rounds 1 \
          2>&1 | tee "$OUT/$name.log" | cut -c1-160 | tail -1
        cut -d, -f1,2,5,6,7,18 "$OUT/$name/summary.csv" | tail -1
        python -m experiments.ffn_deploy.phase_report "$OUT/$name-timers" --skip 64 --output "$OUT/$name-phases.json" | tail -40 | grep -E '"(total|input_wait|mutex_wait|run_wait|gpu_wait|since_prev_end)_ms"' | head -6 | cut -c1-80
    done
done
step "gpu_original with the new kernel, 512-token chunks"
python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$OUT/gpu-fast" --methods gpu_original --gpu-chunk-size 512 \
  --prompt-tokens 1024 --rounds 2 --reps 5 2>&1 | tee "$OUT/gpu-fast.log" | cut -c1-160 | tail -1
cut -d, -f1,2,4,5,6,7 "$OUT/gpu-fast/summary.csv"
step "gpu_original with the old kernel, 512-token chunks, same session"
unset HETEROMEGA_GEMM
python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$OUT/gpu-old" --methods gpu_original --gpu-chunk-size 512 \
  --prompt-tokens 1024 --rounds 2 --reps 5 2>&1 | tee "$OUT/gpu-old.log" | cut -c1-160 | tail -1
cut -d, -f1,2,4,5,6,7 "$OUT/gpu-old/summary.csv"
step "device after"; snapshot
step "done"
