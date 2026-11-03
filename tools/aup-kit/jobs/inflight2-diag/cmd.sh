#!/usr/bin/env bash
# Two-inflight comparison, bottleneck diagnostics, apart from the timing job: merged phase timers, native timed ioctls.
# bg inflight2-diag -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/inflight2-diag/cmd.sh'
set -eo pipefail
source /notebooks/hm-fix/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/inflight2-diag
PREP=$RESULTS/inflight2-prep
mkdir -p "$OUT"
export HETEROMEGA_FFN_FINITE_CHECK=sampled
step "device before"; snapshot
merged() { # method inflight
    name=$1-i$2
    step "timers: $name"
    HETEROMEGA_FFN_TIMERS=$OUT/$name-timers python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$OUT/$name" \
      --methods "$1" --merged-ffn-config "$PWD/deployment.json" --p8192-ffn-config "$PREP/deployment-p8192.json" --gpu-chunk-size 512 \
      --inflight "$2" --prompt-tokens 1024 --rounds 1 2>&1 | tee "$OUT/$name.log" | cut -c1-160 | tail -1
    cut -d, -f1,2,5,6,7,18 "$OUT/$name/summary.csv" | tail -1
    python -m experiments.ffn_deploy.phase_report "$OUT/$name-timers" --skip 64 --output "$OUT/$name-phases.json" | tail -12 | cut -c1-120
}
merged merged_ffn 2
merged merged_ffn_p8192 2
merged merged_ffn_p8192 1
TRACER=/notebooks/hm-results/profile_ioctls.so
native() { # method inflight
    name=$1-i$2
    step "timed ioctls: $name"
    LD_PRELOAD=$TRACER HETEROMEGA_IOCTL_LOG=$OUT/ioctl-$name python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$OUT/$name" \
      --methods "$1" --gateup-config "$PREP/gateup-8192family.config.json" --inflight "$2" --prompt-tokens 1024 --rounds 1 --new-tokens 1 \
      2>&1 | tee "$OUT/$name.log" | cut -c1-160 | tail -1
    cut -d, -f1,2,5,6,7,12 "$OUT/$name/summary.csv" | tail -1
}
native heteromosaic 1
native heteromosaic_gateup 1
native heteromosaic 2
native heteromosaic_gateup 2
step "trace report"
python3 tools/aup-kit/jobs/inflight2-diag/trace_busy.py "$OUT"/ioctl-*.csv | tee "$OUT/trace-busy.json" | cut -c1-200
gzip -f "$OUT"/ioctl-*.csv
step "device after"; snapshot
step "done"
