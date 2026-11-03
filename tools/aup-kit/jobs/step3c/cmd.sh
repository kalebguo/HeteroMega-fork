#!/usr/bin/env bash
# A. Head to head on one timer: heteromosaic vs merged_ffn, alternating processes, native calls counted.
# B. Direct cost of NPU context changes: timed ioctls of heteromosaic, heteromosaic_mdown and merged_ffn_hetero.
# bg step3c -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/step3c/cmd.sh'
set -eo pipefail
source /notebooks/hm-fix/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/fix-step3c
mkdir -p "$OUT"
export HETEROMEGA_FFN_FINITE_CHECK=sampled
step "device before"; snapshot

step "A. heteromosaic vs merged_ffn"
LD_PRELOAD=$COUNT_IOCTLS python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$OUT/head-to-head" \
  --methods heteromosaic merged_ffn --merged-ffn-config "$PWD/deployment.json" --gpu-chunk-size 512 --prompt-tokens 1024 \
  2>&1 | tee "$OUT/head-to-head.log" | cut -c1-160 | tail -1
cut -d, -f1,2,5,6,7,10,11,12,18 "$OUT/head-to-head/summary.csv"
grep -H IOCTL_COUNTS "$OUT"/head-to-head/*.log | sed "s|$OUT/head-to-head/||"

step "B. timed ioctls"
TRACER=/notebooks/hm-results/profile_ioctls.so
gcc -shared -fPIC -I"$HOME_LIBS/xdna-driver-05-19-25/src/include/uapi/drm_local" -I/usr/include/libdrm \
  tools/aup-kit/jobs/step3c/profile_ioctls.c -ldl -o "$TRACER"
CONFIG=$RESULTS/fix-step4-mdown.config.json
for method in heteromosaic heteromosaic_mdown merged_ffn_hetero; do
    LD_PRELOAD=$TRACER HETEROMEGA_IOCTL_LOG=$OUT/ioctl-$method python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build \
      --out "$OUT/trace-$method" --methods $method --mdown-config "$CONFIG" --merged-ffn-config "$PWD/deployment.json" \
      --prompt-tokens 1024 --rounds 1 2>&1 | tee "$OUT/trace-$method.log" | cut -c1-160 | tail -1
    cut -d, -f1,2,5,6,7,12 "$OUT/trace-$method/summary.csv" | tail -1
done
python3 tools/aup-kit/jobs/step3c/context_waits.py "$OUT"/ioctl-*.csv | tee "$OUT/context-waits.json" | tail -80
gzip -f "$OUT"/ioctl-*.csv
step "device after"; snapshot
step "done"
