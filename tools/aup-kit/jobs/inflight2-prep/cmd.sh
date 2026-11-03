#!/usr/bin/env bash
# Two-inflight comparison, preparation. Builds the P8192 merged artifact into a new directory with a new deployment config
# (the P6400 artifact and config stay as they are), writes the all-gate/up-on-NPU config, and tries each once.
# bg inflight2-prep -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/inflight2-prep/cmd.sh'
set -o pipefail
source /notebooks/hm-fix/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/inflight2-prep
mkdir -p "$OUT"
export HETEROMEGA_FFN_FINITE_CHECK=sampled
status=0
step "device before"; snapshot

step "config 3: gate and up artifacts"
FW=hw_bins/npu2
STEM=512x4096x14336_64x128x64_8c_bf16_int4AWQ_bf16
sha256sum $FW/8192x4096x14336/bf16_int4AWQ_bf16_M/{final_$STEM.pdi,insts_$STEM.txt} $FW/16384x4096x14336/bf16_int4AWQ_bf16_M/{final_$STEM.pdi,insts_$STEM.txt} | tee "$OUT/gateup-artifacts.sha256"
python3 tools/split_model/make_gateup_config.py --out "$OUT/gateup.config.json" || status=$?

step "config 3: first run, one process, native calls counted"
LD_PRELOAD=$COUNT_IOCTLS python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$OUT/try-gateup" \
  --methods heteromosaic_gateup --gateup-config "$OUT/gateup.config.json" --prompt-tokens 1024 --rounds 1 --reps 1 \
  2>&1 | tee "$OUT/try-gateup.log" | cut -c1-200 | tail -1 || status=$?
grep -hE 'WARMUP|REP |IOCTL_COUNTS|DONE|rror|Traceback|fall' "$OUT"/try-gateup/*.log | cut -c1-220 | head -20
cut -d, -f1,2,4,5,6,7,14 "$OUT/try-gateup/summary.csv" 2>/dev/null

step "P8192: build"
date -u
python -m experiments.ffn_deploy.prepare --build --iron-python /home/mliraie/mlir-aie/ironenv/bin/python \
  --artifact-dir "$OUT/artifact-p8192" --m 512 --hidden 4096 --intermediate 14336 --npu-channels 8192 \
  --output "$OUT/deployment-p8192.json" 2>&1 | tail -8 || status=$?
date -u
ls -la "$OUT/artifact-p8192" | cut -c1-90 | head -14
tail -3 "$OUT/artifact-p8192/compile.log" | cut -c1-200
git status --short | head -12

step "P8192: smoke, no timers"
python -m experiments.ffn_deploy.smoke --config "$OUT/deployment-p8192.json" --iterations 3 --output "$OUT/smoke-p8192.json" \
  2>&1 | tee "$OUT/smoke-p8192.log" | tail -4 | cut -c1-220 || status=$?
step "device after"; snapshot
step "done status=$status"
exit $status
