#!/usr/bin/env bash
# Step 4 of merged-ffn-fix: the strongest competitor. heteromosaic with and without native M-split down (256 of 512 rows,
# existing artifact), on one timer, with the ioctl counter proving the +64 NPU calls per prefill.
# bg step4-mdown -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/step4-mdown/cmd.sh'
set -eo pipefail
source /notebooks/hm-fix/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/fix-step4-mdown
CONFIG=$RESULTS/fix-step4-mdown.config.json
python3 tools/split_model/make_mdown_config.py --out "$CONFIG"
step "device before"; snapshot
LD_PRELOAD=$COUNT_IOCTLS python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build \
  --out "$OUT" --methods heteromosaic heteromosaic_mdown --mdown-config "$CONFIG" --prompt-tokens 1024 \
  2>&1 | tee "$OUT.log" | grep -vE '^\s*$' | cut -c1-200 | tail -6
echo "--- summary.csv"; cat "$OUT/summary.csv"
step "native calls per process"; grep -H IOCTL_COUNTS "$OUT"/*.log | sed "s|$OUT/||"
step "device after"; snapshot
step "done"
