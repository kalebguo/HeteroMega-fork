#!/usr/bin/env bash
# Config 3 with the 16384x4096x14336 artifact gave wrong tokens on the first try. With minimal_pdi, K=4096 work shares the PDI
# of the 8192x4096x4096 family, so this tries gate and up with the 8192x4096x14336 family's artifact of the same shape.
# bg inflight2-try3 -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/inflight2-try3/cmd.sh'
set -o pipefail
source /notebooks/hm-fix/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/inflight2-prep
python3 tools/split_model/make_gateup_config.py --fw-path hw_bins/npu2/8192x4096x14336/bf16_int4AWQ_bf16_M/ --out "$OUT/gateup-8192family.config.json"
LD_PRELOAD=$COUNT_IOCTLS python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$OUT/try-gateup-8192family" \
  --methods heteromosaic_gateup --gateup-config "$OUT/gateup-8192family.config.json" --prompt-tokens 1024 --rounds 1 --reps 1 \
  2>&1 | tee "$OUT/try-gateup-8192family.log" | cut -c1-200 | tail -1
grep -hE 'WARMUP|REP |IOCTL_COUNTS|DONE|rror|Traceback' "$OUT"/try-gateup-8192family/*.log | cut -c1-220 | head -20
python3 - "$OUT/try-gateup-8192family/heteromosaic_gateup-1024-r0.result.json" <<'P'
import json, sys
r = json.load(open(sys.argv[1]))
print("first generated", r["reps"][0]["generated_ids"][:16], "matches_reference", r["matches_reference"])
P
step "done"
