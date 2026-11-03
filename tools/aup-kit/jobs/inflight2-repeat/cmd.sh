#!/usr/bin/env bash
# Two-inflight comparison, repeat of the P8192 against heteromosaic gap: A B B A A B B A A B B A, 6 processes per method,
# 5 samples per process, native calls counted, nothing else varied. Every process median is reported; none is dropped.
# bg inflight2-repeat -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/inflight2-repeat/cmd.sh'
set -eo pipefail
source /notebooks/hm-fix/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/inflight2-repeat
PREP=$RESULTS/inflight2-prep
mkdir -p "$OUT"
export HETEROMEGA_FFN_FINITE_CHECK=sampled
step "device before"; snapshot
step "A B B A A B B A A B B A"
LD_PRELOAD=$COUNT_IOCTLS python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$OUT/repeat" \
  --merged-ffn-config "$PWD/deployment.json" --p8192-ffn-config "$PREP/deployment-p8192.json" --gpu-chunk-size 512 \
  --methods heteromosaic merged_ffn_p8192 --inflight 2 --prompt-tokens 1024 --rounds 6 --reps 5 \
  2>&1 | tee "$OUT/repeat.log" | cut -c1-160 | tail -1
cut -d, -f1,2,3,4,5,6,9,10,11,18 "$OUT/repeat/summary.csv"
grep -H IOCTL_COUNTS "$OUT"/repeat/*.log | sed "s|$OUT/repeat/||" | cut -c1-150
python3 - "$OUT/repeat" <<'P'
import glob, json, os, statistics, sys
for path in sorted(glob.glob(sys.argv[1] + "/*.result.json"), key=os.path.getmtime):
    run = json.load(open(path))
    samples = [rep["prefill_s"] * 1000 for rep in run["reps"]]
    print(os.path.basename(path), f"median {statistics.median(samples):.1f}", "samples", [round(s, 1) for s in samples], "matches_reference", run["matches_reference"])
P
step "device after"; snapshot
step "done"
