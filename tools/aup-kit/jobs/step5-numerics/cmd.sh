#!/usr/bin/env bash
# Step 5 of merged-ffn-fix: numerics. The steps of tools/npu_error_probe/aup_job.sh, which hardcodes hm-eval-main, on this
# tree, then path_logits.py for the native paths that token_agreement.py cannot reach. Default full finite check.
# bg step5-numerics -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/step5-numerics/cmd.sh'
set -eo pipefail
source /notebooks/hm-fix/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/fix-step5-numerics
export DRIVER_PATH="$HOME_LIBS/xdna-driver-05-19-25" HETEROMEGA_PROBE_BUILD_ROOT=$RESULTS/probe-build
mkdir -p "$OUT/paths"
status=0
step "device before"; snapshot

step "probe sweep: K, sign, metadata (one fresh process per case)"
python3 tools/npu_error_probe/probe_sweep.py --out "$OUT" 2>&1 | cut -c1-200 | grep -v '^$' || status=$?
step "probe report"
python3 tools/npu_error_probe/probe_report.py "$OUT/probe.jsonl" --json "$OUT/verdicts.json" 2>&1 | cut -c1-200 || status=$?

step "token agreement, step-1 logits, perplexity (original vs merged FFN, one process)"
python3 tools/npu_error_probe/token_agreement.py --model "$MODEL" --build build --merged-ffn-config "$PWD/deployment.json" \
  --out "$OUT/token-agreement.json" 2>&1 | tail -40 | cut -c1-200 || status=$?

path() { # method prompt_tokens repeat
  name=$1-$2-r$3
  echo "--- $name"
  LD_PRELOAD=$COUNT_IOCTLS python3 tools/npu_error_probe/path_logits.py --method "$1" --model "$MODEL" --build build \
    --merged-ffn-config "$PWD/deployment.json" --prompt-tokens "$2" --out "$OUT/paths/$name.npz" > "$OUT/paths/$name.log" 2>&1 || status=$?
  grep -E 'IOCTL_COUNTS|^\{"method' "$OUT/paths/$name.log" | cut -c1-220 || true
}
step "one path per process, 1024 tokens"
for method in gpu_original merged_ffn heteromosaic merged_ffn_hetero; do path $method 1024 0; done
for method in merged_ffn_hetero heteromosaic merged_ffn gpu_original; do path $method 1024 1; done
step "one path per process, 512 tokens, next 512 scored"
for method in gpu_original merged_ffn heteromosaic merged_ffn_hetero; do path $method 512 0; done
step "comparison"
python3 tools/npu_error_probe/path_logits.py --compare "$OUT/paths" --json "$OUT/path-comparison.json" 2>&1 | cut -c1-220 || status=$?

step "device after"; snapshot
step "done"
exit $status
