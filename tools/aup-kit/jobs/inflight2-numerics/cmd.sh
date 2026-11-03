#!/usr/bin/env bash
# Two-inflight comparison, numerics. The model-sampled prompts of step 5b, against gpu_original, one fresh process per method.
# bg inflight2-numerics -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/inflight2-numerics/cmd.sh'
set -eo pipefail
source /notebooks/hm-fix/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/inflight2-numerics
PREP=$RESULTS/inflight2-prep
PROMPTS=$RESULTS/fix-step5b-sampled
mkdir -p "$OUT"
status=0
step "device before"; snapshot
for seed in 1 2; do
  DIR=$OUT/seed$seed
  mkdir -p "$DIR"
  path() { # method repeat [extra arguments]
    name=$1-1024-r$2
    method=$1
    shift 2
    echo "--- $name"
    LD_PRELOAD=$COUNT_IOCTLS python3 tools/npu_error_probe/path_logits.py --method "$method" --model "$MODEL" --build build \
      --merged-ffn-config "$PWD/deployment.json" --ref "$PROMPTS/sampled-$seed.json" --prompt-tokens 1024 --out "$DIR/$name.npz" "$@" \
      > "$DIR/$name.log" 2>&1 || status=$?
    grep -E 'IOCTL_COUNTS' "$DIR/$name.log" | cut -c1-200 || true
    tail -1 "$DIR/$name.log" | grep -E 'rror|Traceback' | cut -c1-240 || true
  }
  step "seed $seed"
  path gpu_original 0
  path merged_ffn 0
  path merged_ffn_p8192 0 --merged-ffn-config "$PREP/deployment-p8192.json"
  path heteromosaic 0
  path heteromosaic_gateup 0 --config "$PREP/gateup-8192family.config.json"
  path heteromosaic_gateup16384 0 --config "$PREP/gateup.config.json"
  path gpu_original 1
  step "seed $seed: comparison"
  python3 tools/npu_error_probe/path_logits.py --compare "$DIR" --json "$OUT/comparison-$seed.json" 2>&1 | cut -c1-260 || status=$?
  # The per-position logits are 262 MB per run and only the comparison needs them.
  rm -f "${DIR:?}"/*.npz
done
step "device after"; snapshot
step "done status=$status"
exit $status
