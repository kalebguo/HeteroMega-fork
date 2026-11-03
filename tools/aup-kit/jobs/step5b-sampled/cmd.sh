#!/usr/bin/env bash
# Step 5b of merged-ffn-fix: numerics on model-sampled text. The fixed reference prompt has 12 distinct tokens and a period
# of 10, so it barely tests numerics. Two seeds; every method scores 512 positions after a 1024-token prefill.
# bg step5b-sampled -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/step5b-sampled/cmd.sh'
set -eo pipefail
source /notebooks/hm-fix/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/fix-step5b-sampled
MDOWN=$RESULTS/fix-step4-mdown.config.json
mkdir -p "$OUT"
status=0
step "device before"; snapshot
for seed in 1 2; do
  DIR=$OUT/seed$seed
  mkdir -p "$DIR"
  step "seed $seed: sample prompt"
  python3 tools/npu_error_probe/path_logits.py --sample-prompt "$OUT/sampled-$seed.json" --seed "$seed" --model "$MODEL" --build build \
    > "$OUT/sample-$seed.log" 2>&1 || status=$?
  grep -E '^\{"tokens' "$OUT/sample-$seed.log" | cut -c1-200 || true
  path() { # method repeat [extra arguments]
    name=$1-1024-r$2
    method=$1
    shift 2
    echo "--- $name"
    LD_PRELOAD=$COUNT_IOCTLS python3 tools/npu_error_probe/path_logits.py --method "$method" --model "$MODEL" --build build \
      --merged-ffn-config "$PWD/deployment.json" --ref "$OUT/sampled-$seed.json" --prompt-tokens 1024 --out "$DIR/$name.npz" "$@" \
      > "$DIR/$name.log" 2>&1 || status=$?
    grep -E 'IOCTL_COUNTS' "$DIR/$name.log" | cut -c1-200 || true
  }
  path gpu_original 0
  path merged_ffn 0
  path heteromosaic 0
  path merged_ffn_hetero 0
  path heteromosaic_mdown 0 --config "$MDOWN"
  path gpu_original 1
  step "seed $seed: comparison"
  python3 tools/npu_error_probe/path_logits.py --compare "$DIR" --json "$OUT/comparison-$seed.json" 2>&1 | cut -c1-260 || status=$?
  # The per-position logits are 262 MB per run and only the comparison needs them.
  rm -f "$DIR"/*.npz
done
step "device after"; snapshot
step "done"
exit $status
