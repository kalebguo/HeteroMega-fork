#!/usr/bin/env bash
# First paired prefill evaluation of the merged GPU+NPU FFN (Main C, GPU attention) against the original
# GPU-only path (Main A), with the evaluation runner from the cloud session's branch (eval-runner =
# fork/claude/exciting-faraday-95a0c1) on top of the build-main build. 8 processes: for each prompt length,
# order original/new/new/original, one model load and warmup each, 5 measured generate(1)/generate(64) pairs.
# Log: /notebooks/aup-runs/eval-main/run.log; results: /notebooks/hm-results/2026-11-02/eval-merged-gpu.
set -eo pipefail
step() { echo; echo "=== $(date '+%H:%M:%S') $*"; }

cd /notebooks/hm-eval-main
MODEL=/notebooks/hm-model/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
OUT=/notebooks/hm-results/2026-11-02/eval-merged-gpu
source /notebooks/heteromosaic-venv/bin/activate
unset HETEROMEGA_FFN_CONFIG HETEROMEGA_GPU_FFN HETEROMEGA_GPU_FFN_SPLIT
unset HSA_OVERRIDE_GFX_VERSION PYTHONPATH
export HOME_LIBS=/notebooks/hm-libs
export HETEROMOSAIC_ROOT="$PWD"
export ROCM_PATH=/opt/rocm-7.2.0
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/build/py:$PWD/build/py/unified_llm_w4a16:$PWD"
mkdir -p "$OUT"
echo "HEAD $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
echo "artifact $(python3 -c 'import json;print(json.load(open("deployment.json"))["artifact_dir"])')"

step "device snapshot before"
rocm-smi --showtemp --showpower --showclocks 2>/dev/null | grep -E 'Temperature|Power|sclk|mclk' | head -8 || true

step "evaluate gpu_original vs merged_ffn at 512 and 1024 tokens"
python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out "$OUT" \
  --methods gpu_original merged_ffn --merged-ffn-config "$PWD/deployment.json" \
  --gpu-chunk-size 512 --prompt-tokens 512 1024 \
  2>&1 | tee "$OUT/evaluate.log" | grep -vE '^\s*$' | tail -40

step "summarize"
python3 kernels/gpu/tools/evaluate_prefill.py --out "$OUT" --summarize 2>&1 | tail -30
echo "--- summary.csv"
cat "$OUT/summary.csv"

step "device snapshot after"
rocm-smi --showtemp --showpower 2>/dev/null | grep -E 'Temperature|Power' | head -4 || true

step "done"
