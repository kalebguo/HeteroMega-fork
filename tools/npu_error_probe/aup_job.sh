#!/usr/bin/env bash
# NPU error probes and real-weight token agreement on the AUP box. Run detached (one GPU/NPU job at a
# time), e.g. upload as /notebooks/aup-runs/npu-probe/cmd.sh and start it with the aup-kit `bg` launcher.
# Assumes the /notebooks/hm-eval-main checkout and build from tools/aup-kit/jobs/build-main (branch with
# this directory checked out) and its deployment.json. Writes $OUT; run.log ends with exit=N.
set -eo pipefail
step() { echo; echo "=== $(date -u '+%H:%M:%S') UTC $*"; }

cd /notebooks/hm-eval-main
OUT=${OUT:-/notebooks/hm-results/$(date -u +%F)/npu-error-probe}
MODEL=/notebooks/hm-model/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
source /notebooks/heteromosaic-venv/bin/activate
unset HETEROMEGA_FFN_CONFIG HETEROMEGA_GPU_FFN HETEROMEGA_GPU_FFN_SPLIT HSA_OVERRIDE_GFX_VERSION PYTHONPATH
export HOME_LIBS=/notebooks/hm-libs
export DRIVER_PATH="$HOME_LIBS/xdna-driver-05-19-25"   # pinned UAPI headers for the native binding
export HETEROMOSAIC_ROOT="$PWD"
export ROCM_PATH=/opt/rocm-7.2.0
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/build/py:$PWD/build/py/unified_llm_w4a16:$PWD"
mkdir -p "$OUT"
echo "HEAD $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
ls build/py/unified_llm_w4a16/unified_llm_w4a16_hetero_libtorch*.so
ls hw_bins/npu2/8192x2048x2048/bf16_int4AWQ_bf16_M/final_512x2048x2048_64x128x64_8c_bf16_int4AWQ_bf16.pdi

step "device snapshot before"
rocm-smi --showtemp --showpower 2>/dev/null | grep -E 'Temperature|Power' | head -4 || true

step "probe sweep: K, sign, metadata (one fresh process per case)"
status=0
python3 tools/npu_error_probe/probe_sweep.py --out "$OUT" || status=$?
echo "probe_sweep exit=$status"

step "probe report"
python3 tools/npu_error_probe/probe_report.py "$OUT/probe.jsonl" --json "$OUT/verdicts.json"

step "token agreement, step-1 logits, perplexity (original vs merged FFN, real weights)"
python3 tools/npu_error_probe/token_agreement.py --model "$MODEL" --build build \
  --merged-ffn-config "$PWD/deployment.json" --out "$OUT/token-agreement.json" 2>&1 | tail -40

step "device snapshot after"
rocm-smi --showtemp --showpower 2>/dev/null | grep -E 'Temperature|Power' | head -4 || true
step "done"
exit $status
