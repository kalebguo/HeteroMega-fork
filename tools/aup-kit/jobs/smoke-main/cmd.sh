#!/usr/bin/env bash
# After build-main succeeded: the isolated C++ smoke and the two GPU-attention model smokes from
# docs/ffn-teammate-testing.md sections 4 and 5, on this account's model copy. Diagnostic timing only;
# the paired generate(1) timing comes later from the evaluation runner. Log: /notebooks/aup-runs/smoke-main/run.log.
set -eo pipefail
step() { echo; echo "=== $(date '+%H:%M:%S') $*"; }

cd /notebooks/hm-eval-main
MODEL=/notebooks/hm-model/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
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
echo "HEAD $(git rev-parse --short HEAD)  config $(ls -l deployment.json | cut -c1-60)"
[ -d "$MODEL" ] || { echo "model dir missing: $MODEL"; exit 3; }

step "isolated C++ smoke: expect 30 completed dispatches"
python -m experiments.ffn_deploy.smoke --config deployment.json \
  --iterations 3 --output deployment-results/cpp-smoke.json \
  2>&1 | tee deployment-results/cpp-smoke.log | tail -15

step "model smoke 512 tokens, inflight 1: expect 64 merged completions"
python -m experiments.ffn_deploy.model_smoke --config deployment.json \
  --model-dir "$MODEL" --prompt-tokens 512 --inflight 1 \
  --output deployment-results/model-512.json \
  2>&1 | tee deployment-results/model-512.log | tail -25

step "model smoke 1024 tokens, inflight 2: expect 128 merged completions"
python -m experiments.ffn_deploy.model_smoke --config deployment.json \
  --model-dir "$MODEL" --prompt-tokens 1024 --inflight 2 \
  --output deployment-results/model-1024-inflight2.json \
  2>&1 | tee deployment-results/model-1024-inflight2.log | tail -25

step "summary of the result JSONs"
python - <<'PY'
import json
for f in ("cpp-smoke", "model-512", "model-1024-inflight2"):
    try:
        d = json.load(open(f"deployment-results/{f}.json"))
    except Exception as e:
        print(f, "->", e); continue
    print(f"--- {f}: keys {sorted(d)[:20]}")
    for k, v in d.items():
        s = json.dumps(v)
        if any(w in k.lower() for w in ("status", "dispatch", "complet", "ms", "time", "latency", "agree", "max_abs", "error")):
            print(f"  {k}: {s[:160]}")
PY

step "done"
