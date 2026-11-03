#!/usr/bin/env bash
# Rebuild the model library of /notebooks/hm-gemm as it is, with the flags already configured. It keeps deployment.json as it is.
# bg gemm-model-build -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/build_model.sh'
set -uo pipefail
export SRC=/notebooks/hm-gemm
source /notebooks/hm-gemm/tools/aup-kit/jobs/fix-env.sh
unset PYTHONPATH
step "build unified_llm_w4a16_hetero_libtorch"
cmake --build build -j4 --target unified_llm_w4a16_hetero_libtorch 2>&1 | tee deployment-results/build-model.log | grep -E 'error|Error |Built target unified|Linking' | tail -12
status=${PIPESTATUS[0]}
step "done status=$status"
exit "$status"
