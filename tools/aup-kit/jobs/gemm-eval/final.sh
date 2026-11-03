#!/usr/bin/env bash
# The final code with every switch on: tests, timing of the three best cases, numerics, then where the time goes.
# bg gemm-final -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/final.sh'
set -uo pipefail
J=/notebooks/hm-gemm/tools/aup-kit/jobs
RESULTS=${RESULTS:-/notebooks/hm-results/2026-11-02}
ALL="HETEROMEGA_GEMM_UNCACHED=1 HETEROMEGA_SDPA=fast HETEROMEGA_SDPA_TO_O=1 HETEROMEGA_FFN_FUSE=1 HETEROMEGA_FFN_SLOT_BO=2 HETEROMEGA_RESIDUAL_NORM=1"
status=0
(cd /notebooks/hm-gemm && sha256sum include/heteromega/*.hpp include/unified_llm_w4a16_hetero/unified_llm_w4a16.hpp kernels/gpu/include/heteromega/*.hpp kernels/gpu/src/* src/unified_llm_w4a16_hetero/merged_ffn.cpp \
  src/unified_llm_w4a16_hetero/unified_llm_w4a16.cpp) > "$RESULTS/gemm-final-source-sha256.txt"
bash "$J/gemm-build/cmd.sh" test 2>&1 | grep -E 'ALL PASS|FAIL|rror:' || status=1
cases=()
for p in 8192 7168 6400; do cases+=("p$p-all:p$p:2:${ALL// /,}"); done
bash "$J/gemm-eval/passes.sh" gemm-final 3 "${cases[@]}" || status=1
# shellcheck disable=SC2086
bash "$J/gemm-eval/numerics2.sh" gemm-numerics3 HETEROMEGA_RESIDUAL_NORM=1 2>&1 | grep -E 'max_abs|IOCTL_COUNTS|fallback|done status' | cut -c1-130 || status=1
# shellcheck disable=SC2086
export $ALL
bash "$J/gemm-eval/stages.sh" gemm-stages4 p8192-i2:p8192:2 p7168-i2:p7168:2 p6400-i2:p6400:2 2>&1 | grep -E 'merged_ffn|done' | cut -c1-130 || status=1
echo "final status=$status"
exit $status
