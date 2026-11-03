#!/usr/bin/env bash
# Input reuse and uncached weights at the three best splits, two inflight chunks. Three passes, so that drift of the box does not favor one case.
# bg gemm-ab3 -- 'bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/ab3.sh'
set -uo pipefail
bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/build_model.sh || exit 1
for pass in 1 2 3; do
    bash /notebooks/hm-gemm/tools/aup-kit/jobs/gemm-eval/ab.sh gemm-ab3/pass$pass 1 3 \
      p8192-off:p8192:2:HETEROMEGA_GEMM_REUSE_OFF=1 p8192-reuse:p8192:2 p8192-unc:p8192:2:HETEROMEGA_GEMM_UNCACHED=1 \
      p6400-off:p6400:2:HETEROMEGA_GEMM_REUSE_OFF=1 p6400-reuse:p6400:2 p6400-unc:p6400:2:HETEROMEGA_GEMM_UNCACHED=1 \
      p7168-off:p7168:2:HETEROMEGA_GEMM_REUSE_OFF=1 p7168-reuse:p7168:2 p7168-unc:p7168:2:HETEROMEGA_GEMM_UNCACHED=1 2>&1 | grep -E 'median|FAILED'
done
