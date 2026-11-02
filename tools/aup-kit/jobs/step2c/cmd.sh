#!/usr/bin/env bash
# Attribution of the first evaluation's cost: the no-O tree of step2b with the original per-element scan loop restored, timers
# on, full check. Expect about 33 ms per smoke call and about 2559 ms per 1024-token prefill if nothing else was at play.
# bg step2c -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/step2c/cmd.sh'
set -eo pipefail
export SRC=/notebooks/hm-fix-noopt
source /notebooks/hm-fix/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/fix-step2c
mkdir -p "$OUT"
export HETEROMEGA_FFN_FINITE_CHECK=full
python3 - <<'PY'
p = 'src/unified_llm_w4a16_hetero/merged_ffn.cpp'
s = open(p).read()
old = '''    const auto first = find_nonfinite_bf16(static_cast<const uint16_t *>(host.data_ptr()), host.size(0), host.size(1), mode);
    TORCH_CHECK(first < 0, "merged FFN: nonfinite ", name, " at element ", first);
'''
new = '''    const auto *bits = static_cast<const uint16_t *>(host.data_ptr());
    for (int64_t i = 0; i < host.numel(); ++i)
        TORCH_CHECK((bits[i] & 0x7f80) != 0x7f80, "merged FFN: nonfinite ", name, " at element ", i);
'''
if old in s:
    open(p, 'w').write(s.replace(old, new))
print('loop restored:', new in open(p).read())
PY
step "rebuild"
cmake --build build -j2 --target heteromega_ffn_deploy 2>&1 | tee "$OUT/build-ffn.log" | grep -E 'error|Built target heteromega_merged' || true
cmake --build build -j2 --target unified_llm_w4a16_hetero_libtorch 2>&1 | tee "$OUT/build-model.log" | grep -E 'error|Built target unified_llm_w4a16_hetero_libtorch' || true
grep -h CXX_FLAGS build/src/unified_llm_w4a16_hetero/CMakeFiles/heteromega_merged_ffn.dir/flags.make | grep -c ' -O[0-9s]\b' || true
step "smoke"
HETEROMEGA_FFN_TIMERS=$OUT/smoke-timers python -m experiments.ffn_deploy.smoke --config deployment.json --iterations 3 \
  --output "$OUT/smoke.json" 2>&1 | tee "$OUT/smoke.log" | tail -3
python -m experiments.ffn_deploy.phase_report "$OUT/smoke-timers" --skip 4 --output "$OUT/smoke-phases.json" | tail -3
step "model, 1024 tokens"
HETEROMEGA_FFN_TIMERS=$OUT/model-timers python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build \
  --out "$OUT/model" --methods merged_ffn --merged-ffn-config "$PWD/deployment.json" --gpu-chunk-size 512 \
  --prompt-tokens 1024 --rounds 1 2>&1 | tee "$OUT/model.log" | cut -c1-160 | tail -1
python -m experiments.ffn_deploy.phase_report "$OUT/model-timers" --skip 64 --output "$OUT/model-phases.json" | tail -3
cut -d, -f1,2,5,18 "$OUT/model/summary.csv"
step "done"
