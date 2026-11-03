#!/usr/bin/env bash
# Two-inflight comparison, energy, apart from the timing job because the sampler could disturb latency. First what the reading
# covers, then energy per prefill. Installs nothing and needs no root.
# bg inflight2-energy -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/inflight2-energy/cmd.sh'
set -eo pipefail
source /notebooks/hm-fix/tools/aup-kit/jobs/fix-env.sh
OUT=$RESULTS/inflight2-energy
PREP=$RESULTS/inflight2-prep
mkdir -p "$OUT"
export HETEROMEGA_FFN_FINITE_CHECK=sampled
P=tools/power_probe/power_probe.py
step "device before"; snapshot
step "what the reading covers"
python3 $P coverage --out "$OUT/coverage.json" 2>&1 | tee "$OUT/coverage.log" | tail -60 | cut -c1-120
run() { # method inflight repeat [extra arguments]
    name=$1-i$2-r$3
    method=$1
    inflight=$2
    shift 3
    python3 $P prefill --method "$method" --inflight "$inflight" --model "$MODEL" --merged-ffn-config "$PWD/deployment.json" \
      --out "$OUT/runs/$name.json" "$@" > "$OUT/$name.log" 2>&1
    tail -1 "$OUT/$name.log" | cut -c1-300
}
GATEUP=(--config "$PREP/gateup-8192family.config.json")
P8192=(--merged-ffn-config "$PREP/deployment-p8192.json")
mkdir -p "$OUT/runs"
step "energy, two inflight chunks, A B C D D C B A"
run heteromosaic 2 0
run heteromosaic_gateup 2 0 "${GATEUP[@]}"
run merged_ffn 2 0
run merged_ffn_p8192 2 0 "${P8192[@]}"
run merged_ffn_p8192 2 1 "${P8192[@]}"
run merged_ffn 2 1
run heteromosaic_gateup 2 1 "${GATEUP[@]}"
run heteromosaic 2 1
step "energy, one inflight chunk"
run gpu_original 1 0
run heteromosaic 1 0
run heteromosaic_gateup 1 0 "${GATEUP[@]}"
run merged_ffn 1 0
run merged_ffn_p8192 1 0 "${P8192[@]}"
step "report"
python3 $P report "$OUT/runs" --json "$OUT/report.json" | cut -c1-140 | tail -120
step "device after"; snapshot
step "done"
