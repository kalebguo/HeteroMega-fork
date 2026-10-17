#!/usr/bin/env bash
# Runs the GPU measurements one after another and writes one CSV per run into <out_dir>.
# Nothing else should use the GPU (or heavy CPU) while it runs. Takes about 10 minutes on the HX 370.
#
# Usage: kernels/gpu/tools/run_gpu_suite.sh <out_dir> [build_dir (default build/gpu)]
#   gemv.csv              decode matmuls: gemv_w4a16 (each split_k), HeteroMosaic's GEMV, dense
#   slice*.csv            GPU slice latency; slice_prefix uses one packed copy of all channels (FfnSliceWeights::prefix)
#   bw_*.csv              GPU read bandwidth alone, CPU alone, and both at once (4, 8, 16 CPU threads)
#   ablation.csv          5 rounds of bench_ffn_ablation, variant order shuffled per round
#   ablation_summary.csv  tools/summarize_bench.py of ablation.csv
set -u
out=${1:?usage: run_gpu_suite.sh <out_dir> [build_dir]}
bin=${2:-build/gpu}/bin
tools=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"

run() {
    local name=$1
    shift
    echo "$(date +%T) $name: $*"
    "$@" > "$out/$name.csv" 2> "$out/$name.err"
    echo "$(date +%T) $name exit=$?"
}

run gemv "$bin/bench_gemv"
run slice "$bin/bench_gpu_ffn_slice" --m 1,512 --s 4096,10240,14336
run slice_prefix "$bin/bench_gpu_ffn_slice" --m 1,512 --s 4096,10240,14336 --prefix
run bw_gpu "$bin/bench_bandwidth" --seconds 8
for t in 4 8 16; do
    run "bw_cpu$t" "$bin/bench_bandwidth" --seconds 8 --no-gpu --cpu-threads "$t"
    run "bw_gpu_cpu$t" "$bin/bench_bandwidth" --seconds 8 --cpu-threads "$t"
done
run ablation "$bin/bench_ffn_ablation" --rounds 5
python3 "$tools/summarize_bench.py" "$out/ablation.csv" > "$out/ablation_summary.csv"
echo "$(date +%T) done"
