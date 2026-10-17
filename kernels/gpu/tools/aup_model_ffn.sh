#!/usr/bin/env bash
# End-to-end Llama 3.1 8B runs with and without the HeteroMega GPU FFN, on the AUP cloud image
# (Ryzen AI 9 HX 370 / gfx1150, ROCm 7.2.0, a Torch 2.11 ROCm venv). Run from the repo root:
#   kernels/gpu/tools/aup_model_ffn.sh fetch   model shards (about 5.4 GB) and the pinned xdna-driver headers
#   kernels/gpu/tools/aup_model_ffn.sh build   root project with -DHETEROMEGA_GPU_FFN=ON into build-model/
#   kernels/gpu/tools/aup_model_ffn.sh run     4 processes, HETEROMEGA_GPU_FFN = 0, 1, 1, 0 (run_model_ffn.py)
# The build applies docs/runs/2026-10-12-aup/cloud-build.patch (ROCm 7.2 paths, Torch 2.11 HIP type names) to the
# working tree. Revert it with: git checkout -- CMakeLists.txt src/unified_llm_w4a16_hetero/CMakeLists.txt
# Environment: OUT (results), MODEL_DIR, HOME_LIBS (xdna-driver parent), JOBS, REPS, NEW_TOKENS.
#   DUMMY=1               run with the backend's seeded random weights: no model download, timing only
#   SYSTEM_XDNA_HEADER=1  build against the installed XDNA 2.26 UAPI header instead of the pinned xdna-driver clone.
#                         Only for GPU-only runs (heterogeneity=gpu); the NPU path needs the pinned headers.
set -euo pipefail

OUT=${OUT:-/notebooks/hm-results/g6}
MODEL_DIR=${MODEL_DIR:-/notebooks/hm-model/Meta-Llama-3.1-8B-Instruct-AWQ-INT4}
export HOME_LIBS=${HOME_LIBS:-/notebooks/hm-libs}
MODEL_REV=db1f81ad4b8c7e39777509fac66c652eb0a52f91
XDNA_REV=0e6d303b2cc2b3fe1cf10aba0acbf57a422588fb
RUNS=docs/runs/2026-10-12-aup
ROCM=/opt/rocm-7.2.0
mkdir -p "$OUT"

fetch() {
    mkdir -p "$MODEL_DIR"
    local base=https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/resolve/$MODEL_REV
    curl -fsSL -o "$MODEL_DIR/model.safetensors.index.json" "$base/model.safetensors.index.json"
    local shards
    shards=$(python3 -c 'import json, sys; print(*sorted(set(json.load(open(sys.argv[1]))["weight_map"].values())))' \
        "$MODEL_DIR/model.safetensors.index.json")
    for f in $shards; do
        if [ ! -f "$MODEL_DIR/$f" ]; then
            curl -fL --retry 3 -C - -o "$MODEL_DIR/$f.part" "$base/$f"
            mv "$MODEL_DIR/$f.part" "$MODEL_DIR/$f"
        fi
    done
    (cd "$MODEL_DIR" && sha256sum $shards) | tee "$OUT/model-sha256.txt"

    local xdna=$HOME_LIBS/xdna-driver-05-19-25
    if [ ! -d "$xdna/.git" ]; then
        git clone --filter=blob:none --no-checkout https://github.com/amd/xdna-driver.git "$xdna"
        git -C "$xdna" sparse-checkout set src/include
        git -C "$xdna" checkout "$XDNA_REV"
    fi
    git -C "$xdna" --no-pager log -1 --format='xdna-driver %H' | tee "$OUT/xdna-driver.txt"
}

build() {
    git sparse-checkout list | grep -qx py || git sparse-checkout add py
    local uapi=$HOME_LIBS/xdna-driver-05-19-25/src/include/uapi/drm_local
    if [ "${SYSTEM_XDNA_HEADER:-0}" = 1 ] && [ ! -e "$uapi/amdxdna_accel.h" ]; then
        mkdir -p "$uapi"
        ln -s /usr/src/xrt-amdxdna-2.26.0/include/uapi/drm/amdxdna_accel.h "$uapi/amdxdna_accel.h"
    fi
    git --no-pager show "HEAD:$RUNS/torch211-hip-compat.hpp" > "$OUT/torch211-hip-compat.hpp"
    if git diff --quiet -- CMakeLists.txt src/unified_llm_w4a16_hetero/CMakeLists.txt; then
        git --no-pager show "HEAD:$RUNS/cloud-build.patch" | sed "s#/notebooks/heteromega-baseline-logs/#$OUT/#" | git apply
    fi
    unset PYTHONPATH HSA_OVERRIDE_GFX_VERSION
    export ROCM_PATH=$ROCM LD_LIBRARY_PATH=$ROCM/lib:/opt/xilinx/xrt/lib
    cmake -S . -B build-model -DHETEROMEGA_GPU_FFN=ON \
        -DCMAKE_HIP_ARCHITECTURES=gfx1150 \
        -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++ \
        -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
        -DCMAKE_PREFIX_PATH=$ROCM \
        "-DCMAKE_CXX_FLAGS=-isystem $ROCM/include" \
        "-DCMAKE_HIP_FLAGS=-isystem $ROCM/include --hip-path=$ROCM --rocm-path=$ROCM" \
        "-DCMAKE_EXE_LINKER_FLAGS=-L$ROCM/lib -Wl,-rpath,$ROCM/lib" \
        "-DCMAKE_SHARED_LINKER_FLAGS=-L$ROCM/lib -Wl,-rpath,$ROCM/lib" 2>&1 | tee "$OUT/configure.log"
    cmake --build build-model -j"${JOBS:-4}" --target unified_llm_w4a16_hetero_libtorch 2>&1 | tee "$OUT/build.log"
}

run() {
    unset PYTHONPATH HSA_OVERRIDE_GFX_VERSION
    export ROCM_PATH=$ROCM LD_LIBRARY_PATH=$ROCM/lib:/opt/xilinx/xrt/lib TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
    git --no-pager show "HEAD:$RUNS/gpu-model-result.json" > "$OUT/reference.json"
    local i=0
    for ffn in 0 1 1 0; do
        i=$((i + 1))
        local name=$OUT/run$i-ffn$ffn
        local weights=(--model "$MODEL_DIR")
        [ "${DUMMY:-0}" = 1 ] && weights=(--dummy)
        HETEROMEGA_GPU_FFN=$ffn timeout 1200 python3 -u kernels/gpu/tools/run_model_ffn.py "${weights[@]}" \
            --ref "$OUT/reference.json" --out "$name.json" --reps "${REPS:-3}" --new-tokens "${NEW_TOKENS:-64}" \
            > "$name.log" 2>&1 || echo "run $i failed: see $name.log"
        grep -E "^(WARMUP|REP|DONE|stub)|HeteroMega GPU FFN|Error|Traceback" "$name.log" || true
    done
}

case "${1:-}" in
fetch | build | run) "$1" ;;
*)
    echo "usage: aup_model_ffn.sh fetch|build|run" >&2
    exit 2
    ;;
esac
