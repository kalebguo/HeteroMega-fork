#!/usr/bin/env bash
# Build branch merged-ffn-fix in the separate tree /notebooks/hm-fix (hm-eval-main/build stays as evidence of the
# first evaluation). Flags mirror jobs/build-main/cmd.sh. Upload this file, then:
# bg build-fix -- 'bash build.sh'. Rerun after pushing C++ changes: it fast-forwards and rebuilds in place.
# SRC=<tree> NOOPT=1 builds a separate attribution tree with the -O3 line of the merged FFN target removed.
set -eo pipefail
export SRC=${SRC:-/notebooks/hm-fix}
if [ ! -d "$SRC/.git" ]; then
    git clone --reference /notebooks/hm-eval-main --branch merged-ffn-fix \
        https://github.com/kalebguo/HeteroMega-fork.git "$SRC" 2>&1 | tail -2
else
    git -C "$SRC" checkout -q -- src/unified_llm_w4a16_hetero/CMakeLists.txt
    git -C "$SRC" pull --ff-only 2>&1 | tail -2
fi
if [ -n "${NOOPT:-}" ]; then
    sed -i '/target_compile_options(heteromega_merged_ffn PRIVATE/d' "$SRC/src/unified_llm_w4a16_hetero/CMakeLists.txt"
fi
source "$SRC/tools/aup-kit/jobs/fix-env.sh"
unset PYTHONPATH
mkdir -p deployment-results

step "shared merged weights (read-only use; same 5.4 GB file as the first evaluation)"
W=py/unified_llm_w4a16/model_weights
[ -e "$W" ] || ln -s /notebooks/hm-eval-main/$W $W
ls -lL $W | tail -1

python - <<'PY'
import torch
print('Torch:', torch.__version__, 'HIP:', torch.version.hip)
assert torch.version.hip and torch.cuda.is_available(), 'ROCm Torch/GPU unavailable'
print('GPU:', torch.cuda.get_device_name(0))
PY
gpu_arch=$(python -c 'import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])')
hip_compiler="$ROCM_PATH/lib/llvm/bin/clang++"
if [ ! -x "$hip_compiler" ]; then hip_compiler="$ROCM_PATH/llvm/bin/clang++"; fi

step "cmake configure (HETEROMEGA_MERGED_FFN=ON)"
cmake -S . -B build \
  -DROCM_PATH="$ROCM_PATH" \
  -DCMAKE_HIP_COMPILER="$hip_compiler" \
  -DCMAKE_HIP_ARCHITECTURES="$gpu_arch" \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
  -DPython_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_CXX_FLAGS="-isystem $ROCM_PATH/include" \
  -DCMAKE_HIP_FLAGS="-isystem $ROCM_PATH/include --hip-path=$ROCM_PATH --rocm-path=$ROCM_PATH" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib" \
  -DCMAKE_MODULE_LINKER_FLAGS="-L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib" \
  -DHETEROMEGA_MERGED_FFN=ON \
  -DHETEROMEGA_TORCH_COMPAT_HEADER="$PWD/docs/runs/2026-10-12-aup/torch211-hip-compat.hpp" \
  2>&1 | tee deployment-results/configure.log | tail -3

step "build heteromega_ffn_deploy"
cmake --build build -j2 --target heteromega_ffn_deploy 2>&1 | tee deployment-results/build-ffn.log | grep -E 'error|Built target|Linking' | tail -12
step "build unified_llm_w4a16_hetero_libtorch"
cmake --build build -j2 --target unified_llm_w4a16_hetero_libtorch 2>&1 | tee deployment-results/build-model.log | grep -E 'error|Built target|Linking' | tail -12
if readelf -d build/py/unified_llm_w4a16/*.so | grep -q 'libamdhip64.so.5'; then
    echo "error: the extension links HIP 5.7 (libamdhip64.so.5)" >&2
    exit 1
fi
echo "HIP link check passed"

step "optimization flags of the merged FFN file (expect -O3, or nothing with NOOPT)"
grep -h CXX_FLAGS build/src/unified_llm_w4a16_hetero/CMakeFiles/heteromega_merged_ffn.dir/flags.make | grep -o ' -O[0-9s]\b' | sort -u || true

step "package the retained NPU artifact -> deployment.json"
export PYTHONPATH="$PWD/build/py:$PWD/build/py/unified_llm_w4a16:$PWD"
python -m experiments.ffn_deploy.prepare \
  --artifact-dir "$PWD/docs/runs/2026-10-31-ffn-deploy/deployment-results/artifact" \
  --output deployment.json 2>&1 | tail -3
ls -l deployment.json
ls hw_bins/npu2 | wc -l

step "ioctl counter (docs/ffn-deployment.md)"
if [ ! -e "$COUNT_IOCTLS" ]; then
    gcc -shared -fPIC -I"$HOME_LIBS/xdna-driver-05-19-25/src/include/uapi/drm_local" -I/usr/include/libdrm \
      docs/runs/2026-10-31-ffn-deploy/deployment-results/count_ioctls.c -ldl -o "$COUNT_IOCTLS"
fi
ls -l "$COUNT_IOCTLS"

step "CPU only: cost of the scan variants at the target's flags"
g++ -std=c++17 -O3 -Iinclude tools/aup-kit/jobs/build-fix/scan_bench.cpp -o /tmp/hm_scan_bench
/tmp/hm_scan_bench
step "done"
