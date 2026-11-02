#!/usr/bin/env bash
# Build the published GPU+NPU merged-FFN integration from glassescrab/HeteroMega main on the AUP box.
# Mirrors docs/ffn-deployment.md "Cloud build (validated on AUP, October31)", plus the pinned xdna-driver
# headers that CMakeLists.txt needs under HOME_LIBS (kernels/gpu/tools/aup_model_ffn.sh xdna). Runs detached:
# /notebooks/aup-runs/build-main/run.log, which ends with exit=N.
set -eo pipefail
step() { echo; echo "=== $(date '+%H:%M:%S') $*"; }

SRC=/notebooks/hm-eval-main
export HOME_LIBS=/notebooks/hm-libs
XDNA_REV=0e6d303b2cc2b3fe1cf10aba0acbf57a422588fb

step "clone team main"
if [ -e "$SRC" ]; then echo "$SRC already exists: not touching it"; exit 3; fi
git clone --branch main https://github.com/glassescrab/HeteroMega.git "$SRC" 2>&1 | tail -2
cd "$SRC"
echo "HEAD $(git rev-parse HEAD)"
du -sh hw_bins 2>/dev/null || echo "no hw_bins directory"

step "pinned xdna-driver UAPI headers"
X=$HOME_LIBS/xdna-driver-05-19-25
if [ ! -d "$X/.git" ]; then
    mkdir -p "$HOME_LIBS"
    git clone --filter=blob:none --no-checkout https://github.com/amd/xdna-driver.git "$X" 2>&1 | tail -1
    git -C "$X" sparse-checkout set src/include
    git -C "$X" checkout "$XDNA_REV" 2>&1 | tail -1
fi
git -C "$X" --no-pager log -1 --format='xdna-driver %H %cs'
ls -l "$X/src/include/uapi/drm_local/amdxdna_accel.h"

step "environment"
source /notebooks/heteromosaic-venv/bin/activate
unset HETEROMEGA_FFN_CONFIG HETEROMEGA_GPU_FFN HETEROMEGA_GPU_FFN_SPLIT
unset HSA_OVERRIDE_GFX_VERSION PYTHONPATH
export HETEROMOSAIC_ROOT="$PWD"
export ROCM_PATH=/opt/rocm-7.2.0
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2
mkdir -p deployment-results
python - <<'PY'
import torch, numpy, pybind11
print('Torch:', torch.__version__, 'HIP:', torch.version.hip)
assert torch.version.hip and torch.cuda.is_available(), 'ROCm Torch/GPU unavailable'
print('GPU:', torch.cuda.get_device_name(0))
PY
gpu_arch=$(python -c 'import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])')
hip_compiler="$ROCM_PATH/lib/llvm/bin/clang++"
if [ ! -x "$hip_compiler" ]; then hip_compiler="$ROCM_PATH/llvm/bin/clang++"; fi
echo "gpu_arch=$gpu_arch hip_compiler=$hip_compiler"

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
  2>&1 | tee deployment-results/configure.log | tail -5

step "build heteromega_ffn_deploy"
cmake --build build -j2 --target heteromega_ffn_deploy 2>&1 | tee deployment-results/build-ffn.log | grep -E 'error|warning: unused|Built target|Linking' | tail -20

step "build unified_llm_w4a16_hetero_libtorch"
cmake --build build -j2 --target unified_llm_w4a16_hetero_libtorch 2>&1 | tee deployment-results/build-model.log | grep -E 'error|Built target|Linking' | tail -20
if readelf -d build/py/unified_llm_w4a16/*.so | grep -q 'libamdhip64.so.5'; then
    echo "error: the extension links HIP 5.7 (libamdhip64.so.5)" >&2
    exit 1
fi
echo "HIP link check passed"

step "package the retained NPU artifact -> deployment.json"
export PYTHONPATH="$PWD/build/py:$PWD/build/py/unified_llm_w4a16:$PWD:${PYTHONPATH:-}"
python -m experiments.ffn_deploy.prepare \
  --artifact-dir "$PWD/docs/runs/2026-10-31-ffn-deploy/deployment-results/artifact" \
  --output deployment.json 2>&1 | tail -5
ls -l deployment.json

step "done"
