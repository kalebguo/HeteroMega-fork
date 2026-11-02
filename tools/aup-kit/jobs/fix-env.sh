# Environment of every merged-ffn-fix job on the AUP box (same values as jobs/eval-main/cmd.sh), sourced by the jobs
# in this directory. Sets cwd to the build tree /notebooks/hm-fix; hm-eval-main stays untouched as earlier evidence.
step() { echo; echo "=== $(date '+%H:%M:%S') UTC $*"; }
SRC=/notebooks/hm-fix
MODEL=/notebooks/hm-model/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
RESULTS=/notebooks/hm-results/2026-11-02
COUNT_IOCTLS=/notebooks/hm-results/count_ioctls.so
cd "$SRC"
source /notebooks/heteromosaic-venv/bin/activate
unset HETEROMEGA_FFN_CONFIG HETEROMEGA_GPU_FFN HETEROMEGA_GPU_FFN_SPLIT HETEROMEGA_FFN_FINITE_CHECK HETEROMEGA_FFN_TIMERS
unset HSA_OVERRIDE_GFX_VERSION PYTHONPATH
export HOME_LIBS=/notebooks/hm-libs
export HETEROMOSAIC_ROOT="$PWD"
export ROCM_PATH=/opt/rocm-7.2.0
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2
export PYTHONPATH="$PWD/build/py:$PWD/build/py/unified_llm_w4a16:$PWD"
echo "HEAD $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
snapshot() { rocm-smi --showtemp --showpower --showclocks 2>/dev/null | grep -E 'Temperature|Power|sclk|mclk' | head -8 || true; }
