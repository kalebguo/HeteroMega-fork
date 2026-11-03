# C++ merged FFN deployment

**2026-10-31 status: cloud deployment checks complete under report policy.**
Validated from `codex/ffn-cpp-deploy-20261031`, based on `e6ca183c`; the
required sources and integration are now prepared for publication on `main`.
Compute sources match that tested kernel revision. Full ROCm/XRT executor/model builds,
isolated smoke (30 dispatches), GPU-attention M512/M1024 model checks (64/128
merged FFNs), and native-attention coexistence with one/two inflight slots
(128 merged plus640 native dispatches per job) complete. Strict GPU controls
and structural checks pass; numerical deviations remain. See the
[measured report and raw evidence](runs/2026-10-31-ffn-deploy/README.md).
This does not establish pretrained accuracy or speedup.

## Getting started from GitHub

Start with [the teammate testing guide](ffn-teammate-testing.md) for clone,
prebuilt artifact, model integration, evaluation boundaries and kernel edit
locations. This runbook supplies the complete validated cloud build commands.
Read `AGENTS.md` and the latest `docs/progress.md` entries before development.
The user authorizes publishing this integration to `origin/main` for teammates;
preserve baseline experiments and do not change the shared driver/kernel.

Cloud entry point: <https://dub.aupcloud.io/aipc-16/hub/spawn>. Credentials are
intentionally absent; use your signed-in session or obtain access separately.
The earlier browser/session blockers were resolved in the validation thread.
Archive upload was the historical validation transport; Git is now the normal
source handoff. Its old bundle metadata is retained as provenance, not as an
instruction to use a stale archive or absolute cloud config.

## Manual JupyterLab entry point

After signing in, use **Launcher -> Terminal** for all Bash commands below.
Use a fresh checkout and keep existing experiments intact:

```bash
git clone --branch main https://github.com/glassescrab/HeteroMega.git HeteroMega-evaluation
cd HeteroMega-evaluation
git rev-parse HEAD
```

Keep this terminal open so its environment variables persist. The environment,
model and baseline artifact paths below were rechecked during this validation.
The earlier October28 cached merged artifact was absent and was rebuilt.
Validated staging directory: `/notebooks/heteromega-deploy.eT0uOJ/HeteroMega-ffn-deploy`.
Stop at a failing step and retain its log before proceeding.

## Code and behavior

- `src/unified_llm_w4a16_hetero/merged_ffn.cpp` and
  `include/heteromega/merged_ffn.hpp`: C++ XRT host plus GPU/NPU coordinator.
- `include/heteromega/merged_ffn_contract.hpp`: portable manifest/config ABI
  validation. Binary SHA256 checks use OpenSSL in the executor before BO setup.
- `UnifiedLLMW4A16Impl::heteromega_merged_ffn` and its call immediately after
  FFN RMSNorm in `forward_llama3_with_scratch`: optional whole-FFN hook. The
  original per-operator path is the default and handles unsupported row counts
  and decode. Non-Llama models are rejected when this option is requested.
- `npu_execution_mutex()` in `src/npu_matmul_func/npu_execution.cpp`: shared
  submit/wait ownership for `npuMatmul_zero` and the XRT executor. It serializes
  this application's NPU work; it does not prove two runtime context types can
  coexist, reserve the cloud device or synchronize unrelated processes.
- `experiments/ffn_deploy/bindings.cpp`: test seam for the exact model executor.
  The Python benchmark/sweep framework is not called from model inference.

For the selected artifact M512/H4096/I14336/P6400, NPU owns channels0:6400 and
GPU owns6400:14336, for all512 rows. Both keep their channels through gate/up,
SiLU/multiply and down. The C++ boundary sums two BF16[M,H] outputs. The model
adds the residual once. It uses the teammate's existing three-launch GPU FFN
and one-launch NPU graph; NPU Z still travels through DRAM.

NPU arguments are `3, instructions, instruction_bytes, X, gate, up, down, Z, Y`.
Data BO group IDs are3..8. The pinned runtime's instruction length is **bytes**.
NPU weights are unchanged full4352-byte-tile buffers. GPU gathers only its
suffix using `slice_packed_ffn`. Each model layer gets its own cached NPU
weight BOs and GPU slices, prepared on first use, outside subsequent warm calls.
Budget the added memory: for this shape three NPU weight BOs total about89.25
MiB/layer, about2.79GiB for32 layers, plus GPU suffix copies and scratch.

Each HIP stream has independent GPU and pinned-host scratch. Input readiness
is recorded before GPU FFN submission; a transfer stream stages input while
GPU FFN runs. NPU BO write/launch/wait/readback shares the native execution
lock. Output staging signals the caller stream, which sums partials. No
device-wide synchronization is inserted into an ordinary call; first-use
weight setup and per-slot recycling do wait. C++ host threads block on their
NPU call, as in the initial prototype. This is host staging, not zero-copy.

Launch failures poison the executor, and a started failed/timed-out run is
aborted before releasing NPU ownership. Structural/nonfinite failures remain
errors. Existing BFP16 numerical deviations require explicit report policy;
the wrapper itself does not establish numerical equivalence.

Weight loading/device movement resets cached FFN resources. Do not reload
weights or move a model while inference is running. Keep the deployment config
fixed for the model's lifetime. Partials returned by the test module borrow
per-stream scratch; consume them before another call on that stream.

## Cloud build (validated on AUP, October31)

Use a separate cloud checkout containing the published integration. Do not overwrite a teammate's active checkout. Preserve
the existing IRON and ROCm Python environments and host driver/kernel.

Requires the existing ROCm Torch, pybind11, CMake, matching XRT headers/runtime,
OpenSSL development headers and the baseline driver headers under HOME_LIBS.
The original build instructions and compatibility notes still apply.

```bash
source /notebooks/heteromosaic-venv/bin/activate
set -o pipefail
unset HETEROMEGA_FFN_CONFIG HETEROMEGA_GPU_FFN HETEROMEGA_GPU_FFN_SPLIT
unset HSA_OVERRIDE_GFX_VERSION PYTHONPATH
export HOME_LIBS=/notebooks/heteromosaic-libraries
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
  2>&1 | tee deployment-results/configure.log
cmake --build build -j2 --target heteromega_ffn_deploy \
  2>&1 | tee deployment-results/build-ffn.log
export PYTHONPATH="$PWD/build/py:$PWD/build/py/unified_llm_w4a16:$PWD:${PYTHONPATH:-}"
```

The ROCm include/library flags avoid mixing system HIP5.7 with ROCm7.2.
`CMAKE_MODULE_LINKER_FLAGS` is required for the model's Python MODULE target;
SHARED flags alone did not reach it and the first model run aborted while
loading both HIP versions. The test module uses pybind11 `NO_EXTRAS`: GCC slim
LTO objects were incompatible with the HIP linker and initially omitted its
Python initializer despite a successful link. Both failures and corrected
builds are retained in the run evidence. If a dependency is missing, retain
the configure/build log rather than changing system packages blindly.

Both new deployment and historical GPU-only model opt-ins default OFF at build
time. The merged option builds the reusable GPU library, but does not activate
the historical GPU-only shortcut. Do not set `HETEROMEGA_GPU_FFN=1` together
with `HETEROMEGA_FFN_CONFIG`.

## Artifact preparation

Use the exact matching artifact retained with the cloud evidence. Packaging
verifies file hashes without modifying binaries; it writes fresh absolute paths
for your checkout:

```bash
python -m experiments.ffn_deploy.prepare \
  --artifact-dir $PWD/docs/runs/2026-10-31-ffn-deploy/deployment-results/artifact \
  --output deployment.json
```

This directory contains `final.xclbin`, `insts.bin`, `final.xclbin.json` and
compiler evidence. It targets the documented HX370/MLIR-AIE/XRT stack. The old
October28 cached path was absent during validation and is not a prerequisite.
Do not use the retained cloud `deployment.json` without regenerating its paths.

Or explicitly build using the existing IRON environment. This is a build-time
operation; model inference does not call IRON or the compiler:

```bash
python -m experiments.ffn_deploy.prepare --build \
  --iron-python /home/mliraie/mlir-aie/ironenv/bin/python \
  --artifact-dir "$PWD/artifacts/deploy-m512-h4096-p6400" \
  --m 512 --hidden 4096 --intermediate 14336 --npu-channels 6400 \
  --output deployment.json
```

Verify the IRON path exists on the current cloud; the older cloud used another
path. Compilation logs go to the artifact directory. A smaller initial graph
can use M512/H512/I1024/P512 with the same selected flags. Rebuild and package
the full-size artifact before the model test.

The build helper preserves the supplied IRON Python executable's symlink path
so a virtualenv's `bin/python` is not accidentally replaced by system Python.

Config format (paths relative to the config file are supported):

```json
{
  "artifact_dir": "artifacts/deploy-m512-h4096-p6400",
  "accuracy_policy": "report",
  "timeout_ms": 10000,
  "device_index": 0
}
```

## Validation sequence (completed under report policy)

```bash
python -m experiments.ffn_deploy.smoke --config deployment.json \
  --iterations 3 --output deployment-results/cpp-smoke.json \
  2>&1 | tee deployment-results/cpp-smoke.log
```

This runs the actual C++ executor against quantized synthetic reference
weights: two layers, two streams, serial/concurrent branches, alternating/zero/
repeated inputs, output guards, malformed inputs and counted XRT completions.
GPU-partial checks are strict. NPU/combined results preserve strict numerical
verdicts under report policy; structural errors still stop the run. Timing is
diagnostic, including finite scans, not a reproduction of earlier benchmarks.
Optional `--compare-python` tears down the C++ executor, then requires exact
BF16-bit NPU parity against the current Python host contract. It requires the
existing working pyxrt environment; it does not compile another graph.

Next, with existing local weights and tokenizer (no downloads):

```bash
cmake --build build -j2 --target unified_llm_w4a16_hetero_libtorch \
  2>&1 | tee deployment-results/build-model.log
python -m experiments.ffn_deploy.model_smoke --config deployment.json \
  --model-dir /notebooks/heteromosaic-model --prompt-tokens 512 --inflight 1 \
  --output deployment-results/model-512.json \
  2>&1 | tee deployment-results/model-512.log
python -m experiments.ffn_deploy.model_smoke --config deployment.json \
  --model-dir /notebooks/heteromosaic-model --prompt-tokens 1024 --inflight 2 \
  --output deployment-results/model-1024-inflight2.json \
  2>&1 | tee deployment-results/model-1024-inflight2.log
```

These first use GPU attention to isolate the new FFN bridge. They require32
FFN completions per512-token prefill chunk and zero new completions during
single-token decode/M256 fallback. Original versus candidate logits are
reported, not promoted to an accuracy claim. Cold latency includes weight BO/
GPU-slice initialization; warm measurements reuse them. The config copies
are written beside results; shipped configurations are unchanged.

Only after those pass, validate coexistence with original NPU attention, first
inflight1 then2. The checkout needs the original matching `hw_bins` artifacts.
The model smoke now requires `root/hw_bins/npu2` and explicitly sets
`HETEROMOSAIC_ROOT` to its source root. Without this, the native loader looked
in the original author's checkout, failed imports and silently used GPU
fallback; that initial run is retained as invalid coexistence evidence.

For a fresh staging tree, reuse the preserved baseline artifacts and the
retained observational ioctl counter:

```bash
if [ ! -e hw_bins ]; then ln -s /notebooks/HeteroMosaic-baseline/hw_bins hw_bins; fi
cp docs/runs/2026-10-31-ffn-deploy/deployment-results/count_ioctls.c deployment-results/count_ioctls.c
gcc -shared -fPIC \
  -I"$HOME_LIBS/xdna-driver-05-19-25/src/include/uapi/drm_local" \
  -I/usr/include/libdrm deployment-results/count_ioctls.c -ldl \
  -o deployment-results/count_ioctls.so
env LD_PRELOAD="$PWD/deployment-results/count_ioctls.so" \
  python -m experiments.ffn_deploy.model_smoke --config deployment.json \
  --model-dir /notebooks/heteromosaic-model --base-mode hetero \
  --prompt-tokens 1024 --inflight 1 \
  --output deployment-results/model-hetero-1024-inflight1.json \
  2>&1 | tee deployment-results/model-hetero-1024-inflight1.log
env LD_PRELOAD="$PWD/deployment-results/count_ioctls.so" \
  python -m experiments.ffn_deploy.model_smoke --config deployment.json \
  --model-dir /notebooks/heteromosaic-model --base-mode hetero \
  --prompt-tokens 1024 --inflight 2 \
  --output deployment-results/model-hetero-1024-inflight2.json \
  2>&1 | tee deployment-results/model-hetero-1024-inflight2.log
```

Check both the model report and native evidence. Each measured job had128
merged completions,640 native calls,768 successful submissions/timeline waits,
zero submission/timeline failures, three contexts, zero import failures, and
the expected pipeline setting. The640 unsupported legacy waits used the
documented timeline fallback. Native down GEMV's historical5120>4096 config
fallback remains unchanged. A model report alone does not prove native NPU
execution. Preserve original/native and GPU-only controls before performance
comparisons.

With the default three smoke iterations, the isolated test expects30 successful
NPU dispatches. Model tests expect64 completions for two M512 prefill passes,
or128 for two1024-token prefill passes split into512-token chunks. Decode and
M256 fallback add no merged-FFN completions. Successful isolated status is
`passed` or `completed_with_accuracy_deviations`; the latter retains accepted
NPU numerical deviations, not structural failures. Model status is
`completed_with_report_policy`. None establishes pretrained accuracy.

Download logs and JSONs from `deployment-results` in the JupyterLab file browser.
For a single download, use `tar -czf deployment-results.tar.gz deployment-results`.
Keep original/native NPU attention coexistence as a later step after GPU-attention
model checks pass; it requires the separate original `hw_bins` artifacts.

For normal model use, set `HETEROMEGA_FFN_CONFIG=/absolute/path/deployment.json`
before inference and use the normal frontend. Check
`model.model.merged_ffn_completed_dispatches()`; a config flag alone does not
prove execution. Unset the variable to use the original FFN. Fixed-artifact M
must match actual chunk rows, not total prompt length.

Two opt-in runtime variables (read when the merged FFN is created; the startup log
line states both):

- `HETEROMEGA_FFN_FINITE_CHECK=full|sampled`. The executor checks X and Y_npu for
  nonfinite BF16 on the host. `full` (default) checks every element, about 2.0 ms per
  call at `-O3`. `sampled` checks every row at 1 in 64 columns, about 0.17 ms per call.
  An isolated nonfinite at an unsampled column of layer L reaches layer L+1's input
  across its whole row, because the Q/K/V and O projections mix columns (code-confirmed
  by reading; not injected). In the last layer, only the token checks would see it. The
  smoke's NaN-row control passes in both modes.
- `HETEROMEGA_FFN_TIMERS=<dir>` writes `merged_ffn_phases_<pid>.csv` per process, and
  `python -m experiments.ffn_deploy.phase_report <dir>` prints medians. Timers add host
  waits on the input, the H2D copy, the GPU branch and the sum: use them as diagnostics.

## Local checks

Portable C++ tests do not require Torch/ROCm/XRT:

```bash
cmake -S tests/ffn_deploy -B build/ffn-contract
cmake --build build/ffn-contract
ctest --test-dir build/ffn-contract --output-on-failure
python -m unittest experiments.ffn_deploy.test_prepare -v
```

In this Windows session the C++ test was compiled directly with an isolated
Zig0.13 C++ compiler in the outer workspace `.tools/`; it passed40 invalid-
contract rejections plus valid shape, BO-size, fallback and overflow checks.
The four Python packaging tests passed. An additional IRON symlink-preservation
regression test is skipped on this Windows host because creating a symlink
requires privileges. The later Linux cloud run passed all five packaging tests,
including that regression, and the portable C++ CTest. Full HIP/XRT and model
checks are recorded separately above; portable tests cannot substitute for them.
