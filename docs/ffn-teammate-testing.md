# GPU + NPU FFN: teammate testing guide

This is the entry point for evaluating the new kernels **inside HeteroMosaic**.
The optional C++ executor is integrated into Llama's FFN path. It calls the GPU
kernel and the single-dispatch NPU graph directly; the model does not call the
Python partition benchmark framework.

**Validated on AUP, 2026-10-31:** ROCm 7.2 / Torch 2.11 / Radeon 890M
`gfx1150` / HX370. Isolated and real-weight model deployment checks complete,
including native NPU attention with one/two inflight chunks. Strict GPU and
execution checks pass. NPU/combined numerical deviations are retained under
report policy. This establishes neither pretrained accuracy nor a model speedup.
See [the measured evidence](runs/2026-10-31-ffn-deploy/README.md).

## 1. Get the implementation

On the Linux GPU/NPU machine, use a fresh checkout so an existing experiment
is preserved. In JupyterLab, run these Bash commands in **Launcher -> Terminal**:

```bash
git clone --branch main https://github.com/glassescrab/HeteroMega.git HeteroMega-evaluation
cd HeteroMega-evaluation
git rev-parse HEAD
```

For an existing clean checkout, use `git pull --ff-only origin main` instead.
Record the resulting commit with your measurements. Model weights and Python
environments are not shipped. Use existing cached AWQ weights and tokenizer;
the validated cloud directory is `/notebooks/heteromosaic-model`.

## 2. Build the host integration and GPU code

Follow the complete [cloud build commands](ffn-deployment.md#cloud-build-validated-on-aup-october31)
in the deployment runbook, from the repository root. They activate
`/notebooks/heteromosaic-venv`, set the ROCm/header/runtime paths and run CMake.
The important settings and targets are:

```bash
# Included in the full CMake command in the runbook:
# -DHETEROMEGA_MERGED_FFN=ON
# -DCMAKE_MODULE_LINKER_FLAGS="-L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib"
# -DHETEROMEGA_TORCH_COMPAT_HEADER="$PWD/docs/runs/2026-10-12-aup/torch211-hip-compat.hpp"

# Run only after that configure command succeeds:
cmake --build build -j2 --target heteromega_ffn_deploy unified_llm_w4a16_hetero_libtorch \
  2>&1 | tee deployment-results/build-all.log
export PYTHONPATH="$PWD/build/py:$PWD/build/py/unified_llm_w4a16:$PWD:${PYTHONPATH:-}"
```

Keep `HETEROMEGA_GPU_FFN` and `HETEROMEGA_GPU_FFN_SPLIT` unset for the combined
path. `HETEROMEGA_MERGED_FFN` builds the GPU library as a dependency, so the
separate historical GPU-only shortcut is unnecessary. To run the older
`gpu_ffn_all` control, also build with `-DHETEROMEGA_GPU_FFN=ON` and enable
its runtime switch only in that separate control job. All opt-ins default OFF.
The MODULE linker flag prevents loading system HIP 5 alongside ROCm 7.2.
The test binding disables pybind11's automatic LTO because the mixed HIP/GCC
linker previously produced a module without its Python initializer.

## 3. Select or rebuild the NPU program

The exact artifact used in the cloud check is included with the evidence. On
the same supported platform/runtime, package it without compiling anything:

```bash
python -m experiments.ffn_deploy.prepare \
  --artifact-dir "$PWD/docs/runs/2026-10-31-ffn-deploy/deployment-results/artifact" \
  --output deployment.json
```

This checks XCLBIN/instruction hashes. The C++ loader additionally validates
the ABI, shape, layouts and selected graph flags. Do not use the retained
`deployment-results/deployment.json` directly: it contains the old cloud path.

To rebuild after editing the NPU graph/microkernel, use the existing pinned
IRON environment (MLIR-AIE `7e00b57955e108fe9d8e9419f5828a0c7e650858`):

```bash
python -m experiments.ffn_deploy.prepare --build \
  --iron-python /home/mliraie/mlir-aie/ironenv/bin/python \
  --artifact-dir "$PWD/artifacts/evaluation-m512-h4096-p6400" \
  --m 512 --hidden 4096 --intermediate 14336 --npu-channels 6400 \
  --output deployment.json
```

Verify that Python path exists; compilation logs go to the artifact directory.
Use a new directory per variant. A different platform/toolchain may require
porting; these binaries do not establish support for another NPU generation.

## 4. Run the isolated C++ integration check

```bash
python -m experiments.ffn_deploy.smoke --config deployment.json \
  --iterations 3 --output deployment-results/cpp-smoke.json \
  2>&1 | tee deployment-results/cpp-smoke.log
```

Expect **30 completed dispatches**, strict GPU controls, buffer guards,
zero/repeated/alternating input checks, two layers/streams and concurrent calls.
Successful status is `passed` or `completed_with_accuracy_deviations`.
Report policy preserves failed numerical tolerance verdicts; it does not
accept nonfinite/stale output, corrupted guards, launch errors or timeouts.
Optional `--compare-python` requires a compatible pyxrt environment and was
not part of the completed cloud checks. Initial C++ testing needs no pyxrt.

## 5. Run the model through both kernels

These scripts load existing real weights, compare original/candidate logits,
run cold/warm candidate prefills, and verify decode/unsupported-shape fallback:

```bash
python -m experiments.ffn_deploy.model_smoke --config deployment.json \
  --model-dir /notebooks/heteromosaic-model --prompt-tokens 512 --inflight 1 \
  --output deployment-results/model-512.json \
  2>&1 | tee deployment-results/model-512.log

python -m experiments.ffn_deploy.model_smoke --config deployment.json \
  --model-dir /notebooks/heteromosaic-model --prompt-tokens 1024 --inflight 2 \
  --output deployment-results/model-1024-inflight2.json \
  2>&1 | tee deployment-results/model-1024-inflight2.log
```

Expect `completed_with_report_policy`, **64/128** merged completions respectively
across two prefill passes, finite logits and zero new merged completions during
decode/M256 fallback. The initial jobs use GPU attention to isolate the FFN.
For **original heterogeneous attention plus our FFN**, follow the
[native-attention commands and ioctl counter](ffn-deployment.md#validation-sequence-completed-under-report-policy):
use `--base-mode hetero --prompt-tokens 1024`, inflight1 then2, and original
`hw_bins/npu2` artifacts. A sparse checkout may need that tree restored or a
symlink to `/notebooks/HeteroMosaic-baseline/hw_bins`.

In the validated native jobs, each run had **128 merged + 640 native** calls,
768 successful submissions/timeline waits and zero submission/timeline failures.
The native loader can silently fall back to GPU: verify the actual ioctl counts
and import logs. A `hetero` configuration or successful model JSON is insufficient.

## 6. Use it from your own framework/evaluation driver

The runtime switch is `HETEROMEGA_FFN_CONFIG=/absolute/path/deployment.json`.
Set it before constructing/running the model; do not enable the historical
`HETEROMEGA_GPU_FFN=1` shortcut simultaneously. The model smoke controls this
variable internally, so no export is needed for step 5.

For your own driver using `LLaMA3W4A16Model`, the relevant calls are:

```python
import os
os.environ["HETEROMEGA_FFN_CONFIG"] = "/absolute/path/deployment.json"
# Construct LLaMA3W4A16Model with the same cached weights and a copied config.
# Use the *.model-config.json produced by step 5 as a tested configuration.
before = model.model.merged_ffn_completed_dispatches()
logits = model.model.forward(input_ids, 0)  # input_ids: contiguous GPU [1,512] or [1,1024]
after = model.model.merged_ffn_completed_dispatches()
assert after - before == 32 * (input_ids.shape[1] // 512)
```

For this artifact, **actual FFN chunk rows must be 512**. The M1024 test uses
two M512 chunks. Short prompts, a different chunk schedule, or decode can use
the original FFN fallback. Merely exporting the variable does not prove the
new kernels ran. Unset it for the original FFN control.

The selected split is NPU channels `[0,6400)`, GPU `[6400,14336)`, across all
512 rows. Gate/up ownership persists through SiLU/multiply and down; two
BF16 `[512,4096]` partial outputs are summed, then the model adds the residual
once. GPU uses three launches. NPU uses one launch with two phases and a DRAM
Z intermediate. This is host staging, not zero-copy or fully on-chip fusion.

## Evaluation boundaries and kernel edit locations

- GPU implementation: `kernels/gpu/src/gpu_ffn.cpp` and `src/hipkernels/`.
- NPU dataflow/DMA: `npu/ffn/w4a16_merged_upstream.py`; microkernel:
  `npu/ffn/w4a16_prefill_upstream.cc`; packed ABI: `npu/ffn/upstream_weights.py`.
- C++ host/bridge: `src/unified_llm_w4a16_hetero/merged_ffn.cpp`.
- Model hook: `heteromega_merged_ffn` and `forward_llama3_with_scratch` in
  `src/unified_llm_w4a16_hetero/unified_llm_w4a16.cpp`.
- Contract/partition guards: `include/heteromega/merged_ffn_contract.hpp`.

The original 4352-byte packed-weight tiles interleave quantization metadata
with W4 values. NPU DMA selects its prefix from full original buffers; GPU
`slice_packed_ffn` gathers its suffix. Preserve this ABI when changing kernels.
The NPU BOs add about 89.25 MiB/layer (2.79 GiB for 32 layers), plus GPU slices
and scratch. Changing the split requires a matching rebuilt NPU artifact and
fresh deployment config; the model lifetime's config stays fixed.

`kernels/gpu/tools/evaluate_prefill.py` now has `merged_ffn` (GPU attention) and
`merged_ffn_hetero` (original heterogeneous attention) methods. Pass
`--merged-ffn-config "$PWD/deployment.json"`; only those jobs receive
`HETEROMEGA_FFN_CONFIG`, and every other job has it removed. Each job records
`merged_ffn_completed_dispatches()` before/after every generate call (outside the
timer) and `summary.csv` reports the completion verdict; a wrong count suppresses
the speedup cell. `merged_ffn` needs `--gpu-chunk-size 512`, applied to every
method. Commands and rules: [evaluation](evaluation.md). This runner wiring is
CPU-tested only and has not yet produced AUP measurements. Your own driver must
still **unset `HETEROMEGA_FFN_CONFIG` for every original baseline** so inherited
state cannot activate the candidate in a control job.

Keep weights, exact token IDs, chunk schedule, inflight slots and attention
placement fixed for A/B comparisons. Exclude first-use allocation/weight upload,
synchronize the GPU at the timer boundaries, warm every method, alternate
process order and collect repeated samples. Record commit, artifact hashes,
dispatch evidence, error metrics and device/power conditions. The model smoke
has only one cold/warm sample and extra validation work; its timing is diagnostic.
Do not compare it directly with the existing `generate(1)` evaluation timer or
the earlier isolated FFN benchmark. No new end-to-end speedup is asserted here.

Save logs/JSONs and download with:

```bash
tar -czf deployment-results.tar.gz deployment-results
```
