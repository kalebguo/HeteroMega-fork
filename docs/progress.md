# HeteroMega progress

## 2026-10-17: GPU P0/P1 work (`kernels/gpu/`, branch `gpu-p0p1`)

All runs on the AUP cloud (HX 370, Radeon 890M `gfx1150`, ROCm 7.2, Torch 2.11).
Details and tables: [kernels/gpu/README.md](../kernels/gpu/README.md#results-2026-10-17-hx-370).

- **Executed, pass:** `test_gpu_ffn` quick and full, 0 failed. New checks:
  `slice_packed_ffn` byte-equal to `pack_ffn_slice` (20 cases), `GpuFfnModel`
  bit-equal to the prefix path, prefix GEMM/GEMV blocks vs fp32.
- **Repeated measurements (G1):** `bench_ffn_ablation` 5 rounds, shuffled order,
  GPU clock/power/temperature per row. Per-round ratio to HeteroMosaic's order:
  fused SiLU-and-multiply about −2 % at every M; merged gate/up GEMM 0 % at
  M ≥ 128; `gemv_w4a16` −13 to −14 % of the decode FFN. Round 0 of each run was
  up to 10 % slower (package 51 to 53 W, GPU about 2.3 GHz, for about 50 s).
- **Bandwidth (G2, CPU as the second agent):** GPU alone 95.9 GB/s; GPU + 8 CPU
  threads 67.6 + 27.1 GB/s. The sum stays at 93 to 97 GB/s. **Not measured with
  the NPU** (needs the NPU side).
- **One weight copy, two split points (G3):** `kInterleaved` gate/up layout and
  `FfnSliceWeights::prefix`; same latency as a separately packed slice.
- **GEMM (G4):** AUTOTUNE kept `generic` in all 24 FFN shapes, so it gives
  no gain. `generic` is compute-bound at 9.1 to 9.5 TFLOPS from M = 128. Static
  ISA count of its K loop: 64 WMMA, 756 other VALU (about 10 per unpacked
  weight, incl. a software fp32 → bf16 rounding) and 487 LDS instructions per
  wave per K step. Proposed fix (not implemented): exact integer weights
  `0x4300 | q` into WMMA with zero/scale applied per K step, and B-fragment order
  in LDS. `rocprofv3` aborts with Torch in the process, so no counters.
- **Decode GEMV (G5):** new `gemv_w4a16` (fp32 accumulation): per layer at
  s = 14336, 1.090 → 0.979 ms; 93 to 96 GB/s.
- **Model integration (G6):** `slice_packed_ffn`, `GpuFfnModel`, and a switch in
  `forward_llama3_with_scratch` (CMake `-DHETEROMEGA_GPU_FFN=ON`, environment
  `HETEROMEGA_GPU_FFN=1`). **Not built or run in the model yet:** this AUP
  instance has no model weights, XDNA headers or `HeteroMosaic-baseline` copy.
- `src/hipkernels/w4a16_gemm_packed.cu`: new `w4a16_gemm_fused_packed_tiles`
  entry point, K-loop bound fix for K prefixes in the K2048 kernel, `#ifndef`
  guard on `AUTOTUNE`. Existing entry points are unchanged in behavior.

## 2026-10-16: GPU FFN-slice module proposed (`kernels/gpu/`)

- Added `kernels/gpu/`, a separate CMake project for the GPU half of the
  persistent intermediate-channel FFN split: slice packer, gate/up GEMM,
  SiLU-and-multiply and down GEMM on one HIP stream, boundary sum on the GPU or
  the CPU, tests and benchmarks. It compiles the existing `src/hipkernels/`
  packed GEMM/GEMV unchanged. Root build, model path and baseline are unchanged.
- **Code-confirmed:** the GPU branch contract (shapes, packed layout, metadata,
  dtype/strides, buffer ownership, completion) is in `kernels/gpu/README.md`.
  Tests include the proposed 4096 NPU / 10240 GPU split and M=512.
- **Executed (macOS, no GPU, 2026-10-06):** numpy and C++ packers are byte-equal
  to a line-by-line translation of `_pack_weights_packed`; emulated GEMV
  addressing on slices and `Y_G + Y_N` == full FFN for 3 split points pass.
- **Not executed:** CMake configure, HIP/C++ compile, GPU tests or benchmarks on
  the AUP cloud. No performance or GPU numerical result is claimed.
- Next: build and run `test_gpu_ffn` in the `cloud-baseline.md` environment,
  then `bench_gpu_ffn_slice --m 512 --s 10240,14336`.

## 2026-10-15: GitHub kernel teammate handoff

- User explicitly requested committing/pushing the work so the GPU/NPU kernel
  teammate can refer to it. Remote `origin/main` was checked before publication
  and still pointed to starting commit `61714a3147f34857aa5dfef21a28e762777a2df0`.
- Prepared separate runtime-port and documentation/evidence commits. Added
  `docs/kernel-handoff.md` and README entry links; preserved upstream setup.
  Runtime port commit: `20e4d3052ac21d95ae03f70ad916fab6c75de632`.
  Saved plans, cloud instructions, compatibility evidence, profiling report and
  reproduction helpers are included. Earlier dated references to uncommitted
  changes describe historical session state.
- No new math kernels, persistent-FFN implementation, dependencies or model
  downloads were introduced for this handoff. Cloud-only build adaptations
  remain documented reproduction artifacts. Shared-host drivers were untouched.
- Checks: source patch matches the saved tested compatibility patch; whitespace
  checks pass for runtime edits and maintained Markdown. Preserved user-plan
  Markdown hard breaks and raw evidence retain their original whitespace.
  Text/archive credential scans and local Markdown link checks passed; saved
  cloud screenshots were visually reviewed. Prior cloud build/run evidence is
  retained; no fresh GPU/NPU run is implied by publication.
- Next: coordinate kernel buffer/packing/completion contracts, investigate the
  random packed-W4A16 numerical mismatch, then implement an isolated FFN shape
  and separate down-placement benefits from persistent-partition benefits.

## 2026-10-14: stage profiling and bottleneck report

- User authorized profiling latency/stages/bottlenecks to guide project work.
  Reused AUP hardware, weights, dependencies and the October 13 compatibility
  port. HEAD still `61714a3147f34857aa5dfef21a28e762777a2df0`.
- **Measured:** four configurations, one excluded warmup plus five timed calls
  each, full 32 layers, fixed 1024-token input and eight greedy output tokens.
  Median prefill: GPU native chunks 1745.1 ms, GPU matched 512-token chunks
  1780.2 ms, heterogeneous serial 1327.3 ms, same partitions/two inflight chunks
  1316.6 ms. All 20 measured baseline calls matched prior GPU token IDs.
  Conditional timing results only; numerical fidelity remains unvalidated.
- **Measured:** serial heterogeneous decode loop 439.9 ms for seven steps,
  versus GPU matched 415.3 ms. Complete generation medians 1766.8 vs 2196.7 ms.
  Two inflight chunks improve complete latency only 0.3% on this workload.
  Final repetitions supersede an early commentary impression of a regression.
- Added temporary cloud-only Llama FFN substage scopes and NPU host-phase
  timestamps; existing tracer captures decode too because its guard lives
  through `generate`. Separate phase by sequence length. Two measured traced
  calls per diagnostic after warmup; an untraced instrumentation pass checks
  overhead. Added timed-ioctl interposer and separate Torch/Kineto GPU traces.
- **Measured:** synchronized serial prefill 1340.8 ms: gate/up 437.1 ms, down
  422.2 ms, SiLU/gating 49.4 ms, FFN norm 9.0 ms; FFN total 917.7 ms (68.4%).
  G1 158.5 ms, attention 158.0 ms, O/layout/residual 95.0 ms. Stage barriers
  add about 1% to prefill but 10% to decode; pipeline host intervals are not
  comparable device-completion measurements.
- **Measured:** 288 prefill +224 gate-decode NPU submissions per generation;
  successful submit/timeline waits. Untraced serial prefill host sums: command
  build 18.9 ms, readiness polling 1171.9 ms, mutex queue 0.5 ms, submission
  3.2 ms, completion wait 587.1 ms. Pipeline readiness/queue sums rise to
  2602.3/181.5 ms while completion stays 589.4 ms. These overlap, are not
  additive latency or hardware utilization, and are not all removable overhead.
- **Code-confirmed/measured:** active prefill down is GPU-only; M-split uses
  shared output regions, not a mandatory concat. Down decode's invalid
  cpuN+npuN causes 224 warnings/fallbacks per generation; Q decode also uses
  CPU channels. Do not describe the decode comparison as purely GPU/NPU.
- Saved report, PNG/SVG figure, normal summary CSV, configs/results, raw
  stage/host/ioctl traces, kernel tables and losslessly zipped GPU traces in
  `docs/profiling-2026-10-14.md` and `docs/runs/2026-10-14-profile/`.
  Reproduction scripts and restoration checks are included. No packages were
  installed; used the existing IRON matplotlib only for the report figure.
- Restored cloud model/NPU source from pre-profile snapshots and restored its
  baseline module, then rebuilt the ordinary extension successfully. Hash/content
  checks passed. No profiling instrumentation was added to local implementation
  files; the earlier four-file compatibility patch remains uncommitted. No host
  driver/firmware changes, inference algorithm changes, commits or pushes.
- Next: resolve numerical mismatch; isolate a 512x4096x14336 FFN and test down
  placement before attributing gains to persistence. A proposed feasibility
  partition is 4096 NPU /10240 GPU channels using staged NPU shape families,
  with activation/layout/packing correctness still to establish. Preserve a
  token-row comparison and measure readiness/queueing with controlled schedules.

## 2026-10-13: direct-XDNA runtime port and confirmed GPU-NPU model run

- User explicitly authorized resolving cloud compatibility. Source HEAD remains
  `61714a3147f34857aa5dfef21a28e762777a2df0`; changes are uncommitted.
- **Measured:** explicitly aligned device-heap mapping resolves context EINVAL.
  Zeroed padded QoS/CU parameters resolve PDI configuration failure; unique BO
  residency handles resolve submission EALREADY; DRM timeline waits resolve
  unsupported legacy WAIT_CMD. Changed four runtime/header files locally and
  in the cloud, preserving model math, kernels, splits and scheduling.
- **Executed:** patched M sweep and model-extension builds, exit 0. Original
  random-weight sweep now submits/completes NPU work but fails existing
  numerical tolerance (first shape max difference 1.5). An independent float
  product puts the larger error on the NPU side. Exact scales alone do not fix
  it; intermediate accumulation/rounding is a hypothesis, not a confirmed cause.
- **Executed:** cloud-only short diagnostic sweep, 2048 cubed: exact activation
  1/16, qweight 9, zero point 8, scale 1/16 => output exactly 8. Eight row splits
  from NPU M=256 to 2048 pass with zero error, including concurrent shared-buffer
  GPU-NPU work and NPU-only execution. Random activations with those constant
  weights pass loose existing tolerances but have max difference 0.25.
- **Executed:** full GPU model regression, same fixed 1024-token input and eight
  greedy tokens as October 12; one cold generation call 2.436990 s.
- **Executed:** true heterogeneous model run with shipped 512-token serial
  chunk schedule, same weights/input/output settings. Two contexts; 512 NPU
  submissions and 512 timeline waits succeeded, zero submission/timeline
  failures. Generated token IDs match GPU-only; cold wall time 2.005948 s.
  This is functional evidence, not a repeated speedup or accuracy result.
- Found/corrected sparse artifact staging: GEMV needs `_K_8c` folders. Earlier
  apparent hetero model success actually used GPU fallback (zero submissions)
  after missing files disabled the driver. Preserve those logs as rejected
  evidence. Existing down-GEMV cpuN+npuN=5120>4096 warning/fallback remains;
  the other NPU paths are confirmed by counts.
- Rejected START_NPU packet adapters after worse numerical results. Retained
  original START_CU encoding. Restored the cloud sweep source after temporary
  diagnostics; its original-benchmark rebuild is recorded separately.
- Saved raw probe/build/model logs, source variants, configs/results and patches
  locally in `docs/runs/2026-10-13-compat/evidence/`; current source patch is
  `docs/runs/2026-10-13-compat/runtime-compat.patch`. Updated session instructions,
  plan, cloud runbook and `docs/runtime-compatibility.md`. Host/IRON unchanged;
  no commits/pushes. Cloud session expired during a stalled browser call;
  recovered persistent files through a new session.
- Next: characterize NPU numerical behavior without relaxing checks, establish
  random packed-W4A16 correctness and broader real-weight accuracy, and make
  setup failures explicit before persistent-FFN implementation/performance work.

## 2026-10-12: baseline build and real-weight GPU model run

- User explicitly requested running HeteroMosaic, authorizing isolated cloud
  dependency installation, model downloads, and compatibility investigation.
- Created a separate ROCm PyTorch environment (2.11.0+rocm7.2) and pinned source
  checkout in `/notebooks`; kept IRON environment and host drivers intact.
- **Executed:** CMake configure and build of GPU layout test, original M-split
  sweep, and Python model extension; exit 0 after two CMake path adaptations
  and a build-time alias header for renamed Torch HIP stream/guard types.
  No inference/kernel/scheduler/runtime source changes.
- **Executed:** existing GPU layout test; unpacked GEMM/GEMV matched references.
  Its source selects unpacked mode; this is not packed-kernel numerical validation.
- **Executed:** original heterogeneous sweep, bounded at 180 s. 88 split entries
  failed NPU context creation (`EINVAL`), four GPU reference entries completed,
  and a fifth GPU reference was interrupted. No successful heterogeneous result.
- **Executed:** full 32-layer Llama 3.1 8B AWQ GPU inference with pinned real
  model/tokenizer revision `db1f81ad4b8c7e39777509fac66c652eb0a52f91`; loaded 291
  parameter groups, zero skipped. Fixed 1,024-token input, context capacity 1,536,
  eight greedy generated tokens; synchronized generation-call wall time
  2.669836 s in one cold run, excluding construction/weight loading. Exit 0.
  Not a statistical performance or model accuracy/perplexity study.
- **Executed:** process-local extended-QoS adapter retry; still `EINVAL`, bounded
  at 90 s. Confirmed installed XDNA UAPI differs; complete cause remains unresolved.
  Kernel log access is unavailable inside the container.
- Saved exact commands, patch, adapter/driver sources, configs, package pins,
  fixed input/output IDs, and logs locally in `docs/runs/2026-10-12-aup/`.
  Updated `docs/cloud-baseline.md` with reproducible setup and execution steps.
- Local implementation unchanged; source HEAD still
  `61714a3147f34857aa5dfef21a28e762777a2df0`; no commits/pushes.
- Next: provider kernel diagnostics/current-XDNA compatibility work or matching
  host stack; validate original packed shared-buffer NPU launch before full
  heterogeneous inference or persistent-FFN implementation.

## 2026-10-12: AUP cloud access and device smoke checks

- Authenticated to the supplied AUP cloud and opened a JupyterLab terminal in
  the `mlir-aie:latest` environment. Credentials were not saved to the repository.
- Observed Ryzen AI 9 HX 370/Radeon 890M (`gfx1150`), NPU Strix (`aie2p`),
  exposed GPU/NPU devices, Ubuntu 24.04.4/kernel 6.17.0-42, ROCm 7.2 and XRT 2.26.
- **Executed:** official MLIR-AIE single-core i16/i32 matmul, M=K=N=512,
  `make ... NPU2=1 run`; exit 0 and `PASS!`.
- **Executed:** scratch HIP kernel on gfx1150; 16/16 outputs correct. Fixed its
  build invocation to select ROCm 7.2 headers/libraries instead of stale system
  HIP 5.7. No installed files or repository implementation were changed.
- Existing `ironenv` Torch is `2.13.0+cpu`; GPU tensor check failed because it
  has no HIP support. Separate ROCm PyTorch is required for HeteroMosaic.
- Saved connection steps, version differences, working smoke commands and the
  staged baseline procedure in `docs/cloud-baseline.md`. Remote diagnostic
  source/logs are under `/notebooks/heteromega-baseline-logs`; example generated
  its build artifacts. One transient connection error recovered through hub.
- **Not executed:** HeteroMosaic configure/build, W4A16 kernels, DMA-BUF sharing,
  concurrent GPU-NPU tests, model generation, or performance baseline. No
  dependencies installed, weights downloaded, commits/pushes, or implementation
  edits. HEAD still `61714a3147f34857aa5dfef21a28e762777a2df0`.
- Next: resolve supported runtime/image versus a documented 7.2/XDNA 2.26 port;
  establish isolated ROCm Torch; validate original packed W4A16 GPU/NPU split
  with correctness and sharing before full-model downloads or FFN changes.

## 2026-10-12: revised plan saved for future sessions

- Saved the complete user-provided plan verbatim in repository `agent.md`, with
  provenance and separately labeled assistant review recommendations.
- Current planning scope is FFN prefill on one model/platform; attention remains
  on its existing path. Attention partitioning, decode optimization, speculative
  decoding, and broader model support are deferred.
- Updated AGENTS to read `agent.md` and marked the earlier project-plan scope
  and milestones as historical. Preserved the existing architecture/build notes.
- Implementation remains unstarted. HEAD remains
  `61714a3147f34857aa5dfef21a28e762777a2df0`; documentation remains uncommitted.
- Verification: checked saved plan against the supplied text and inspected Git
  status. No implementation, dependency, model, build, or benchmark changes.
- Next step when development is requested: use `agent.md` for the audit,
  NPU activation/layout feasibility investigation, and baseline experiment.
  The September next-session list below is historical where it differs.

## 2026-09-18: repository onboarding and scope capture

**Status:** documentation prepared; implementation and hardware validation have
not started. No performance or numerical correctness result has been reproduced.

### Starting state

- Repository: `C:/Users/pangq/Desktop/HeteroMega/HeteroMega`, branch `main`.
- Starting HEAD: `61714a3147f34857aa5dfef21a28e762777a2df0`,
  `updated readme (#3)`, authored 2026-09-04.
- Origin: `https://github.com/glassescrab/HeteroMega.git`;
  upstream: `https://github.com/UIUC-ChenLab/heteroMosaic.git`.
- Working tree was clean before documentation changes. No existing AGENTS.md
  was found in the workspace/repository or checked ancestor locations.
- Reference PDFs live in sibling `../Documents/`, outside Git. The supplied
  proposal matches the user's goal; whether it is the final version is unknown.
- Session host is Windows/PowerShell; target execution recipe is Linux/AMD.

### Completed

- Read README, build/setup files, Llama frontend/bindings, main model execution,
  quantized linear dispatch, heterogeneous compute helpers, NPU config/import/
  launch code, and relevant M/K/N microbenchmark and tuning locations.
- Read the complete one-page proposal and relevant Megatron-LM/HeteroMosaic
  sections; visually checked the proposal and key architecture figures.
- Created repository `AGENTS.md`, `docs/project-plan.md`, and this log. Added a
  small outer-workspace `../AGENTS.md` entry point to identify the nested repo.
- Recorded documented commands, source-based corrections, reuse opportunities,
  feasibility gates, and a small first development milestone in the plan.

### Findings that affect the research

- Existing code supports operator splits plus causal chunk scheduling. Preserve
  both in the original baseline and control scheduling in causal comparisons.
- Main Llama attention and FFN pointwise work are GPU-side. Persistent device
  ownership across those operations is not implemented in the inspected path.
- M-split GEMM writes into a shared output; do not assume every operator performs
  a gather. K-split stages/copies and sums; N-split benchmark explicitly assembles.
- N-split GEMM is benchmarked but lacks a dedicated model dispatch branch.
- NPU execution uses shared DMA-BUF handles and a mutex-protected submit/wait
  path. KV caches are currently excluded from XDNA buffer import.
- Shipped Strix Point Llama config selects GPU. README's config filename and
  model example differ from the actual frontend; see plan before running.
- Existing traces can either perturb execution with synchronization or omit
  device completion timing. Fine-grained overhead evidence needs more work.

### Verification and boundaries

Static source/path checks only. No setup/build/model/benchmark/tuning script was
run. No dependencies installed, weights/tokenizers downloaded, implementation
files modified, PDFs moved, or commits/pushes made. Documentation links/anchors
and final Git diff/status were checked. Reference-render scratch images are in
the app visualization directory, outside the Git repository.

### Next session

1. Read AGENTS and the plan; incorporate a later final proposal if supplied.
2. Identify the target Ryzen AI system and available Linux runtime/NPU kernel
   generation tools. Do not treat documentation as a successful setup.
3. When implementation is authorized, implement M1: synthetic FFN partition
   metadata/reference correctness harness with unequal channel splits and one
   boundary sum, plus a shape/layout compatibility audit. No model needed.
4. Record actual commands/results and calibration tolerances here. Hardware
   FFN integration and NPU attention remain subsequent work, not completed work.
