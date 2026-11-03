# Host-side serialization review (report only, 2026-11-02)

> **Correction, later on 2026-11-02 (measured on AUP).** The estimates in this review for `-O3` and for removing the
> scans were pessimistic (856 / 1757 and 799 / 1643 ms estimated; 709.2 / 1473.8 and 649.4 / 1351.2 ms measured at 512 /
> 1024 tokens), and native M-split down did not save time with the existing artifact (1518.3 vs 1333.8 ms). See
> `docs/runs/2026-11-02-fix-step1/README.md`, `docs/runs/2026-11-02-fix-step4-mdown/README.md` and `docs/runs/2026-11-02-fix-step3c/README.md`.

Scope: `generate`, `forward_llama3_chunked`, `forward_llama3_with_scratch` and chunk
scheduling in `src/unified_llm_w4a16_hetero/unified_llm_w4a16.cpp`; the merged FFN
executor in `src/unified_llm_w4a16_hetero/merged_ffn.cpp`; `npu_execution_mutex()` in
`src/npu_matmul_func/npu_execution.cpp`; and the native NPU call path it guards
(`hetero_compute.cpp`, `npuMatmul_zero` in `npuSetup.cpp`). No behavior was changed.
Labels: **code-confirmed** (read in source), **measured** (cited evidence file),
**estimated** (arithmetic on measured numbers), **hypothesis** (needs an AUP run).
Nothing here was executed on hardware.

## 1. Why two inflight chunks gain so little

Measured (`docs/profiling-2026-10-14.md`, 1024 tokens, two 512-row chunks): prefill
1327.3 ms serial vs 1316.6 ms with two inflight chunks (0.8%); complete generation
1766.8 vs 1761.9 ms (0.3%). Host sums over 288 NPU calls: readiness polling 1171.9 →
2602.3 ms, mutex wait 0.5 → 181.5 ms, completion wait 587.1 → 589.4 ms.

1. **The heterogeneous prefill is GPU-bound** (estimated from the measured stage table).
   Every synchronized stage of the heterogeneous run is about the GPU share of the
   GPU-only stage: gate/up 800.7 ms GPU-only → 437.1 ms with 256 of 512 rows on the GPU;
   G1 210.4 → 158.5 (Q fully on the NPU); O 135.9 → 95.0; attention 152.6 → 158.0 and
   down 409.9 → 422.2 stay GPU-only. The total, 1340.8 ms synchronized and 1327.3 ms
   normal, is about 1780 ms GPU-only minus the GPU work moved to the NPU (about 450 ms).
   The NPU finishes inside each stage: its completion waits sum to 587 ms, under half
   the prefill. With a second chunk, both chunks' GPU work goes to **one GPU** on two
   streams (`init_llama_chunk_async_runtime`, `unified_llm_w4a16.cpp:1856-1859`). Total
   GPU work is unchanged and the GEMMs are compute-bound (9.1-9.5 TFLOPS,
   `kernels/gpu/README.md`), so only the small non-GPU gaps can be filled (about 1%).
2. **The polling and completion sums overlap and are not additive** (code-confirmed). Each
   NPU call runs on its own `std::async` thread (`hetero_compute.cpp:243`) that spins on
   `hipEventQuery` (`npuSetup.cpp:1860-1870`) until the GPU stream reaches the event
   recorded at call time (`hetero_compute.cpp:116`). The host enqueues far ahead of the
   GPU, so "readiness" is mostly the NPU waiting for the GPU, which is the GPU-bound
   symptom of item 1. With two chunks twice as many calls wait at once, so the sum doubles
   (2602 ms) while elapsed time barely moves. Completion waits (587 → 589 ms) show that
   the NPU work itself is unchanged.
3. **One NPU, one lock** (code-confirmed). `npu_execution_mutex()` is a single
   process-wide `static std::mutex` (`npu_execution.cpp:3-6`), held from submission
   through completion (`npuSetup.cpp:1880-1903`) and, for the merged FFN, from BO write
   to BO readback (`merged_ffn.cpp:262-285`). Calls from both chunks serialize on it: the
   181.5 ms mutex wait is the second chunk queuing behind the first on the same device.
4. **The pipeline is shallow by design** (code-confirmed). Chunk c waits, per layer, for
   chunk c-1's KV-ready progress with a `yield` spin plus `hipStreamWaitEvent`
   (`unified_llm_w4a16.cpp:2495-2505`). Every projection's host thread blocks on its NPU
   future before continuing (`:2447-2452` q/k/v, `:2632-2633` o, `:2682-2685` gate/up,
   `:2694-2695` down). With two 512-row chunks only one chunk boundary can overlap.

Consequence (hypothesis to measure): host-side serialization is **not** the first-order
limit of the original heterogeneous path; GPU work is. Only moving work off the GPU or
making GPU kernels faster can shorten it.

The merged FFN is the opposite case. At P6400 one call takes 32.5 ms when GPU and NPU
overlap and 44.8 ms in serial mode, so the GPU branch is about 12.3 ms and the NPU path
about 32 ms (measured: `docs/runs/2026-10-31-ffn-deploy/deployment-results/cpp-smoke.json`,
layer 0/stream 0; derived split). It is **NPU-bound**, and the NPU path holds the global
mutex, so a second inflight chunk cannot overlap two NPU paths either. The local session
reports the first paired evaluation as slower than `gpu_original` (about 1.3 vs 0.9 s at
512 tokens). That is consistent, but not yet in this repository.

Unexplained (hypothesis): in the same smoke, layer 1 on stream 1 shows no overlap
(concurrent 44.3 ms ≈ serial 44.6 ms). One candidate is that the transfer stream's D2H
copy shares a hardware queue with stream 1's GPU FFN kernels (HIP multiplexes streams
onto a few hardware queues). Measure with per-stream timestamps, or by varying
`GPU_MAX_HW_QUEUES`.

## 2. Serialization points and busy-wait loops

| # | Location | What serializes | Kind |
| --- | --- | --- | --- |
| S1 | `npu_execution.cpp:3-6`, `npuSetup.cpp:1880-1903` | One global mutex around submit + completion wait for every native NPU call | lock, blocking ioctl wait (`npu_matmul_func.cpp:30-52`) |
| S2 | `merged_ffn.cpp:262-285` | Same mutex held across BO write+sync, run, wait, BO sync+read (host copies of 4 MiB in and out) | lock |
| S3 | `npuSetup.cpp:1860-1870` | `hipEventQuery` spin per NPU call, one CPU core each | **busy wait** |
| S4 | `unified_llm_w4a16.cpp:2500-2502` | `yield` spin on previous chunk's layer progress | **busy wait** (yielding) |
| S5 | `hetero_compute.cpp:116` + `unified_llm_w4a16.cpp:2680-2681` (also q/k/v at `:2444-2446`) | Readiness event recorded after the previous sibling's GPU slice: up's NPU waits for gate's GPU rows, k/v for q's | conservative ordering |
| S6 | `unified_llm_w4a16.cpp:2447-2452, 2632-2633, 2682-2685, 2694-2695` | Host thread blocks on each projection's NPU future | host blocking |
| S7 | `merged_ffn.cpp:241-242, 259` | Host waits for the slot's previous call and for the input D2H before the NPU can start | host blocking |
| S8 | `merged_ffn.cpp:260, 286` | Scalar finite scans of 2 x 2.1M BF16 values on the critical path | host CPU work |
| S9 | `merged_ffn.cpp:255` | `serial=true` waits for the GPU branch before the NPU starts (test option; the model passes the default) | optional |
| S10 | `npuSetup.cpp:1843` | `create_cmd_packet_mutex` around command construction (measured 18.9 ms total) | lock, minor |
| S11 | `hetero_compute.cpp:243` (and the other `std::async` sites) | One new thread per NPU call (288 per prefill) instead of the existing `g_thread_pool` (`:31`) | thread creation |
| S12 | `unified_llm_w4a16.cpp:2331`, `merged_ffn.cpp:174` | Setup mutexes taken once per chunk / per call | negligible (µs) |
| S13 | `unified_llm_w4a16.cpp:2256-2262, 2284-2295` | A slot's next chunk waits for that slot's previous chunk; final per-stream synchronize | scheduling |
| S14 | `unified_llm_w4a16.cpp:3946, 3952` | Device synchronize around prefill in `generate` | timer boundary, keep |

## 3. Proposed changes, ranked by expected benefit and risk

All benefits are estimates, and each comes with the AUP measurement that would confirm it.
Items 1 and 2 are placement rather than serialization changes. They come first because
section 1 shows placement, not locking, sets the critical resource.

1. **Merged FFN: pick a smaller NPU share** (benefit: large, likely turning a loss into
   a gain; risk: low, no code change). At P6400 the NPU path (~32 ms) is about 2.6× the
   GPU branch (~12.3 ms). The task-2 cost model predicts the crossover. *Measure:*
   `experiments.ffn_deploy.prepare --build --npu-channels P` for P near the predicted
   balance, then `evaluate_prefill.py --methods gpu_original merged_ffn --gpu-chunk-size 512`.
2. **Original heterogeneous path: offload half of down** (benefit: up to about 200 ms of
   1327 ms, estimated: down is 422 ms GPU-only and the NPU's 256-row gate/up waits are
   3.24 ms per call, comparable to half a down; risk: low, config only). The shipped
   1024-token group disables down (`"use": false`, `npuM: 0`), but
   `hw_bins/npu2/16384x14336x4096/bf16_int4AWQ_bf16_M/final_256x14336x4096_*` exists. The
   original authors may have rejected this after tuning, so measure before adopting.
   *Measure:* a copied config with that entry enabled (`npuM: 256`, that `fw_path`),
   `evaluate_prefill.py --methods heteromosaic` against the shipped config, plus the ioctl
   counter for +64 NPU calls.
3. **Merged FFN: take the finite scans off the critical path** (S8; benefit about
   1-4 ms per call, i.e. 64-256 ms per 1024-token prefill, estimated; risk: low-medium
   because it weakens a structural guard). Replace with an asynchronous GPU-side
   `isfinite` check on the summed output, or make the host scan sampled. *Measure:*
   wall time of `forward` with and without the scan in `experiments.ffn_deploy.smoke`,
   then the evaluator.
4. **Merged FFN: shrink the critical section and stop host staging** (S2, S7; benefit
   about 1.5-2 ms per call, estimated from the earlier design's D2H 0.24 ms, H2D 0.19 ms
   and BO IO 1.1-1.4 ms, `experiments/ffn_partition/results/timeline_m512_p512_array8.json`
   and `bottleneck_analysis.json`; risk: medium-high). Option (a): per-slot x/z/y BOs, so
   BO write/read stays outside the mutex and only run+wait holds it; this helps only with
   two or more slots. Option (b): DMA-BUF import of the GPU buffers, as the native path does
   (`import_dma_buf_to_xdna`), removing all four 4 MiB copies. *Measure:* smoke
   concurrent/serial per call, then evaluator inflight 1 and 2.
5. **Record one readiness event per shared input** (S5; benefit small while the NPU
   mutex serializes gate→up anyway; becomes relevant once NPU calls are shorter; risk:
   low). *Measure:* per-instruction readiness waits with the profiling instrumentation
   (`docs/runs/2026-10-14-profile/`), expecting up's 3.4 ms to drop toward gate's 0.17 ms,
   and the prefill median.
6. **Replace spin waits with blocking waits** (S3, S4; latency benefit < 1%, estimated;
   risk: low). Use events created with `hipEventBlockingSync` plus `hipEventSynchronize`,
   or spin briefly and then sleep. CPU and iGPU share the package power budget
   (44-53 W and clock changes observed in `kernels/gpu/README.md`), so freeing cores
   might raise GPU clocks (hypothesis). *Measure:* package power, GPU clock and prefill
   median with spin vs blocking.
7. **Reuse the thread pool for NPU futures** (S11; about 6-15 ms per prefill at
   20-50 µs per thread creation, estimated; risk: low). *Measure:* prefill median.

## 4. Fewer kernels, and an attention-style fused FFN (user request)

Launches per FFN per 512-row chunk (code-confirmed):

| Path | GPU launches / stream ops | NPU launches |
| --- | --- | --- |
| Original HeteroMosaic (`unified_llm_w4a16.cpp:2680-2699`) | gate GEMM, up GEMM, `silu_`, `mul_`, down GEMM, residual add = 6 | 2 (gate, up M-split) in hetero mode |
| HeteroMega GPU FFN (`kernels/gpu/src/gpu_ffn.cpp:85-97`) | gate_up GEMM, `silu_mul`, down GEMM + residual add = 4 | 0 |
| Merged FFN (`merged_ffn.cpp:246-292`, `unified_llm_w4a16.cpp:2652-2653`) | 3 GPU-branch kernels + D2H + H2D + `add_out` + residual add = 7 | 1 (two phases, Z through DRAM) |

Launch overhead itself is small. At about 5-10 µs per launch, removing three launches
from 64 layer-chunks saves about 1-2 ms per 1024-token prefill (estimated). Fusion pays
through memory traffic and synchronization, not launch count:

- **F1. SiLU×up in the gate/up GEMM epilogue.** This already exists but is off:
  `FUSE_ACT` in `src/hipkernels/w4a16_gemm_packed.cu:380,549`, selected by
  `HETEROMEGA_GPU_FUSE_ACT` in `gpu_ffn.cpp:5-6,90-96`, default 0, never measured. It
  removes the `silu_mul` kernel and one write+read of the [512, 2c] gate_up tensor.
  Expected at most about 0.35 ms per full-width layer-chunk, roughly 20 ms per 1024-token
  prefill (estimated from Kineto `silu` 0.29 ms + `mul` 0.47 ms and the measured 2% saving
  of `silu_mul`). *Measure:* configure `kernels/gpu` with
  `-DCMAKE_CXX_FLAGS=-DHETEROMEGA_GPU_FUSE_ACT=1` (no CMake file edit), then run
  `test_gpu_ffn` and `bench_gpu_ffn_slice --m 512 --s 7936,14336`.
- **F2. One pass for the boundary sum and the residual.** `add_out(y_gpu, y_npu)` then
  `x.add_(y)` makes two passes over [512, 4096], about 0.13 ms each (measured 132.9 µs
  reduction). A single `x += y_gpu + y_npu`, or writing the residual from the down-GEMM
  epilogue, saves about 0.13-0.26 ms per call, roughly 8-17 ms per 1024-token prefill
  (estimated). The summation order changes, so it needs report-policy numerics.
- **F3. A FlashAttention-style fused FFN** (tile over intermediate channels, keep gate/up/Z
  on chip, accumulate Y) is **not recommended now** (estimated):
  - *GPU:* a 64-row block's FP32 Y accumulator is [64, 4096] = 1 MiB, far beyond LDS and
    registers, so the down reduction needs a cross-workgroup split-K pass. The FFN GEMMs
    are compute-bound (180 GFLOP per layer-chunk ≈ 19.8 ms), while all intermediate DRAM
    traffic (gate_up write/read, Z write/read ≈ 88 MB) costs about 0.9 ms at about
    96 GB/s. The ceiling is therefore about 5% of GPU FFN time, before any efficiency
    loss, for a large rewrite.
  - *NPU:* Z for M512 × P6400 is 6.5 MB, larger than the memory tiles of an 8-column
    array (assumed 8 × 512 KiB; verify). Row blocking that fits (M32: 410 KB) would
    re-stream the down weights 16 times (about 14 MB → 220 MB per call). The merged graph
    already fuses SwiGLU into phase 1 in one launch, and its remaining cost is projection
    arithmetic, not Z traffic.
- **"Attention-like" overlap** of one chunk's FFN with another chunk's attention is
  already HeteroMosaic's micro-batch pipeline. It gains about 1% because the GPU is the
  bottleneck (section 1); it pays off only after work moves to the NPU (items 1-2).
