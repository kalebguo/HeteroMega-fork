# 2026-11-02: input wait, two inflight chunks, attribution (AUP)

**Measured.** The headline is now 1.36x at 1024 tokens and 1.38x at 512 tokens at commit `9dac460`. A second
inflight chunk gains nothing. Job `aup-runs/step2b`, 20:51-21:11 UTC (14:51-15:11 CST).

## A. Paired evaluation at `9dac460`, `sampled` check, one inflight chunk

| tokens | `gpu_original` median (min-max) ms | `merged_ffn` median (min-max) ms | speedup |
| ---: | --- | --- | ---: |
| 512 | 891.7 (882.8-969.0) | 644.7 (640.8-668.5) | 1.383 |
| 1024 | 1827.8 (1826.1-1871.7) | 1340.8 (1335.8-1376.5) | 1.363 |

10 samples per cell; 32 / 64 completions per pass, 22 passes per cell, 0 wrong; controls clean; 64 greedy tokens
identical; 8 reference tokens match at 1024. Commands in `tools/aup-kit/jobs/step2b/cmd.sh`.

Smoke without timers, `smoke-sampled.json`, median ms per call: slot 0 concurrent 13.11, serial 25.38;
slot 1 concurrent 13.50 (25.53 before this commit), serial 25.69. 30 dispatches.

## B. Two inflight chunks (`--inflight 2`, 1024 tokens)

| Method | Inflight 1 (above) | Inflight 2 median (min-max) ms |
| --- | ---: | --- |
| `gpu_original` | 1827.8 | 1840.0 (1829.9-1957.5) |
| `merged_ffn` | 1340.8 | 1352.5 (1348.3-2386.7) |

Tokens identical. Timers (`inflight2-phases.json`): each of the two slots takes 320 of the 640 calls, the NPU wait
stays 11.06 ms, and the GPU branch now outlasts the NPU path by 10.34 ms because GPU attention of the other chunk
queues on the same GPU. The GPU carries about 20.2 ms per layer-chunk (7.82 ms attention, 12.37 ms branch), so
the lower bound is 1293 ms (estimated). A second chunk at this split can gain at most 4%, and it gained none.

## C. First attribution attempt: same commit without `-O3`

The tree `/notebooks/hm-fix-noopt` removes the `-O3` line and keeps everything else, including the rewritten scan.

| Phase, median ms per call, model, full check | No `-O` | `-O3` |
| --- | ---: | ---: |
| Input scan | 1.860 | 0.935 |
| Output scan | 1.990 | 1.062 |
| NPU wait | 11.487 | 11.401 |
| Whole call | 23.934 | 22.353 |
| Prefill median, 5 samples, ms | 1588.4 | 1480.6 |

This does not reproduce the first evaluation's 2558.7 ms. The difference is the scan itself: this commit's
`find_nonfinite_bf16` keeps the element count in a local, while the original loop calls `host.numel()` per element,
which is many non-inlined calls without optimization. `../2026-11-02-fix-step2c/` restores the original loop.
