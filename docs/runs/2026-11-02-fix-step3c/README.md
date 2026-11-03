# 2026-11-02: head-to-head and NPU context-change cost (AUP)

**Measured.** With GPU attention, `merged_ffn` ties `heteromosaic`: 1336.2 ms against 1317.6 ms (0.986x), with 64
instead of 288 NPU calls per prefill and one hardware context instead of two. Each NPU context change costs 2.0-3.0 ms.
That cost decides both failed combinations: `merged_ffn_hetero` and `heteromosaic_mdown`.

Job `aup-runs/step3c`, 00:26-00:34 UTC Nov 3 (18:26-18:34 CST Nov 2), commit `e16a98d`, C++ build `9dac460`, `sampled`
check. Part A: `evaluate_prefill.py`, 10 samples per cell, 2 processes per method in ABBA order. Part B: same methods with
`profile_ioctls.c` preloaded, which records every submit and wait. Part B medians include trace cost: heteromosaic 1342.4,
heteromosaic_mdown 1524.5, merged_ffn_hetero 1626.5 ms. Traces (6 `ioctl-*.csv.gz`) stay on the box in
`hm-results/2026-11-02/fix-step3c`; `context-waits.json` is the summary.

## A. Head to head

| Method | Prefill median ms (min-max) | Vs heteromosaic | NPU submissions per process | Hardware contexts |
| --- | ---: | ---: | ---: | ---: |
| `heteromosaic` | 1317.6 (1290.7-1341.9) | 1.000x | 13472 | 2 |
| `merged_ffn` | 1336.2 (1329.1-1384.1) | 0.986x | 704 | 1 |

Tokens match each other and the reference. `merged_ffn` submissions: 704 = 64 per prefill x 11 passes. Process medians of
`heteromosaic` differ by up to 36 ms (1310.8 vs 1335.2, 1313.5 vs 1349.4), so a 19 ms difference is close to the
between-process spread.

## B. NPU completion wait by call position in a layer-chunk

Median ms per call, 11 passes, second process of each method, computed from the traces. The prefill has two chunks. A
layer in the first chunk has 4 native calls; a layer in the second chunk has 5. Bold calls are the two sides of a context
change. The gate and up labels are inferred from the equal waits and the Oct 14 kernel map.

| Call | `heteromosaic` | `heteromosaic_mdown` | `merged_ffn_hetero` |
| --- | ---: | ---: | ---: |
| Chunk 0: first attention call | 1.911 | **4.744** | **4.858** |
| Chunk 0: second attention call | 1.013 | 1.029 | 1.083 |
| Chunk 0: gate, up | 3.241, 3.241 | 3.255, 3.253 | not on the NPU |
| Chunk 0: down (K=14336), or merged kernel | not on the NPU | **6.173** | **13.182** |
| Chunk 1: first attention call | 1.002 | **3.877** | **3.919** |
| Chunk 1: second, third attention calls | 0.330, 1.001 | 0.352, 1.015 | 0.383, 1.045 |
| Chunk 1: gate, up | 3.233, 3.235 | 3.247, 3.241 | not on the NPU |
| Chunk 1: down (K=14336), or merged kernel | not on the NPU | **6.161** | **13.103** |
| Decode step | 0.646 | 0.646 | 0.645 |

Only the bold calls change between the columns (code-confirmed why: the native `minimal_pdi` cache key is K, tile size,
column and dtype, so all K=4096 prefill work shares one context, and native down at K=14336 or the XRT merged kernel needs
another). The merged kernel alone waits 11.098 ms in `../2026-11-02-fix-step2/`.

## What this settles

| Context change | Cost per change | Evidence |
| --- | ---: | --- |
| Back to native K=4096 after down | +2.83, +2.88 ms | chunk 0, chunk 1 |
| Back to native K=4096 after the merged kernel | +2.95, +2.92 ms | chunk 0, chunk 1 |
| Native K=4096 to native K=14336 | +2.93 ms | 6.17 ms vs 3.24 ms, same MAC count; assumes the same rate |
| Native K=4096 to the merged kernel | +2.04 ms | 13.14 ms vs 11.10 ms, same kernel |

Per layer-chunk (estimated, arithmetic on these medians):

1. `heteromosaic_mdown`: 2.85 + 2.93 = 5.78 ms, or 370 ms per prefill. The measured net loss is 184 ms. So M-split down
   would save about 186 ms if both changes were free: 1518.3 - 370 = 1148 ms.
2. `merged_ffn_hetero`: 2.93 + 2.04 = 4.97 ms, or 318 ms per prefill: 1612.3 - 318 = 1294 ms. The Oct 14 stage sum gives
   a lower floor of 1227 ms, so 67 ms stay unexplained.
3. Why a change costs this much is a hypothesis: the driver loads the other PDI's instructions and tile state. It was not
   separated here.
4. Whether one PDI can serve K=4096 and K=14336, or the attention GEMMs and the merged graph, is unknown. No such artifact
   exists in `hw_bins/npu2` or in the deployment artifact.
