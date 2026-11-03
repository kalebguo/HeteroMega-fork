# 2026-11-02: two-inflight comparison, bottleneck timers (AUP)

**Measured.** Neither new configuration reaches its estimate, and neither device is saturated at two inflight chunks. Config 3 moved the
work as expected: gate and up waits are 6.4 ms instead of 3.24 ms. The NPU is then busy for 74.8% of the prefill and idle for the rest.
P8192 removes most of the wait on the GPU branch, and its cycle of 19.85 ms stays above the GPU load of about 17.6 ms (estimated).

Job `aup-runs/inflight2-diag`, 03:10-03:16 UTC Nov 3 (21:10-21:16 CST Nov 2), commit `9c86438`, C++ build `9dac460`. Separate from the
timing job, because timers and tracers change latency. Config 3 uses the 8192 family's artifact.

## Merged path, median ms per FFN call (`*-phases.json`, 640 calls)

| Phase | P6400, inflight 2 | P8192, inflight 2 | P8192, inflight 1 |
| --- | ---: | ---: | ---: |
| Input wait: attention, and the other chunk's GPU work | 18.24 | 16.41 | 7.91 |
| NPU wait | 11.04 | 14.04 | 14.30 |
| GPU wait after the NPU path | 10.48 | 2.42 | 0.00 |
| Whole call | 41.77 | 39.69 | 23.43 |
| Whole call / inflight chunks | 20.89 | 19.85 | 23.43 |
| Prefill of this diagnostic run, median | 1358.3 | 1276.4 | 1535.3 |

## Native path, NPU busy time per prefill (`npu-busy.json`, the last 10 prefills, 288 calls each)

| Method, inflight | Span ms | NPU busy ms | NPU idle ms | Busy | gate/up calls over 5 ms, and their sum |
| --- | ---: | ---: | ---: | ---: | --- |
| `heteromosaic`, 1 | 1323.3 | 586.0 | 736.4 | 44.3% | none |
| `heteromosaic`, 2 | 1305.5 | 588.6 | 717.4 | 45.1% | none |
| Config 3, 1 | 1692.3 | 1003.8 | 689.9 | 59.3% | 128, 825.2 ms |
| Config 3, 2 | 1322.9 | 987.7 | 334.2 | 74.8% | 128, 816.0 ms |

`trace-busy.json` has the waits by call position at inflight 1. Config 3: chunk 0 [1.93, 1.04, 6.39, 6.40], chunk 1 [1.02, 0.35, 1.02, 6.36,
6.37] ms. `heteromosaic`: chunk 0 [1.90, 1.00, 3.23, 3.24], chunk 1 [1.00, 0.33, 1.00, 3.23, 3.23] ms. The traces stay on the box in
`/notebooks/hm-results/2026-11-02/inflight2-diag/`. `tools/aup-kit/jobs/inflight2-diag/npu_busy.py` computes the busy table.

## Against the estimates (estimated arithmetic on these medians)

1. The estimates assumed the prefill equals max(GPU, NPU) per layer-chunk times a factor of 1.008-1.046.
2. P8192: GPU 17.62 ms and NPU about 15.4 ms including host work, so 1140-1180 ms. The cycle is 19.85 ms, a factor of 1.127 over the GPU load.
   That leaves 2.2 ms per layer-chunk, or about 140 ms per prefill, unattributed. At P6400 the same difference is about 0.7 ms.
3. Config 3: GPU 14.12 ms and NPU 15.65 ms, so 1010-1050 ms. The NPU part holds: 987.7 ms is 15.4 ms per layer-chunk. The cycle is 20.98 ms, so
   the NPU is idle for 5.2 ms per layer-chunk.

## Not established

- **Hypothesis:** the factor grows as the two devices approach balance, because one chunk's next step then often waits on the device that the
  other chunk holds. No GPU busy timer exists, so GPU idle time is not measured.
- **Hypothesis:** the GPU runs slower while the NPU is busy, through shared DRAM or the power budget. `kernels/gpu/README.md` records that the
  bandwidth under an NPU load was not measured.
