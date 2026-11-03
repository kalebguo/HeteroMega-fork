# 2026-11-02: the first evaluation's cost, attributed (AUP)

**Measured.** The two finite scans account for the whole gap. Without optimization each original scan takes
9.8-10.2 ms in the model, not the 7.16 ms that the CPU-only test gave. That difference is the "remaining 5 ms".

Job `aup-runs/step2c`, 21:39-21:41 UTC (15:39-15:41 CST). Tree `/notebooks/hm-fix-noopt`: commit `4924976`,
with the `-O3` line removed and the original per-element loop restored (`tools/aup-kit/jobs/step2c/cmd.sh`), full
check, timers on.

| Median ms per call | No `-O`, original loop | `-O3`, `sampled` (`../2026-11-02-fix-step2/`) | Difference |
| --- | ---: | ---: | ---: |
| Input scan, model | 9.836 | 0.113 | 9.723 |
| Output scan, model | 10.200 | 0.056 | 10.144 |
| NPU wait, model | 11.417 | 11.098 | 0.319 |
| Whole call, model | 40.416 | 20.422 | 19.994 |
| Input / output scan, smoke | 8.984 / 9.388 | 0.094 / 0.060 | |
| Whole call, smoke slot 0 concurrent | 31.36 | 13.31 | 18.05 |

The two scan rows sum to 19.87 ms of the 19.99 ms difference per call.

This tree reproduces the first evaluation: 2615.3 ms per 1024-token prefill with timers (5 samples), against
2558.7 ms without timers. The smoke's 31.36 ms compares with 33.02 ms.

Why a scan is 33% slower in the model than in the CPU-only test is a hypothesis: the CPU shares package power and
DRAM with the GPU while the GPU branch runs. It was not separated here.
