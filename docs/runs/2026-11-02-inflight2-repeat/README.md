# 2026-11-02: P8192 against heteromosaic at two inflight chunks, repeat (AUP)

**Measured.** `merged_ffn_p8192` is faster than `heteromosaic` by 56.2 ms, 1.044x: 1274.8 ms against 1331.0 ms. All 6 P8192 process
medians are below all 6 `heteromosaic` process medians. The smallest gap between any two is 23.9 ms.

Job `aup-runs/inflight2-repeat`, 03:35-03:48 UTC Nov 3 (21:35-21:48 CST Nov 2), commit `b035e2e`, C++ build `9dac460`. Both methods at
`--inflight 2`, 1024 tokens, `sampled` check, `count_ioctls.so` preloaded. 6 processes per method in the order A B B A A B B A A B B A, 5
samples per process, nothing else varied, no process dropped.

| Process | `heteromosaic` median ms | `merged_ffn_p8192` median ms |
| ---: | ---: | ---: |
| r0 | 1300.0 | 1276.1 |
| r1 | 1331.3 | 1273.6 |
| r2 | 1333.3 | 1273.8 |
| r3 | 1329.5 | 1273.6 |
| r4 | 1336.4 | 1276.0 |
| r5 | 1333.2 | 1273.0 |
| Pooled, 30 samples | 1331.0 (min 1283.4) | 1274.8 (min 1264.3) |
| Spread of process medians | 36.4 | 3.1 |

## Checks (measured)

1. NPU submissions per process, `exec_fail=0`: 13472 and 704 in all 12 processes.
2. One 64-token sequence per method over all processes; the methods agree with each other, and every process matches the reference.
3. The first sample of a process is usually the slowest for P8192, up to 1604.0 ms. It is kept in the pooled median.

## Limits

- `heteromosaic` varies more than P8192 between processes, and its low processes reach 1277.6 ms (`../2026-11-02-inflight2-timing/`) and
  1300.0 ms (here). On such a process the gap is 8 to 24 ms.
- This compares latency at two inflight chunks only. At one inflight chunk P8192 is slower than both P6400 and `heteromosaic`.

## Latest code against the archived heteromosaic records

`latest-vs-heteromosaic-records.csv` compares `merged_ffn` P6400 and P8192 at this build with every `heteromosaic` record in
`docs/runs`, without running `heteromosaic` again. Generate it with `tools/split_model/heteromosaic_records.py`. Records from runs with a
tracer are left out. These rows are not paired, and the records themselves range over 1309.6-1355.5 ms.

| `heteromosaic` record | Inflight | Record ms | P8192, inflight 2, 1274.8 ms: gain, speedup | P6400, inflight 1, 1340.9 ms: gain, speedup |
| --- | ---: | ---: | ---: | ---: |
| `2026-11-02-fix-step3c/head-to-head` | 1 | 1317.6 | 42.8 ms, 1.034 | -23.3 ms, 0.983 |
| `2026-11-02-fix-step3-h2` | 1 | 1322.3 | 47.5 ms, 1.037 | -18.5 ms, 0.986 |
| `2026-11-02-fix-step4-mdown` | 1 | 1333.8 | 59.0 ms, 1.046 | -7.1 ms, 0.995 |
| `2026-11-02-inflight2-timing/inflight1-1024` | 1 | 1355.5 | 80.7 ms, 1.063 | 14.7 ms, 1.011 |
| `2026-11-02-inflight2-timing/inflight2-1024` | 2 | 1309.6 | 34.8 ms, 1.027 | -31.3 ms, 0.977 |
| `2026-11-02-inflight2-repeat/repeat` | 2 | 1331.0 | 56.2 ms, 1.044 | -9.9 ms, 0.993 |

An earlier two-inflight record from October, 1316.6 ms in `docs/runs/2026-10-14-profile/evidence/baseline-summary.csv`, has another
format, so the script does not read it. Against it P8192 gains 41.8 ms, 1.033.
