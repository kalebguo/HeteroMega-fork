# 2026-11-02: two-inflight comparison, timing (AUP)

**Measured.** At two inflight chunks, `merged_ffn` P8192 is the fastest of the four configurations: 1269.2 ms. Its cost shows at one
inflight chunk: 1534.6 ms at 1024 tokens and 747.5 ms at 512 tokens. Config 3 gains nothing over `heteromosaic` at either setting.

Job `aup-runs/inflight2-timing`, 02:41-03:09 UTC Nov 3 (20:41-21:09 CST Nov 2), commit `9c86438`, C++ build `9dac460`, `sampled` check,
`--gpu-chunk-size 512`, `count_ioctls.so` preloaded, no timers and no power sampling. 10 samples per cell: 2 processes per method, 5
samples per process, order A B C D D C B A. Config 3 uses the 8192 family's artifact (`../2026-11-02-inflight2-prep/README.md`).

| Method, 1024 tokens | Inflight 2, median (min-max) ms | Inflight 1, median (min-max) ms | Estimate before the run (inflight 2 / inflight 1) |
| --- | ---: | ---: | --- |
| `heteromosaic` | 1309.6 (1259.8-1337.0) | 1355.5 (1337.4-1379.9) | |
| `heteromosaic_gateup` (config 3) | 1342.5 (1337.5-1358.8) | 1748.8 (1705.4-1779.6) | 1010-1050 / none |
| `merged_ffn` P6400 | 1355.5 (1346.1-1837.2) | 1340.9 (1335.6-1386.9) | |
| `merged_ffn_p8192` | 1269.2 (1263.2-1330.2) | 1534.6 (1524.4-1541.0) | 1140-1180 / about 1540 |

The estimates are from the task brief, not from this run. `tools/split_model/REPORT.md` predicted 1540 ms for P8192 at one inflight
chunk, 0.4% from the measurement. Both estimates for two inflight chunks are too low; `../2026-11-02-inflight2-diag/README.md` has the
timers.

| Method, 512 tokens, inflight 1 | Median (min-max) ms | vs `gpu_original` |
| --- | ---: | ---: |
| `gpu_original` | 895.0 (893.3-971.9) | 1.000 |
| `merged_ffn` P6400 | 649.3 (644.9-669.1) | 1.378 |
| `merged_ffn_p8192` | 747.5 (743.3-760.4) | 1.197 |

So P8192 costs 98.1 ms (15.1%) on a one-chunk prompt against P6400.

The 16384 family's artifact, timed apart after the paired cells and so not paired: 1378.8 ms (1371.2-1397.5) at inflight 2 and
1752.1 ms (1721.0-1803.0) at inflight 1. Its tokens are wrong; these are not results for a usable configuration.

## Checks (measured)

1. NPU submissions per process, `exec_fail=0`: `heteromosaic` and config 3 both 13472 at 1024 tokens, which equals 288 per prefill and 32
   per decode step; `merged_ffn` and P8192 both 704 at 1024 tokens and 352 at 512 tokens, which equals 64 and 32 per prefill. So config 3
   moved rows, not calls.
2. All four 1024-token methods give the same tokens over all passes and match the reference. The reference prompt is too easy to show
   accuracy; `../2026-11-02-inflight2-numerics/README.md` has the numerics.
3. The gap of 40 ms between P8192 and `heteromosaic` is not settled by this table: the process medians are 1277.6 and 1319.8 ms for
   `heteromosaic`, 1269.4 and 1269.0 ms for P8192. `../2026-11-02-inflight2-repeat/README.md` repeats it.
