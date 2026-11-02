# 2026-11-02: merged FFN with -O3, full vs sampled finite checks (AUP)

**Measured.** The merged GPU+NPU FFN is now faster than `gpu_original`: 1.35x at 1024 tokens with the
sampled finite check, 1.24x with the full check. The earlier 0.71x came from one host source file
that was compiled without optimization. Job `aup-runs/step1-fix`, 20:26-20:44 UTC (14:26-14:44 CST).

## Code and artifact

- Build tree `/notebooks/hm-fix`, branch `merged-ffn-fix`. C++ from `2ea8e6c`; the runner at `6dc9c35`
  (Python-only difference). `tools/aup-kit/jobs/build-fix/cmd.sh`, flags as in `build-main`, plus
  `-O3` on the `heteromega_merged_ffn` target (`flags.make` checked). `/notebooks/hm-eval-main/build`
  stays as the evidence of the first evaluation.
- Same P6400 artifact, same weights (`model_weights` symlink to the 5.4 GB file of the first evaluation).
- `HETEROMEGA_FFN_FINITE_CHECK`: `full` is every element of the input and the NPU output; `sampled` is
  every row at 1 in 64 columns. Same binary for both.

## Commands (`tools/aup-kit/jobs/step1-fix/cmd.sh`)

```bash
HETEROMEGA_FFN_FINITE_CHECK=$mode python -m experiments.ffn_deploy.smoke --config deployment.json --iterations 3 --output smoke-$mode.json
HETEROMEGA_FFN_FINITE_CHECK=$mode python3 kernels/gpu/tools/evaluate_prefill.py --model "$MODEL" --build build --out eval-$mode \
  --methods gpu_original merged_ffn --merged-ffn-config "$PWD/deployment.json" --gpu-chunk-size 512 --prompt-tokens 512 1024
```

GPU attention, M512 serial chunks, order original/new/new/original per length, 10 samples per cell.

## Paired prefill, synchronized `generate(1)` (median, min-max, ms)

| Build | Check | tokens | `gpu_original` | `merged_ffn` | speedup |
| --- | --- | ---: | --- | --- | ---: |
| `8e5c804`, no `-O` (first evaluation) | full | 512 | 889.3 (880.0-967.8) | 1257.1 (1248.1-1621.9) | 0.707 |
| `8e5c804`, no `-O` (first evaluation) | full | 1024 | 1829.0 (1825.6-1875.4) | 2558.7 (2543.4-2621.3) | 0.715 |
| this, `-O3` | full | 512 | 891.1 (882.4-966.8) | 709.2 (700.2-911.5) | 1.257 |
| this, `-O3` | full | 1024 | 1829.5 (1827.6-1877.7) | 1473.8 (1438.5-1480.6) | 1.241 |
| this, `-O3` | sampled | 512 | 894.4 (892.4-970.9) | 649.4 (644.6-1246.7) | 1.377 |
| this, `-O3` | sampled | 1024 | 1829.7 (1826.9-1876.4) | 1351.2 (1346.6-1400.6) | 1.354 |

Checks on every cell of this table's new rows:

- Merged FFN active: 32 / 64 completions per prefill pass, 22 passes per cell, 0 wrong. Controls clean (0).
- The 64 greedy tokens are identical to `gpu_original` in every repetition (`tokens_match_gpu=True`),
  and the 8 reference tokens match at 1024.
- `gpu_original` reproduces the first evaluation within 0.6% at both lengths, so the box did not drift.
- The first repetition of a process is sometimes slow (1246.7 ms at 512 tokens); medians are not affected.

## Measured against the estimates made before this run

| Configuration | 512 estimated | 512 measured | 1024 estimated | 1024 measured |
| --- | ---: | ---: | ---: | ---: |
| `-O3`, scans on the critical path | 856 | 709.2 | 1757 | 1473.8 |
| `-O3`, scans off the critical path | 799 | 649.4 | 1643 | 1351.2 |
| Oct 28 harness rate | 657 | - | 1359 | - |

The linear estimates were 17-22% too high. Sampled now runs at the Oct 28 harness rate. See
`../2026-11-02-fix-step2/README.md` for the phases.

## C++ smoke (`smoke-*.json`, wall ms per call, median)

| Layer / slot | Mode | full, n | full | sampled | No `-O` (`../2026-11-02-smoke-main`) |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 / 0 | concurrent | 9 | 14.57 | 13.13 | 33.02 |
| 0 / 0 | serial | 8 | 26.54 | 25.28 | 43.95 |
| 1 / 1 | concurrent | 5 | 25.53 | 25.53 | 44.53 |
| 1 / 1 | serial | 4 | 26.10 | 25.61 | - |

30 dispatches in both. Status `completed_with_accuracy_deviations`, 24 of 90 checks outside tolerance
as before; complete-output relative L2 median 0.0148, as on Oct 28 and Oct 31. Both modes reject a
whole NaN input row before dispatching (`finite_check.nan_row_rejected`). Layer 1 / slot 1 keeps the
missing overlap (25.53 ms is about its serial time).

## CPU cost of one scan (`run.log` of `aup-runs/build-fix`, warm memory, target's flags)

| Variant | Median ms per [512, 4096] scan |
| --- | ---: |
| `full` (`find_nonfinite_bf16`) | 0.415 |
| `sampled` | 0.0135 |
| `full_branchless` (same detection, vectorizable; not in the executor) | 0.105 |

In the model `full` costs 1.9 ms more per call than `sampled` (estimated from 1473.8 - 1351.2 ms over
64 calls). The phase timers measure 0.94 + 1.06 ms for the two scans in the model
(`../2026-11-02-fix-step2b/`); the CPU-only `full` figure varies between runs (0.41-0.90 ms).

## Why sampled, and what it can miss

Chosen over a GPU-side `isfinite`, which needs a blocking readback or more kernels on the same stream
and so stops the host from queueing the next layer. In the model the two sampled scans cost 0.169 ms
per call (measured with the phase timers, `../2026-11-02-fix-step2/`; 1 in 64 columns touches one cold
cache line per element) and do not block GPU work.

- Whole nonfinite row: caught in the same call.
- Isolated nonfinite element at an unsampled column of layer L: missed at L. Layer L+1's input has it
  in the whole row, because the Q/K/V and O GEMMs multiply it into every output channel of the row and
  the residual adds that to x (code-confirmed by reading; not injected on hardware). Caught at L+1.
- Same at the last layer: not caught by this check. It reaches the logits of that row only, and the
  token checks are the only remaining guard.
