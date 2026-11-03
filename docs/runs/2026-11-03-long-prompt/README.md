# 2026-11-03: `merged_ffn` prefill at 2048 and 4096 tokens (AUP)

**Measured.** `merged_ffn` is not limited to 1024 tokens. P8192 is 1.85x faster than `gpu_original` at 2048 tokens and 1.84x at 4096 tokens, and its output stays near the GPU-only control. At 4096 tokens all switches do not fit in GTT; without `HETEROMEGA_GEMM_UNCACHED` it fits, with a margin of 5 MB in the worst process.

`heteromosaic` with its shipped groups is slower than P8192 at both lengths, by 1.13x and 1.09x, but its output deviates strongly at these lengths. These ratios therefore do not compare equal outputs.

## Timing (measured)

Prefill, median of 9 samples from 3 processes, except `gpu_original`: 3 samples from 1 process. Same job, same build, same sampled prompt (a prefix of 4160 tokens, seed 1), two inflight chunks unless stated. New kernels and switches are on for `merged_ffn` and off for the other two. All columns are in `summary.csv`.

| Method | 2048 tokens, ms | ms per token | 4096 tokens, ms | ms per token |
| --- | ---: | ---: | ---: | ---: |
| `gpu_original` | 3790.9 | 1.851 | 8184.7 | 1.998 |
| `heteromosaic`, shipped default: chunks of 1024 / 2048 rows | 2318.0 | 1.132 | 4863.9 | 1.187 |
| `heteromosaic`, shipped schedule: four 512-row chunks, 4 inflight | 2460.4 | 1.201 | not run | |
| `merged_ffn` P8192 | 2053.9 | 1.003 | 4452.3 | 1.087 |
| `merged_ffn` P6400 | 2107.7 | 1.029 | 4484.9 | 1.095 |
| `merged_ffn` P7168 | 2140.2 | 1.045 | 4602.6 | 1.124 |

The `merged_ffn` rows at 2048 tokens use all switches. The rows at 4096 tokens omit `HETEROMEGA_GEMM_UNCACHED`. P8192 at 1024 tokens on this prompt is 1010.4 ms (0.987 ms per token; 996-1012 ms by process), against 1005.5 ms on the fixed reference prompt in `docs/runs/2026-11-03-gemm/`, so the prompt has little effect on speed.

Native NPU calls per process: 896 at 2048 tokens and 1792 at 4096 tokens for `merged_ffn`, 8288-11200 for `heteromosaic`. Uncached fallbacks: 0 in every `merged_ffn` run reported above.

The `heteromosaic` schedule drifted upward within the job, 2382.7, 2460.4 and 2586.5 ms by process. The cause is not known. Its other setting stayed within 0.4%.

Whole request, prefill plus 64 decoded tokens: 5.80 / 8.37 s for P8192, 6.22 / 8.96 s for `heteromosaic`, 7.59 / 12.19 s for `gpu_original`.

## Numerics (measured)

Against `gpu_original` at the same length and in a fresh process. `gpu_chunk*` is `gpu_original` with larger chunks; it shows how far a harmless change moves the output. 256 scored positions at 2048 tokens; none at 4096 tokens because the prompt ends at 4160. Seed 1 / seed 2 where two values appear; 4096 tokens has seed 1 only.

| Run | Tokens | Step 1 relative L2 | Top-5 overlap, step 1 | Greedy, 64 tokens | Argmax agreement, scored | Mean KL, scored, nats |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpu_chunk1024` | 2048 | 0.0030 | 5 | 64 | 96.5% | 0.00006 |
| `merged_ffn` P8192, all switches | 2048 | 0.0058 / 0.0055 | 5 / 5 | 30 / 64 | 96.9% / 98.8% | 0.00007 / 0.00014 |
| `merged_ffn` P7168, all switches | 2048 | 0.0043 | 5 | 64 | 96.9% | 0.00006 |
| `merged_ffn` P6400, all switches | 2048 | 0.0049 | 5 | 63 | 96.9% | 0.00006 |
| `heteromosaic`, 1024-row chunks | 2048 | 0.2294 / 0.1386 | 2 / 0 | 0 / 0 | 58.6% / 62.5% | 0.00832 / 0.01281 |
| `heteromosaic`, four 512-row chunks | 2048 | 0.0644 / 0.0239 | 3 / 3 | 0 / 18 | 64.8% / 84.4% | 0.00834 / 0.00643 |
| `gpu_chunk2048` | 4096 | 0.0029 | 5 | 64 | | |
| `merged_ffn` P8192, without `GEMM_UNCACHED` | 4096 | 0.0041 | 4 | 64 | | |
| `merged_ffn` P7168, without `GEMM_UNCACHED` | 4096 | 0.0040 | 4 | 64 | | |
| `merged_ffn` P6400, without `GEMM_UNCACHED` | 4096 | 0.0055 | 5 | 64 | | |
| `heteromosaic`, 2048-row chunks | 4096 | 0.2932 | 1 | 0 | | |

For comparison, archived `heteromosaic` at 1024 tokens, on the first 1024 tokens of these two prompts: relative L2 0.0204 / 0.0619, top-5 overlap 4 / 5, greedy 64 / 4, argmax agreement 93.6% / 75.2%. So its deviation already exists at 1024 tokens and is larger with the groups of the longer prompts. At 4096 tokens its greedy output repeats a two-token loop.

P8192 at 2048 tokens, seed 1, first differs from the reference at token 30 and P6400 at token 0 with the other 63 positions equal. Both have a KL within a factor of 1.2 of `gpu_chunk1024`. **Hypothesis:** these are near ties, as with the 96.5% of `gpu_chunk1024`. Not separated.

**The cause of the `heteromosaic` deviation is not determined.** `gpu_chunk*` shows that the chunk size alone does not cause it. **Hypotheses, not tested:** the artifacts for larger chunks, `down` on the NPU in these groups, or this repository's runtime port of the original code. So this result describes `heteromosaic` on this box with this port, not necessarily the original work.

## Memory (measured)

GTT limit 11818 MB. Peak of the sampled `mem_info_gtt_used` per case, every 0.5 s.

| Case | 2048 tokens, MB | 4096 tokens, MB |
| --- | ---: | ---: |
| `gpu_original` | 8471-8795 | 9967 |
| `heteromosaic` | 8510-9009 | 10042 |
| P8192, all switches | 10981-11067 | failed at 11818, `out of memory` |
| P8192, without `GEMM_UNCACHED` | not run | 11753-11813 |
| P8192, without `SDPA` and `SDPA_TO_O` | not run | failed at 11814, `out of memory` |
| P8192, without all three | not run | 11523 |
| P7168 at the switches of the column above | 11181 | 11330-11440 |
| P6400 at the switches of the column above | 11385-11479 | 11446-11796 |

Without all three switches, P8192 at 4096 tokens took 5271.4 ms in one sample, against 4521.8 ms without `GEMM_UNCACHED` in the same stage. I estimated that both `GEMM_UNCACHED` and the V tiles had to go at 4096 tokens; dropping `GEMM_UNCACHED` alone was enough, so that estimate was wrong.

## Limits

- One prompt for timing, sampled from the model at temperature 1. Two prompts for numerics at 2048 tokens, one at 4096 tokens. They do not establish accuracy. No natural text.
- `heteromosaic` was not run at 1024 tokens on this prompt, at the user's earlier request, so there is no paired ratio at 1024 tokens. The 1.30-1.35x of `docs/runs/2026-11-03-gemm/` is against archived records and is not comparable with the paired 1.13x and 1.09x here.
- `heteromosaic` runs its shipped kernels; new kernels apply only to `merged_ffn`. Chunks of `merged_ffn` stay at M=512; larger artifacts were not tried.
- `HETEROMEGA_FFN_FINITE_CHECK=sampled`. On the HX 370 only. Energy was not measured.
- The margin of 5 MB at 4096 tokens is not usable in practice. No process failed in the 5 runs of P8192 without `GEMM_UNCACHED`, but a small extra allocation would end the run.

## Files

`summary.csv`: pooled rows, from `tools/aup-kit/jobs/long-prompt/summarize.py`. `results/`: result JSON of each stage. `numerics/`: comparison JSON. `prompts/`: the two sampled prompts and their logs. `runlogs/`: `long-probe`, `long-timing` (4096 probe, 2048 timing), `long-timing4096` (4096 timing, numerics), `long-numerics2` (merged at 4096, seed 2). Jobs: `tools/aup-kit/jobs/long-prompt/`, 07:17-10:10 CST Nov 3. The build is the one of `gemm-final`; C++ sources are unchanged since `ebee3bfb`.
