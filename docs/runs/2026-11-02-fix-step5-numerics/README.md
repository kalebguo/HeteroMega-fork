# 2026-11-02: NPU error probes and token and logit agreement (AUP)

**Measured.** The native NPU GEMM's error grows with K, reaching relative L2 0.285 at K=14336, while the GPU control stays at
0.0017. On the fixed reference prompt every method agrees on all 64 tokens, but this prompt is too easy to test numerics, so
`../2026-11-02-fix-step5b-sampled/README.md` repeats the comparison on harder prompts.

Job `aup-runs/step5-numerics`, 00:45-00:57 UTC Nov 3 (18:45-18:57 CST Nov 2), commit `6291e3d`, C++ build `9dac460`, default
`full` check. Steps of `tools/npu_error_probe/aup_job.sh` on this tree, plus `path_logits.py` for the native paths.

## A. Native GEMM against FP32 (`probe.jsonl`, `verdicts.json`)

M=512, all rows on the NPU through the original M-split dispatcher; GPU control on the same packed bytes; inputs per
`probe_cases.py`. e = mean absolute error, r = e_npu / e_gpu. Repeat runs are bit-exact in every case.

| Case | K | N | NPU relative L2 | GPU relative L2 | e_npu | e_gpu | r | Max abs NPU - GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K sweep | 1536 | 2048 | 0.0534 | 0.0017 | 0.298 | 0.0071 | 42 | 1.125 |
| K sweep | 2048 | 2048 | 0.0649 | 0.0017 | 0.425 | 0.0084 | 51 | 1.375 |
| K sweep | 10240 | 2048 | 0.2186 | 0.0017 | 3.142 | 0.0188 | 167 | 10.375 |
| K sweep | 11264 | 2048 | 0.2312 | 0.0017 | 3.540 | 0.0199 | 178 | 13.500 |
| K sweep | 4096 | 4096 | 0.1049 | 0.0017 | 0.976 | 0.0119 | 82 | 3.250 |
| K sweep | 14336 | 4096 | 0.2854 | 0.0017 | 4.862 | 0.0222 | 219 | 17.750 |
| Sign, all positive | 2048 | 2048 | 0.0649 | 0.0017 | 0.425 | 0.0084 | 51 | 1.375 |
| Sign, all negative | 2048 | 2048 | 0.0206 | 0.0017 | 0.122 | 0.0084 | 15 | 0.875 |
| Sign, mixed | 2048 | 2048 | 0.0533 | 0.0023 | 0.043 | 0.0018 | 24 | 0.297 |
| Metadata, equal scales, zeros 8 | 2048 | 2048 | 0.0183 | 0.0016 | 0.056 | 0.0044 | 13 | 0.219 |
| Metadata, equal scales, random zeros | 2048 | 2048 | 0.0299 | 0.0016 | 0.185 | 0.0082 | 23 | 1.000 |
| Metadata, random scales, zeros 8 | 2048 | 2048 | 0.0828 | 0.0017 | 0.221 | 0.0037 | 60 | 0.438 |
| Metadata, random scales and zeros | 2048 | 2048 | 0.0649 | 0.0017 | 0.425 | 0.0084 | 51 | 1.375 |

The decision rule in `probe_report.py`, fixed before this data, supports three causes at once:

1. **Accumulation or partial-sum rounding:** log-r slope against log K is 0.73 at N=2048 and 0.78 at N=4096 (limit 0.25). The raw
   error slope is 1.24-1.28, above linear.
2. **Sign handling:** r(negative) / r(positive) = 0.288, and NPU(-a) is not the exact negation of NPU(a).
3. **Metadata, factor scales:** random scales give 4.7x the r of equal scales.

Against the CPU emulation written before this run (`tools/npu_error_probe/README.md`, estimated): BF16 partials every K16 predicted
the 1.25 maximum at K=2048 (measured 1.375), but r of 7-20 (measured 42-178), and it is sign-symmetric. **Hypothesis, not
tested:** the native binary rounds partial sums asymmetrically, for example by truncation. The binary's source is not in
this repository.

## B. Token and logit agreement on the fixed reference prompt (`token-agreement.json`, `path-comparison.json`)

Against `gpu_original`, 1024-token prefill unless stated. A second process of every method reproduces its logits exactly.

| Method | Step-1 logits max abs | Step-1 relative L2 | Argmax equal | Greedy tokens equal | Perplexity, 512-token prefill, next 512 scored |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gpu_original` | 0 | 0 | yes | 64 / 64 | 1.0007 |
| `merged_ffn` | 0.469 | 0.0376 | yes | 64 / 64 | 1.0007 |
| `heteromosaic` | 2.406 | 0.1550 | yes | 64 / 64 | 1.0014 |
| `merged_ffn_hetero` | 1.906 | 0.1227 | yes | 64 / 64 | 1.0019 |

Native execution with `exec_fail=0`: 2880 (`heteromosaic`), 2688 (`merged_ffn_hetero`), 192 (`merged_ffn`) submissions at 1024 tokens.

**Limitation (measured):** this prompt has 12 distinct tokens and a period of 10, so every path predicts it with a perplexity
near 1. All "tokens match" results in these runs, and in earlier ones, are weak evidence of accuracy.
