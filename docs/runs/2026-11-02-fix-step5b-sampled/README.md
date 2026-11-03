# 2026-11-02: numerics on model-sampled prompts (AUP)

**Measured.** `merged_ffn` stays closest to `gpu_original`. `heteromosaic` deviates about 4x more in logit L2 and 20-30x more
in KL. Native M-split down fails: on both seeds its first predicted token is wrong, and it shares no top-5 token with the
reference. The earlier token checks did not show this, because the fixed reference prompt is too easy
(`../2026-11-02-fix-step5-numerics/README.md`).

Job `aup-runs/step5b-sampled`, 01:09-01:27 UTC Nov 3 (19:09-19:27 CST Nov 2), commit `7016bf6`, C++ build `9dac460`, default
`full` check. `tools/npu_error_probe/path_logits.py`, one fresh process per method. `sampled-1.json` and `sampled-2.json`:
1536 tokens sampled from `gpu_original` at temperature 1 from BOS, seeds 1 and 2. Each method prefills the first 1024 and
then scores the next 512 with single-token decode steps. All numbers are against the first `gpu_original` process.

## Results, seed 1 / seed 2

| Method | Step-1 logits relative L2 | Step-1 KL, nats | Step-1 argmax equal | Step-1 top-5 overlap | Greedy tokens equal, of 64 | Mean KL over 512 positions, nats | Argmax agreement over 512 positions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpu_original`, second process | 0 / 0 | 0 / 0 | yes / yes | 5 / 5 | 64 / 64 | 0 / 0 | 100% / 100% |
| `merged_ffn` | 0.0050 / 0.0173 | 0.00016 / 0.00069 | yes / yes | 5 / 5 | 64 / 9 | 0.00008 / 0.00006 | 99.2% / 97.9% |
| `heteromosaic` | 0.0204 / 0.0619 | 0.00441 / 0.01352 | yes / yes | 4 / 5 | 64 / 4 | 0.00262 / 0.00151 | 93.6% / 75.2% |
| `merged_ffn_hetero` | 0.0187 / 0.0508 | 0.00539 / 0.01256 | yes / yes | 4 / 4 | 64 / 4 | 0.00242 / 0.00121 | 94.7% / 76.8% |
| `heteromosaic_mdown` | 0.3281 / 0.5661 | 0.63951 / 1.19736 | **no / no** | **0 / 0** | **0 / 1** | 0.01300 / 0.01461 | 81.6% / 37.1% |

`comparison-1.json` and `comparison-2.json` have every field. The per-position logits (262 MB per run) stayed on the box
only until the comparison ran.

## Checks (measured)

1. Native execution per process, with `exec_fail=0`: 19520 (`heteromosaic`), 19264 (`merged_ffn_hetero`), 19776
   (`heteromosaic_mdown`), 256 (`merged_ffn`), 0 (`gpu_original`). These equal 4 prefills at 288 / 224 / 352 / 64 calls, plus
   574 decode steps at 32 calls on the native methods.
2. Control: a second `gpu_original` process reproduces all logits exactly on both seeds.

## How to read this

1. These prompts are a stress test, not natural text. Mean NLL is 10.29 / 10.12 nats, close to a uniform 11.76 nats, and
   1456 / 1321 of 1536 tokens are distinct. Sampling at temperature 1 appears to run away to near-random tokens; the box has
   no tokenizer, so the text was not read. Do not read perplexity here as model quality.
2. Argmax agreement and greedy tokens are over-sensitive on nearly flat distributions. KL and step-1 L2 are steadier. They
   are also what separates the methods by the largest factor.
3. Scores after the first position include each path's decode: the NPU on native methods, the original GPU FFN on GPU-attention
   methods. Only the step-1 columns compare prefill alone.
4. `merged_ffn_hetero` is about equal to `heteromosaic` on every column. So native attention dominates the native path's
   deviation, and most of `merged_ffn`'s lead comes from GPU attention (estimated from this comparison; there is no
   configuration of native gate/up with GPU attention). Per FFN alone, Oct 28 measured relative L2 0.0148 for merged
   against 0.1167 for native on synthetic weights.
5. M-split down is the largest single deviation. It agrees with the K=14336 native GEMM probe: relative L2 0.285 and maximum
   absolute error 17.8 on synthetic inputs (`../2026-11-02-fix-step5-numerics/README.md`).
6. Two seeds only. No real-text perplexity or task accuracy was measured.
