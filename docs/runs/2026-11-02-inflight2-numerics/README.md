# 2026-11-02: two-inflight comparison, numerics (AUP)

**Measured.** Neither new configuration changes the picture. Config 3 stays at `heteromosaic`'s level of deviation, so the hypothesis
that more native rows raise its error is not supported at step 1. P8192 stays at `merged_ffn`'s level, with no consistent direction
between the two seeds. The 16384 family's artifact gives unusable output.

Job `aup-runs/inflight2-numerics`, 03:17-03:33 UTC Nov 3 (21:17-21:33 CST Nov 2), commit `b035e2e`, C++ build `9dac460`, default `full`
check, default one inflight chunk. `tools/npu_error_probe/path_logits.py`, one fresh process per method, on the two model-sampled
prompts of `../2026-11-02-fix-step5b-sampled/`, against the first `gpu_original` process.

## Results, seed 1 / seed 2

| Method | Step-1 logits relative L2 | Step-1 KL, nats | Step-1 argmax equal | Step-1 top-5 overlap | Greedy tokens equal, of 64 | Mean KL over 512 positions, nats | Argmax agreement over 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpu_original`, second process | 0 / 0 | 0 / 0 | yes / yes | 5 / 5 | 64 / 64 | 0 / 0 | 100% / 100% |
| `merged_ffn` P6400 | 0.0050 / 0.0173 | 0.00016 / 0.00069 | yes / yes | 5 / 5 | 64 / 9 | 0.00008 / 0.00006 | 99.2% / 97.9% |
| `merged_ffn_p8192` | 0.0073 / 0.0105 | 0.00032 / 0.00059 | yes / yes | 5 / 5 | 64 / 64 | 0.00009 / 0.00006 | 98.4% / 96.3% |
| `heteromosaic` | 0.0204 / 0.0619 | 0.00441 / 0.01352 | yes / yes | 4 / 5 | 64 / 4 | 0.00262 / 0.00151 | 93.6% / 75.2% |
| `heteromosaic_gateup` (config 3) | 0.0190 / 0.0620 | 0.00398 / 0.01637 | yes / yes | 4 / 5 | 2 / 5 | 0.00300 / 0.00164 | 92.8% / 72.1% |
| `heteromosaic_gateup16384` (16384 family's artifact) | 0.8258 / 0.9291 | 2.66625 / 3.98544 | **no / no** | **0 / 0** | **0 / 0** | 2.45837 / 2.96521 | 0.2% / 0.4% |

`comparison-1.json` and `comparison-2.json` have every field. The per-position logits stayed on the box only until the comparison ran.

## Checks (measured)

1. A second `gpu_original` process reproduces all logits exactly on both seeds.
2. `merged_ffn` and `heteromosaic` reproduce the step 5b values to the printed digits.
3. Native execution per process, `exec_fail=0`: 19520 on all three native methods, 256 on both merged methods, 0 on `gpu_original`.

## How to read this

These are the stress-test prompts of step 5b, not natural text; its "How to read this" section applies. Two seeds cannot establish a
direction for P8192. Argmax over 512 positions is lower for P8192 on both seeds, step-1 L2 is higher on seed 1 and lower on seed 2, and
mean KL is equal. Scores after step 1 include each path's decode, and `merged_ffn` decodes on the original GPU FFN at both splits.
