# 2026-11-02: native M-split down on heteromosaic (AUP)

**Measured.** Native M-split down makes `heteromosaic` slower, not faster: 1518.3 ms against 1333.8 ms per 1024-token
prefill (0.878x, +184 ms). The earlier estimates of a 12.3 ms FFN and "up to 200 ms saved" did not hold with the existing
artifact, because that artifact needs a third NPU context (`../2026-11-02-fix-step3c/README.md`). It also fails numerically
on harder prompts (`../2026-11-02-fix-step5b-sampled/README.md`).

Job `aup-runs/step4-mdown`, 21:33-21:38 UTC (15:33-15:38 CST), commit `4924976`, C++ build `9dac460`. Settings:
`tools/split_model/make_mdown_config.py` enables the four `down` entries at prompt length 1024 with `npuM: 256` and
`hw_bins/npu2/16384x14336x4096/bf16_int4AWQ_bf16_M/` (`mdown.config.json` here). `evaluate_prefill.py`, synchronized
`generate(1)`, 10 samples per cell, 2 processes per method in ABBA order, `count_ioctls.so` preloaded.

| Method | Prefill median ms (min-max) | Vs heteromosaic | NPU submissions per process | Hardware contexts |
| --- | ---: | ---: | ---: | ---: |
| `heteromosaic` | 1333.8 (1284.6-1363.5) | 1.000x | 13472 | 2 |
| `heteromosaic_mdown` | 1518.3 (1510.5-1579.2) | 0.878x | 14176 | 3 |

## Checks (measured)

1. +64 NPU calls per prefill confirmed: 14176 - 13472 = 704 = 64 x 11 passes, with `exec_fail=0`.
2. Tokens of both methods are the same over 22 passes, and both match the reference at 1024 tokens. This does not
   show that M-split down is accurate: the reference prompt is too easy. On model-sampled prompts its first predicted token
   is wrong (`../2026-11-02-fix-step5b-sampled/README.md`).
3. All decode rows keep their previous path; only prefill `down` entries at 1024 changed.
