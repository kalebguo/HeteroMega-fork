# 2026-11-02: H2, heteromosaic vs merged_ffn_hetero (AUP)

**Measured.** With native NPU attention, the merged FFN loses: 1612.3 ms against 1322.3 ms per 1024-token prefill
(0.820x). `../2026-11-02-fix-step3c/README.md` shows why: each NPU context change costs 2-3 ms, and this
combination changes context twice per layer-chunk.

Job `aup-runs/step3-h2`, 21:27-21:32 UTC (15:27-15:32 CST), commit `9dac460`, `sampled` check, Llama 3.1 8B AWQ INT4,
P6400 merged artifact. `evaluate_prefill.py`, synchronized `generate(1)`, 10 samples per cell, 2 processes per method
in ABBA order, 1 warmup excluded, `count_ioctls.so` preloaded.

| Method | Prefill median ms (min-max) | Vs heteromosaic | NPU submissions per process | Hardware contexts |
| --- | ---: | ---: | ---: | ---: |
| `heteromosaic` | 1322.3 (1291.4-1379.1) | 1.000x | 13472 | 2 |
| `merged_ffn_hetero` | 1612.3 (1594.3-1641.9) | 0.820x | 12768 | 3 |

## Checks (measured)

1. Native execution: `exec_ok` is 13472 and 12768 with `exec_fail=0`. No silent GPU fallback.
2. The difference is -704 submissions, which is 64 per prefill x 11 passes: the merged FFN makes 64 calls, and native
   gate and up make 128.
3. Merged completions: 1408 in 22 passes (64 per pass), 0 wrong.
4. Tokens: both methods match each other, and both match the reference at 1024 tokens.
5. This timer pairs only these two methods. The GPU-attention head-to-head is in `../2026-11-02-fix-step3c/`.
