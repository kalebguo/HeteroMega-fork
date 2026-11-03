# 2026-11-02: phase timers of merged_ffn_hetero (AUP)

**Measured.** With native NPU attention the time between two merged FFN calls is 10.31 ms, and the merged NPU wait
rises from 11.10 ms to 13.16 ms. The merged kernel itself did not change, so this points at the NPU context change.
`../2026-11-02-fix-step3c/README.md` measures that directly.

Job `aup-runs/step3b`, ended 21:39 UTC (15:39 CST), commit `4924976`, `sampled` check, timers on, 1 process, 1024 tokens.
Diagnostic only: prefill 1643.4 ms (1618.5-1652.7) with timers, against 1612.3 ms without timers in
`../2026-11-02-fix-step3-h2/`.

| Median ms per call | `merged_ffn_hetero` | `merged_ffn` (`../2026-11-02-fix-step2/`) |
| --- | ---: | ---: |
| Since the previous call ended | 10.311 | 0.519 |
| Wait for input | 0.203 | 7.820 |
| D2H | 0.150 | 0.133 |
| Input scan | 0.116 | 0.113 |
| BO write + sync + run start + y sync + y read | 0.636 | 0.626 |
| NPU wait | 13.156 | 11.098 |
| Output scan | 0.057 | 0.056 |
| H2D | 0.342 | 0.182 |
| Sum | 0.338 | 0.235 |
| Whole call | 15.059 | 20.422 |
| Whole call minus input wait (FFN proper) | 14.856 | 12.602 |

## What this settles (attention numbers: measured here; comparison: estimated)

1. Attention plus norm take about 10.51 ms per layer-chunk here (10.311 + 0.203). The native path takes about 6.57 ms in
   the Oct 14 synchronized profile (G1 2.48, A 2.47, O 1.48, norm 0.14).
2. FFN proper grows by 2.25 ms, and the NPU wait explains 2.06 ms of that.
3. Estimated floor without any context-change cost: (6.57 + 12.60) x 64 = 1227 ms. Measured: 1612.3 ms.
   `../2026-11-02-fix-step3c/` assigns 4.97 ms per layer-chunk (318 ms per prefill) of the 385 ms gap to context changes.
