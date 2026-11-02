# 2026-11-02: per-phase timers of the merged FFN (AUP)

**Measured.** No 5 ms remains. In the model one merged FFN takes 12.60 ms beyond waiting for attention,
which is faster than the Oct 28 Python harness's 14.27-14.59 ms. Per-call phases add up to the measured
prefill: 64 x (0.519 + 20.422) ms = 1340 ms against 1339.6 ms.

Job `aup-runs/step2-phases`, 20:46-20:47 UTC (14:46-14:47 CST), build `4b31b72`, `sampled` check,
`HETEROMEGA_FFN_TIMERS=<dir>`. Follow-up `../2026-11-02-fix-step2b/` (`9dac460`) and
`../2026-11-02-fix-step2c/`. Timers add host waits, so these are diagnostics; the headline numbers are
from runs without timers.

## Phases, median ms per call

| Phase | Model, 1024 tokens (n=640, warmup dropped) | Smoke slot 0 concurrent (n=9) | Smoke slot 0 serial (n=8) | Oct 28 harness concurrent (100 samples per block) |
| --- | ---: | ---: | ---: | ---: |
| Since the previous call ended (host enqueues attention) | 0.519 | - | - | - |
| Wait for input (GPU attention and norm) | 7.820 | 0 | 0 | - |
| GPU branch alone | - | - | 12.370 | - |
| D2H | 0.133 | 0.344 | 0.125 | 0.267 / 0.269 |
| Input scan (`sampled`) | 0.113 | 0.094 | 0.091 | - |
| BO write + sync + run start + y sync + y read | 0.626 | 0.589 | 0.555 | 0.950 / 1.005 (dispatch IO minus wait) |
| NPU wait | 11.098 | 11.377 | 11.324 | 11.514 / 11.710 |
| Output scan (`sampled`) | 0.056 | 0.060 | 0.057 | - |
| H2D | 0.182 | 0.193 | 0.337 | 0.740 / 0.734 (H2D, sum and boundary wait) |
| GPU branch still running after the NPU path | 0.002 | 0.254 | 0.002 | - |
| Sum | 0.235 | 0.238 | 0.336 | in the row above |
| Whole call | 20.422 | 13.314 | 25.067 | 14.274 / 14.585 |
| Whole call minus input wait | 12.602 | 13.314 | - | 14.274 / 14.585 |

Oct 28 source: `b0_p6400.json`, `b1_p6400.json` in `docs/runs/2026-10-28-merged-npu/` on
`origin/codex/npu-merged-kernel-parallel` (`e6ca183`), medians of `samples[].diagnostics` here.

## What this settles

1. **The 5 ms candidates are gone (measured).** Per-layer weight BO switching: NPU wait is
   11.01-11.36 ms across all 32 layers, with the layer changing on every call; the Oct 28 harness used one
   layer at 11.51-11.71 ms. Copies, BO work and the sum total 1.18 ms. Inside the call, apart from the two
   waits, no single phase exceeds 0.24 ms.
2. **The FFN is balanced, just on the NPU side (measured).** NPU path up to the y read and output scan: about
   12.03 ms. GPU branch alone: 12.37 ms in the smoke. When the NPU path ended, the GPU branch was already
   done (median wait 0.002 ms, no layer median above 0.025 ms), but the margin is under 1 ms.
3. **The rest of the layer-chunk is attention (measured).** 7.82 ms of the 20.42 ms is GPU attention.
4. **Layer 1 / stream 1 loses overlap only without the input wait (measured).** Smoke slot 1 concurrent:
   25.53 ms in step 1 without timers, 13.03 ms with timers. The one timer-only wait before the D2H copy is
   on input readiness. Commit `9dac460` makes it unconditional, and without timers slot 1 is now 13.50 ms
   (`../2026-11-02-fix-step2b/smoke-sampled.json`). Why this wait restores overlap is a hypothesis
   (a shared HIP hardware queue or a delayed flush); it is not established here.
5. **That effect does not reach the single-chunk model path (measured).** With one inflight chunk all 640
   calls use slot 0. Two inflight chunks use both slots; see `../2026-11-02-fix-step2b/README.md`.
