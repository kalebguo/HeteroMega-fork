# 2026-11-02: two-inflight comparison, preparation (AUP)

**Measured.** The P8192 artifact builds and runs. Config 3 (every gate/up row on the NPU) runs with the 8192 family's 512-row
artifact and gives the reference tokens. With the 16384 family's artifact, named in the task, it runs without an error and gives
wrong tokens.

Jobs `aup-runs/inflight2-prep` (02:32-02:34 UTC Nov 3, 20:32-20:34 CST Nov 2) and `aup-runs/inflight2-try3` (02:37 UTC, 20:37 CST).
C++ build `9dac460`.

## P8192

`experiments.ffn_deploy.prepare --build --npu-channels 8192` into a new directory; the P6400 artifact and `deployment.json` stayed as
they were. The build took 11 s with an empty `compile.log`. `artifact-p8192/final.xclbin.json` and `deployment-p8192.json` are here;
the binaries stay on the box in `/notebooks/hm-results/2026-11-02/inflight2-prep/artifact-p8192/`. Smoke: 30 dispatches, 24 outside
tolerance, `reported_deviations`, the same counts as P6400 (`smoke-p8192.json`).

## Config 3: which artifact

| Config | gate/up `fw_path` | 8 warmup tokens | Prefill, 1 sample | NPU submissions per process |
| --- | --- | --- | ---: | ---: |
| `gateup.config.json` | `hw_bins/npu2/16384x4096x14336/bf16_int4AWQ_bf16_M/` | `2628, 596, 11, 323, 11, 323, 11, 323`: differs from the reference | 1720.1 ms | 3104 |
| `gateup-8192family.config.json` | `hw_bins/npu2/8192x4096x14336/bf16_int4AWQ_bf16_M/` | reference | 1754.6 ms | 3104 |

Both runs are `final_512x4096x14336_64x128x64_8c_bf16_int4AWQ_bf16`, and the two families' PDI and instruction files differ
(`gateup-artifacts.sha256`). Neither crashed, so the zero-row GPU path is not the cause of the wrong tokens.

- **Code-confirmed:** with `minimal_pdi: true` (shipped), `npuSetup.cpp:1388-1398` caches the PDI by K, tile, column and dtype, and
  takes only the instructions from each entry's `fw_path`.
- **Measured:** both logs load one K=4096 PDI, `8192x4096x4096/.../final_512x4096x4096_*`, and reuse it for gate and up (`Reusing PDI`).
  So the failing run executes the 16384 family's instructions on an 8192 family PDI.
- **Hypothesis:** that mismatch causes the wrong tokens. It was not tested with `minimal_pdi: false`.
- **Code-confirmed:** every shipped entry at 512x4096x14336 uses the 8192 family's directory; none uses the 16384 family's file.

So "config 3" in the two-inflight results means the 8192 family's artifact. The other one is measured apart and labeled
(`heteromosaic_gateup16384` in `../2026-11-02-inflight2-numerics/`, `family16384-*` in `../2026-11-02-inflight2-timing/`).
