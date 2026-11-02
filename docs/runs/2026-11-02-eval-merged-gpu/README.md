# 2026-11-02: first paired prefill evaluation, gpu_original vs merged_ffn (AUP)

**Measured.** Run on the AUP cloud box (Ryzen AI 9 HX 370, Radeon 890M `gfx1150`, NPU Strix
`aie2p`, ROCm 7.2.0, Torch 2.11.0+rocm7.2) on 2026-11-02, 17:43 to 17:52 UTC, by the local
session through `tools/aup-kit` (job `aup-runs/eval-main`, script in
`tools/aup-kit/jobs/eval-main/cmd.sh` on the `aup-kit` branch). Raw files are in this directory.

## Code and artifact

- Build: `main` `8e5c804` built with `tools/aup-kit/jobs/build-main/cmd.sh`
  (`-DHETEROMEGA_MERGED_FFN=ON`, pinned xdna-driver headers `0e6d303b`, HIP link check passed;
  logs in `../2026-11-02-build-main/`). The C++ was not rebuilt afterwards.
- Runner: branch `claude/exciting-faraday-95a0c1` at `0190cd8a` ("Add merged GPU+NPU FFN methods
  to the prefill evaluation runner"), checked out on top of that build as `eval-runner`.
- NPU artifact: the retained P6400 artifact
  `docs/runs/2026-10-31-ffn-deploy/deployment-results/artifact`, packaged by
  `experiments.ffn_deploy.prepare` into `deployment.json` (NPU channels `[0,6400)`, GPU `[6400,14336)`).
- Model: `Meta-Llama-3.1-8B-Instruct-AWQ-INT4`, the two cached safetensors shards; fixed token IDs
  from the bundled reference `docs/runs/2026-10-12-aup/gpu-model-result.json`.

## Command

```bash
python3 kernels/gpu/tools/evaluate_prefill.py --model /notebooks/hm-model/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --build build --out /notebooks/hm-results/2026-11-02/eval-merged-gpu \
  --methods gpu_original merged_ffn --merged-ffn-config "$PWD/deployment.json" \
  --gpu-chunk-size 512 --prompt-tokens 512 1024
```

GPU attention for both methods, M512 serial chunks, process order original/new/new/original per
length, one model load and one excluded warmup per process, 5 measured `generate(1)` /
`generate(64)` pairs per process (10 samples per cell). `HETEROMEGA_FFN_CONFIG` was set only in
the `merged_ffn` processes. Device before: 37 C, 5 W, sclk 600 MHz; after: 62 C.

## Result (`summary.csv`)

| method | tokens | samples | prefill median ms | min | max | vs gpu_original | tokens match | merged completions (expected/pass, passes, wrong) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpu_original | 512 | 10 | 889.3 | 880.0 | 967.8 | 1.00 | - | 0 |
| merged_ffn | 512 | 10 | 1257.1 | 1248.1 | 1621.9 | 0.707 | yes, 64/64 | 32, 22 passes, 0 wrong |
| gpu_original | 1024 | 10 | 1829.0 | 1825.6 | 1875.4 | 1.00 | reference 8/8 | 0 |
| merged_ffn | 1024 | 10 | 2558.7 | 2543.4 | 2621.3 | 0.715 | yes, 64/64; reference 8/8 | 64, 22 passes, 0 wrong |

- The merged FFN is confirmed active (`merged_ffn_verdict = active`, `npu_execution =
  merged_ffn_counted`) and produces the same 64 greedy tokens as the GPU-only path in every
  repetition; decode stays at about 17 tok/s for both (the merged FFN applies only to M512 chunks).
- Peak allocation 7.20 GiB (gpu_original) vs 7.85 GiB (merged_ffn); model load 27 to 31 s.
- The merged FFN at P6400 makes prefill about 1.4x slower at both lengths. This is a
  measurement of the current merged graph and split, not of the GPU+NPU idea: the original
  heterogeneous path measured 1327 ms at 1024 tokens on 2026-10-14 (C++ forward timer, not
  directly comparable to this runner's `generate(1)` boundary).

## Files

- `summary.csv`, `evaluate.log`, `run.log` (job log with device snapshots).
- `<method>-<tokens>-r<round>.result.json`: per-process results with the five repetitions.
- `gpu_original-512-r0.result.config.json`: the effective model config of one process; the other
  seven differ only in the method switches and are not stored.
- Not stored: the eight `*.result.log` files (165 KB each, almost entirely weight-loading lines).
  They stay on the box under `/notebooks/hm-results/2026-11-02/eval-merged-gpu/`.
- `../2026-11-02-smoke-main/`: the isolated C++ smoke on the same build (30 completed dispatches,
  `completed_with_accuracy_deviations`, 24 of 90 checks outside tolerance under report policy) and
  the failed `model_smoke` attempt (`ModuleNotFoundError: transformers`; this account's venv has no
  `transformers` and the model directory has no tokenizer files).
- `../2026-11-02-build-main/`: configure and build logs.
