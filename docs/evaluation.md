# Prefill evaluation

> **2026-11-02 update:** the batch runner now has two Main C methods for the
> cloud-validated C++ endpoint ([GPU + NPU FFN testing](ffn-teammate-testing.md)):
> `merged_ffn` (GPU attention) and `merged_ffn_hetero` (original heterogeneous
> attention/projections). Only those jobs receive `HETEROMEGA_FFN_CONFIG`, taken
> from `--merged-ffn-config`; every other job has it and
> `HETEROMEGA_GPU_FFN_SPLIT` removed. Each job records merged completions per
> prefill pass. Model smoke timing remains diagnostic and differs from this
> runner's synchronized generate(1) measurement boundary.
>
> **Later on 2026-11-02:** this wiring has now run on AUP, measured results in
> `docs/runs/2026-11-02-fix-*/` and `docs/progress.md`. `heteromosaic_mdown`
> (`--mdown-config`, from `tools/split_model/make_mdown_config.py`) adds native
> M-split down. `HETEROMEGA_FFN_FINITE_CHECK=sampled` and
> `HETEROMEGA_FFN_TIMERS=<dir>` pass through to the jobs; see
> [ffn-deployment.md](ffn-deployment.md). Timers add host waits, so report
> headline numbers from runs without them.

Reuse `kernels/gpu/tools/run_model_ffn.py` from GPU PR #1: one excluded warmup,
then five measured pairs of synchronized `generate(1)` / `generate(64)` calls.
The primary time includes first-token selection and Python call overhead; it is
**not** the older profiler's C++ forward-only timer. Model loading is excluded.
No GPU/NPU kernels or model execution code are changed by this evaluation work.

The full experiment plan includes pending implementations, not just CLI choices:

| Experiment | Comparison / method | Current integration |
| --- | --- | --- |
| Main A | Original GPU-only (`gpu_original`) | Wired; no new hardware measurements |
| Main B | Original HeteroMosaic GPU+NPU (`heteromosaic`) | Wired; working NPU runtime required |
| Main C | Our persistent FFN GPU+NPU (`merged_ffn`, `merged_ffn_hetero`) | C++ endpoint and cloud deployment checks complete; runner methods/counter wired, not yet run on AUP |
| GPU sanity | Original GPU FFN vs our all-GPU FFN (`gpu_original` vs `gpu_ffn_all`) | Model runner wired; reuse PR #1 FFN tests/benchmarks |
| NPU sanity | Original NPU FFN vs our all-NPU FFN | Pending both equivalent full-FFN endpoints |

Main A/B/C compare complete prefill latency and output agreement on identical
weights/inputs. GPU sanity reuses Main A rather than duplicating that baseline.
Both sanity checks compare identical FFN work: output error against the same
reference and complete FFN latency, then model logits/tokens where integrated.
Token agreement is only a regression check. These controls distinguish device
implementation changes from gains due to GPU/NPU cooperation.

All-GPU/all-NPU here means all FFN channels; attention keeps its existing path.
The original runtime's `npu` setting does not establish a complete NPU-only FFN:
its GPU activation/down work must be accounted for in an equivalent endpoint.
Pending entries are documented plans, not runnable CLI methods or measured results.
A Main C channel-split sweep needs one rebuilt NPU artifact and deployment config
per split (`--merged-ffn-config` per output directory); only P6400 exists today.
PR #1's partial-GPU mode omits the NPU output, so this runner clears its override.
Existing kernel tests/microbenchmarks stay in their current tools.

From the repository root (a fresh output directory for each experiment):

```bash
# Mac/CPU: inspect the commands without importing Torch or loading a model.
python3 kernels/gpu/tools/evaluate_prefill.py --out build/eval-preview --prompt-tokens 128 512 1024 --dry-run
python3 -m unittest discover -s kernels/gpu/tools -p test_evaluate_prefill.py -v
# AMD/Linux: use the existing PR #1 build/environment and real AWQ model directory.
python3 kernels/gpu/tools/evaluate_prefill.py --model /path/to/AWQ --out build/eval-gpu --prompt-tokens 128 512 1024
# Add the original heterogeneous baseline only with a working NPU runtime/firmware.
python3 kernels/gpu/tools/evaluate_prefill.py --model /path/to/AWQ --out build/eval-three --methods gpu_original gpu_ffn_all heteromosaic
python3 kernels/gpu/tools/evaluate_prefill.py --out build/eval-gpu --summarize
# Main C preview (CPU): deployment.json may be absent in a dry run; the run itself requires it.
python3 kernels/gpu/tools/evaluate_prefill.py --out build/eval-merged-preview --methods gpu_original merged_ffn \
  --merged-ffn-config deployment.json --gpu-chunk-size 512 --prompt-tokens 512 1024 --dry-run
# AMD/Linux, -DHETEROMEGA_MERGED_FFN=ON build: GPU attention, M512 chunks for both methods.
python3 kernels/gpu/tools/evaluate_prefill.py --model /path/to/AWQ --build build --out build/eval-merged-gpu \
  --methods gpu_original merged_ffn --merged-ffn-config "$PWD/deployment.json" --gpu-chunk-size 512 --prompt-tokens 512 1024
# Original heterogeneous attention: compare B and C with the shipped 1024-token 512-row chunk schedule.
python3 kernels/gpu/tools/evaluate_prefill.py --model /path/to/AWQ --build build --out build/eval-merged-hetero \
  --methods heteromosaic merged_ffn_hetero --merged-ffn-config "$PWD/deployment.json" --prompt-tokens 1024
```

Main C rules (code-confirmed in `evaluate_prefill.py`/`run_model_ffn.py`, CPU-tested):

- Merged methods require `--merged-ffn-config` and prompt lengths that are multiples
  of 512; the config's artifact manifest must be M512/H4096/I14336 (a dry run only
  prints a note when it is missing). `merged_ffn` also requires `--gpu-chunk-size 512`;
  this and `--inflight` are applied to **every** method in the batch so the chunk
  schedule stays fixed. The shipped GPU setting (one 1024-row chunk) would bypass
  the M512 artifact. For heterogeneous methods the shipped `kernels_gemm_chunked`
  group selects 512-row chunks only for 1024-token prompts; other lengths are untested.
- `heteromosaic` and `merged_ffn_hetero` jobs get `HETEROMOSAIC_ROOT=<repo root>` unless
  it is already set; without it the native loader silently fell back to the GPU in the
  October 31 invalid run. Native NPU attention calls are still not counted here; use the
  ioctl counter in [ffn-deployment.md](ffn-deployment.md) (`npu_execution=unverified`).
- Each generate call (warmup, every `generate(1)` and `generate(64)`) is one prefill
  pass. The counter is read before and after each call, outside the synchronized
  timer window. Expected per pass: `32 * (prompt_tokens // 512)` for merged methods
  and 0 for every other method (decode rows use the original FFN).
- `summary.csv` adds `merged_ffn_expected_per_pass`, `merged_ffn_passes`,
  `merged_ffn_completions`, `merged_ffn_wrong_passes` and `merged_ffn_verdict`:
  `active` / `wrong_count` / `unrecorded` for merged methods, `clean` / `contaminated` /
  `unrecorded` for controls (`unrecorded` = binding without the counter).
  `speedup_vs_gpu` and the new `speedup_vs_heteromosaic` are suppressed unless a
  merged row is `active`, the baseline is not `contaminated`, and the existing token,
  reference and GPU-FFN rules pass. Greedy-token differences caused by accepted NPU
  numerical deviations therefore suppress the speedup cell; the medians stay in the CSV
  and any hand-computed ratio must be reported together with the token mismatch.

Defaults alternate process order (original/new/new/original), with one model
load and warmup per process; `--rounds`, `--reps`, `--build`, `--config` and
`--new-tokens` are adjustable. Configuration scheduling is preserved. All methods
use the same token prefix, weights, quantization and requested output count.
The bundled reference has 1024 tokens; 2048/4096 require a longer `--ref` with
real input IDs and its corresponding reference output, not repeated/padded IDs.
Shortened prompts have no historical output reference; compare their methods.

Each process saves a log, effective config and result JSON. `summary.csv` groups
by method/length: sample count, median/min/max ms, speedup versus original GPU
(and versus `heteromosaic` when present), token agreement, GPU FFN activation and
the merged-FFN completion verdict. A speedup of 1.10 means 1.10x as fast.
Token mismatch, a failed historical reference, an inactive new FFN or a wrong
merged completion count suppresses the speedup cell. Token agreement is a
regression check, not numerical accuracy.
NPU execution stays `unverified`: enabling `hetero` does not establish successful
NPU completions. Confirm these separately before reporting a heterogeneous gain;
use pinned XDNA headers, not PR #1's GPU-only `SYSTEM_XDNA_HEADER=1` build shortcut.
Keep power/thermal conditions and background load comparable. `--dummy` is only
a timing smoke check. This local change has no new AMD measurements.
