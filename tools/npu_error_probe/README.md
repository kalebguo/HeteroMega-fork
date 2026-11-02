# NPU W4A16 error probes (hardware) and arithmetic emulation (secondary)

Status 2026-11-02: **not executed on hardware.** This cloud sandbox cannot reach the AUP
hub (its egress proxy refuses `dub.aupcloud.io` with HTTP 403 at CONNECT). Scripts pass
CPU unit tests and dry runs here. All emulation numbers below are **estimated** (CPU NumPy).

## Why probes first

The October 13 random-weight sweep left one open question: the original precompiled
HeteroMosaic NPU GEMM (`hw_bins/npu2/8192x2048x2048/bf16_int4AWQ_bf16_M`) differs from
the GPU by up to **1.5** on the first sweep shape (M=N=K=2048; measured,
`docs/runs/2026-10-13-compat/evidence/m-sweep-v3.json`). Its source is not in this
repository, so its accumulation precision and rounding order cannot be read from code.
The probes measure it directly; the emulation only supplies predictions to compare with.

## Files

| File | Role |
| --- | --- |
| `probe_cases.py` | Case list, seeded inputs (sweep distributions), FP32 reference from `W4A16.dequantize`, error statistics. NumPy only. |
| `native_matmul_bindings.cpp` | Thin adapter (compiled with `torch.utils.cpp_extension`, linked to the existing model extension) that runs **one** native M-split GEMM with all 512 rows on the NPU through `hetero_matmul_out_gemm_packed_M`, plus the original GPU kernel as a control. GPU fallback is an error. No kernel, artifact or CMake change. |
| `probe_sweep.py` | Runs the cases in order, one fresh process per case (the native runtime is single-initialization); appends one JSON line per case to `probe.jsonl`, plus blocked lines. `--dry-run` works on CPU. |
| `probe_report.py` | Table + verdicts with a fixed decision rule (docstring). `--emulate OUT` writes CPU predictions. |
| `token_agreement.py` | Real weights, one process, `HETEROMEGA_FFN_CONFIG` switched per pass: greedy-token agreement over 64 tokens, first disagreement, step-1 logit max difference (+ original-vs-original control), small teacher-forced perplexity. Built on `run_model_ffn.py` because the box has no `transformers`/tokenizer (so `model_smoke` cannot import). |
| `aup_job.sh` | Detached AUP job: probes, report, token agreement, device snapshots. |
| `emulation/` | Secondary: NumPy emulation of `w4a16_prefill_upstream.cc` + `silu_mul.cc`, its experiments and results. |

## Cases (M=512) and blocked requests

| Series | Cases | Native artifact family |
| --- | --- | --- |
| K sweep, N=2048 | K = 1536, 2048, 10240, 11264 | `8192x{K}x2048/bf16_int4AWQ_bf16_M` |
| K sweep, N=4096 | K = 4096, 14336 | `8192x4096x4096`, `16384x14336x4096` (`_M`) |
| Sign, K=N=2048 | activations all-positive, all-negative (exact negation), mixed | `8192x2048x2048_M` |
| Metadata, K=N=2048 | scales {all 1/16, random} x zero points {all 8, random} | `8192x2048x2048_M` |

Inputs follow `tensor_parallel_gemm_npu_gpu_sweepM.cpp:308-326` (activations U[0,0.1),
random q/scales/zeros); K2048/N2048 positive is the 1.5 anchor shape (512 of its rows).
Blocked (recorded as `status: blocked` lines, never faked):

- **K=64**: below the 128-element quantization group; the 4352-byte K128/N64 tile ABI
  and every artifact require K % 128 == 0.
- **K=256, 1024, 2048 at N=4096** and **K=16384 at N=2048**: no native M-split
  artifact (`final_512xKxN` exists only as K-split `_K` slices for the K-split
  dispatcher, which this probe does not drive). Hence the N=2048 K series.

## Decision rule (fixed before data; full text in `probe_report.py`)

r = e_npu / e_gpu (mean absolute errors vs FP32, same inputs and packed bytes).
1. slope of log r vs log K >= 0.25 → accumulation precision / partial-sum rounding;
2. |slope| < 0.25 and median r > 1.5 → per-element dequantization / operand format;
3. NPU(-a) != -NPU(a) bitwise, or r_neg / r_pos outside [0.8, 1.25] → sign handling;
4. r(random scales+zeros) > 2 x r(1/16, 8) → metadata addressing (single factors named);
5. all r <= 1.5 → no NPU-specific error.

## Emulation findings (secondary; estimated, CPU)

Arithmetic of the **deployed merged artifact** (code-confirmed from
`w4a16_prefill_upstream.cc`, flags in its `final.xclbin.json`): BF16 dequantization
`BF16((q-z)*scale)`; BFP16 (8-bit mantissa, 8-element shared exponent along K) for both
operands; FP32 MAC per K8 block; **C stored as BF16 after every K128 group**; LUT SiLU.
Assumptions the code does not determine: bfp16ebs8 mantissa width/rounding/overflow,
the MMUL's internal block accumulation, `linear_approx` indexing (see the module docstring).

On the exact Oct-31 smoke inputs (input SHA256 matches `cpp-smoke.json`), the emulated NPU
P6400 branch reproduces the **measured** hardware error:

| Smoke case | Measured rel. L2 / RMS / max | Emulated rel. L2 / RMS / max |
| --- | --- | --- |
| layer 0, input 0 | 0.02144 / 0.00978 / 0.0625 | 0.02143 / 0.00977 / 0.0703 |
| layer 0, input 1 | 0.02142 / 0.00982 / 0.0703 | 0.02144 / 0.00983 / 0.0781 |
| layer 1, input 0 | 0.02142 / 0.00978 / 0.0625 | 0.02140 / 0.00977 / 0.0703 |
| layer 1, input 1 | 0.02150 / 0.00981 / 0.0625 | 0.02148 / 0.00980 / 0.0781 |

Which operation dominates (layer 0 / input 0, relative L2, each alone vs all): BF16 C per
K128 0.01405, BFP16 activations 0.01397, BFP16 weights 0.01136, SiLU LUT 0.00353; all
together 0.02143 (roughly the quadrature sum). **No single operation dominates**; the
three projection effects are comparable. FP32 partials (the earlier non-native8 layout)
are predicted at 0.01714, which is below the existing 0.02 limit; a 9-bit mantissa
assumption predicts 0.0158, inconsistent with the measurement (supports 8-bit).

**The 1.5 hypothesis.** With the same arithmetic on the first sweep shape (256 NPU rows,
3 seeds), the emulated max |NPU - GPU| is 0.375 vs **1.5** measured (exact scales: 0.375 vs
1.25; constant weights: 0.06-0.09 vs 0.25; exact matrix: 0 vs 0). K128 BF16 partials
therefore explain only about a quarter of the native error. A scan of the BF16-store
interval (`emulation` with `partial_interval`) fits all three non-zero measurements only
near **K16** (1.375-1.5 / 1.0 / 0.31-0.34). The constant-metadata case is underpredicted
by the same factor, so scale/zero addressing is **not required** to explain the sweep.
Hypothesis, not verified: the native binary rounds partial sums to BF16 far more often
than every K128. Predicted probe outcomes (`emulation/results/probe-predictions.jsonl`):

| Hypothesis | r at K1536 → K11264 (N2048) | log-r slope | Max abs NPU-GPU, K2048 | Verdict |
| --- | --- | --- | --- | --- |
| BF16 C every K128 + BFP16 | 2.9 → 6.5 | 0.41 | 0.375 | accumulation |
| BF16 C every K16 + BFP16 | 7.1 → 20.3 | 0.53 | 1.25 | accumulation |
| FP32 C + BFP16 | 1.74 → 1.75 | 0.00 | 0.125 | per-element |

Failing emulated configurations (contract `error_metrics`, rel. L2 limit 0.02, unchanged)
are kept in `emulation/results/failing-examples.json` with their worst elements.

## Run on the AUP box (one GPU/NPU job at a time)

From `/notebooks/hm-eval-main` with this branch checked out and the `build-main` build:

```bash
python3 tools/npu_error_probe/probe_sweep.py --dry-run
OUT=/notebooks/hm-results/$(date -u +%F)/npu-error-probe bash tools/npu_error_probe/aup_job.sh
```

Run the second command as a detached job (`aup_cli.py bg npu-probe -- '...'` or the
README launcher). Expect 13 cases × (binding load + 3 GEMMs), then a 1-2 minute model
load and about 512 + 2 × 64 decode steps per path for `token_agreement.py`.

CPU checks: `python3 -m unittest discover -s tools/npu_error_probe -p 'test_*.py' -v`.
