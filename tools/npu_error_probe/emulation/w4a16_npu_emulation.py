"""NumPy emulation of the NPU W4A16 projection and SiLU arithmetic (CPU only).

Source of each step (code-confirmed unless marked ASSUMPTION):

* Dequantization, `npu/ffn/w4a16_prefill_upstream.cc:166-174,230-235`:
  BF16(128+q) - BF16(128+z) is the exact integer q-z; it is multiplied by the
  BF16 scale (exact in FP32) and rounded once to BF16 with conv_even
  (round-to-nearest-even; rounding mode set at :422/:501/:577/:609).
* Deployed artifact flags, `docs/runs/2026-10-31-ffn-deploy/deployment-results/
  artifact/final.xclbin.json`: arithmetic=bfp16, bfp16_native_layout=true,
  preconvert_b=true, split_m32_a=true, preconvert_a=false, accumulator_format=bf16.
* B operand: dequantized BF16 weights are transposed to N8/K8 and converted to
  bfp16ebs8 with conv_even (:236-244). Each block of 8 consecutive elements
  therefore shares one exponent: one output channel n, eight consecutive K.
* A operand: the mixed `mmul<8,8,8,bfloat16,bfp16ebs8,accfloat>` converts BF16
  activations; :512-523 documents that conversion as row-major 8x8 tiles, i.e.
  one block = one row m, eight consecutive K.
* Accumulation: FP32 (`accfloat`) accumulators are loaded from C, receive one
  8x8x8 MAC per K8 step for the 16 K8 steps of a K128 group in increasing K
  order, then are stored back as BF16 with conv_even (:445-494, :488-491).
  C starts at zero (:664-669); the output Y is a copy of the BF16 C (:693-699).
* SiLU*up, `npu/ffn/silu_mul.cc`: clamp gate to [-16, 15.875], 128-segment
  piecewise-linear sigmoid (width 1/4) from the source table, round sigmoid to
  BF16, BF16(gate*sigmoid), then BF16(silu*up); sigmoid=0 below -16, 1 at >=16.

ASSUMPTIONS (not determined by code in this repository; the AIE API and
hardware conversion units are outside it):

1. bfp16ebs8 layout: 8-bit signed mantissa per element (7 magnitude bits) plus
   one shared 8-bit exponent per 8 elements (72 bytes per 64 elements is
   code-confirmed at :133-134). Shared exponent = exponent of the block's
   largest magnitude; mantissas are rounded half-to-even (conv_even) and
   saturated at +-127 if rounding overflows. `mantissa_bits` and `overflow`
   expose the unknowns for sensitivity runs.
2. One 8x8x8 BFP16 MAC adds the exact eight-term block dot product to the FP32
   accumulator with one round-to-nearest-even (`k8_rounding="fp32"`). Hardware
   may align or truncate differently; `k8_rounding="exact"` brackets it.
3. silu_mul.cc's `linear_approx(table, -2, 64, 0)` selects segment
   floor(4*x)+64 and computes slope*x+intercept in FP32 before BF16 rounding.

The precompiled native HeteroMosaic binaries under hw_bins/npu2 have no source
here; this module does not determine their arithmetic.
"""
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
_NPU_DIR = str(ROOT / "npu" / "ffn")
if _NPU_DIR not in sys.path:
    sys.path.insert(0, _NPU_DIR)
from reference import bf16, unpack_int4  # noqa: E402  (repository RNE helper)

GROUP = 128
K8 = 8
DEPLOYED = dict(bfp16_a=True, bfp16_b=True, bf16_partials=True)


def bfp16_ebs8(values, mantissa_bits=8, overflow="saturate"):
    """Round to block floating point, blocks of 8 along the last axis (ASSUMPTION 1)."""
    x = np.asarray(values, dtype=np.float32)
    if x.shape[-1] % K8:
        raise ValueError("the last axis must be a multiple of 8")
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // K8, K8).astype(np.float64)
    amax = np.max(np.abs(blocks), axis=-1, keepdims=True)
    _, exponent = np.frexp(np.where(amax > 0, amax, 1.0))  # amax in [2^(e-1), 2^e)
    step = np.ldexp(1.0, exponent - mantissa_bits + 1)  # amax/step in [2^(b-2), 2^(b-1))
    limit = 2 ** (mantissa_bits - 1) - 1
    mantissa = np.rint(blocks / step)  # numpy rint: half to even (conv_even)
    if overflow == "saturate":
        mantissa = np.clip(mantissa, -limit, limit)
    elif overflow == "renormalize":
        over = np.max(np.abs(mantissa), axis=-1, keepdims=True) > limit
        step = np.where(over, step * 2, step)
        mantissa = np.clip(np.rint(blocks / step), -limit, limit)
    else:
        raise ValueError("overflow must be 'saturate' or 'renormalize'")
    return (mantissa * step).reshape(x.shape).astype(np.float32)


def dequantize(q, scales, zeros):
    """BF16((q - z) * BF16(scale)) per output channel and K128 group; q: [N, K]."""
    q = np.asarray(q)
    reps = q.shape[1] // np.asarray(zeros).shape[1]
    centered = q.astype(np.float32) - np.repeat(np.asarray(zeros, np.float32), reps, axis=1)
    return bf16(centered * np.repeat(bf16(scales), reps, axis=1))


def npu_projection(a, q, scales, zeros, bfp16_a=True, bfp16_b=True, bf16_partials=True,
                   k8_rounding="fp32", mantissa_bits=8, overflow="saturate", partial_interval=GROUP):
    """Y[M,N] = A[M,K] @ W[N,K]^T in the kernel's order; returns BF16 values as float32.

    partial_interval is the K distance between stores of C (128 in w4a16_prefill_upstream.cc).
    Other values are hypotheses for binaries without source here (e.g. the native hw_bins kernel)."""
    a = bf16(np.asarray(a, dtype=np.float32))
    w = dequantize(q, scales, zeros)
    if bfp16_b:
        w = bfp16_ebs8(w, mantissa_bits, overflow)  # blocks: one n, eight consecutive K
    if bfp16_a:
        a = bfp16_ebs8(a, mantissa_bits, overflow)  # blocks: one m, eight consecutive K
    m, k = a.shape
    if w.shape[1] != k or k % GROUP or partial_interval % K8 or GROUP % partial_interval:
        raise ValueError("K must match and be a multiple of 128; the C interval must divide 128 in K8 steps")
    a64, wt64 = a.astype(np.float64), np.ascontiguousarray(w.T, dtype=np.float64)
    c = np.zeros((m, w.shape[0]), dtype=np.float32)
    for g in range(0, k, partial_interval):
        acc = c  # FP32 accumulator loaded from C (BF16 values are exact in FP32)
        if k8_rounding == "fp32":
            for k0 in range(g, g + partial_interval, K8):
                # Exact: within a block all products share one exponent (bfp16) or fit FP64.
                acc = (acc.astype(np.float64) + a64[:, k0:k0 + K8] @ wt64[k0:k0 + K8]).astype(np.float32)
        elif k8_rounding == "exact":
            acc = (acc.astype(np.float64) + a64[:, g:g + partial_interval] @ wt64[g:g + partial_interval]).astype(np.float32)
        else:
            raise ValueError("k8_rounding must be 'fp32' or 'exact'")
        c = bf16(acc) if bf16_partials else acc
    return c if bf16_partials else bf16(c)


def gpu_like_projection(a, q, scales, zeros):
    """BF16 inputs and dequantized weights, FP32 accumulation, one BF16 output rounding."""
    a = bf16(np.asarray(a, dtype=np.float32))
    return bf16(a @ dequantize(q, scales, zeros).T)


def exact_projection(a, q, scales, zeros):
    """FP64 product of the BF16 activations and exact (q - z) * scale; no rounding."""
    q = np.asarray(q)
    reps = q.shape[1] // np.asarray(zeros).shape[1]
    w = (q.astype(np.float64) - np.repeat(np.asarray(zeros, np.float64), reps, axis=1)) \
        * np.repeat(np.asarray(bf16(scales), np.float64), reps, axis=1)
    return bf16(np.asarray(a, np.float32)).astype(np.float64) @ w.T


def sigmoid_table(path=ROOT / "npu" / "ffn" / "silu_mul.cc"):
    """(slope, intercept) for the 128 segments, parsed from the kernel source."""
    text = Path(path).read_text()
    body = text[text.index("#define SIGMOID_TABLE"):text.index("alignas(")]
    rows = re.findall(r"LUT_ROW\(([^)]*)\)", body)
    values = np.array([float(v.rstrip("f")) for row in rows for v in row.split(",")], dtype=np.float32)
    if values.size != 256:
        raise ValueError(f"expected 128 slope/intercept pairs, found {values.size // 2}")
    return values[0::2], values[1::2]


def npu_silu_mul(gate, up, table=None, index_rounding="floor"):
    """silu_mul_bf16 from npu/ffn/silu_mul.cc on BF16 gate/up values (ASSUMPTION 3)."""
    slope, intercept = sigmoid_table() if table is None else table
    g, u = bf16(gate), bf16(up)
    clamped = np.minimum(np.maximum(g, np.float32(-16.0)), np.float32(15.875))
    scaled = clamped.astype(np.float64) * 4
    index = (np.floor(scaled) if index_rounding == "floor" else np.trunc(scaled)).astype(np.int64) + 64
    index = np.clip(index, 0, 127)
    linear = (slope[index].astype(np.float64) * clamped + intercept[index]).astype(np.float32)
    sigmoid = bf16(linear)
    sigmoid = np.where(g < -16, np.float32(0), sigmoid)
    sigmoid = np.where(g >= 16, np.float32(1), sigmoid)
    silu = bf16(g * sigmoid)
    return bf16(silu * u)


def npu_ffn_branch(x, branch, silu="npu", **arith):
    """NPU merged-graph branch: gate/up projections, SiLU*up, down projection (BF16 values)."""
    gate = npu_projection(x, unpack_int4(branch.gate.packed), branch.gate.scales, branch.gate.zeros, **arith)
    up = npu_projection(x, unpack_int4(branch.up.packed), branch.up.scales, branch.up.zeros, **arith)
    if silu == "npu":
        z = npu_silu_mul(gate, up)
    else:
        from reference import silu_multiply
        z = silu_multiply(gate, up, bf16_io=True)
    return npu_projection(z, unpack_int4(branch.down.packed), branch.down.scales, branch.down.zeros, **arith)
