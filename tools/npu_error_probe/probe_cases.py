"""Probe cases, seeded inputs, FP32 reference and error statistics (NumPy only).

Every runnable case is one packed W4A16 GEMM with all M=512 rows on the NPU, using
an existing native HeteroMosaic M-split artifact (hw_bins/npu2/<family>/
bf16_int4AWQ_bf16_M/final_512xKxN_64x128x64_8c_bf16_int4AWQ_bf16.pdi) and the same
packed bytes the GPU kernel consumes (experiments.ffn_partition.gpu_backend).

Series (all at M=512):
  k_n2048  K = 1536, 2048, 10240, 11264 at N=2048 (widest native M-split K range at one N)
  k_n4096  K = 4096, 14336 at N=4096 (the only native M-split K values at N=4096)
  sign     K=2048, N=2048: activations all-positive, all-negative, mixed sign
  metadata K=2048, N=2048: scales equal (1/16) or random x zero points all 8 or random

The K series and the positive sign/metadata cases use the original M-sweep input
distribution (src/microbenchmarks/tensor_parallel_gemm_npu_gpu_sweepM.cpp:308-326):
activations U[0,0.1), q uniform in [0,15], scales U(0,0.1) rounded to BF16 (drawn here
from [0.001,0.1) because W4A16 rejects zero scales), zero points uniform in [0,15].
K2048/N2048 is the shape of the October 13 observation (max GPU/NPU difference 1.5).
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "npu" / "ffn"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from reference import W4A16, bf16, pack_int4  # noqa: E402

M = 512
GROUP = 128
TILE = "64x128x64"
COL = "8c"
DTYPE = "bf16_int4AWQ_bf16"
SEED = 20261102
SWEEP_RTOL = SWEEP_ATOL = 0.2  # the original sweep's allclose limits (sweepM.cpp:40-41); report only

# Requested by the probe plan but not runnable with existing native artifacts.
BLOCKED = [
    dict(series="k_n4096", k=64, n=4096,
         reason="K=64 is below the 128-element quantization group; the packed 4352-byte K128/N64 tile "
                "ABI (and every native artifact) requires K % 128 == 0"),
    dict(series="k_n4096", k=256, n=4096,
         reason="no native M-split artifact with K=256 (no hw_bins/npu2/*x256x*/bf16_int4AWQ_bf16_M); "
                "K-split (_K) artifacts start at K=512 and serve the K-split dispatcher, not driven here"),
    dict(series="k_n4096", k=1024, n=4096,
         reason="no native M-split artifact 512x1024x4096; final_512x1024x4096 exists only in "
                "8192x14336x4096/bf16_int4AWQ_bf16_K, a K-split slice for the K-split dispatcher"),
    dict(series="k_n4096", k=2048, n=4096,
         reason="no native M-split artifact 512x2048x4096 (only the K-split slice in 8192x14336x4096/_K); "
                "the K sweep therefore uses N=2048, where four M-split K values exist"),
    dict(series="k_n2048", k=16384, n=2048,
         reason="8192x16384x2048 has only K-split (_K) artifacts; no M-split final_512x16384x2048"),
]


def _case(name, series, k, n, act="positive", scales="random", zeros="random"):
    return dict(name=name, series=series, m=M, k=k, n=n, act=act, scales=scales, zeros=zeros, seed=SEED)


def cases():
    """All runnable cases, in execution order: K sweep, sign sweep, metadata sweep."""
    result = [_case(f"k{k}_n2048", "k_n2048", k, 2048) for k in (1536, 2048, 10240, 11264)]
    result += [_case(f"k{k}_n4096", "k_n4096", k, 4096) for k in (4096, 14336)]
    result += [_case(f"sign_{act}", "sign", 2048, 2048, act=act) for act in ("positive", "negative", "mixed")]
    result += [_case(f"meta_scales_{s}_zeros_{z}", "metadata", 2048, 2048, scales=s, zeros=z)
               for s in ("equal", "random") for z in ("eight", "random")]
    return result


def artifact(root, m, k, n):
    """Relative fw_path of a native M-split artifact holding final_<m>x<k>x<n>, or None."""
    stem = f"{m}x{k}x{n}_{TILE}_{COL}_{DTYPE}"
    for family in sorted(Path(root, "hw_bins", "npu2").glob(f"*x{k}x{n}"), key=lambda p: (len(p.name), p.name)):
        directory = family / f"{DTYPE}_M"
        if (directory / f"final_{stem}.pdi").is_file() and (directory / f"insts_{stem}.txt").is_file():
            return directory.relative_to(root).as_posix() + "/"
    return None


def native_config(case, fw_path):
    """One-kernel native configuration: all rows on the NPU (npuM == forM, M-split)."""
    kernel = dict(use=True, layer="gate", forM=case["m"], forK=case["k"], forN=case["n"],
                  npuM=case["m"], npuK=case["k"], npuN=case["n"], chunk_id=0, config=-1, num_tiles=32,
                  fw_path=fw_path, tile_size=TILE, col=COL, dtype=DTYPE)
    return dict(heterogeneity="hetero", warmup=False, dummy_weights=False, debug_verbosity=0,
                usePackedWeights=True, padPackedWeights=False, usePreSavedWeights=False,
                prompt_len=case["m"], minimal_pdi=True, cpu_decode=False, gemv_driven_split_K=False,
                chunking=False, kernels_gemm=[kernel], kernels_gemv=[])


def make_inputs(case):
    """Seeded activations [M,K] (BF16 values) and canonical W4A16 weights [N,K]."""
    m, k, n = case["m"], case["k"], case["n"]
    weights_rng = np.random.default_rng(case["seed"] + 1000 * k + n)
    activations_rng = np.random.default_rng(case["seed"] + 7 + 1000 * k + n)
    u = activations_rng.random((m, k), dtype=np.float32)
    signed = {"positive": u, "negative": -u, "mixed": 2 * u - 1}[case["act"]]
    a = bf16(bf16(signed) * np.float32(0.1))
    q = weights_rng.integers(0, 16, (n, k), dtype=np.uint8)
    random_scales = bf16(weights_rng.uniform(0.001, 0.1, (n, k // GROUP)).astype(np.float32))
    random_zeros = weights_rng.integers(0, 16, (n, k // GROUP), dtype=np.uint8)
    scales = random_scales if case["scales"] == "random" else np.full((n, k // GROUP), 1 / 16, np.float32)
    zeros = random_zeros if case["zeros"] == "random" else np.full((n, k // GROUP), 8, np.uint8)
    return a, W4A16(pack_int4(q), scales, zeros, GROUP)


def reference(a, weight):
    """FP32 reference from the repository's dequantization helper (FP64 product, FP32 storage)."""
    return (a.astype(np.float64) @ weight.dequantize(False).astype(np.float64).T).astype(np.float32)


def error_stats(actual, expected):
    actual64, expected64 = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    delta = np.abs(actual64 - expected64)
    row, col = np.unravel_index(int(np.argmax(delta)), delta.shape)
    norm = float(np.linalg.norm(expected64))
    return dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()), rms=float(np.sqrt(np.mean(delta ** 2))),
                max_row=int(row), max_col=int(col), relative_l2=float(np.linalg.norm(delta) / norm) if norm else None,
                finite=bool(np.all(np.isfinite(actual64))))


def compare(npu, gpu, expected):
    """Statistics recorded per case: NPU and GPU control vs FP32, and NPU vs GPU as in the sweep."""
    diff = np.abs(np.asarray(npu, np.float64) - np.asarray(gpu, np.float64))
    allclose = bool(np.all(diff <= SWEEP_ATOL + SWEEP_RTOL * np.abs(np.asarray(gpu, np.float64))))
    return dict(npu=error_stats(npu, expected), gpu=error_stats(gpu, expected),
                npu_vs_gpu_max_abs=float(diff.max()), sweep_allclose_rtol0p2_atol0p2=allclose,
                reference_abs_max=float(np.max(np.abs(expected))), reference_abs_mean=float(np.mean(np.abs(expected))))
