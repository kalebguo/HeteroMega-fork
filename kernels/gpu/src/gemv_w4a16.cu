// W4A16 GEMV (M = 1) on the HeteroMosaic packed layout. See gemv_w4a16() in gpu_ffn.hpp.
//
// Differences from hipkernels::w4a16_gemv_fused_packed, which reads 76–90 GB/s of a 96 GB/s ceiling on the HX 370:
//   - 16-byte weight loads (that kernel loads one byte per thread per step)
//   - fp32 accumulation (that kernel accumulates in fp16)
//   - no __syncthreads in the K loop
//   - an optional split of K over blocks, for matrices with few tile columns (down: N = 4096 is only 64 columns)
#include "bf16_device.hpp"
#include "heteromega/gpu_ffn.hpp"

#include <algorithm>

namespace heteromega {
namespace {

constexpr int kGroups = 8; // groups of 32 threads per block

// Tile layout (see pack_w4a16): byte a*256 + cc*32 + b*4 + j of the 4096 weight bytes holds row a*8 + b, column
// cc*8 + 2j (low nibble) and column cc*8 + 2j + 1 (high nibble). Bytes 4096..4223 are 64 bf16 scales. Bytes
// 4224..4351 are int8 zeros; column c is at (c / 8) * 16 + c % 8.
//
// Block (x, y) computes the 64 columns of tile column x over the K tiles of split y. Thread group g (32 threads)
// takes tiles kb + g, kb + g + 8, ... Lane l loads bytes 16 * l + 512 * j (j = 0..7) of a tile: 4 rows
// 16 * j + 8 * (l / 16) + 4 * (l % 2) + {0..3}, and columns cc * 8 + {0..7} with cc = (l % 16) / 2. Each word of the
// load is one row and 8 columns. The lane keeps fp32 sums for its 8 columns.
//
// Per tile, sum_r (q - z) * s * x_r = s * (sum_r q * x_r - z * sum_r x_r), so the zero and scale are applied once per
// tile and column, not per weight.
template <bool kPartial>
__global__ void __launch_bounds__(256)
    gemv_w4a16_kernel(const uint8_t *__restrict__ packed, const uint16_t *__restrict__ x, uint16_t *__restrict__ y,
                      float *__restrict__ partial, int tiles_k_used, int tiles_k, int tiles_n, int tiles_per_split, int n) {
    const int col_tile = blockIdx.x;
    int64_t base = static_cast<int64_t>(col_tile / 8) * tiles_k;
    for (int g = 0; g < col_tile % 8; ++g) {
        base += static_cast<int64_t>((tiles_n + 7 - g) / 8) * tiles_k;
    }
    const int kb = blockIdx.y * tiles_per_split;
    const int ke = min(kb + tiles_per_split, tiles_k_used);

    const int group = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int cc = (lane % 16) / 2;
    const int row0 = 8 * (lane / 16) + 4 * (lane % 2);

    float acc[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        acc[c] = 0.0f;
    }

    for (int kt = kb + group; kt < ke; kt += kGroups) {
        const uint8_t *tile = packed + (base + kt) * kTileBytes;
        const uint16_t *xt = x + static_cast<int64_t>(kt) * kTileK + row0;
        uint4 q[8];
        uint2 xv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            q[j] = *reinterpret_cast<const uint4 *>(tile + 16 * lane + 512 * j);
            xv[j] = *reinterpret_cast<const uint2 *>(xt + 16 * j);
        }
        const uint4 sv = *reinterpret_cast<const uint4 *>(tile + 4096 + 16 * cc);
        const uint2 zv = *reinterpret_cast<const uint2 *>(tile + 4224 + 16 * cc);

        float t[8], xs = 0.0f;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            t[c] = 0.0f;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t words[4] = {q[j].x, q[j].y, q[j].z, q[j].w};
            const float xr[4] = {bf16_to_float(static_cast<uint16_t>(xv[j].x)), bf16_to_float(static_cast<uint16_t>(xv[j].x >> 16)),
                                 bf16_to_float(static_cast<uint16_t>(xv[j].y)), bf16_to_float(static_cast<uint16_t>(xv[j].y >> 16))};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t lo = words[i] & 0x0F0F0F0Fu;
                const uint32_t hi = (words[i] >> 4) & 0x0F0F0F0Fu;
                xs += xr[i];
                t[0] = fmaf(static_cast<float>(lo & 0xFFu), xr[i], t[0]);
                t[1] = fmaf(static_cast<float>(hi & 0xFFu), xr[i], t[1]);
                t[2] = fmaf(static_cast<float>((lo >> 8) & 0xFFu), xr[i], t[2]);
                t[3] = fmaf(static_cast<float>((hi >> 8) & 0xFFu), xr[i], t[3]);
                t[4] = fmaf(static_cast<float>((lo >> 16) & 0xFFu), xr[i], t[4]);
                t[5] = fmaf(static_cast<float>((hi >> 16) & 0xFFu), xr[i], t[5]);
                t[6] = fmaf(static_cast<float>(lo >> 24), xr[i], t[6]);
                t[7] = fmaf(static_cast<float>(hi >> 24), xr[i], t[7]);
            }
        }

        const uint32_t sw[4] = {sv.x, sv.y, sv.z, sv.w};
        const uint32_t zw[2] = {zv.x, zv.y};
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const float scale = bf16_to_float(static_cast<uint16_t>(sw[c / 2] >> (16 * (c % 2))));
            const float zero = static_cast<float>(static_cast<int8_t>(zw[c / 4] >> (8 * (c % 4))));
            acc[c] = fmaf(scale, t[c] - zero * xs, acc[c]);
        }
    }

    // Lanes l, l ^ 1, l ^ 16 and l ^ 17 hold the same 8 columns. These shuffles stay inside 32 lanes for wave32 and wave64.
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        acc[c] += __shfl_xor(acc[c], 1);
        acc[c] += __shfl_xor(acc[c], 16);
    }

    __shared__ float sums[kGroups][64];
    if ((lane & 17) == 0) {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            sums[group][cc * 8 + c] = acc[c];
        }
    }
    __syncthreads();
    if (threadIdx.x < 64) {
        float v = 0.0f;
#pragma unroll
        for (int g = 0; g < kGroups; ++g) {
            v += sums[g][threadIdx.x];
        }
        const int col = col_tile * 64 + threadIdx.x;
        if constexpr (kPartial) {
            partial[static_cast<int64_t>(blockIdx.y) * n + col] = v;
        } else {
            y[col] = float_to_bf16(v);
        }
    }
}

// y[c] = bf16(sum over splits of partial[s, c]).
__global__ void __launch_bounds__(256) sum_splits_kernel(const float *__restrict__ partial, uint16_t *__restrict__ y, int splits, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) {
        float v = 0.0f;
        for (int s = 0; s < splits; ++s) {
            v += partial[static_cast<int64_t>(s) * n + c];
        }
        y[c] = float_to_bf16(v);
    }
}

} // namespace

void gemv_w4a16(torch::Tensor &out, const torch::Tensor &x, const torch::Tensor &packed, int64_t k, int64_t n, int64_t packed_k,
                int64_t packed_n, int split_k, hipStream_t stream) {
    TORCH_CHECK(k > 0 && n > 0 && k % kTileK == 0 && n % kTileN == 0, "gemv_w4a16: k must be a multiple of 128 and n of 64");
    TORCH_CHECK(k <= packed_k && n <= packed_n && packed_k % kTileK == 0 && packed_n % kTileN == 0,
                "gemv_w4a16: packed matrix [", packed_k, ", ", packed_n, "] does not contain [", k, ", ", n, "]");
    TORCH_CHECK(packed.numel() == (packed_k / kTileK) * (packed_n / kTileN) * kTileBytes, "gemv_w4a16: packed size does not match ",
                packed_k, "x", packed_n);
    TORCH_CHECK(x.numel() == k && x.is_contiguous() && x.scalar_type() == torch::kBFloat16, "gemv_w4a16: x must be contiguous bf16 with ", k,
                " elements");
    TORCH_CHECK(out.numel() == n && out.is_contiguous() && out.scalar_type() == torch::kBFloat16, "gemv_w4a16: out must be contiguous bf16 with ",
                n, " elements");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(packed.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0,
                "gemv_w4a16: packed must be 16-byte and x 8-byte aligned");

    const int tiles_k_used = static_cast<int>(k / kTileK);
    const int col_tiles = static_cast<int>(n / kTileN);
    if (split_k <= 0) {
        // Fewer than 128 blocks leave the 16 CUs of the 890M with few waves to hide memory latency.
        // Keep at least one tile per thread group in each split.
        split_k = 1;
        while (col_tiles * split_k < 128 && tiles_k_used / (2 * split_k) >= kGroups) {
            split_k *= 2;
        }
    }
    split_k = std::max(1, std::min(split_k, tiles_k_used));
    const int tiles_per_split = (tiles_k_used + split_k - 1) / split_k;
    const int splits = (tiles_k_used + tiles_per_split - 1) / tiles_per_split;

    const auto *pk = packed.data_ptr<uint8_t>();
    const auto *xp = reinterpret_cast<const uint16_t *>(x.data_ptr());
    auto *yp = reinterpret_cast<uint16_t *>(out.data_ptr());
    const int tk = static_cast<int>(packed_k / kTileK), tn = static_cast<int>(packed_n / kTileN);
    if (splits == 1) {
        hipLaunchKernelGGL(gemv_w4a16_kernel<false>, dim3(col_tiles), dim3(256), 0, stream, pk, xp, yp, nullptr, tiles_k_used, tk, tn,
                           tiles_per_split, static_cast<int>(n));
        check_launch("gemv_w4a16");
        return;
    }
    auto partial = torch::empty({splits, n}, x.options().dtype(torch::kFloat32));
    hipLaunchKernelGGL(gemv_w4a16_kernel<true>, dim3(col_tiles, splits), dim3(256), 0, stream, pk, xp, nullptr,
                       partial.data_ptr<float>(), tiles_k_used, tk, tn, tiles_per_split, static_cast<int>(n));
    check_launch("gemv_w4a16");
    hipLaunchKernelGGL(sum_splits_kernel, dim3((n + 255) / 256), dim3(256), 0, stream, partial.data_ptr<float>(), yp, splits,
                       static_cast<int>(n));
    check_launch("gemv_w4a16 sum");
}

} // namespace heteromega
