// Causal attention for prefill on bf16 Q, K and V, with WMMA. See sdpa_bf16() in gpu_ffn.hpp.
//
// One wave computes 16 rows of one Q head, with no barriers:
//   for each fragment of 16 keys:
//     S = K Q^T (8 WMMA, one per 16 dimensions), P = exp(scale * S - m), O += P V (8 WMMA)
//   output = O / sum(P)
// m is a value per row that keeps P in the range of fp32. Unlike the usual streaming softmax, m changes only if an exponent exceeds
// kRescaleAbove, because a change must scale all of O for that row. This is rare.
//
// WMMA registers: see gemm_w4a16.cu. half = lane / 16, dim = lane % 16.
//   S: A holds key `dim`, B holds Q row `dim`, and C[key 2 * e + half, Q row dim] is in element e. So all exponents of a row are in 2 lanes.
//   O: A holds P of Q row `dim` in lane half == dim % 2, B holds V of dimension `dim` over the keys, and C[Q row 2 * e + half, dimension dim] is in element e.
// A and B of the second step hold their 16 keys in the order 0, 2, .. 14, 1, 3, .. 15. A lane has P of the keys of its half, and takes the others from the
// other lane of the row. V is stored in this order by sdpa_store_v, because V is row-major in the cache and WMMA needs it per dimension.
#include "bf16_device.hpp"
#include "heteromega/gpu_ffn.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <tuple>

namespace heteromega {
namespace {

typedef float v4f __attribute__((ext_vector_type(4)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef uint32_t v4u __attribute__((ext_vector_type(4)));

constexpr int kFrag = 16;
constexpr int kDim = 128;
constexpr int kDimFrags = kDim / kFrag;
constexpr float kLog2E = 1.44269504f;
constexpr float kRescaleAbove = 40.0f;
constexpr float kMasked = -1000.0f; // exp(-1000) is 0 in fp32.

__device__ __forceinline__ v8f load_8f(const uint16_t *p) {
    const v4f a = *reinterpret_cast<const v4f *>(p);
    const v4f b = *reinterpret_cast<const v4f *>(p + 8);
    return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7);
}

// Lanes of a wave read in LDS what the other lanes wrote. A wave runs its LDS operations in order, so only the compiler must keep the order.
__device__ __forceinline__ void keep_lds_order() { __asm__ volatile("" ::: "memory"); }

// tiles[((g * key_frags + f) * 8 + j) * 256 + d * 16 + (k & 1) * 8 + k / 2] = V[g, 16 * f + k, 16 * j + d], k and d in 0..15.
__global__ void __launch_bounds__(32) store_v_kernel(uint16_t *__restrict__ tiles, const uint16_t *__restrict__ v, int first_key, int capacity, int key_frags) {
    const int key = first_key + blockIdx.x * 32 + threadIdx.x;
    const int j = blockIdx.y % kDimFrags;
    const int g = blockIdx.y / kDimFrags;
    const v4u a = *reinterpret_cast<const v4u *>(v + (static_cast<int64_t>(g) * capacity + key) * kDim + j * kFrag);
    const v4u b = *reinterpret_cast<const v4u *>(v + (static_cast<int64_t>(g) * capacity + key) * kDim + j * kFrag + 8);
    const int k = key % kFrag;
    uint16_t *cell = tiles + ((static_cast<int64_t>(g) * key_frags + key / kFrag) * kDimFrags + j) * 256 + (k & 1) * 8 + k / 2;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        cell[(2 * w) * 16] = static_cast<uint16_t>(a[w]);
        cell[(2 * w + 1) * 16] = static_cast<uint16_t>(a[w] >> 16);
        cell[(2 * w + 8) * 16] = static_cast<uint16_t>(b[w]);
        cell[(2 * w + 9) * 16] = static_cast<uint16_t>(b[w] >> 16);
    }
}

// With ToGemm, `out` and `sums` are the prepared input of the next GEMM, in the layout of prepare_input_kernel in gemm_w4a16.cu. A wave then fills exactly one
// (fragment of rows, K tile) of it, so the output does not go to memory twice.
template <bool ToGemm>
__global__ void __launch_bounds__(32)
    sdpa_kernel(uint16_t *__restrict__ out, const uint16_t *__restrict__ q, const uint16_t *__restrict__ k, const uint16_t *__restrict__ tiles, int64_t out_row_stride,
                int64_t q_row_stride, int group, int capacity, int key_frags, int keys, int mask_offset, float scale, float *__restrict__ sums, int q_heads) {
    __shared__ __align__(16) v4u p_other[2][kFrag];
    __shared__ __align__(16) float value_other[2][kFrag];
    __shared__ __align__(16) float per_row[2][8]; // [half][e]: the value of Q row 2 * e + half
    __shared__ __align__(16) float row_part[kFrag][kFrag]; // [row of the wave][lane dim]: the sum of the lane's dimensions, for ToGemm

    const int lane = threadIdx.x;
    const int half = lane / kFrag;
    const int dim = lane % kFrag;
    const int first_row = blockIdx.x * kFrag;
    const int head = blockIdx.y;
    const int g = head / group;
    const int row = first_row + dim;

    v8f q_frag[kDimFrags];
    {
        const uint16_t *p = q + row * q_row_stride + head * kDim;
#pragma unroll
        for (int j = 0; j < kDimFrags; ++j) {
            q_frag[j] = load_8f(p + j * kFrag);
        }
    }
    v8f o[kDimFrags];
#pragma unroll
    for (int j = 0; j < kDimFrags; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            o[j][e] = 0.0f;
        }
    }
    float sum_p = 0.0f;
    float m = 0.0f;

    const uint16_t *k_lane = k + (static_cast<int64_t>(g) * capacity + dim) * kDim;
    const uint16_t *v_lane = tiles + static_cast<int64_t>(g) * key_frags * kDimFrags * 256 + dim * kFrag;
    const int visible = row + mask_offset; // the last key that this row sees
    const int last_key = first_row + kFrag - 1 + mask_offset;
    const int frags = (last_key < keys ? last_key : keys - 1) / kFrag + 1;

    for (int f = 0; f < frags; ++f) {
        // 1. Exponents, relative to m.
        v8f s;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            s[e] = 0.0f;
        }
        {
            const uint16_t *p = k_lane + static_cast<int64_t>(f) * kFrag * kDim;
#pragma unroll
            for (int j = 0; j < kDimFrags; ++j) {
                s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(load_8f(p + j * kFrag), q_frag[j], s);
            }
        }
        float x[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            x[e] = s[e] * scale - m;
        }
        if ((f + 1) * kFrag - 1 > first_row + mask_offset) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                x[e] = f * kFrag + 2 * e + half <= visible ? x[e] : kMasked;
            }
        }
        float x_max = x[0];
#pragma unroll
        for (int e = 1; e < 8; ++e) {
            x_max = fmaxf(x_max, x[e]);
        }

        // 2. A new m at the first keys, or where the exponent is too large. Then O and sum_p change to the units of the new m.
        const uint32_t exceeds = x_max > kRescaleAbove ? 1u : 0u;
        if (f == 0) {
            value_other[half][dim] = x_max;
            keep_lds_order();
            m = fmaxf(x_max, value_other[1 - half][dim]);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                x[e] -= m;
            }
        } else if (__builtin_amdgcn_uicmp(exceeds, 0u, 33 /* not equal */) != 0) {
            value_other[half][dim] = x_max;
            keep_lds_order();
            const float change = fmaxf(0.0f, fmaxf(x_max, value_other[1 - half][dim]));
            const float ratio = __builtin_amdgcn_exp2f(-change * kLog2E);
            keep_lds_order();
            per_row[dim & 1][dim / 2] = ratio;
            keep_lds_order();
            const v4f r0 = *reinterpret_cast<const v4f *>(&per_row[half][0]);
            const v4f r1 = *reinterpret_cast<const v4f *>(&per_row[half][4]);
#pragma unroll
            for (int j = 0; j < kDimFrags; ++j) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    o[j][e] *= r0[e];
                    o[j][e + 4] *= r1[e];
                }
            }
            sum_p *= ratio;
            m += change;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                x[e] -= change;
            }
        }

        // 3. P, as bf16 in the order of V.
        uint32_t words[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float p0 = __builtin_amdgcn_exp2f(x[2 * w] * kLog2E);
            const float p1 = __builtin_amdgcn_exp2f(x[2 * w + 1] * kLog2E);
            sum_p += p0 + p1;
            words[w] = float_to_bf16(p0) | (static_cast<uint32_t>(float_to_bf16(p1)) << 16);
        }
        const v4u mine = {words[0], words[1], words[2], words[3]};
        p_other[half][dim] = mine;
        keep_lds_order();
        const v4u other = p_other[1 - half][dim];
        keep_lds_order();
        v8f a;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            a[w] = __uint_as_float(half == 0 ? mine[w] : other[w]);
            a[w + 4] = __uint_as_float(half == 0 ? other[w] : mine[w]);
        }

        // 4. O += P V.
        const uint16_t *v_frag = v_lane + static_cast<int64_t>(f) * kDimFrags * 256;
#pragma unroll
        for (int j = 0; j < kDimFrags; ++j) {
            o[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, load_8f(v_frag + j * 256), o[j]);
        }
    }

    // 5. Output.
    value_other[half][dim] = sum_p;
    keep_lds_order();
    const float inverse = 1.0f / (sum_p + value_other[1 - half][dim]);
    keep_lds_order();
    per_row[dim & 1][dim / 2] = inverse;
    keep_lds_order();
    const v4f r0 = *reinterpret_cast<const v4f *>(&per_row[half][0]);
    const v4f r1 = *reinterpret_cast<const v4f *>(&per_row[half][4]);
    if (ToGemm) {
        const int64_t place = static_cast<int64_t>(first_row / kFrag) * q_heads + head;
        uint16_t *cell = out + place * 2048 + dim % 8;
        float part[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            part[e] = 0.0f;
        }
#pragma unroll
        for (int j = 0; j < kDimFrags; ++j) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float r = e < 4 ? r0[e & 3] : r1[e & 3];
                const uint16_t value = float_to_bf16(o[j][e] * r);
                part[e] += bf16_to_float(value); // Of the rounded values, because WMMA of the next GEMM reads those.
                cell[(2 * j + dim / 8) * kDim + (2 * e + half) * 8] = value;
            }
        }
        keep_lds_order();
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            row_part[2 * e + half][dim] = part[e];
        }
        keep_lds_order();
        if (half == 0) {
            float sum = 0.0f;
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                const v4f v = *reinterpret_cast<const v4f *>(&row_part[dim][a * 4]);
                sum += v.x + v.y + v.z + v.w;
            }
            sums[place * kFrag + (dim & 1) * 8 + dim / 2] = sum;
        }
    } else {
        uint16_t *out_lane = out + head * kDim + dim;
#pragma unroll
        for (int j = 0; j < kDimFrags; ++j) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float r = e < 4 ? r0[e & 3] : r1[e & 3];
                out_lane[(first_row + 2 * e + half) * out_row_stride + j * kFrag] = float_to_bf16(o[j][e] * r);
            }
        }
    }
}

} // namespace

bool sdpa_bf16_supports(int64_t q_rows, int64_t keys, int64_t q_heads, int64_t kv_heads, int64_t head_dim) {
    return q_rows > 0 && q_rows % kFrag == 0 && keys % kFrag == 0 && head_dim == kDim && kv_heads > 0 && q_heads % kv_heads == 0;
}

uint16_t *sdpa_v_tiles(int64_t layer, int64_t kv_heads, int64_t capacity, int64_t head_dim) {
    TORCH_CHECK(head_dim == kDim && capacity % kFrag == 0, "sdpa_v_tiles: unsupported shape");
    static std::mutex mutex;
    static std::map<std::tuple<int64_t, int64_t, int64_t>, uint16_t *> buffers;
    std::lock_guard<std::mutex> lock(mutex);
    uint16_t *&buffer = buffers[std::make_tuple(layer, kv_heads, capacity)];
    if (buffer == nullptr) {
        const size_t bytes = static_cast<size_t>(kv_heads) * capacity * kDim * sizeof(uint16_t);
        TORCH_CHECK(hipMalloc(&buffer, bytes) == hipSuccess, "sdpa_v_tiles: cannot allocate ", bytes, " bytes");
    }
    return buffer;
}

void sdpa_store_v(uint16_t *tiles, const uint16_t *v_cache, int64_t kv_heads, int64_t capacity, int64_t first_key, int64_t keys, hipStream_t stream) {
    TORCH_CHECK(first_key % 32 == 0 && keys % 32 == 0, "sdpa_store_v: keys must be multiples of 32");
    hipLaunchKernelGGL(store_v_kernel, dim3(static_cast<unsigned>(keys / 32), static_cast<unsigned>(kv_heads * kDimFrags)), dim3(32), 0, stream, tiles, v_cache,
                       static_cast<int>(first_key), static_cast<int>(capacity), static_cast<int>(capacity / kFrag));
    check_launch("sdpa_store_v");
}

void sdpa_bf16(uint16_t *out, int64_t out_row_stride, const uint16_t *q, int64_t q_row_stride, const uint16_t *k_cache, const uint16_t *v_tiles, int64_t q_rows,
               int64_t q_heads, int64_t kv_heads, int64_t capacity, int64_t keys, int64_t mask_offset, hipStream_t stream, GemmInputReuse *gemm_x) {
    TORCH_CHECK(sdpa_bf16_supports(q_rows, keys, q_heads, kv_heads, kDim) && capacity % kFrag == 0, "sdpa_bf16: unsupported shape");
    static std::mutex seen_mutex;
    static std::set<int64_t> seen_keys;
    {
        // The log proves which chunks ran here: the model falls back to the torch attention for shapes outside the check above.
        std::lock_guard<std::mutex> lock(seen_mutex);
        if (seen_keys.insert(keys).second) {
            std::fprintf(stderr, "sdpa_bf16: first launch with %lld keys, %lld rows\n", static_cast<long long>(keys), static_cast<long long>(q_rows));
        }
    }
    const float scale = 1.0f / 11.3137085f; // 1 / sqrt(128)
    const dim3 grid(static_cast<unsigned>(q_rows / kFrag), static_cast<unsigned>(q_heads));
    if (gemm_x != nullptr) {
        float *sums = nullptr;
        uint16_t *prepared = nullptr;
        gemm_input_buffers(q_rows, q_heads * kDim, stream, &sums, &prepared);
        hipLaunchKernelGGL(sdpa_kernel<true>, grid, dim3(32), 0, stream, prepared, q, k_cache, v_tiles, out_row_stride, q_row_stride, static_cast<int>(q_heads / kv_heads),
                           static_cast<int>(capacity), static_cast<int>(capacity / kFrag), static_cast<int>(keys), static_cast<int>(mask_offset), scale, sums,
                           static_cast<int>(q_heads));
        gemm_x->mark_prepared(out, q_heads * kDim, stream);
    } else {
        hipLaunchKernelGGL(sdpa_kernel<false>, grid, dim3(32), 0, stream, out, q, k_cache, v_tiles, out_row_stride, q_row_stride, static_cast<int>(q_heads / kv_heads),
                           static_cast<int>(capacity), static_cast<int>(capacity / kFrag), static_cast<int>(keys), static_cast<int>(mask_offset), scale,
                           static_cast<float *>(nullptr), static_cast<int>(q_heads));
    }
    check_launch("sdpa_bf16");
}

} // namespace heteromega
