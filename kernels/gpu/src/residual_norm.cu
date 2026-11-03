// The work between two GEMMs of the model as one dispatch: x += a + b, RMSNorm of x, and the input layout of gemm_w4a16. See residual_norm_bf16() in gpu_ffn.hpp.
//
// As separate kernels this reads and writes the activations 4 times between two layers: one wave per row and one lane per K tile avoid that.
// The layout of the prepared input: see prepare_input_kernel in gemm_w4a16.cu.
#include "bf16_device.hpp"
#include "heteromega/gpu_ffn.hpp"

namespace heteromega {
namespace {

typedef float v4f __attribute__((ext_vector_type(4)));
typedef uint32_t v4u __attribute__((ext_vector_type(4)));

constexpr int kFrag = 16;
constexpr int kTiles = 32;
constexpr int kOctets = kTileK / 8;
constexpr int kHidden = kTiles * kTileK;

__device__ __forceinline__ float low_float(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float high_float(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
// Lanes of a wave read in LDS what the other lanes wrote. A wave runs its LDS operations in order, so only the compiler must keep the order.
__device__ __forceinline__ void keep_lds_order() { __asm__ volatile("" ::: "memory"); }

// a and b can be null. The value of `normed` is optional too: a null pointer means that only the prepared input holds the result.
__global__ void __launch_bounds__(32)
    residual_norm_kernel(uint16_t *__restrict__ x, const uint16_t *__restrict__ a, const uint16_t *__restrict__ b, const uint16_t *__restrict__ weight, uint16_t *__restrict__ normed,
                         float *__restrict__ sums, uint16_t *__restrict__ xt, int64_t x_stride, int64_t a_stride, int64_t b_stride, int64_t normed_stride, float eps) {
    __shared__ __align__(16) float parts[kTiles];
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const bool has_a = a != nullptr, has_b = b != nullptr;

    v4u values[kOctets];
    float sum_sq = 0.0f;
    uint16_t *x_lane = x + row * x_stride + t * kTileK;
#pragma unroll
    for (int i = 0; i < kOctets; ++i) {
        v4u v = *reinterpret_cast<const v4u *>(x_lane + i * 8);
        if (has_a) {
            const v4u va = *reinterpret_cast<const v4u *>(a + row * a_stride + t * kTileK + i * 8);
            v4u vb = {0u, 0u, 0u, 0u};
            if (has_b) {
                vb = *reinterpret_cast<const v4u *>(b + row * b_stride + t * kTileK + i * 8);
            }
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t lo = float_to_bf16(low_float(v[w]) + low_float(va[w]) + low_float(vb[w]));
                const uint32_t hi = float_to_bf16(high_float(v[w]) + high_float(va[w]) + high_float(vb[w]));
                v[w] = lo | (hi << 16);
            }
            *reinterpret_cast<v4u *>(x_lane + i * 8) = v;
        }
        values[i] = v;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float lo = low_float(v[w]), hi = high_float(v[w]);
            sum_sq += lo * lo + hi * hi;
        }
    }

    parts[t] = sum_sq;
    keep_lds_order();
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < kTiles / 4; ++i) {
        const v4f v = *reinterpret_cast<const v4f *>(&parts[i * 4]);
        total += v.x + v.y + v.z + v.w;
    }
    const float rsqrt_value = rsqrtf(total / kHidden + eps);

    const int64_t place = static_cast<int64_t>(row / kFrag) * kTiles + t;
    uint16_t *q = xt + place * 2048 + (row % kFrag) * 8;
    uint16_t *normed_lane = normed == nullptr ? nullptr : normed + row * normed_stride + t * kTileK;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kOctets; ++i) {
        const v4u w8 = *reinterpret_cast<const v4u *>(weight + t * kTileK + i * 8);
        v4u out;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint16_t lo = float_to_bf16(low_float(values[i][w]) * rsqrt_value * low_float(w8[w]));
            const uint16_t hi = float_to_bf16(high_float(values[i][w]) * rsqrt_value * high_float(w8[w]));
            sum += bf16_to_float(lo) + bf16_to_float(hi); // Of the rounded values, because WMMA of the next GEMM reads those.
            out[w] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
        }
        *reinterpret_cast<v4u *>(q + i * kFrag * 8) = out;
        if (normed != nullptr) {
            *reinterpret_cast<v4u *>(normed_lane + i * 8) = out;
        }
    }
    sums[place * kFrag + (row & 1) * 8 + (row & 15) / 2] = sum;
}

} // namespace

bool residual_norm_supports(int64_t m, int64_t h) { return m > 0 && m % kFrag == 0 && h == kHidden; }

void residual_norm_bf16(uint16_t *x, const uint16_t *a, const uint16_t *b, const uint16_t *weight, uint16_t *normed, const void *gemm_x, GemmInputReuse &gemm_input, int64_t m, int64_t h,
                        int64_t x_stride, int64_t a_stride, int64_t b_stride, int64_t normed_stride, float eps, hipStream_t stream) {
    TORCH_CHECK(residual_norm_supports(m, h), "residual_norm_bf16: unsupported shape ", m, ", ", h);
    float *sums = nullptr;
    uint16_t *xt = nullptr;
    gemm_input_buffers(m, h, stream, &sums, &xt);
    hipLaunchKernelGGL(residual_norm_kernel, dim3(static_cast<unsigned>(m)), dim3(32), 0, stream, x, a, b, weight, normed, sums, xt, x_stride, a_stride, b_stride, normed_stride, eps);
    check_launch("residual_norm_bf16");
    gemm_input.mark_prepared(gemm_x, h, stream);
}

} // namespace heteromega
