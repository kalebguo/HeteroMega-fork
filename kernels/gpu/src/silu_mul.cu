#include "bf16_device.hpp"
#include "heteromega/gpu_ffn.hpp"

namespace heteromega {
namespace {

// Each work item handles 8 consecutive channels of one row (one 16-byte load each from gate and up).
// Blocked: gate of channel c is column c, up is column n + c. Interleaved: gate is column (c / 64) * 128 + c % 64,
// up is 64 columns after it.
__global__ void __launch_bounds__(256)
    silu_mul_kernel(const uint16_t *__restrict__ gate_up, uint16_t *__restrict__ act, int64_t rows, int64_t n, int64_t ld_in,
                    int64_t ld_out, bool interleaved) {
    const int64_t vec_per_row = n / 8;
    const int64_t total = rows * vec_per_row;
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t r = i / vec_per_row;
        const int64_t c = (i - r * vec_per_row) * 8;
        const int64_t gc = interleaved ? (c / 64) * 128 + c % 64 : c;
        const int64_t uc = interleaved ? gc + 64 : n + c;
        float g[8], u[8], o[8];
        load8(gate_up + r * ld_in + gc, g);
        load8(gate_up + r * ld_in + uc, u);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            o[k] = g[k] / (1.0f + __expf(-g[k])) * u[k];
        }
        store8(act + r * ld_out + c, o);
    }
}

} // namespace

void silu_mul(const torch::Tensor &gate_up, torch::Tensor &act, hipStream_t stream, GateUpLayout layout) {
    TORCH_CHECK(gate_up.dim() == 2 && act.dim() == 2, "silu_mul: expected 2-D tensors");
    TORCH_CHECK(gate_up.scalar_type() == torch::kBFloat16 && act.scalar_type() == torch::kBFloat16, "silu_mul: expected bf16");
    TORCH_CHECK((gate_up.is_cuda() || gate_up.is_pinned()) && (act.is_cuda() || act.is_pinned()),
                "silu_mul: tensors must be GPU or pinned host memory");
    const int64_t rows = gate_up.size(0);
    const int64_t n = act.size(1);
    TORCH_CHECK(act.size(0) == rows && gate_up.size(1) == 2 * n, "silu_mul: gate_up must be [M, 2n] and act [M, n]");
    const bool interleaved = layout == GateUpLayout::kInterleaved;
    TORCH_CHECK(n % (interleaved ? 64 : 8) == 0, "silu_mul: channel count must be a multiple of ", interleaved ? 64 : 8);
    TORCH_CHECK(gate_up.stride(1) == 1 && act.stride(1) == 1, "silu_mul: last dimension must be contiguous");
    TORCH_CHECK(gate_up.stride(0) % 8 == 0 && act.stride(0) % 8 == 0, "silu_mul: row stride must be a multiple of 8");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(gate_up.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(act.data_ptr()) % 16 == 0,
                "silu_mul: tensors must be 16-byte aligned");

    const int64_t work = rows * (n / 8);
    if (work == 0) {
        return;
    }
    hipLaunchKernelGGL(silu_mul_kernel, grid_for(work), dim3(256), 0, stream,
                       reinterpret_cast<const uint16_t *>(gate_up.data_ptr()), reinterpret_cast<uint16_t *>(act.data_ptr()), rows, n,
                       gate_up.stride(0), act.stride(0), interleaved);
    check_launch("silu_mul");
}

} // namespace heteromega
