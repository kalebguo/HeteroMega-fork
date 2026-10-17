#include "heteromega/gpu_ffn.hpp"

#include "hipkernels/w4a16_gemm_packed.hpp"

namespace heteromega {
namespace {

// True if t's storage holds `rows` rows starting at t's first element.
bool storage_has_rows(const torch::Tensor &t, int64_t rows) {
    const int64_t needed = (t.storage_offset() + rows * t.stride(0)) * t.element_size();
    return static_cast<int64_t>(t.storage().nbytes()) >= needed;
}

// Contiguous [rows, width] view of the first rows * width elements of `buffer`.
torch::Tensor leading(const torch::Tensor &buffer, int64_t rows, int64_t width) {
    return buffer.view(-1).narrow(0, 0, rows * width).view({rows, width});
}

} // namespace

void packed_linear(torch::Tensor &out, const torch::Tensor &in, const torch::Tensor &packed, int64_t k, int64_t n,
                   hipStream_t stream, const LinearOptions &opt) {
    const int64_t pk = opt.packed_k > 0 ? opt.packed_k : k;
    const int64_t pn = opt.packed_n > 0 ? opt.packed_n : n;
    TORCH_CHECK(in.dim() == 2 && in.size(1) == k && in.is_contiguous(), "packed_linear: input must be contiguous [M, ", k, "]");
    const int64_t m = in.size(0);
    TORCH_CHECK(m > 0, "packed_linear: M must be > 0");
    TORCH_CHECK(k % kTileK == 0 && n % kTileN == 0, "packed_linear: k must be a multiple of 128 and n of 64");
    TORCH_CHECK(m == 1 || k <= 1024 || n % 128 == 0, "packed_linear: the GEMM needs n % 128 == 0 when k > 1024");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == in.size(0) && out.size(1) == n && out.is_contiguous(),
                "packed_linear: output must be contiguous [M, ", n, "]");
    TORCH_CHECK(in.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kBFloat16, "packed_linear: expected bf16");
    TORCH_CHECK(k <= pk && n <= pn && pk % kTileK == 0 && pn % kTileN == 0, "packed_linear: packed matrix [", pk, ", ", pn,
                "] does not contain [", k, ", ", n, "]");
    TORCH_CHECK(packed.numel() == (pk / kTileK) * (pn / kTileN) * kTileBytes, "packed_linear: packed size does not match ", pk, "x", pn);
    TORCH_CHECK(packed.device() == in.device() && out.device() == in.device(), "packed_linear: all tensors must be on one device");

    if (m == 1 && opt.gemv == GemvKernel::kHeteroMega) {
        gemv_w4a16(out, in, packed, k, n, pk, pn, opt.gemv_split_k, stream);
    } else if (m == 1) {
        TORCH_CHECK(pk == k, "packed_linear: the HeteroMosaic GEMV cannot skip rows of the packed matrix");
        hipkernels::w4a16_gemv_fused_packed(out, in, packed, k, n, pn, 0, stream);
    } else {
        TORCH_CHECK(storage_has_rows(in, padded_rows(m)), "packed_linear: input storage must hold ", padded_rows(m),
                    " rows for M = ", m, " (the GEMM reads whole 128-row tiles)");
        hipkernels::w4a16_gemm_fused_packed_tiles(out, in, packed, k, n, pk / kTileK, pn / kTileN, stream);
    }
}

GpuFfnBuffers GpuFfnBuffers::allocate(int64_t max_m, int64_t channels, int64_t hidden, const torch::Device &device) {
    const auto opt = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    const int64_t rows = padded_rows(max_m);
    return {torch::empty({rows, 2 * channels}, opt), torch::empty({rows, channels}, opt), torch::empty({rows, hidden}, opt)};
}

void gpu_ffn_slice_forward(const FfnSliceWeights &w, const torch::Tensor &x_norm, GpuFfnBuffers &buf, torch::Tensor &y,
                           hipStream_t stream) {
    const int64_t n = w.channels();
    const int64_t stored = w.stored_channels();
    const int64_t h = w.hidden;
    const int64_t m = x_norm.size(0);
    const int64_t rows = padded_rows(m);
    TORCH_CHECK(n == stored || w.layout == GateUpLayout::kInterleaved,
                "gpu_ffn_slice_forward: a prefix of the packed channels needs GateUpLayout::kInterleaved");
    TORCH_CHECK(buf.gate_up.numel() >= rows * 2 * n && buf.act.numel() >= rows * n && buf.x_pad.size(0) >= rows &&
                    buf.x_pad.size(1) == h,
                "gpu_ffn_slice_forward: buffers do not fit M = ", m, ", ", n, " channels, H = ", h);
    TORCH_CHECK(x_norm.dim() == 2 && x_norm.size(1) == h && x_norm.is_contiguous(), "gpu_ffn_slice_forward: x_norm must be contiguous [M, H]");

    torch::Tensor x = x_norm;
    if (m > 1 && !storage_has_rows(x_norm, padded_rows(m))) {
        x = buf.x_pad.narrow(0, 0, m);
        const hipError_t err = hipMemcpyAsync(x.data_ptr(), x_norm.data_ptr(), x_norm.nbytes(), hipMemcpyDefault, stream);
        TORCH_CHECK(err == hipSuccess, "gpu_ffn_slice_forward: copy to x_pad failed: ", hipGetErrorString(err));
    }

    auto gate_up = leading(buf.gate_up, m, 2 * n);
    auto act = leading(buf.act, m, n);
    LinearOptions gate_up_opt, down_opt;
    gate_up_opt.packed_n = 2 * stored;
    down_opt.packed_k = stored;
    packed_linear(gate_up, x, w.gate_up, h, 2 * n, stream, gate_up_opt);
    silu_mul(gate_up, act, stream, w.layout);
    packed_linear(y, act, w.down, n, h, stream, down_opt);
}

} // namespace heteromega
