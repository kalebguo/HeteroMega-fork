// GPU side of the HeteroMega FFN split. See kernels/gpu/README.md.
//
// One device owns intermediate channels [begin, end) of an FFN layer and computes
//   Y = (SiLU(X W_gate[:, begin:end]) * (X W_up[:, begin:end])) W_down[begin:end, :]
// The partial outputs of the GPU and the NPU are added by residual_add2().
#pragma once

#include <hip/hip_runtime.h>
#include <torch/torch.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace heteromega {

constexpr int64_t kTileK = 128;      // rows per packed tile; also the quantization group size
constexpr int64_t kTileN = 64;       // columns per packed tile
constexpr int64_t kTileBytes = 4352; // 4096 weight bytes + 64 bf16 scales + 128 zero bytes

// HeteroMosaic's GEMM reads whole 128-row tiles of its input (only the output writes are guarded).
// For M > 1, input storage must therefore hold padded_rows(M) rows.
inline int64_t padded_rows(int64_t m) { return (m + kTileK - 1) / kTileK * kTileK; }

// ---- Weight packing (CPU) --------------------------------------------------

// Packs a [K, N] W4A16 matrix into the HeteroMosaic tile layout. Both
// hipkernels::w4a16_{gemm,gemv}_fused_packed and the NPU binaries read this layout.
//   qweight: [K, N] uint8, values 0..15
//   scales:  [K / 128, N] bfloat16
//   zeros:   [K / 128, N] int8
// K must be a multiple of 128 and N a multiple of 64. Returns a flat uint8 CPU tensor.
torch::Tensor pack_w4a16(const torch::Tensor &qweight, const torch::Tensor &scales, const torch::Tensor &zeros);

// AWQ tensors of one FFN layer in [in_features, out_features] order, as returned by
// HeteroMosaic _unpack_awq_qweight / _unpack_awq_qzeros. H = hidden, I = intermediate.
struct FfnLayerQuant {
    torch::Tensor gate_q, gate_s, gate_z; // [H, I], [H/128, I], [H/128, I]
    torch::Tensor up_q, up_s, up_z;       // [H, I], [H/128, I], [H/128, I]
    torch::Tensor down_q, down_s, down_z; // [I, H], [I/128, H], [I/128, H]
};

// Column order of a packed gate_up matrix of n channels.
enum class GateUpLayout {
    kBlocked,     // [gate 0..n) | up 0..n)]
    kInterleaved, // per 64 channels: their 64 gate columns, then their 64 up columns
};

// One device's packed share of an FFN layer: intermediate channels [begin, end).
// The packed tensors can hold more channels than [begin, end) (see prefix()). The forward pass then uses only the
// first channels() of them.
struct FfnSliceWeights {
    torch::Tensor gate_up; // packed [H, 2 * stored_channels()], column order `layout`
    torch::Tensor down;    // packed [stored_channels(), H]
    int64_t begin = 0;
    int64_t end = 0;
    int64_t hidden = 0;
    GateUpLayout layout = GateUpLayout::kBlocked;
    int64_t packed_channels = 0; // channels in the packed tensors; 0 means channels()

    int64_t channels() const { return end - begin; }
    int64_t stored_channels() const { return packed_channels > 0 ? packed_channels : channels(); }
    FfnSliceWeights to(const torch::Device &device) const;

    // The same tensors, used for channels [begin, begin + n) only. No copy.
    // This lets one packed copy serve several split points, for example one for prefill and one for decode.
    // n must be a multiple of 128. n < stored_channels() needs GateUpLayout::kInterleaved, because only then are the
    // gate and up columns of the first n channels the first 2n columns.
    FfnSliceWeights prefix(int64_t n) const;
};

// Packs channels [begin, end). begin and end must be multiples of 128.
// Split points that are multiples of 512 (8 tile columns) keep each slice a tile subset of the full packed buffer.
FfnSliceWeights pack_ffn_slice(const FfnLayerQuant &layer, int64_t begin, int64_t end,
                               GateUpLayout layout = GateUpLayout::kBlocked);

// Position of tile (tile column c, tile row r) in a packed matrix of tiles_k x tiles_n tiles (order: see pack_w4a16).
int64_t packed_tile_index(int64_t c, int64_t r, int64_t tiles_k, int64_t tiles_n);

// pack_ffn_slice from matrices that are already packed, as HeteroMosaic's QuantizedLinear holds them (packed_params_,
// padPackedWeights off). Copies whole tiles on the tensors' device and gives the same bytes as pack_ffn_slice.
//   gate, up: packed [hidden, inter]; down: packed [inter, hidden]
// begin and end must be multiples of 128. For [0, inter) the result's down is `down` itself, not a copy.
FfnSliceWeights slice_packed_ffn(const torch::Tensor &gate, const torch::Tensor &up, const torch::Tensor &down, int64_t hidden,
                                 int64_t inter, int64_t begin, int64_t end, GateUpLayout layout = GateUpLayout::kBlocked);

// ---- GPU kernels -------------------------------------------------------------

// act[m, j] = SiLU(gate[m, j]) * up[m, j] for j < n, where gate and up are columns of gate_up in `layout` order.
// bf16 in and out, fp32 math, one rounding. n must be a multiple of 8 (of 64 for kInterleaved).
void silu_mul(const torch::Tensor &gate_up, torch::Tensor &act, hipStream_t stream,
              GateUpLayout layout = GateUpLayout::kBlocked);

// W4A16 GEMV on the packed layout: out[1, n] = x[1, k] @ W[:k, :n], where W is a packed [packed_k, packed_n] matrix.
// Unlike HeteroMosaic's GEMV, it loads 16 weight bytes per thread, accumulates in fp32, can split K over blocks,
// and can skip rows of W (packed_k > k). split_k = 0 picks a split from the shape.
void gemv_w4a16(torch::Tensor &out, const torch::Tensor &x, const torch::Tensor &packed, int64_t k, int64_t n,
                int64_t packed_k, int64_t packed_n, int split_k, hipStream_t stream);

// Reducer R-GPU: x += y_g + y_n. x is bf16 [M, H]. y_g and y_n are bf16 or fp32 [M, H].
void residual_add2(torch::Tensor &x, const torch::Tensor &y_g, const torch::Tensor &y_n, hipStream_t stream);

// Reducer R-CPU: the same math on CPU tensors (for example pinned host memory).
// The caller must make sure that both devices have finished writing y_g and y_n.
void residual_add2_cpu(torch::Tensor &x, const torch::Tensor &y_g, const torch::Tensor &y_n);

// ---- GPU slice forward -------------------------------------------------------

// Scratch buffers for gpu_ffn_slice_forward(). Allocate once for the largest M and the most channels.
// Rows are padded to a multiple of 128 (see padded_rows()). Smaller M or fewer channels use the leading elements.
struct GpuFfnBuffers {
    torch::Tensor gate_up; // [rows, 2n] bf16
    torch::Tensor act;     // [rows, n] bf16
    torch::Tensor x_pad;   // [rows, H] bf16: copy of x_norm when its storage is not padded

    static GpuFfnBuffers allocate(int64_t max_m, int64_t channels, int64_t hidden, const torch::Device &device);
};

// GPU share of one FFN sublayer. Enqueues 3 kernels on `stream` and does not synchronize.
// If M > 1 and x_norm's storage is not padded to 128 rows, it first copies x_norm into buf.x_pad.
//   x_norm: [M, H] bf16, contiguous (the RMSNorm output)
//   y:      [M, H] bf16, contiguous (the partial output Y_G)
// M == 1 uses gemv_w4a16. M > 1 uses the HeteroMosaic GEMM kernel.
void gpu_ffn_slice_forward(const FfnSliceWeights &w, const torch::Tensor &x_norm, GpuFfnBuffers &buf, torch::Tensor &y,
                           hipStream_t stream);

enum class GemvKernel {
    kHeteroMega,   // gemv_w4a16
    kHeteroMosaic, // hipkernels::w4a16_gemv_fused_packed (the baseline's kernel)
};

struct LinearOptions {
    int64_t packed_k = 0; // rows of the whole packed matrix; 0 means k
    int64_t packed_n = 0; // columns of the whole packed matrix; 0 means n
    GemvKernel gemv = GemvKernel::kHeteroMega;
    int gemv_split_k = 0; // see gemv_w4a16
};

// Packed matmul: out[M, n] = in[M, k] @ W[:k, :n]. GEMV for M == 1, GEMM otherwise.
// W is the whole packed matrix, [packed_k, packed_n] (by default [k, n]).
// For M > 1, in's storage must hold padded_rows(M) rows; the extra rows can hold any values.
void packed_linear(torch::Tensor &out, const torch::Tensor &in, const torch::Tensor &packed, int64_t k, int64_t n,
                   hipStream_t stream, const LinearOptions &opt = {});

// ---- Model integration -------------------------------------------------------

// The GPU FFN of every layer of a model, for the model's forward loop.
// Each layer keeps one packed copy of all its channels (gate/up kInterleaved). A forward pass uses the first
// s_prefill channels when M > 1 and the first s_decode channels when M == 1 (FfnSliceWeights::prefix, no copy).
// The other channels are the NPU's share. This class does not compute them.
class GpuFfnModel {
  public:
    // Adds the next layer from HeteroMosaic's packed gate [H, I], up [H, I] and down [I, H] (see slice_packed_ffn).
    // Copies gate and up into one interleaved gate_up tensor. Uses down as it is.
    void add_layer(const torch::Tensor &gate, const torch::Tensor &up, const torch::Tensor &down, int64_t hidden, int64_t inter);

    // GPU channels for prefill (M > 1) and decode (M == 1). Multiples of 128 in (0, I]. Default: all I channels.
    void set_split(int64_t s_prefill, int64_t s_decode);

    int64_t layers() const { return static_cast<int64_t>(layers_.size()); }
    int64_t channels(int64_t m) const { return m > 1 ? s_prefill_ : s_decode_; }

    // y = the GPU share of FFN layer `layer` for x_norm. x_norm and y: bf16 [M, H], contiguous.
    // Enqueues 3 kernels on `stream` and does not synchronize. Pass the current stream: the scratch buffers come
    // from the caching allocator on it. Threads with different streams get different scratch buffers.
    void forward(int64_t layer, const torch::Tensor &x_norm, torch::Tensor &y, hipStream_t stream);

  private:
    std::vector<FfnSliceWeights> layers_;
    int64_t s_prefill_ = 0;
    int64_t s_decode_ = 0;
    std::mutex mutex_;
    std::unordered_map<hipStream_t, GpuFfnBuffers> buffers_;
};

} // namespace heteromega
