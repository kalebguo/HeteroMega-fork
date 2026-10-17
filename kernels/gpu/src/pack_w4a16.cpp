// Port of HeteroMosaic _pack_weights_packed (py/unified_llm_w4a16/llama3_8b_w4a16_model.py)
// without the padding path. tools/check_packing.py checks that both produce the same bytes.
#include "heteromega/gpu_ffn.hpp"

#include <cstring>

namespace heteromega {

torch::Tensor pack_w4a16(const torch::Tensor &qweight, const torch::Tensor &scales, const torch::Tensor &zeros) {
    TORCH_CHECK(qweight.dim() == 2, "pack_w4a16: qweight must be [K, N]");
    const int64_t K = qweight.size(0);
    const int64_t N = qweight.size(1);
    TORCH_CHECK(K % kTileK == 0 && N % kTileN == 0, "pack_w4a16: K must be a multiple of 128 and N of 64, got ", K, "x", N);
    const int64_t groups = K / kTileK;
    TORCH_CHECK(scales.dim() == 2 && scales.size(0) == groups && scales.size(1) == N, "pack_w4a16: scales must be [K/128, N]");
    TORCH_CHECK(zeros.dim() == 2 && zeros.size(0) == groups && zeros.size(1) == N, "pack_w4a16: zeros must be [K/128, N]");

    const auto q = qweight.to(torch::kCPU, torch::kUInt8).contiguous();
    const auto s = scales.to(torch::kCPU, torch::kBFloat16).contiguous();
    const auto z = zeros.to(torch::kCPU, torch::kInt8).contiguous();
    const uint8_t *qp = q.data_ptr<uint8_t>();
    const auto *sp = reinterpret_cast<const uint16_t *>(s.data_ptr<at::BFloat16>());
    const int8_t *zp = z.data_ptr<int8_t>();

    const int64_t tiles_k = K / kTileK;
    const int64_t tiles_n = N / kTileN;
    auto out = torch::empty({tiles_k * tiles_n * kTileBytes}, torch::kUInt8);
    uint8_t *op = out.data_ptr<uint8_t>();

    // Tile order: column group (tile column % 8), then tile column, then tile row.
    int64_t t = 0;
    for (int64_t g = 0; g < 8; ++g) {
        for (int64_t c = g; c < tiles_n; c += 8) {
            for (int64_t r = 0; r < tiles_k; ++r, ++t) {
                uint8_t *tile = op + t * kTileBytes;
                const uint8_t *src = qp + (r * kTileK) * N + c * kTileN;

                // Byte ((a*8 + cc)*8 + b)*4 + j holds row a*8 + b, columns cc*8 + 2j (low) and cc*8 + 2j + 1 (high).
                for (int a = 0; a < 16; ++a) {
                    for (int cc = 0; cc < 8; ++cc) {
                        for (int b = 0; b < 8; ++b) {
                            const uint8_t *row = src + (a * 8 + b) * N + cc * 8;
                            uint8_t *dst = tile + ((a * 8 + cc) * 8 + b) * 4;
                            for (int j = 0; j < 4; ++j) {
                                dst[j] = static_cast<uint8_t>((row[2 * j] & 0x0F) | ((row[2 * j + 1] & 0x0F) << 4));
                            }
                        }
                    }
                }

                // 64 bf16 scales of group r (group size == tile height).
                std::memcpy(tile + 4096, sp + r * N + c * kTileN, 64 * sizeof(uint16_t));

                // Zeros: each run of 8 is stored twice, so column i is at (i / 8) * 16 + i % 8.
                uint8_t *tz = tile + 4096 + 128;
                const int8_t *zr = zp + r * N + c * kTileN;
                for (int i = 0; i < 64; ++i) {
                    const auto v = static_cast<uint8_t>(zr[i]);
                    tz[(i / 8) * 16 + i % 8] = v;
                    tz[(i / 8) * 16 + 8 + i % 8] = v;
                }
            }
        }
    }
    return out;
}

FfnSliceWeights pack_ffn_slice(const FfnLayerQuant &layer, int64_t begin, int64_t end, GateUpLayout layout) {
    TORCH_CHECK(layer.gate_q.dim() == 2, "pack_ffn_slice: gate_q must be [H, I]");
    const int64_t hidden = layer.gate_q.size(0);
    const int64_t inter = layer.gate_q.size(1);
    TORCH_CHECK(0 <= begin && begin < end && end <= inter, "pack_ffn_slice: bad channel range [", begin, ", ", end, ")");
    TORCH_CHECK(begin % kTileK == 0 && end % kTileK == 0, "pack_ffn_slice: begin and end must be multiples of 128");
    TORCH_CHECK(layer.down_q.size(0) == inter && layer.down_q.size(1) == hidden, "pack_ffn_slice: down_q must be [I, H]");

    auto cols = [&](const torch::Tensor &t) { return t.slice(1, begin, end); };
    auto rows = [&](const torch::Tensor &t, int64_t per_row) { return t.slice(0, begin / per_row, end / per_row); };

    // [rows, n] gate and up -> [rows, 2n] in `layout` order.
    const int64_t n = end - begin;
    auto join = [&](const torch::Tensor &g, const torch::Tensor &u) {
        if (layout == GateUpLayout::kBlocked) {
            return torch::cat({g, u}, 1);
        }
        const int64_t r = g.size(0);
        return torch::stack({g.reshape({r, n / kTileN, kTileN}), u.reshape({r, n / kTileN, kTileN})}, 2).reshape({r, 2 * n});
    };

    FfnSliceWeights w;
    w.gate_up = pack_w4a16(join(cols(layer.gate_q), cols(layer.up_q)), join(cols(layer.gate_s), cols(layer.up_s)),
                           join(cols(layer.gate_z), cols(layer.up_z)));
    w.down = pack_w4a16(rows(layer.down_q, 1), rows(layer.down_s, kTileK), rows(layer.down_z, kTileK));
    w.begin = begin;
    w.end = end;
    w.hidden = hidden;
    w.layout = layout;
    return w;
}

int64_t packed_tile_index(int64_t c, int64_t r, int64_t tiles_k, int64_t tiles_n) {
    // Column group g holds tile columns g, g + 8, ...: (tiles_n - g + 7) / 8 of them.
    int64_t columns_before = c / 8;
    for (int64_t g = 0; g < c % 8; ++g) {
        columns_before += (tiles_n - g + 7) / 8;
    }
    return columns_before * tiles_k + r;
}

FfnSliceWeights slice_packed_ffn(const torch::Tensor &gate, const torch::Tensor &up, const torch::Tensor &down, int64_t hidden,
                                 int64_t inter, int64_t begin, int64_t end, GateUpLayout layout) {
    TORCH_CHECK(hidden % kTileK == 0 && inter % kTileK == 0, "slice_packed_ffn: hidden and inter must be multiples of 128");
    TORCH_CHECK(0 <= begin && begin < end && end <= inter, "slice_packed_ffn: bad channel range [", begin, ", ", end, ")");
    TORCH_CHECK(begin % kTileK == 0 && end % kTileK == 0, "slice_packed_ffn: begin and end must be multiples of 128");
    const int64_t matrix_tiles = (hidden / kTileK) * (inter / kTileN);
    for (const auto *t : {&gate, &up, &down}) {
        TORCH_CHECK(t->defined(), "slice_packed_ffn: a packed matrix is undefined");
        TORCH_CHECK(t->scalar_type() == torch::kUInt8 && t->is_contiguous() && t->numel() == matrix_tiles * kTileBytes,
                    "slice_packed_ffn: expected contiguous uint8 packed matrices of ", matrix_tiles * kTileBytes, " bytes, got ",
                    t->numel());
        TORCH_CHECK(t->device() == gate.device(), "slice_packed_ffn: all matrices must be on one device");
    }

    const int64_t n = end - begin;
    const auto gather = [&](const torch::Tensor &tiles, const std::vector<int64_t> &idx) {
        const auto i = torch::from_blob(const_cast<int64_t *>(idx.data()), {static_cast<int64_t>(idx.size())}, torch::kInt64)
                           .to(tiles.device());
        return tiles.index_select(0, i).view(-1);
    };

    // gate_up [hidden, 2n]: output tile column c comes from tile column src of gate or up. up's tiles follow gate's.
    std::vector<int64_t> idx;
    const int64_t tiles_k = hidden / kTileK, tiles_n = inter / kTileN, out_tiles_n = 2 * n / kTileN, half = n / kTileN;
    for (int64_t g = 0; g < 8; ++g) {
        for (int64_t c = g; c < out_tiles_n; c += 8) {
            const bool is_up = layout == GateUpLayout::kBlocked ? c >= half : c % 2 == 1;
            const int64_t src = begin / kTileN + (layout == GateUpLayout::kBlocked ? c % half : c / 2);
            for (int64_t r = 0; r < tiles_k; ++r) {
                idx.push_back((is_up ? matrix_tiles : 0) + packed_tile_index(src, r, tiles_k, tiles_n));
            }
        }
    }
    FfnSliceWeights w;
    w.gate_up = gather(torch::cat({gate.view({-1, kTileBytes}), up.view({-1, kTileBytes})}), idx);

    // down [n, hidden]: tile rows [begin / 128, end / 128) of every tile column.
    if (begin == 0 && end == inter) {
        w.down = down;
    } else {
        idx.clear();
        const int64_t down_tiles_k = inter / kTileK, down_tiles_n = hidden / kTileN;
        for (int64_t g = 0; g < 8; ++g) {
            for (int64_t c = g; c < down_tiles_n; c += 8) {
                for (int64_t r = begin / kTileK; r < end / kTileK; ++r) {
                    idx.push_back(packed_tile_index(c, r, down_tiles_k, down_tiles_n));
                }
            }
        }
        w.down = gather(down.view({-1, kTileBytes}), idx);
    }
    w.begin = begin;
    w.end = end;
    w.hidden = hidden;
    w.layout = layout;
    return w;
}

FfnSliceWeights FfnSliceWeights::prefix(int64_t n) const {
    TORCH_CHECK(n > 0 && n <= stored_channels() && n % kTileK == 0, "FfnSliceWeights::prefix: n must be a multiple of 128 in (0, ",
                stored_channels(), "], got ", n);
    TORCH_CHECK(n == stored_channels() || layout == GateUpLayout::kInterleaved,
                "FfnSliceWeights::prefix: a shorter prefix needs GateUpLayout::kInterleaved");
    FfnSliceWeights w = *this;
    w.packed_channels = stored_channels();
    w.end = begin + n;
    return w;
}

FfnSliceWeights FfnSliceWeights::to(const torch::Device &device) const {
    FfnSliceWeights w = *this;
    w.gate_up = gate_up.to(device);
    w.down = down.to(device);
    return w;
}

} // namespace heteromega
