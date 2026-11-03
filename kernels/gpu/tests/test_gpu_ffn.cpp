// Correctness tests for the GPU side of the FFN split.
//   Level 1: each kernel against an fp32 reference (random signed inputs, nonuniform weights, scales and zeros).
//            slice_packed_ffn must give the bytes of pack_ffn_slice.
//   Level 2: Y_G + Y_N of two channel slices against the unsplit fp32 FFN, with the GPU slice packed on its own
//            and as a prefix of all channels (FfnSliceWeights::prefix). GpuFfnModel must give the prefix's bytes.
// Run on the AUP cloud in the environment of docs/cloud-baseline.md:
//   ./test_gpu_ffn          (full: M in {1, 37, 128, 512, 2048}, s in {512, 7168, 10240, 13824})
//   ./test_gpu_ffn --quick  (M in {1, 37, 128}, s = 7168)
// Exit code 0 means every check passed.
#include "heteromega/gpu_ffn.hpp"

#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <c10/hip/HIPStream.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace heteromega;

namespace {

constexpr int64_t kHidden = 4096;
constexpr int64_t kInter = 14336;
int g_failures = 0;

struct Tol {
    double max_rel;
    double rel_l2;
    double cos;
};
constexpr Tol kExact{1e-2, 5e-3, 0.99999}; // one bf16 rounding
constexpr Tol kLevel{1e-2, 1e-2, 0.9999};  // levels 1 and 2

// max_rel = max |got - ref| / max |ref|.
void check(const std::string &name, const torch::Tensor &got, const torch::Tensor &ref, Tol tol) {
    const auto g = got.to(torch::kCPU, torch::kFloat64).flatten();
    const auto r = ref.to(torch::kCPU, torch::kFloat64).flatten();
    const double max_rel = (g - r).abs().max().item<double>() / (r.abs().max().item<double>() + 1e-30);
    const double rel_l2 = ((g - r).norm() / (r.norm() + 1e-30)).item<double>();
    const double cos = (torch::dot(g, r) / (g.norm() * r.norm() + 1e-30)).item<double>();
    const bool ok = max_rel <= tol.max_rel && rel_l2 <= tol.rel_l2 && cos >= tol.cos;
    std::printf("%-52s %s  max_rel=%.2e  rel_l2=%.2e  cos=%.7f\n", name.c_str(), ok ? "PASS" : "FAIL", max_rel, rel_l2, cos);
    if (!ok) {
        ++g_failures;
    }
}

// Bitwise equality, for two paths that must give the same bytes.
void check_equal(const std::string &name, const torch::Tensor &got, const torch::Tensor &ref) {
    const bool ok = got.sizes() == ref.sizes() && torch::equal(got.cpu(), ref.cpu());
    std::printf("%-52s %s  (bitwise)\n", name.c_str(), ok ? "PASS" : "FAIL");
    if (!ok) {
        ++g_failures;
    }
}

struct Quant {
    torch::Tensor q, s, z;
};

// AWQ-like random weights: q in [0, 15], zeros in [6, 9], scales in [0.002, 0.02].
Quant random_quant(int64_t k, int64_t n) {
    const auto cpu = torch::TensorOptions().device(torch::kCPU);
    Quant t;
    t.q = torch::randint(0, 16, {k, n}, cpu.dtype(torch::kUInt8));
    t.s = (torch::rand({k / kTileK, n}, cpu.dtype(torch::kFloat32)) * 0.018 + 0.002).to(torch::kBFloat16);
    t.z = torch::randint(6, 10, {k / kTileK, n}, cpu.dtype(torch::kInt8));
    return t;
}

// fp32 [k, n] weight: (q - zero) * scale, one scale and zero per 128-row group.
torch::Tensor dequant(const Quant &t, const torch::Device &dev) {
    const auto z = t.z.to(dev, torch::kFloat32).repeat_interleave(kTileK, 0);
    const auto s = t.s.to(dev, torch::kFloat32).repeat_interleave(kTileK, 0);
    return (t.q.to(dev, torch::kFloat32) - z) * s;
}

torch::Tensor bf16(const torch::Tensor &t) { return t.to(torch::kBFloat16).contiguous(); }

// Random bf16 [m, k] input whose storage holds padded_rows(m) rows, as packed_linear() requires.
torch::Tensor padded_input(int64_t m, int64_t k, const torch::Device &dev) {
    return bf16(torch::randn({padded_rows(m), k}, torch::TensorOptions().device(dev)) * 0.5).narrow(0, 0, m);
}

void sync() { (void)hipDeviceSynchronize(); }

// Level 1: packer + HeteroMosaic GEMM/GEMV on full matrices and on the slices that HeteroMega uses.
void test_packed_linear(const torch::Device &dev, hipStream_t stream, const std::vector<int64_t> &ms) {
    struct Case {
        const char *name;
        int64_t k, n;
    };
    const Case cases[] = {{"gate_up slice [4096 x 2*7168]", kHidden, 2 * 7168},
                          {"gate_up slice [4096 x 2*512]", kHidden, 2 * 512},
                          {"gate_up slice [4096 x 2*10240]", kHidden, 2 * 10240},
                          {"down slice [7168 x 4096]", 7168, kHidden},
                          {"down slice [10240 x 4096]", 10240, kHidden},
                          {"down slice [512 x 4096]", 512, kHidden}};
    for (const auto &c : cases) {
        const Quant t = random_quant(c.k, c.n);
        const auto w = dequant(t, dev);
        const auto packed = pack_w4a16(t.q, t.s, t.z).to(dev);
        for (int64_t m : ms) {
            const auto x = padded_input(m, c.k, dev);
            auto y = torch::empty({m, c.n}, x.options());
            packed_linear(y, x, packed, c.k, c.n, stream);
            sync();
            check(std::string("L1 linear ") + c.name + " M=" + std::to_string(m), y, x.to(torch::kFloat32).matmul(w), kLevel);
        }
    }
}

// Level 1: gemv_w4a16 with each K split, and HeteroMosaic's GEMV, on the shapes of the decode step.
void test_gemv(const torch::Device &dev, hipStream_t stream) {
    struct Case {
        const char *name;
        int64_t k, n;
    };
    const Case cases[] = {{"gate_up [4096 x 2*7168]", kHidden, 2 * 7168}, {"down [14336 x 4096]", kInter, kHidden},
                          {"down [7168 x 4096]", 7168, kHidden}, {"down [512 x 4096]", 512, kHidden}};
    for (const auto &c : cases) {
        const Quant t = random_quant(c.k, c.n);
        const auto w = dequant(t, dev);
        const auto packed = pack_w4a16(t.q, t.s, t.z).to(dev);
        const auto x = padded_input(1, c.k, dev);
        const auto ref = x.to(torch::kFloat32).matmul(w);
        auto y = torch::empty({1, c.n}, x.options());
        for (int split : {0, 1, 2, 3, 8}) {
            gemv_w4a16(y, x, packed, c.k, c.n, c.k, c.n, split, stream);
            sync();
            check(std::string("L1 gemv_w4a16 split_k=") + std::to_string(split) + " " + c.name, y, ref, kLevel);
        }
        LinearOptions hm;
        hm.gemv = GemvKernel::kHeteroMosaic;
        packed_linear(y, x, packed, c.k, c.n, stream, hm);
        sync();
        check(std::string("L1 HeteroMosaic GEMV ") + c.name, y, ref, kLevel);
    }
}

// Level 1: the leading [k, n] block of a larger packed matrix, as FfnSliceWeights::prefix() uses it.
// Covers the GEMV, the generic GEMM (K >= 2048) and the small-K GEMM (K <= 1024).
void test_packed_prefix(const torch::Device &dev, hipStream_t stream, const std::vector<int64_t> &ms) {
    const int64_t full_k = 4096, full_n = 1024;
    const Quant t = random_quant(full_k, full_n);
    const auto w = dequant(t, dev);
    const auto packed = pack_w4a16(t.q, t.s, t.z).to(dev);
    LinearOptions opt;
    opt.packed_k = full_k;
    opt.packed_n = full_n;
    const int64_t blocks[][2] = {{2048, 512}, {4096, 256}, {1024, 768}, {512, 1024}};
    for (const auto &b : blocks) {
        const int64_t k = b[0], n = b[1];
        const auto ref_w = w.narrow(0, 0, k).narrow(1, 0, n);
        for (int64_t m : ms) {
            const auto x = padded_input(m, k, dev);
            auto y = torch::empty({m, n}, x.options());
            packed_linear(y, x, packed, k, n, stream, opt);
            sync();
            check("L1 linear prefix [" + std::to_string(k) + " x " + std::to_string(n) + "] of [4096 x 1024] M=" + std::to_string(m), y,
                  x.to(torch::kFloat32).matmul(ref_w), kLevel);
        }
    }
}

// Level 1: gemm_w4a16 on whole matrices and on the leading [k, n] block of a larger packed matrix.
// Its result has one bf16 rounding per output, so it must meet kExact, which HeteroMosaic's GEMM need not.
void test_gemm_w4a16(const torch::Device &dev, hipStream_t stream) {
    struct Case {
        const char *name;
        int64_t k, n, packed_k, packed_n;
    };
    const Case cases[] = {{"gate_up [4096 x 2*7168]", kHidden, 2 * 7168, kHidden, 2 * 7168},
                          {"gate_up [4096 x 2*512]", kHidden, 2 * 512, kHidden, 2 * 512},
                          {"down [7168 x 4096]", 7168, kHidden, 7168, kHidden},
                          {"down [512 x 4096]", 512, kHidden, 512, kHidden},
                          {"attention k [4096 x 1024]", kHidden, 1024, kHidden, 1024},
                          {"prefix gate_up [4096 x 2*1024] of 2*2048", kHidden, 2 * 1024, kHidden, 2 * 2048},
                          {"prefix down [1024 x 4096] of 2048", 1024, kHidden, 2048, kHidden},
                          {"prefix [4096 x 1152] of 1216 (19 tile columns)", kHidden, 1152, kHidden, 1216}};
    for (const auto &c : cases) {
        const Quant t = random_quant(c.packed_k, c.packed_n);
        const auto w = dequant(t, dev).narrow(0, 0, c.k).narrow(1, 0, c.n);
        const auto packed = pack_w4a16(t.q, t.s, t.z).to(dev);
        for (int64_t m : {128L, 512L, 2048L}) {
            const auto x = padded_input(m, c.k, dev);
            const auto ref = x.to(torch::kFloat32).matmul(w);
            auto y = torch::empty({m, c.n}, x.options());
            {
                gemm_w4a16(y, x, packed, c.k, c.n, c.packed_k, c.packed_n, stream);
                sync();
                check(std::string("L1 gemm_w4a16 ") + c.name + " M=" + std::to_string(m), y, ref, kExact);
            }
        }
    }
    // Three GEMMs of one input with GemmInputReuse, as Q, K and V, then one without it on the same stream.
    const Quant a = random_quant(kHidden, 4096), b = random_quant(kHidden, 1024);
    const auto packed_a = pack_w4a16(a.q, a.s, a.z).to(dev), packed_b = pack_w4a16(b.q, b.s, b.z).to(dev);
    const auto x = padded_input(512, kHidden, dev), other = padded_input(512, kHidden, dev);
    auto y_a = torch::empty({512, 4096}, x.options()), y_b = torch::empty({512, 1024}, x.options()), y_c = torch::empty_like(y_b);
    auto y_other = torch::empty_like(y_b);
    {
        GemmInputReuse reuse;
        gemm_w4a16(y_a, x, packed_a, kHidden, 4096, kHidden, 4096, stream);
        gemm_w4a16(y_b, x, packed_b, kHidden, 1024, kHidden, 1024, stream);
        gemm_w4a16(y_c, x, packed_b, kHidden, 1024, kHidden, 1024, stream);
    }
    gemm_w4a16(y_other, other, packed_b, kHidden, 1024, kHidden, 1024, stream);
    sync();
    const auto x_fp = x.to(torch::kFloat32);
    check("L1 gemm_w4a16 reuse first", y_a, x_fp.matmul(dequant(a, dev)), kExact);
    check("L1 gemm_w4a16 reuse second", y_b, x_fp.matmul(dequant(b, dev)), kExact);
    check("L1 gemm_w4a16 reuse third", y_c, x_fp.matmul(dequant(b, dev)), kExact);
    check("L1 gemm_w4a16 after reuse", y_other, other.to(torch::kFloat32).matmul(dequant(b, dev)), kExact);
}

// Level 2: gemm_w4a16_ffn against the fp32 FFN and against its three kernels one after the other, with a prefix of the packed channels too.
void test_gemm_w4a16_ffn(const torch::Device &dev, hipStream_t stream) {
    FfnLayerQuant layer;
    const Quant g = random_quant(kHidden, kInter), u = random_quant(kHidden, kInter), d = random_quant(kInter, kHidden);
    layer.gate_q = g.q, layer.gate_s = g.s, layer.gate_z = g.z;
    layer.up_q = u.q, layer.up_s = u.s, layer.up_z = u.z;
    layer.down_q = d.q, layer.down_s = d.s, layer.down_z = d.z;
    const auto wg = dequant(g, dev), wu = dequant(u, dev), wd = dequant(d, dev);
    LinearOptions gate_up_opt, down_opt;
    gate_up_opt.gemm = down_opt.gemm = GemmKernel::kHeteroMega;
    for (const int64_t stored : {6144L, 7936L}) {
        const auto w = pack_ffn_slice(layer, 0, stored, GateUpLayout::kInterleaved).to(dev);
        for (const int64_t n : {stored, stored - 512}) {
            for (const int64_t m : {128L, 512L}) {
                const auto x = padded_input(m, kHidden, dev);
                const auto xf = x.to(torch::kFloat32);
                const auto ref = (torch::silu(xf.matmul(wg.narrow(1, 0, n))) * xf.matmul(wu.narrow(1, 0, n))).matmul(wd.narrow(0, 0, n));
                auto y = torch::empty({m, kHidden}, x.options()), y_steps = torch::empty_like(y);
                gemm_w4a16_ffn(y, x, w.gate_up, w.down, kHidden, n, stored, stream);
                auto gate_up = torch::empty({m, 2 * n}, x.options()), act = torch::empty({m, n}, x.options());
                gate_up_opt.packed_n = 2 * stored;
                down_opt.packed_k = stored;
                packed_linear(gate_up, x, w.gate_up, kHidden, 2 * n, stream, gate_up_opt);
                silu_mul(gate_up, act, stream, GateUpLayout::kInterleaved);
                packed_linear(y_steps, act, w.down, n, kHidden, stream, down_opt);
                sync();
                const std::string tag = " stored=" + std::to_string(stored) + " n=" + std::to_string(n) + " M=" + std::to_string(m);
                check("L2 gemm_w4a16_ffn vs fp32 FFN" + tag, y, ref, kLevel);
                check("L2 gemm_w4a16_ffn vs three kernels" + tag, y, y_steps.to(torch::kFloat32), kLevel);
            }
        }
    }
}

// Level 1: residual_norm_bf16 against the same math in fp32, in row order and through the next GEMM, with 2, 1 and 0 addends.
void test_residual_norm(const torch::Device &dev, hipStream_t stream) {
    const int64_t m = 512;
    const float eps = 1e-5f;
    const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);
    const auto random = [&](int64_t rows, double scale) { return (torch::randn({rows, kHidden}, torch::TensorOptions().device(dev)) * scale).to(torch::kBFloat16); };
    const auto weight = (torch::randn({kHidden}, torch::TensorOptions().device(dev)) * 0.2 + 1.0).to(torch::kBFloat16);
    const Quant w = random_quant(kHidden, 1024);
    const auto packed = pack_w4a16(w.q, w.s, w.z).to(dev);
    const auto wf = dequant(w, dev);
    for (int addends = 2; addends >= 0; --addends) {
        auto x = random(m, 1.0);
        const auto a = random(m, 1.0), b = random(m, 1.0);
        const auto expected_x = (addends == 0 ? x.to(torch::kFloat32) : addends == 1 ? x.to(torch::kFloat32) + a.to(torch::kFloat32) : x.to(torch::kFloat32) + a.to(torch::kFloat32) + b.to(torch::kFloat32));
        const auto rounded_x = expected_x.to(torch::kBFloat16).to(torch::kFloat32);
        const auto expected = rounded_x * (rounded_x.square().mean(-1, true) + eps).rsqrt() * weight.to(torch::kFloat32);
        auto normed = torch::empty({m, kHidden}, bf16), unwritten = torch::empty_like(normed);
        auto y_row = torch::empty({m, int64_t(1024)}, bf16), y_direct = torch::empty_like(y_row);
        {
            GemmInputReuse input;
            residual_norm_bf16(reinterpret_cast<uint16_t *>(x.data_ptr()), addends > 0 ? reinterpret_cast<const uint16_t *>(a.data_ptr()) : nullptr,
                               addends > 1 ? reinterpret_cast<const uint16_t *>(b.data_ptr()) : nullptr, reinterpret_cast<const uint16_t *>(weight.data_ptr()),
                               reinterpret_cast<uint16_t *>(normed.data_ptr()), normed.data_ptr(), input, m, kHidden, kHidden, kHidden, kHidden, kHidden, eps, stream);
            gemm_w4a16(y_row, normed, packed, kHidden, 1024, kHidden, 1024, stream);
            TORCH_CHECK(input.hits() == 1, "the GEMM did not use the prepared input");
        }
        auto x_again = random(m, 1.0);
        {
            GemmInputReuse input;
            sync();
            x_again.copy_(x);
            residual_norm_bf16(reinterpret_cast<uint16_t *>(x_again.data_ptr()), nullptr, nullptr, reinterpret_cast<const uint16_t *>(weight.data_ptr()), nullptr, unwritten.data_ptr(), input, m,
                               kHidden, kHidden, kHidden, kHidden, kHidden, eps, stream);
            gemm_w4a16(y_direct, unwritten, packed, kHidden, 1024, kHidden, 1024, stream);
            TORCH_CHECK(input.hits() == 1, "the GEMM did not use the prepared input");
        }
        sync();
        const std::string tag = " addends=" + std::to_string(addends);
        check("L1 residual_norm_bf16 x" + tag, x, expected_x, kExact);
        check("L1 residual_norm_bf16 normed" + tag, normed, expected, kExact);
        check("L1 residual_norm_bf16 through a GEMM" + tag, y_row, expected.to(torch::kBFloat16).to(torch::kFloat32).matmul(wf), kExact);
        check("L1 residual_norm_bf16 without row order" + tag, y_direct, expected.to(torch::kBFloat16).to(torch::kFloat32).matmul(wf), kExact);
    }
}

// Level 1: sdpa_bf16 against a reference with an explicit mask in fp32, and which keys PyTorch's flash attention shows to a later chunk.
void test_sdpa_bf16(const torch::Device &dev, hipStream_t stream) {
    const int64_t q_heads = 32, kv_heads = 8, dim = 128, capacity = 1024;
    const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);
    torch::manual_seed(11);
    const auto q_all = (torch::randn({1, capacity, q_heads, dim}, torch::TensorOptions().dtype(torch::kFloat32)) * 2.0f).to(dev).to(torch::kBFloat16);
    const auto k_cache = (torch::randn({1, kv_heads, capacity, dim}, torch::TensorOptions().dtype(torch::kFloat32)) * 2.0f).to(dev).to(torch::kBFloat16);
    const auto v_cache = torch::randn({1, kv_heads, capacity, dim}, torch::TensorOptions().dtype(torch::kFloat32)).to(dev).to(torch::kBFloat16);
    auto *tiles = sdpa_v_tiles(-1, kv_heads, capacity, dim);
    sdpa_store_v(tiles, reinterpret_cast<const uint16_t *>(v_cache.data_ptr()), kv_heads, capacity, 0, capacity, stream);
    sync();
    const auto reference = [&](const torch::Tensor &q, int64_t keys, int64_t offset) {
        // q: [1, rows, heads, 128]. Returns [rows, heads * 128].
        const auto qf = q.to(torch::kFloat32).transpose(1, 2);
        const auto kf = k_cache.narrow(2, 0, keys).to(torch::kFloat32).repeat_interleave(q_heads / kv_heads, 1);
        const auto vf = v_cache.narrow(2, 0, keys).to(torch::kFloat32).repeat_interleave(q_heads / kv_heads, 1);
        auto s = qf.matmul(kf.transpose(-2, -1)) / std::sqrt(static_cast<float>(dim));
        const auto rows = torch::arange(q.size(1), dev).unsqueeze(1), key = torch::arange(keys, dev).unsqueeze(0);
        s = s.masked_fill(key > rows + offset, -std::numeric_limits<float>::infinity());
        const auto o = torch::softmax(s, -1).matmul(vf);
        return o.transpose(1, 2).contiguous().view({q.size(1), q_heads * dim});
    };
    struct Case {
        const char *name;
        int64_t first_row, rows, keys, offset;
    };
    const Case cases[] = {{"chunk 0", 0, 512, 512, 0}, {"chunk 1", 512, 512, 1024, 512}, {"small", 0, 32, 64, 0}};
    for (const auto &c : cases) {
        const auto q = q_all.narrow(1, c.first_row, c.rows).contiguous();
        auto out = torch::empty({c.rows, q_heads * dim}, bf16);
        sdpa_bf16(reinterpret_cast<uint16_t *>(out.data_ptr()), q_heads * dim, reinterpret_cast<const uint16_t *>(q.data_ptr()), q_heads * dim,
                  reinterpret_cast<const uint16_t *>(k_cache.data_ptr()), tiles, c.rows, q_heads, kv_heads, capacity, c.keys, c.offset, stream);
        sync();
        check(std::string("L1 sdpa_bf16 ") + c.name, out, reference(q, c.keys, c.offset), Tol{0.5, 0.05, 0.998});
    }
    // The output as the prepared input of the next GEMM gives the result of the output through memory.
    const Quant w = random_quant(q_heads * dim, q_heads * dim);
    const auto packed = pack_w4a16(w.q, w.s, w.z).to(dev);
    for (const auto &c : cases) {
        if (c.rows % 128 != 0) {
            continue;
        }
        const auto q = q_all.narrow(1, c.first_row, c.rows).contiguous();
        auto out = torch::empty({c.rows, q_heads * dim}, bf16), unwritten = torch::empty_like(out);
        auto y_memory = torch::empty_like(out), y_direct = torch::empty_like(out);
        sdpa_bf16(reinterpret_cast<uint16_t *>(out.data_ptr()), q_heads * dim, reinterpret_cast<const uint16_t *>(q.data_ptr()), q_heads * dim,
                  reinterpret_cast<const uint16_t *>(k_cache.data_ptr()), tiles, c.rows, q_heads, kv_heads, capacity, c.keys, c.offset, stream);
        gemm_w4a16(y_memory, out, packed, q_heads * dim, q_heads * dim, q_heads * dim, q_heads * dim, stream);
        {
            GemmInputReuse input;
            sdpa_bf16(reinterpret_cast<uint16_t *>(unwritten.data_ptr()), q_heads * dim, reinterpret_cast<const uint16_t *>(q.data_ptr()), q_heads * dim,
                      reinterpret_cast<const uint16_t *>(k_cache.data_ptr()), tiles, c.rows, q_heads, kv_heads, capacity, c.keys, c.offset, stream, &input);
            gemm_w4a16(y_direct, unwritten, packed, q_heads * dim, q_heads * dim, q_heads * dim, q_heads * dim, stream);
            TORCH_CHECK(input.hits() == 1, "the GEMM did not use the prepared input");
        }
        sync();
        check(std::string("L1 sdpa_bf16 to gemm_w4a16 ") + c.name, y_direct, y_memory.to(torch::kFloat32), kExact);
    }
    // The torch calls of the model: scaled_dot_product_attention for the first chunk, and flash attention with is_causal for later ones.
    const auto q1 = q_all.narrow(1, 512, 512).contiguous();
    const auto q0 = q_all.narrow(1, 0, 512).contiguous();
    const auto flash = std::get<0>(at::_scaled_dot_product_flash_attention(q1.transpose(1, 2), k_cache, v_cache, 0.0, true, false, c10::nullopt));
    const auto got = flash.transpose(1, 2).contiguous().view({512, q_heads * dim}).to(torch::kFloat32);
    for (int64_t offset : {int64_t(0), int64_t(512)}) {
        const auto ref = reference(q1, 1024, offset).to(torch::kFloat32);
        std::printf("flash attention, chunk 1, rows see keys up to row + %ld: relative L2 %.4f\n", static_cast<long>(offset),
               ((got - ref).norm() / ref.norm()).item<float>());
    }
    const auto k0 = k_cache.narrow(2, 0, 512).repeat_interleave(q_heads / kv_heads, 1), v0 = v_cache.narrow(2, 0, 512).repeat_interleave(q_heads / kv_heads, 1);
    const auto sdpa = torch::scaled_dot_product_attention(q0.transpose(1, 2), k0, v0, c10::nullopt, 0.0, true, std::nullopt, false);
    const auto got0 = sdpa.transpose(1, 2).contiguous().view({512, q_heads * dim}).to(torch::kFloat32);
    const auto ref0 = reference(q0, 512, 0).to(torch::kFloat32);
    std::printf("scaled_dot_product_attention, chunk 0: relative L2 %.4f\n", ((got0 - ref0).norm() / ref0.norm()).item<float>());
}

// Level 1: slice_packed_ffn (tile gather from packed whole matrices) gives the bytes of pack_ffn_slice.
// hidden / 64 = 18 and inter / 64 = 46 tile columns are not multiples of 8, so the column groups differ in size.
void test_slice_packed(const torch::Device &dev) {
    const int64_t hidden = 1152, inter = 2944;
    FfnLayerQuant layer;
    const Quant g = random_quant(hidden, inter), u = random_quant(hidden, inter), d = random_quant(inter, hidden);
    layer.gate_q = g.q, layer.gate_s = g.s, layer.gate_z = g.z;
    layer.up_q = u.q, layer.up_s = u.s, layer.up_z = u.z;
    layer.down_q = d.q, layer.down_s = d.s, layer.down_z = d.z;
    const auto pg = pack_w4a16(g.q, g.s, g.z).to(dev), pu = pack_w4a16(u.q, u.s, u.z).to(dev), pd = pack_w4a16(d.q, d.s, d.z).to(dev);
    const int64_t ranges[][2] = {{0, inter}, {0, 512}, {512, 1920}, {1280, inter}, {128, 256}};
    for (const auto &rg : ranges) {
        for (auto layout : {GateUpLayout::kBlocked, GateUpLayout::kInterleaved}) {
            const auto got = slice_packed_ffn(pg, pu, pd, hidden, inter, rg[0], rg[1], layout);
            const auto ref = pack_ffn_slice(layer, rg[0], rg[1], layout);
            const std::string tag = " [" + std::to_string(rg[0]) + ", " + std::to_string(rg[1]) + ")" +
                                    (layout == GateUpLayout::kBlocked ? " blocked" : " interleaved");
            check_equal("L1 slice_packed_ffn gate_up" + tag, got.gate_up, ref.gate_up);
            check_equal("L1 slice_packed_ffn down" + tag, got.down, ref.down);
        }
    }
}

// Level 1: SiLU-and-multiply.
void test_silu_mul(const torch::Device &dev, hipStream_t stream) {
    for (int64_t m : {1L, 37L, 2048L}) {
        const int64_t n = 7168;
        const auto gu = bf16(torch::randn({m, 2 * n}, torch::TensorOptions().device(dev)) * 4.0);
        auto act = torch::empty({m, n}, gu.options());
        silu_mul(gu, act, stream);
        sync();
        const auto g = gu.narrow(1, 0, n).to(torch::kFloat32);
        const auto u = gu.narrow(1, n, n).to(torch::kFloat32);
        check("L1 silu_mul M=" + std::to_string(m), act, torch::silu(g) * u, kExact);

        // Interleaved: per 64 channels, 64 gate columns then 64 up columns.
        auto act_i = torch::empty({m, n}, gu.options());
        silu_mul(gu, act_i, stream, GateUpLayout::kInterleaved);
        sync();
        const auto pairs = gu.to(torch::kFloat32).view({m, n / 64, 2, 64});
        const auto gi = pairs.select(2, 0).reshape({m, n}), ui = pairs.select(2, 1).reshape({m, n});
        check("L1 silu_mul interleaved M=" + std::to_string(m), act_i, torch::silu(gi) * ui, kExact);
    }
}

// Level 1: reducers R-GPU and R-CPU, bf16 and fp32 partials.
void test_residual_add2(const torch::Device &dev, hipStream_t stream) {
    for (int64_t m : {1L, 2048L}) {
        for (auto pdt : {torch::kBFloat16, torch::kFloat32}) {
            const auto opt = torch::TensorOptions().device(dev);
            const auto x0 = bf16(torch::randn({m, kHidden}, opt));
            const auto yg = bf16(torch::randn({m, kHidden}, opt) * 0.3);
            const auto yn = (torch::randn({m, kHidden}, opt) * 0.3).to(pdt).contiguous();
            const auto ref = x0.to(torch::kFloat32) + (yg.to(torch::kFloat32) + yn.to(torch::kFloat32));
            const std::string tag = std::string(" M=") + std::to_string(m) + (pdt == torch::kFloat32 ? " y_n=fp32" : " y_n=bf16");

            auto xg = x0.clone();
            residual_add2(xg, yg, yn, stream);
            sync();
            check("L1 residual_add2 GPU" + tag, xg, ref, kExact);

            auto xc = x0.cpu();
            residual_add2_cpu(xc, yg.cpu(), yn.cpu());
            check("L1 residual_add2 CPU" + tag, xc, ref, kExact);
        }
    }
}

// Level 2: Y_G + Y_N == full FFN. The NPU slice runs on the GPU here as a stand-in.
// x is not padded, so M = 37 also tests the copy into GpuFfnBuffers::x_pad.
void test_ffn_split(const torch::Device &dev, hipStream_t stream, const std::vector<int64_t> &ms,
                    const std::vector<int64_t> &splits) {
    FfnLayerQuant layer;
    const Quant g = random_quant(kHidden, kInter), u = random_quant(kHidden, kInter), d = random_quant(kInter, kHidden);
    layer.gate_q = g.q, layer.gate_s = g.s, layer.gate_z = g.z;
    layer.up_q = u.q, layer.up_s = u.s, layer.up_z = u.z;
    layer.down_q = d.q, layer.down_s = d.s, layer.down_z = d.z;
    const auto wg = dequant(g, dev), wu = dequant(u, dev), wd = dequant(d, dev);

    const int64_t max_m = *std::max_element(ms.begin(), ms.end());
    const auto full = pack_ffn_slice(layer, 0, kInter).to(dev);
    auto full_buf = GpuFfnBuffers::allocate(max_m, kInter, kHidden, dev);
    // All channels packed once with interleaved gate/up. Each split's GPU part is a prefix of it.
    const auto all = pack_ffn_slice(layer, 0, kInter, GateUpLayout::kInterleaved).to(dev);
    auto all_buf = GpuFfnBuffers::allocate(max_m, kInter, kHidden, dev);
    // The model path: GpuFfnModel built from the packed whole matrices, as HeteroMosaic's QuantizedLinear holds them.
    GpuFfnModel model;
    model.add_layer(pack_w4a16(g.q, g.s, g.z).to(dev), pack_w4a16(u.q, u.s, u.z).to(dev), pack_w4a16(d.q, d.s, d.z).to(dev), kHidden,
                    kInter);

    for (int64_t s : splits) {
        const auto gpu_part = pack_ffn_slice(layer, 0, s).to(dev);
        const auto npu_part = pack_ffn_slice(layer, s, kInter).to(dev);
        auto gpu_buf = GpuFfnBuffers::allocate(max_m, s, kHidden, dev);
        auto npu_buf = GpuFfnBuffers::allocate(max_m, kInter - s, kHidden, dev);

        for (int64_t m : ms) {
            const auto x = bf16(torch::randn({m, kHidden}, torch::TensorOptions().device(dev)) * 0.5);
            const auto xf = x.to(torch::kFloat32);
            const auto ref = (torch::silu(xf.matmul(wg)) * xf.matmul(wu)).matmul(wd);

            auto y_g = torch::empty({m, kHidden}, x.options());
            auto y_n = torch::empty({m, kHidden}, x.options());
            auto y_full = torch::empty({m, kHidden}, x.options());
            auto y_prefix = torch::empty({m, kHidden}, x.options());
            gpu_ffn_slice_forward(gpu_part, x, gpu_buf, y_g, stream);
            gpu_ffn_slice_forward(npu_part, x, npu_buf, y_n, stream);
            gpu_ffn_slice_forward(full, x, full_buf, y_full, stream);
            gpu_ffn_slice_forward(all.prefix(s), x, all_buf, y_prefix, stream);

            // Reducer path: zero residual, so x_out == Y_G + Y_N.
            auto x_out = torch::zeros({m, kHidden}, x.options());
            residual_add2(x_out, y_g, y_n, stream);
            auto x_out_prefix = torch::zeros({m, kHidden}, x.options());
            residual_add2(x_out_prefix, y_prefix, y_n, stream);

            // GpuFfnModel with s channels, then with s for this M's phase only (s_prefill for M > 1, s_decode for M == 1).
            auto y_model = torch::empty({m, kHidden}, x.options());
            auto y_phase = torch::empty({m, kHidden}, x.options());
            model.set_split(s, s);
            model.forward(0, x, y_model, stream);
            model.set_split(m > 1 ? s : kInter, m > 1 ? kInter : s);
            model.forward(0, x, y_phase, stream);
            sync();

            const std::string tag = " s=" + std::to_string(s) + " M=" + std::to_string(m);
            check("L2 split Y_G + Y_N vs fp32 FFN" + tag, x_out, ref, kLevel);
            check("L2 prefix split Y_G + Y_N vs fp32 FFN" + tag, x_out_prefix, ref, kLevel);
            check("L2 prefix Y_G vs separately packed Y_G" + tag, y_prefix, y_g, kExact);
            check("L2 unsplit GPU FFN vs fp32 FFN" + tag, y_full, ref, kLevel);
            check_equal("L2 GpuFfnModel vs prefix Y_G" + tag, y_model, y_prefix);
            check_equal("L2 GpuFfnModel phase split vs prefix Y_G" + tag, y_phase, y_prefix);
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    torch::manual_seed(0);
    const torch::Device dev(torch::kCUDA, 0);
    const hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    // On an APU, "device memory" can be a small BIOS carve-out. The full test needs about 2 GiB.
    hipDeviceProp_t prop;
    size_t free_bytes = 0, total_bytes = 0;
    if (hipGetDeviceProperties(&prop, 0) == hipSuccess && hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
        std::printf("Device: %s (%s), %.1f GiB free of %.1f GiB\n", prop.name, prop.gcnArchName, free_bytes / 1073741824.0,
                    total_bytes / 1073741824.0);
    }

    const std::vector<int64_t> ms = quick ? std::vector<int64_t>{1, 37, 128} : std::vector<int64_t>{1, 37, 128, 512, 2048};
    const std::vector<int64_t> splits = quick ? std::vector<int64_t>{7168} : std::vector<int64_t>{512, 7168, 10240, 13824};

    test_packed_linear(dev, stream, ms);
    test_gemv(dev, stream);
    test_packed_prefix(dev, stream, ms);
    test_gemm_w4a16(dev, stream);
    test_gemm_w4a16_ffn(dev, stream);
    test_residual_norm(dev, stream);
    test_sdpa_bf16(dev, stream);
    test_slice_packed(dev);
    test_silu_mul(dev, stream);
    test_residual_add2(dev, stream);
    test_ffn_split(dev, stream, ms, splits);

    std::printf("%s (%d failed)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
